"""5단계 학습: 정글러 하나만 PPO로 학습, 나머지 9명은 규칙 봇.

    python -m sim.teacher                                 :: 규칙 봇 모방으로 시작 정책 만들기
    python -m sim.train --init models/stage5/policy_bc.npz --iters 300 --workers 4   :: models/stage5/ 에 정책·학습 기록 저장
    python -m sim.train --iters 100 --resume              :: 마지막 정책에서 이어서

경기마다 학습 정글러의 팀(블루/레드)을 번갈아 둔다. 보상 = 결정 사이 예측 승률 변화(합 = 20분 예측 승률 − 시작 승률).
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .agent import N_MAIN, ACTION_NAMES, obs_size
from .data import ROOT
from .ppo import Policy, Adam, clip_grads, gae, masked_log_softmax, mlp_forward
from .rollout import play
from .teacher import bc_grads

OUT = os.path.join(ROOT, "models", "stage5")

_policy = None


def _worker_init(obs_dim):
    global _policy
    _policy = Policy(obs_dim)


def _worker_run(args):
    state, jobs, wseed = args
    _policy.load(state)
    _policy.rng = np.random.default_rng(wseed)
    out = []
    for seed, team in jobs:
        r = play(seed, team, _policy.act)
        traj = r.pop("traj")
        r["obs"] = np.array([t["obs"] for t in traj], dtype=np.float32)
        r["m"] = np.array([t["m"] for t in traj])
        r["a"] = np.array([t["a"] for t in traj])
        r["logp"] = np.array([t["logp"] for t in traj])
        r["v"] = np.array([t["v"] for t in traj])
        r["r"] = np.array([t["r"] for t in traj])
        out.append(r)
    return out


def collect(pool, policy, seeds, workers, it):
    jobs = [(s, 100 if s % 2 == 0 else 200) for s in seeds]
    chunks = [jobs[i::workers] for i in range(workers)]
    state = policy.state()
    res = pool.map(_worker_run, [(state, ch, it * 1000 + i) for i, ch in enumerate(chunks)])
    return [g for r in res for g in r]


def update(policy, opt_pi, opt_vf, games, cfg):
    X, M, A, LP, ADV, RET = [], [], [], [], [], []
    for g in games:
        rew = g["r"] * cfg.reward_scale
        adv, ret = gae(rew, g["v"], cfg.gamma, cfg.lam)
        X.append(g["obs"]); M.append(g["m"]); A.append(g["a"]); LP.append(g["logp"]); ADV.append(adv); RET.append(ret)
    X = np.concatenate(X).astype(np.float64)
    M, A, LP, ADV, RET = (np.concatenate(v) for v in (M, A, LP, ADV, RET))
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    n = len(X)
    rng = np.random.default_rng(len(X))
    infos = []
    for _ in range(cfg.epochs):
        idx = rng.permutation(n)
        for s in range(0, n, cfg.minibatch):
            b = idx[s:s + cfg.minibatch]
            gpi, gvf, info = policy.loss_grads(X[b], M[b], A[b], LP[b], ADV[b], RET[b], clip=cfg.clip, ent_coef=cfg.ent_coef)
            if cfg.real is not None:
                # 갈래 B: 실제 천상계 정글러가 그 상태에서 한 행동을 따라 하는 손실을 함께 줄인다(실제 표본 상태에서만)
                Xr, Mr, Ar = cfg.real
                r = rng.integers(0, len(Xr), min(256, len(Xr)))
                gbc, bc_loss, bc_acc = bc_grads(policy, Xr[r], Mr[r], Ar[r], heads=(0,))   # 와드·강타 선택은 정답 없음
                gpi = [g + cfg.real_coef * h for g, h in zip(gpi, gbc)]
                info = info | {"bc_loss": bc_loss, "bc_acc": bc_acc}
            gpi, _ = clip_grads(gpi, cfg.max_grad_norm)
            gvf, _ = clip_grads(gvf, cfg.max_grad_norm)
            opt_pi.step(policy.pi, gpi)
            opt_vf.step(policy.vf, gvf)
            infos.append(info)
        if np.mean([i["approx_kl"] for i in infos[-max(1, n // cfg.minibatch):]]) > cfg.target_kl:
            break
    return {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}, n


def revive_rare(policy, games, min_legal=0.05, max_shift=10.0, sample=20000):
    """굳은 행동 방지: 이번 반복 표본에서 주 행동마다 고를 수 있을 때의 평균 확률을 재고,
    policy.floor보다 낮으면 그 행동의 출력 편향만 log(floor / 평균)만큼 올린다(다른 행동 로짓은 그대로).
    상태마다 바닥을 두는 손실은 상황에 따라 드문 행동(먼 캠프 등)까지 올려 분포 전체를 평평하게 만들어 쓰지 않음."""
    X = np.concatenate([g["obs"] for g in games]).astype(np.float64)
    M = np.concatenate([g["m"] for g in games])[:, :N_MAIN]
    if len(X) > sample:
        i = np.random.default_rng(len(X)).choice(len(X), sample, replace=False)
        X, M = X[i], M[i]
    z, _ = mlp_forward(policy.pi, X)
    _, p = masked_log_softmax(z[:, :N_MAIN], M)
    legal = M.mean(0)
    mean_p = (p * M).sum(0) / np.maximum(M.sum(0), 1)
    n = 0
    for a in range(N_MAIN):
        if legal[a] >= min_legal and mean_p[a] < policy.floor:
            policy.pi[-1][a] += min(max_shift, float(np.log(policy.floor / max(mean_p[a], 1e-12))))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=96, help="반복마다 경기 수")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", dest="ent_coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--reward-scale", dest="reward_scale", type=float, default=10.0)
    ap.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=0.5)
    ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.03)
    ap.add_argument("--seed0", type=int, default=100000, help="학습 경기 시드 시작(평가 시드 0~9999와 겹치지 않게)")
    ap.add_argument("--floor", type=float, default=None,
                    help="주 행동 평균 확률 바닥(예: 0.002). 반복마다 고를 수 있을 때 평균 확률이 이보다 낮은 행동의 출력 편향을 올려 바닥에 맞춘다(굳은 행동 방지)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="시작 정책(예: models/stage5/policy_bc.npz, sim/teacher.py 모방 학습 결과)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--real", help="실제 정글러 행동 표본(sim/imitate.py가 만든 npz: X, M, A). 주면 PPO와 함께 모방 손실을 줄인다")
    ap.add_argument("--real-coef", dest="real_coef", type=float, default=1.0, help="모방 손실 기울기 배율")
    cfg = ap.parse_args()
    os.makedirs(cfg.out, exist_ok=True)
    real_path = cfg.real
    if cfg.real:
        d = np.load(cfg.real)
        cfg.real = (d["X"].astype(np.float64), d["M"], d["A"])
    latest = os.path.join(cfg.out, "policy_latest.npz")
    log_path = os.path.join(cfg.out, "train_log.csv")
    obs_dim = obs_size()
    policy = Policy.from_file(cfg.init) if cfg.init else Policy(obs_dim, seed=0)
    if cfg.floor is not None:
        policy.floor = cfg.floor
    start_it = 0
    if cfg.resume and os.path.exists(latest):
        policy = Policy.from_file(latest)
        with open(log_path, encoding="utf-8") as f:
            start_it = sum(1 for _ in f) - 1
    opt_pi, opt_vf = Adam(policy.pi, cfg.lr), Adam(policy.vf, cfg.lr)
    with open(os.path.join(cfg.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(cfg) | {"real": real_path, "obs_dim": obs_dim, "actions": ACTION_NAMES}, f, ensure_ascii=False, indent=1)
    new_log = not (cfg.resume and os.path.exists(log_path))
    logf = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    fields = ["iter", "games", "samples", "phi_mean", "cs10", "gold10", "k10", "d10", "dragons", "ganks", "invade_camps",
              "lanefarm", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "bc_loss", "bc_acc", "revived", "sec"]
    w = csv.DictWriter(logf, fieldnames=fields)
    if new_log:
        w.writeheader()
    with Pool(cfg.workers, initializer=_worker_init, initargs=(obs_dim,)) as pool:
        for it in range(start_it, start_it + cfg.iters):
            t0 = time.time()
            seeds = range(cfg.seed0 + it * cfg.games, cfg.seed0 + (it + 1) * cfg.games)
            games = collect(pool, policy, seeds, cfg.workers, it)
            info, n = update(policy, opt_pi, opt_vf, games, cfg)
            if policy.floor > 0:
                info["revived"] = revive_rare(policy, games)
            st = lambda k: float(np.mean([g["stats"][k] for g in games]))
            row = {"iter": it, "games": len(games), "samples": n, "phi_mean": round(float(np.mean([g["phi"] for g in games])), 4),
                   "cs10": st("cs10"), "gold10": round(st("gold10")), "k10": round(st("k10"), 2), "d10": round(st("d10"), 2),
                   "dragons": round(st("dragons"), 2), "ganks": round(st("ganks"), 2), "invade_camps": round(st("invade_camps"), 2),
                   "lanefarm": round(st("lanefarm"), 2), **{k: round(v, 4) for k, v in info.items()}, "sec": round(time.time() - t0, 1)}
            w.writerow(row)
            logf.flush()
            policy.save(latest)
            if it % 25 == 0:
                policy.save(os.path.join(cfg.out, f"policy_{it:04d}.npz"))
            counts = np.sum([g["counts"] for g in games], axis=0)
            top = ", ".join(f"{ACTION_NAMES[i]} {c}" for i, c in sorted(enumerate(counts), key=lambda x: -x[1])[:6])
            print(f"[{it}] Φ {row['phi_mean']:.3f} CS10 {row['cs10']:.1f} 골드10 {row['gold10']} K/D10 {row['k10']}/{row['d10']} "
                  f"드래곤 {row['dragons']} 되살림 {row.get('revived', 0)} 엔트로피 {row['entropy']:.2f} kl {row['approx_kl']:.3f} {row['sec']}s | {top}", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
