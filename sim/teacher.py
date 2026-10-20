"""5단계 시작점: 규칙 봇 정글러를 흉내 내는 정책(모방 학습)을 먼저 만들고 PPO는 거기서 출발한다.

    python -m sim.teacher --games 3000 --epochs 30     :: → models/stage5/policy_bc.npz
    python -m sim.train --init models/stage5/policy_bc.npz ...

교사 = 학습 정글러와 같은 행동 인터페이스(AgentJungler)로 규칙 봇의 선택(JunglerBot.choose)을 행동 번호로 바꿔 실행하는 봇.
교사로 돈 경기의 Φ가 규칙 봇 그대로의 Φ와 비슷하면, 행동 인터페이스가 규칙 봇이 하던 운영을 표현할 수 있다는 확인도 된다.
무작위 정책에서 PPO를 시작하면 첫 캠프를 끝까지 잡는 것조차 우연에 맡겨야 해서 학습이 매우 느렸다(반복 15회에 10분 CS 4).
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from .agent import A_CAMP, A_INVADE, A_SCUTTLE, A_GANK, A_OBJ, A_RECALL, CAMP_KINDS, LANES, OBJS, AgentJungler, obs_size
from .bots import JunglerBot
from .data import ROOT
from .game import Game, SIDE
from .ppo import Policy, Adam, clip_grads, HEADS, masked_log_softmax, mlp_forward, mlp_backward
from .rollout import winprob, jungler_stats


class Teacher:
    def __init__(self):
        self.bot = None
        self.violations = 0

    def action_of(self, choice):
        b = self.bot
        team = b.c.team
        kind = choice[0]
        if kind == "recall":
            return A_RECALL
        if kind == "gank":
            return A_GANK + LANES.index(choice[2]["lane"])
        if kind == "objective":
            return A_OBJ + OBJS.index(choice[2]["name"])
        cid = choice[2]["camp"]
        if cid.startswith("river_"):
            return A_SCUTTLE + (0 if "top" in cid else 1)
        side, k = cid.split("_", 1)
        return (A_CAMP if side == SIDE[team] else A_INVADE) + CAMP_KINDS.index(k)

    def __call__(self, obs, m, mw, ms):
        b = self.bot
        g, c = b.g, b.c
        if c.task is not None and b.cur_action is not None and m[b.cur_action] and b.still_valid(c.task):
            a = b.cur_action
        else:
            for _ in range(20):
                choice = JunglerBot.choose(b)
                if choice[0] != "skip":
                    break
            a = self.action_of(choice)
            b.extra_kw = {"route": True} if choice[0] == "camp" and choice[2].get("route") else {}
        s = 0 if ms[0] else 1
        if a < A_GANK:
            s = 0 if JunglerBot.use_smite(b, g.camps[b.action_target(a)]) else 1
        if not m[a] or not ms[s]:
            self.violations += 1
        return a, 0, s, {"obs": obs, "m": np.concatenate([m, mw, ms]), "a": np.array([a, 0, s])}


def play_teacher(seed, team):
    wp = winprob()
    t = Teacher()

    def make(g, c):
        t.bot = AgentJungler(g, c, t, wp, keep_route=True)
        return t.bot

    g = Game(seed=seed, jungler_override={team: make}).run()
    phi = t.bot.finish_episode()
    return {"seed": seed, "team": team, "phi": phi, "traj": t.bot.traj, "violations": t.violations,
            "stats": jungler_stats(g, team)}


def _job(args):
    return play_teacher(*args)


def bc_grads(policy, X, M, A):
    z, acts = mlp_forward(policy.pi, X)
    n = len(X)
    gz = np.zeros_like(z)
    loss, correct = 0.0, 0
    use_smite = (A[:, 0] < A_GANK).astype(float)
    for k, (a, b) in enumerate(HEADS):
        lp, p = masked_log_softmax(z[:, a:b], M[:, a:b])
        w = use_smite if k == 2 else np.ones(n)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), A[:, k]] = 1
        loss += -(w * lp[np.arange(n), A[:, k]]).mean()
        gz[:, a:b] = (p - onehot) * (w / n)[:, None]
        if k == 0:
            correct = int((p.argmax(1) == A[:, 0]).sum())
    return mlp_backward(policy.pi, acts, gz), loss, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "stage5", "policy_bc.npz"))
    a = ap.parse_args()
    jobs = [(s, 100 if s % 2 == 0 else 200) for s in range(a.seed0, a.seed0 + a.games)]
    with Pool(a.workers) as pool:
        games = pool.map(_job, jobs, chunksize=16)
    print(f"교사 경기 {len(games)}판: Φ 평균 {np.mean([g['phi'] for g in games]):.3f}, 10분 정글 CS {np.mean([g['stats']['cs10'] for g in games]):.1f}, "
          f"마스크 위반 {sum(g['violations'] for g in games)}건")
    X = np.concatenate([[t["obs"] for t in g["traj"]] for g in games]).astype(np.float64)
    M = np.concatenate([[t["m"] for t in g["traj"]] for g in games])
    A = np.concatenate([[t["a"] for t in g["traj"]] for g in games])
    # 가치망도 교사 경기의 반환값(남은 Φ 변화의 합 × 보상 배율 10)으로 미리 맞춘다
    R = np.concatenate([np.cumsum([t["r"] for t in g["traj"]][::-1])[::-1] * 10 for g in games])
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    val, tr = idx[: n // 10], idx[n // 10:]
    pol = Policy(X.shape[1], seed=0)
    opt, optv = Adam(pol.pi, 1e-3), Adam(pol.vf, 1e-3)
    rng = np.random.default_rng(1)
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), 1024):
            b = perm[s:s + 1024]
            g, loss, acc = bc_grads(pol, X[b], M[b], A[b])
            g, _ = clip_grads(g, 1.0)
            opt.step(pol.pi, g)
            v, vacts = mlp_forward(pol.vf, X[b])
            gv = mlp_backward(pol.vf, vacts, (v[:, 0] - R[b])[:, None] / len(b))
            gv, _ = clip_grads(gv, 1.0)
            optv.step(pol.vf, gv)
        _, vloss, vacc = bc_grads(pol, X[val], M[val], A[val])
        vv = mlp_forward(pol.vf, X[val])[0][:, 0]
        print(f"에폭 {ep}: 검증 손실 {vloss:.3f}, 주 행동 일치 {vacc:.3f}, 가치 오차 {np.mean((vv - R[val]) ** 2):.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pol.save(a.out)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
