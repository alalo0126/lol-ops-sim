"""천상계 1초 동선으로 (1) 실제 정글러 행동을 따라 하는 정책을 만들고 (2) 실제 행동이 시뮬에서 손해로 나오는 곳을 찾는다.

    python -m sim.imitate [--policy models/stage6c/policy_latest.npz] [--seeds 2]

표본: 리플레이 경기마다 양 팀 정글러 × 3~17분 시작 시각. 실제 상태를 시뮬레이터로 옮기고(stage6.reconstruct),
캠프 상태는 두 정글러의 1초 동선(캠프에 5초 이상 머문 끝 = 처치)으로 덮어쓴다(분 단위 CS 추정보다 정확).
정답 행동 = sim/real_actions.py가 그 초에 붙인 행동(이동 중이면 다음에 머문 곳).
(1) 모방: 경기 5개 중 1개를 시험용으로 떼고, 학습 정책·규칙 봇 모방 정책·실제 모방 정책의 첫 행동 일치율을 잰다.
    실제 모방 정책은 models/stage6c/policy_real_bc.npz, 표본은 models/stage6c/real_train.npz·real_test.npz.
(2) 반사실: 같은 상태·같은 시드에서 첫 결정만 실제 행동으로 묶은 것과 학습 정책 그대로를 20분까지 돌려
    20분 예측 승률 차이(실제 − 학습)를 행동 묶음별로 낸다. 크게 음수인 묶음 = 시뮬이 그 행동 가치를 낮게 보는 곳.
결과는 reports/imitate.md.
"""
import argparse
import collections
import glob
import os
from multiprocessing import Pool

import numpy as np

from .agent import ACTION_NAMES, N_MAIN
from .data import ROOT
from .game import Game
from .ppo import Adam, Policy
from .real_actions import GROUP_KO, GROUPS, group, real_tracks, segments
from .stage6 import RealMatch, reconstruct
from .stage6_track import compact_raw
from .teacher import bc_grads
from .winprob import WinProb

MINUTES = range(3, 18)
CAMP_DWELL = 5
SIDE = {100: "B", 200: "R"}
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_NAMES)}
WITHIN = True   # 분 시작만(False) / 그 1분의 모든 결정 시각(True). 2026-10-22 기준점(stage6d)은 분 시작만


def camp_kills(seg, team):
    """{초: 행동} → {캠프 id: [처치 시각]} (캠프에 CAMP_DWELL초 이상 연속으로 머문 구간의 끝)."""
    out = collections.defaultdict(list)
    run, start = None, None
    for t in sorted(seg) + [None]:
        a = seg.get(t) if t is not None else None
        if a != run or (t is not None and start is not None and t != prev + 1):
            if run and run.split("_")[0] in ("camp", "invade") and prev - start + 1 >= CAMP_DWELL:
                side = SIDE[team] if run.startswith("camp_") else SIDE[300 - team]
                out[f"{side}_{run.split('_', 1)[1]}"].append(prev)
            run, start = a, t
        prev = t
    return out


def apply_camps(g, kills, t):
    """1초 동선으로 본 처치 시각으로 캠프 상태를 덮어쓴다(동선이 있는 캠프만)."""
    for cid, ts in kills.items():
        past = [k for k in ts if k <= t]
        if not past or cid not in g.camps:
            continue
        camp = g.camps[cid]
        camp.first = False
        camp.spawn_at = past[-1] + g.rules.camps[camp.kind]["respawn_s"]
        camp.alive = t >= camp.spawn_at
        if camp.alive:
            camp.level = max(1, round(g.avg_level()))


# ---------------------------------------------------------------------- 표본
def match_inputs():
    """경기마다 (경기 파일, {팀: 1초 행동}, 두 정글러 캠프 처치)."""
    nodes = Game(seed=0).map.nodes
    herald = Game(seed=0).objectives["herald"]["spawn_at"]
    by_match = collections.defaultdict(dict)
    for mid, team, pid, pos in real_tracks():
        by_match[mid][team] = segments(pos, team, nodes, herald, 60, 1200)
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        mid = os.path.basename(path)[:-5]
        if mid not in by_match:
            continue
        kills = collections.defaultdict(list)
        for team, seg in by_match[mid].items():
            for cid, ts in camp_kills(seg, team).items():
                kills[cid] += ts
        out.append((path, by_match[mid], {k: sorted(v) for k, v in kills.items()}))
    return out


def _state(rm, team, minute, kills, policy_fn, seed):
    g, bot = reconstruct(rm, team, minute, policy_fn, WinProb(), seed=seed)
    apply_camps(g, kills, minute * 60)
    return g, bot


def _samples(args):
    path, segs, kills = args
    rm = RealMatch(compact_raw(path))
    if not rm.valid:
        return []
    out = []
    for team, seg in segs.items():
        for minute in MINUTES:
            if len(rm.frames) <= minute + 1:
                continue
            a = seg.get(minute * 60)
            if a not in ACTION_INDEX:
                continue
            got = []
            box = {}

            def grab(o, m, mw, ms):
                # 그 초의 실제 행동을 정답으로 기록하고, 그 행동을 그대로 따라 해서 상태가 실제 동선을 따라가게 한다.
                # WITHIN이면 분 시작뿐 아니라 그 1분 동안의 모든 결정 시각(이동 중 5초마다·새 정보)을 표본으로 씀
                lab = seg.get(box["g"].t) if "g" in box else a
                if lab in ACTION_INDEX and m[ACTION_INDEX[lab]]:
                    got.append({"obs": o.copy(), "m": m.copy(), "mw": mw.copy(), "ms": ms.copy(), "action": lab,
                                "t": box["g"].t if "g" in box else minute * 60})
                    return ACTION_INDEX[lab], 0, 0, None
                return int(np.flatnonzero(m)[0]), 0, 0, None
            g, _ = _state(rm, team, minute, kills, grab, 0)
            box["g"] = g
            end = minute * 60 + (60 if WITHIN else 1)
            while g.t < end and (WITHIN or not got):
                g.step()
            for x in got[:(None if WITHIN else 1)]:
                out.append({"match": os.path.basename(path)[:-5], "team": team, "minute": minute, **x})
    return out


# ---------------------------------------------------------------------- 반사실
_CF = {}


def _cf_init(policy_path, inputs):
    _CF["pol"] = Policy.from_file(policy_path)
    _CF["inputs"] = {os.path.basename(p)[:-5]: (p, k) for p, _, k in inputs}
    _CF["rm"] = {}


def _rollout(match, team, minute, forced, seed):
    path, kills = _CF["inputs"][match]
    if match not in _CF["rm"]:
        _CF["rm"][match] = RealMatch(compact_raw(path))
    rm, pol = _CF["rm"][match], _CF["pol"]
    pol.rng = np.random.default_rng(seed)
    state = {"first": forced is not None}

    def fn(o, m, mw, ms):
        if state["first"]:
            state["first"] = False
            mm = np.zeros_like(m)
            mm[forced] = True
            return pol.act(o, mm, mw, ms)
        return pol.act(o, m, mw, ms)
    g, bot = _state(rm, team, minute, kills, fn, seed)
    wp = WinProb()
    while g.t < 1200:
        g.step()
    return bot.finish_episode() if bot.traj else wp.for_team(g, team)


def _cf_job(job):
    match, team, minute, a, seed = job
    return _rollout(match, team, minute, ACTION_INDEX[a], seed) - _rollout(match, team, minute, None, seed)


# ---------------------------------------------------------------------- 실행
def train_bc(X, M, A, init, epochs=200, lr=1e-3, seed=0):
    pol = Policy.from_file(init)
    opt = Adam(pol.pi, lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for b in range(0, len(X), 256):
            i = idx[b:b + 256]
            grads, _, _ = bc_grads(pol, X[i], M[i], A[i], heads=(0,))
            opt.step(pol.pi, grads)
    return pol


def accuracy(pol, S):
    hit = 0
    for s in S:
        a, _, _, _ = pol.act(s["obs"], s["m"], s["mw"], s["ms"], greedy=True)
        hit += ACTION_NAMES[a] == s["action"]
    return hit / max(1, len(S))


def group_accuracy(pol, S):
    hit = 0
    for s in S:
        a, _, _, _ = pol.act(s["obs"], s["m"], s["mw"], s["ms"], greedy=True)
        hit += group(ACTION_NAMES[a]) == group(s["action"])
    return hit / max(1, len(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage6c", "policy_latest.npz"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "imitate.md"))
    ap.add_argument("--minute-start-only", action="store_true", help="표본을 분 시작 시각만(2026-10-22 방식)")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "models", "stage6c"), help="real_train/test.npz·policy_real_bc.npz 저장 위치")
    args = ap.parse_args()
    global WITHIN
    WITHIN = not args.minute_start_only
    inputs = match_inputs()
    with Pool(args.workers) as pool:
        S = [s for ss in pool.map(_samples, inputs) for s in ss]
    matches = sorted({s["match"] for s in S})
    test_m = set(matches[::5])
    tr = [s for s in S if s["match"] not in test_m]
    te = [s for s in S if s["match"] in test_m]

    def arr(ss):
        X = np.array([s["obs"] for s in ss], np.float32)
        M = np.array([np.concatenate([s["m"], s["mw"], s["ms"]]) for s in ss])
        A = np.array([[ACTION_INDEX[s["action"]], 0, 0] for s in ss])
        return X, M, A
    for name, ss in (("train", tr), ("test", te)):   # sim/train.py --real 용 표본
        X, M, A = arr(ss)
        np.savez(os.path.join(args.save_dir, f"real_{name}.npz"), X=X, M=M, A=A)
    real_bc = train_bc(*arr(tr), init=os.path.join(ROOT, "models", "stage5", "policy_bc.npz"))
    real_bc.save(os.path.join(args.save_dir, "policy_real_bc.npz"))
    pols = {"학습 정글러(stage6c)": Policy.from_file(args.policy),
            "규칙 봇 모방(stage5 bc)": Policy.from_file(os.path.join(ROOT, "models", "stage5", "policy_bc.npz")),
            "실제 모방(이번)": real_bc}

    first = [s for s in S if s["t"] == s["minute"] * 60]   # 반사실은 분 시작 표본만(그 상태가 실제 그대로)
    jobs = [(s["match"], s["team"], s["minute"], s["action"], 1000 * k + i) for i, s in enumerate(first) for k in range(args.seeds)]
    with Pool(args.workers, initializer=_cf_init, initargs=(args.policy, inputs)) as pool:
        d = pool.map(_cf_job, jobs, chunksize=8)
    by = collections.defaultdict(list)
    for (m, t, mi, a, _), v in zip(jobs, d):
        by[group(a)].append(v)

    lines = ["# 천상계 동선 모방과 반사실", "",
             f"- 표본: 리플레이 {len(matches)}판, 정글러 × 3~17분 시작 시각 {len(S)}개(학습 {len(tr)} / 시험 {len(te)}, 경기 단위로 나눔).",
             "- 정답 = 그 초에 실제 정글러가 하던 행동(이동 중이면 다음에 머문 곳). 캠프 상태는 두 정글러 1초 동선으로 덮어씀.", "",
             "## 첫 행동 일치율(시험 경기)", "", "| 정책 | 세부 행동 | 행동 묶음 |", "|---|---|---|"]
    for name, p in pols.items():
        lines.append(f"| {name} | {accuracy(p, te):.0%} | {group_accuracy(p, te):.0%} |")
    cnt = collections.Counter(group(s["action"]) for s in S)
    lines += ["", "## 첫 결정을 실제 행동으로 묶었을 때 20분 예측 승률 차이(실제 − 학습 정책)", "",
              f"같은 상태·같은 시드 {args.seeds}개. 음수 = 시뮬에서는 실제 행동이 손해로 나옴.", "",
              "| 실제 행동 | 표본 | 평균 차이 | 95% 구간 |", "|---|---|---|---|"]
    for k in GROUPS:
        v = np.array(by.get(k, []))
        if len(v) < 5:
            continue
        se = v.std(ddof=1) / np.sqrt(len(v))
        lines.append(f"| {GROUP_KO[k]} | {cnt[k]} | {100 * v.mean():+.1f}%p | {100 * (v.mean() - 1.96 * se):+.1f} ~ {100 * (v.mean() + 1.96 * se):+.1f} |")
    allv = np.array(d)
    lines.append(f"| 전체 | {len(S)} | {100 * allv.mean():+.1f}%p | {100 * (allv.mean() - 1.96 * allv.std(ddof=1) / np.sqrt(len(allv))):+.1f} ~ "
                 f"{100 * (allv.mean() + 1.96 * allv.std(ddof=1) / np.sqrt(len(allv))):+.1f} |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
