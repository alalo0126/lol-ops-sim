"""6단계 실행: 실제 상황마다 학습 정글러·규칙 봇의 다음 1분 선택을 뽑고, 차이가 큰 상황에서 반사실 실험을 돌린다.

    python -m sim.compare situations              :: → data/stage6_situations.json (실제 1,000판 × 양 팀 × 3~17분)
    python -m sim.compare counterfactual          :: → data/stage6_cf.json (표본 상황 × 범주 6개 × 반복)
    python -m sim.compare counterfactual --set objective... --tag xxx   :: 가정 바꿔 다시(민감도)
    python -m sim.compare report                  :: → reports/stage6_compare.md, reports/stage6_compare.json

입력은 data/stage6_compact.json.gz(tools/stage6_compact.py).
"""
import argparse
import collections
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from .data import ROOT, Rules
from .evaluate import apply_overrides
from .game import Game
from .ppo import Policy
from .stage6 import CATS, CAT_KO, RealMatch, label, load_compact, run_sim_minute, rollout_value, situation
from .winprob import WinProb

MINUTES = range(3, 18)
POLICY = os.path.join(ROOT, "models", "stage5", "policy_final.npz")
_S = {}


def _init(overrides=None, camp_shift_s=0.0):
    _S["matches"] = load_compact()
    _S["wp"] = WinProb()
    _S["pol"] = Policy.from_file(POLICY)
    _S["nodes"] = Game(seed=0).map.nodes
    _S["rules"] = apply_overrides(Rules(), overrides) if overrides else None
    _S["camp_shift"] = camp_shift_s


def _fn(seed):
    pol = _S["pol"]
    pol.rng = np.random.default_rng(seed)
    return lambda o, m, mw, ms: pol.act(o, m, mw, ms)


# ---------------------------------------------------------------------- 1. 상황
def _situations(mi):
    m = _S["matches"][mi]
    rm = RealMatch(m)
    if not rm.valid:
        return []
    wp, nodes = _S["wp"], _S["nodes"]
    out = []
    for team in (100, 200):
        for minute in MINUTES:
            if len(rm.frames) <= minute + 1 or m["duration"] <= (minute + 1) * 60:
                continue
            seed = mi * 1000 + team + minute
            real_w = rm.window(team, minute, nodes)
            j = rm.jungler(team)
            a, b = rm.pf(minute, j), rm.pf(minute + 1, j)
            s = 1 if team == 100 else -1
            phi = lambda k: wp.blue_from_diffs(rm.diffs(k), k * 60) if s == 1 else 1 - wp.blue_from_diffs(rm.diffs(k), k * 60)
            row = {"mi": mi, "team": team, "minute": minute, **situation(rm, team, minute),
                   "real": label(real_w, team, nodes),
                   "real_gain": {"gold": b["total_gold"] - a["total_gold"], "jcs": b["jcs"] - a["jcs"], "xp": b["xp"] - a["xp"]},
                   "phi": phi(minute), "phi3": phi(minute + 3) if len(rm.frames) > minute + 3 and m["duration"] > (minute + 3) * 60 else None,
                   "win": m["win100"] if team == 100 else 1 - m["win100"]}
            row["ai"], row["ai_gain"] = [], []
            for r in range(2):
                lab, gain = run_sim_minute(rm, team, minute, _fn(seed + 7919 * r), wp, seed=seed + 7919 * r)
                row["ai"].append(lab)
                row["ai_gain"].append(gain)
            row["rule"], row["rule_gain"] = run_sim_minute(rm, team, minute, None, wp, seed=seed)
            out.append(row)
    return out


def cmd_situations(args):
    _init()
    n = len(_S["matches"])
    t = time.time()
    with Pool(args.workers, initializer=_init) as pool:
        rows = [r for part in pool.imap(_situations, range(n), chunksize=4) for r in part]
    dst = os.path.join(ROOT, "data", "stage6_situations.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{len(rows)}개 상황 → {dst} ({time.time() - t:.0f}초)")


# ---------------------------------------------------------------------- 2. 반사실
def feasible(row):
    cats = list(CATS)
    return cats


def _cf(job):
    mi, team, minute, reps, seed0 = job
    rm = RealMatch(_S["matches"][mi])
    wp = _S["wp"]
    res = {"mi": mi, "team": team, "minute": minute, "q": {}, "lab": {}}
    for cat in [None] + CATS:
        vals, labs = [], []
        for r in range(reps):
            seed = seed0 + r
            v, lab, phi0 = rollout_value(rm, team, minute, _fn(seed), wp, seed=seed, cat=cat, rules=_S["rules"],
                                         camp_shift_s=_S["camp_shift"])
            vals.append(v)
            labs.append(lab)
        key = cat or "free"
        res["q"][key], res["lab"][key] = vals, labs
    res["phi0"] = phi0
    return res


def pick_samples(rows, per_cell, rng):
    """반사실 표본: 시기 × 드래곤 상태 × 실제 범주 칸마다 최대 per_cell개(드문 선택도 표본에 들어가게)."""
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["phase"], r["dragon"], r["real"])].append(r)
    out = []
    for key in sorted(cells):
        v = cells[key]
        rng.shuffle(v)
        out += v[:per_cell]
    return out


def cmd_counterfactual(args):
    with open(os.path.join(ROOT, "data", "stage6_situations.json"), encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(6)
    sample = pick_samples(rows, args.per_cell, rng)
    if args.limit:
        sample = sample[:args.limit]
    jobs = [(r["mi"], r["team"], r["minute"], args.reps, 500000 + i * 100) for i, r in enumerate(sample)]
    t = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.set, args.camp_shift)) as pool:
        res = list(pool.imap(_cf, jobs, chunksize=2))
    dst = os.path.join(ROOT, "data", f"stage6_cf{('_' + args.tag) if args.tag else ''}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"overrides": args.set, "reps": args.reps, "results": res}, f)
    print(f"{len(res)}개 상황 × {len(CATS) + 1}갈래 × {args.reps}회 → {dst} ({time.time() - t:.0f}초)")


# ---------------------------------------------------------------------- 3. 범주별 1분 결과
def _outcome(job):
    mi, team, minute, seed = job
    rm = RealMatch(_S["matches"][mi])
    nodes, wp = _S["nodes"], _S["wp"]
    rw = rm.window(team, minute, nodes)
    out = {"real": [label(rw, team, nodes), rw["kills_for"] > 0, rw["died"], rw["elite"] > 0]}
    lab, _, w = run_sim_minute(rm, team, minute, _fn(seed), wp, seed=seed, with_window=True)
    out["ai"] = [lab, w["kills_for"] > 0, w["died"], w["elite"] > 0]
    lab, _, w = run_sim_minute(rm, team, minute, None, wp, seed=seed, with_window=True)
    out["rule"] = [lab, w["kills_for"] > 0, w["died"], w["elite"] > 0]
    return out


def cmd_outcomes(args):
    """범주로 붙은 1분 동안 킬 관여·사망·에픽 몬스터 처치 비율(실제 / 학습 정글러 / 규칙 봇). 결과 모델 대조용."""
    with open(os.path.join(ROOT, "data", "stage6_situations.json"), encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(66)
    sample = rng.sample(rows, min(args.n, len(rows)))
    jobs = [(r["mi"], r["team"], r["minute"], 900000 + i) for i, r in enumerate(sample)]
    with Pool(args.workers, initializer=_init) as pool:
        res = list(pool.imap(_outcome, jobs, chunksize=16))
    table = {}
    for who in ("real", "ai", "rule"):
        by = collections.defaultdict(list)
        for x in res:
            by[x[who][0]].append(x[who][1:])
        table[who] = {c: {"n": len(v), "kill": float(np.mean([a[0] for a in v])) if v else None,
                          "death": float(np.mean([a[1] for a in v])) if v else None,
                          "elite": float(np.mean([a[2] for a in v])) if v else None} for c, v in by.items()}
    dst = os.path.join(ROOT, "data", "stage6_outcomes.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print(json.dumps(table, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("situations")
    a.add_argument("--workers", type=int, default=os.cpu_count())
    b = sub.add_parser("counterfactual")
    b.add_argument("--workers", type=int, default=os.cpu_count())
    b.add_argument("--per-cell", type=int, default=12)
    b.add_argument("--reps", type=int, default=6)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--set", action="append", default=None)
    b.add_argument("--tag", default="")
    b.add_argument("--camp-shift", type=float, default=0.0, help="캠프 처치 추정 시각을 이만큼 이르게(초)")
    o = sub.add_parser("outcomes")
    o.add_argument("--workers", type=int, default=os.cpu_count())
    o.add_argument("--n", type=int, default=6000)
    c = sub.add_parser("report")
    c.add_argument("--extra", action="append", default=[], help="민감도 결과 태그")
    args = ap.parse_args()
    if args.cmd == "situations":
        cmd_situations(args)
    elif args.cmd == "counterfactual":
        cmd_counterfactual(args)
    elif args.cmd == "outcomes":
        cmd_outcomes(args)
    else:
        from .stage6_report import report
        report(args.extra)


if __name__ == "__main__":
    main()
