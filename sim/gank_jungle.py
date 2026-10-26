"""라인 방문 결과를 적 정글러 거리별로: 실제(리플레이) vs 시뮬(학습 정글러·규칙 봇).

    python -m sim.gank_jungle [--policy models/stage12_a/policy_latest.npz] [--games 200] [--set gank.enemy_jg_death.mult=[...]]

실제는 sim/guide_check.py와 같은 방문 정의(sim/lane_visits.py)·같은 거리(방문 시작 초, 두 정글러 좌표). 결과는 reports/gank_jungle.md.
"""
import argparse
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT, Rules
from .evaluate import apply_overrides
from .game import Game
from .lane_visits import T0, T1, lane_of, outcome, visits
from .stage6 import zone

EDGES = [0, 2000, 4000, 7000, 1e9]
_S = {}


def _init(policy, overrides):
    from .ppo import Policy
    _S["pol"] = Policy.from_file(policy) if policy else None
    _S["ov"] = overrides


def _one(seed):
    from .agent import AgentJungler
    from .winprob import WinProb
    rules = apply_overrides(Rules(), _S["ov"]) if _S["ov"] else None
    team = 100 if seed % 2 == 0 else 200
    override = {}
    if _S["pol"] is not None:
        pol = _S["pol"]
        pol.rng = np.random.default_rng(seed)
        override = {team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())}
    g = Game(seed=seed, rules=rules, jungler_override=override)
    js = {c.team: c for c in g.champs if c.role == "JUNGLE"}
    xy = {t: {} for t in js}
    while g.t < 1200:
        g.step()
        if T0 <= g.t < T1:
            for t, c in js.items():
                p = g.xy(c)
                if p:
                    xy[t][g.t] = p
    kills = [{"t": e["t"], "killer": e["killer"], "victim": e["victim"], "assists": e["assists"]} for e in g.events if e["type"] == "kill"]
    out = {}
    for t, c in js.items():
        who = "ai" if (_S["pol"] is not None and t == team) else "rule"
        zones = {s: zone(x, y, t, g.map.nodes) for s, (x, y) in xy[t].items()}
        tp = {x.pid for x in g.champs if x.team == t}
        rows = []
        for span in visits(zones):
            o = xy[300 - t].get(span[0])
            d = math.dist(xy[t][span[0]], o) if o else None
            rows.append(outcome(span, kills, c.pid, tp) | {"lane": lane_of(*xy[t][span[0]]), "enemy_jg_dist": d})
        out.setdefault(who, []).extend(rows)
    return out


def bins(rows):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        rs = [r for r in rows if r["enemy_jg_dist"] is not None and lo <= r["enemy_jg_dist"] < hi]
        out.append((len(rs), float(np.mean([r["kill"] for r in rs])) if rs else float("nan"),
                    float(np.mean([r["death"] for r in rs])) if rs else float("nan")))
    rs = [r for r in rows if r["enemy_jg_dist"] is None]
    out.append((len(rs), float(np.mean([r["kill"] for r in rs])) if rs else float("nan"),
                float(np.mean([r["death"] for r in rs])) if rs else float("nan")))
    return out


def simulate(policy, games, overrides=None, workers=None):
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(policy, overrides)) as pool:
        res = pool.map(_one, range(games))
    out = {}
    for r in res:
        for k, v in r.items():
            out.setdefault(k, []).extend(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--set", action="append", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "gank_jungle.md"))
    args = ap.parse_args()
    from .guide_check import gank_rows
    real = gank_rows(Game(seed=0).map.nodes)
    sim = simulate(args.policy, args.games, args.set)
    labels = [f"{lo:g}~{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(EDGES[:-1], EDGES[1:])] + ["적 정글러 사망 중"]
    L = ["# 라인 방문 결과 × 적 정글러 거리", "",
         f"- 시뮬 {args.games}판, 정책 `{os.path.relpath(args.policy, ROOT)}`" + (f", 바꾼 값 {args.set}" if args.set else "") + ".",
         "- 방문 = 3~18분 라인 구역 연속 5초 이상. 거리 = 방문 시작 초 두 정글러 거리. 결과 = 방문 시작~끝+10초 정글러 킬 관여 / 사망.", "",
         "| 적 정글러 거리 | 실제 방문 | 실제 킬 | 실제 사망 | 학습 방문 | 학습 킬 | 학습 사망 | 규칙 봇 방문 | 규칙 봇 킬 | 규칙 봇 사망 |", "|---|---|---|---|---|---|---|---|---|---|"]
    bs = {k: bins(v) for k, v in [("real", real)] + list(sim.items())}
    for i, lab in enumerate(labels):
        cells = []
        for k in ("real", "ai", "rule"):
            if k in bs:
                n, kr, dr = bs[k][i]
                cells += [str(n), f"{kr:.0%}" if n else "-", f"{dr:.0%}" if n else "-"]
            else:
                cells += ["-", "-", "-"]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    for k, name in (("real", "실제"), ("ai", "학습"), ("rule", "규칙 봇")):
        v = real if k == "real" else sim.get(k, [])
        if v:
            L.append(f"\n{name} 전체: 방문 {len(v)}번, 킬 {np.mean([r['kill'] for r in v]):.0%}, 사망 {np.mean([r['death'] for r in v]):.0%}.")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
