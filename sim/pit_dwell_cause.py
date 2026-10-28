"""학습 정글러가 둥지 근처에 오래 있는 이유: 처치 횟수 vs 처치당 시간 (3~18분).

    python -m sim.pit_dwell_cause [--policy models/stage17_a/policy_latest.npz] [--games 160]

실제는 리플레이 70판 타임라인(data/replay_raw)의 ELITE_MONSTER_KILL 팀당 수. 처치 시간·인원 실제값은 reports/objective_time_notes.md.
"""
import argparse
import collections
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game

T0, T1 = 180, 1080
_S = {}


def real_kills():
    k, n = collections.Counter(), 0
    for f in glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        fr = d.get("info", {}).get("frames") or d.get("timeline", {}).get("info", {}).get("frames")
        if not fr:
            continue
        n += 1
        for F in fr:
            for e in F["events"]:
                if e["type"] == "ELITE_MONSTER_KILL" and T0 * 1000 <= e["timestamp"] < T1 * 1000:
                    k[e["monsterType"]] += 1
    return n, k


def _one(seed):
    from .agent import AgentJungler
    from .ppo import Policy
    from .winprob import WinProb
    team = 100 if seed % 2 == 0 else 200
    pol = Policy.from_file(_S["policy"])
    pol.rng = np.random.default_rng(seed)
    g = Game(seed=seed, jungler_override={team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())})
    starts, orig = [], g.objective_time

    def timed(c, name, helpers):
        d = orig(c, name, helpers)
        starts.append((g.t, "ai" if c.team == team else "rule", name, d, len(helpers)))
        return d
    g.objective_time = timed
    while g.t < T1:
        g.step()
    kills = [("ai" if e["team"] == team else "rule", e["name"]) for e in g.events if e["type"] == "objective" and T0 <= e["t"] < T1]
    return [s for s in starts if T0 <= s[0] < T1], kills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage17_a", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=160)
    args = ap.parse_args()
    _S["policy"] = args.policy
    with Pool(os.cpu_count()) as pool:
        res = pool.map(_one, range(args.games))
    n = args.games
    starts = [s for r in res for s in r[0]]
    kills = collections.Counter(k for r in res for k in r[1])
    rn, rk = real_kills()
    real = {"dragon": rk["DRAGON"] / rn / 2, "herald": rk["RIFTHERALD"] / rn / 2}
    print(f"실제 리플레이 {rn}판 팀당 3~18분 처치: 용 {real['dragon']:.2f}, 전령 {real['herald']:.2f}, 공허 유충 {rk['HORDE'] / rn / 2:.2f}마리")
    for who in ("ai", "rule"):
        for nm in ("dragon", "grubs", "herald"):
            s = [x for x in starts if x[1] == who and x[2] == nm]
            if not s:
                continue
            d = np.array([x[3] for x in s])
            h = np.array([x[4] for x in s])
            print(f"{who} {nm}: 시작 {len(s) / n:.2f}/판, 처치 {kills[(who, nm)] / n:.2f}/판, 처치 시간 중앙 {np.median(d):.1f}초, 도우미 평균 {h.mean():.2f}, 혼자 {np.mean(h == 0):.0%}")


if __name__ == "__main__":
    main()
