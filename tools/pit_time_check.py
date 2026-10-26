"""둥지 근처 초(행동 라벨 없이, 반경 1,200, 3~18분) 실제 vs 시뮬, 판당 평균과 정글러 단위 부트스트랩 95% 구간.

    python tools/pit_time_check.py
"""
import collections, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import objective_check as oc
from sim.data import Rules
from sim.game import Game
nodes = Game(seed=0).map.nodes; rules = Rules()
real = oc.real(nodes, rules)
oc._POLICY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stage16_a", "policy_latest.npz")
with Pool(4, initializer=oc._init, initargs=(oc._POLICY,)) as pool:
    sims = pool.map(oc._sim_one, range(200))
groups = {"real": [r[3] for r in real], "ai": [s["ai"][3] for s in sims], "rule": [s["rule"][3] for s in sims]}
rng = np.random.default_rng(0)
for nm in ("dragon", "grubs", "herald"):
    for st in ("alive", "after"):
        row = []
        for k, rs in groups.items():
            v = np.array([r[(nm, st)] for r in rs], float)
            b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
            row.append(f"{k} {v.mean():.1f} ({np.percentile(b,2.5):.1f}~{np.percentile(b,97.5):.1f})")
        print(nm, st, " | ".join(row))
