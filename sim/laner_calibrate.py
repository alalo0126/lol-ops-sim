"""라이너 행동 모델 보정: 시뮬 라이너 분포를 1단계 실제 분포(sim/targets_laner.json)에 맞추는 좌표 하강.

    python -m sim.laner_calibrate --games 96 --rounds 4 --write   :: 결과 → sim/laner_calibrated.json

목적 함수 = Σ((시뮬 중앙 − 실제 중앙) / 실제 사분위 범위)² + 위치 비율 차이 / 0.05 의 제곱 + 포탑 수 차이.
조정하는 값은 'calibrated' 표시가 붙은 라인 모듈·라이너 값만(PARAMS). 검증 지표로 쓰지 않는다.
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics

from .data import ROOT, Rules
from .laner_eval import simulate, real_zone, replay_rows, rp_value, kill_rows, ZONES, MINUTES

PARAMS = [
    ("lanes", "last_hit", "TOP"), ("lanes", "last_hit", "MIDDLE"), ("lanes", "last_hit", "BOTTOM"), ("lanes", "last_hit", "UTILITY"),
    ("lanes", "backlog", "keep"), ("lanes", "support_item_gold_per_s"),
    ("lanes", "lane_fight", "rate_per_s", "top", 0), ("lanes", "lane_fight", "rate_per_s", "top", 1),
    ("lanes", "lane_fight", "rate_per_s", "mid", 0), ("lanes", "lane_fight", "rate_per_s", "mid", 1),
    ("lanes", "lane_fight", "rate_per_s", "bot", 0), ("lanes", "lane_fight", "rate_per_s", "bot", 1),
    ("lanes", "tower_damage", "minions"), ("lanes", "tower_damage", "per_strength"), ("lanes", "siege_at"), ("lanes", "tower_damage", "early_power"),
    ("lanes", "xp_share_mult", 1), ("lanes", "push", "rate"), ("lanes", "push", "sigma"),
    ("laner", "roam", "rate_per_min", "TOP", 0), ("laner", "roam", "rate_per_min", "TOP", 1),
    ("laner", "roam", "rate_per_min", "MIDDLE", 0), ("laner", "roam", "rate_per_min", "MIDDLE", 1),
    ("laner", "roam", "rate_per_min", "BOTTOM", 0), ("laner", "roam", "rate_per_min", "BOTTOM", 1),
    ("laner", "roam", "rate_per_min", "UTILITY", 0), ("laner", "roam", "rate_per_min", "UTILITY", 1),
    ("laner", "rotate", "after_bot_outer_down"), ("laner", "rotate", "after_bot_outer_down_early"), ("laner", "rotate", "after_s_rate"),
    ("laner", "support_ward_per_min", "early"), ("laner", "support_ward_per_min", "late"),
    ("laner", "recall_mult", "TOP"), ("laner", "recall_mult", "MIDDLE"), ("laner", "recall_mult", "BOTTOM"), ("laner", "recall_mult", "UTILITY"),
    ("laner", "roam", "fight_on_arrival", "TOP"), ("laner", "roam", "fight_on_arrival", "MIDDLE"),
    ("laner", "roam", "fight_on_arrival", "BOTTOM"), ("laner", "roam", "fight_on_arrival", "UTILITY"),
    ("laner", "roam", "resolve", "loser_death"), ("laner", "roam", "resolve", "trade_death"),
    ("laner", "roam", "encounter", "prob_per_s"),
    ("group_fight", "rate_per_s", 0), ("group_fight", "rate_per_s", 1), ("group_fight", "resolve", "lose_death"), ("group_fight", "join_prob"), ("group_fight", "join_travel_s"),
    ("objective", "rally", "respond_prob", 1), ("skirmish", "assist_near", "prob"),
    ("laner", "top_group", "rate_per_min"), ("laner", "top_group", "stay_s", 1),
    ("laner", "recall_walk", "TOP"), ("laner", "recall_walk", "UTILITY"),
    ("laner", "roam", "detour", "UTILITY", 0), ("laner", "roam", "detour", "UTILITY", 1), ("laner", "roam", "detour", "MIDDLE", 1),
    ("laner", "roam", "linger_mult", "TOP", 0), ("laner", "roam", "linger_mult", "TOP", 1),
    ("laner", "roam", "linger_mult", "MIDDLE", 0), ("laner", "roam", "linger_mult", "MIDDLE", 1),
    ("laner", "roam", "linger_mult", "BOTTOM", 0), ("laner", "roam", "linger_mult", "BOTTOM", 1),
    ("laner", "roam", "linger_mult", "UTILITY", 0), ("laner", "roam", "linger_mult", "UTILITY", 1),
]


def get(tree, path):
    for k in path:
        tree = tree[k]
    return tree


def put(tree, path, v):
    for k in path[:-1]:
        tree = tree[k]
    tree[path[-1]] = v


def evaluate(args):
    values, seeds = args
    import sim.laner_eval as le
    rules = Rules()
    for path, v in zip(PARAMS, values):
        put(rules.params, path, v)
    return le.simulate(len(seeds), seeds[0], {"rules": rules})


def merge(parts):
    import collections
    vals, where = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    for v, w in parts:
        for k, x in v.items():
            vals[k] += x
        for k, c in w.items():
            where[k].update(c)
    return vals, where


def loss(vals, where, t):
    rp = json.load(open(os.path.join(ROOT, "sim", "targets_laner_replay.json"), encoding="utf-8"))
    tot = 0.0
    for r in ("TOP", "MIDDLE", "BOTTOM", "UTILITY"):
        for m in MINUTES[1:]:
            for k in ("lane_cs", "gold", "level"):
                real = t["by_minute"][r][m][k]
                iqr = max(1.0, real["p75"] - real["p25"])
                tot += ((statistics.median(vals[(r, m, k)]) - real["p50"]) / iqr) ** 2
        for key, real in ((("first_shop_s",), t["first_shop_s"][r]), (("shops_by_20",), t["shops_by_20"][r])):
            iqr = max(1.0, real["p75"] - real["p25"])
            tot += ((statistics.median(vals[(r,) + key]) - real["p50"]) / iqr) ** 2
        for m in (10, 20):
            for k, tk in (("deaths", "deaths"), ("wards", "wards")):
                real = t[tk][f"{r}_{m}"]
                tot += ((statistics.mean(vals[(r, f"{k}_{m}")]) - real["mean"]) / max(0.5, 0.25 * real["mean"])) ** 2
        for m in (6, 10, 15, 20):
            n = sum(where[(r, m)].values()) or 1
            rz = real_zone(t["zone_by_minute"], r, m)
            for z in ZONES:
                tot += ((where[(r, m)][z] / n - rz[z]) / 0.05) ** 2 * 0.25
        # 리플레이 1초 동선(3~18분) 목표: 기지 방문·로밍 수·로밍당 킬·자기 라인 비율
        for key, real, tol, _ in replay_rows(vals, rp, r):
            tot += ((rp_value(vals, r, key) - real) / tol) ** 2
    for key, _, real, tol in kill_rows(vals, rp):
        tot += ((statistics.mean(vals[key]) - real) / tol) ** 2
    b20 = t["by_20_per_game"]
    tot += ((statistics.mean(vals[("plates_by_20",)]) - b20["plates"]["mean"]) / 4) ** 2
    tot += ((statistics.mean(vals[("outer_by_20",)]) - b20["outer"]["mean"]) / 0.6) ** 2
    tot += ((statistics.mean(vals[("inner_by_20",)]) - b20["inner"]["mean"]) / 0.3) ** 2
    tot += ((statistics.median(vals[("first_tower",)]) - t["first_tower_s"]["p50"]) / 60) ** 2
    if vals[("plate", "outer")]:
        tot += ((statistics.median(vals[("plate", "outer")]) - t["plate_time_by_tier_s"]["outer"]["p50"]) / 60) ** 2
    return tot


def run(values, games, pool, seed0=1000):
    per = games // 4
    parts = pool.map(evaluate, [(values, list(range(seed0 + i * per, seed0 + (i + 1) * per))) for i in range(4)])
    return merge(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t = json.load(open(os.path.join(ROOT, "sim", "targets_laner.json"), encoding="utf-8"))
    base = Rules().params
    x = [float(get(base, p)) for p in PARAMS]   # 이미 laner_calibrated.json이 있으면 그 값에서 출발
    with mp.Pool(4) as pool:
        best = loss(*run(x, a.games, pool), t)
        print(f"start loss {best:.1f}", flush=True)
        step = 0.3
        for rnd in range(a.rounds):
            for i, p in enumerate(PARAMS):
                for f in (1 + step, 1 / (1 + step)):
                    y = list(x)
                    y[i] = x[i] * f
                    l = loss(*run(y, a.games, pool), t)
                    if l < best:
                        best, x = l, y
                        print(f"r{rnd} {'.'.join(map(str, p))} → {x[i]:.4g}  loss {best:.1f}", flush=True)
                        break
            step *= 0.6
    out = {".".join(map(str, p)): round(v, 5) for p, v in zip(PARAMS, x)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.write:
        with open(os.path.join(ROOT, "sim", "laner_calibrated.json"), "w", encoding="utf-8") as f:
            json.dump({"_doc": f"sim/laner_calibrate.py 결과(손실 {best:.1f}, {a.games}판). sim/data.py가 params.json 위에 덮어씀", "values": out},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
