"""위쪽 가치 검수: 시뮬 N판에 1,000판과 같은 팀-분 푸아송 회귀(tools/upper_value_fit)를 맞춰 계수를 비교한다.

    python -m sim.upper_check --games 200        :: → reports/upper_check.md

실제 계수는 sim/upper_value.json(PC에서 python -m tools.upper_value_fit). 시뮬 계수가 실제 ±2 표준오차 안이면 OK.
"""
import argparse
import json
import os

from .data import ROOT
from .game import Game
from tools.upper_value_fit import rows_from_sim, fit

FEATS = ["g_us", "h_us", "h_op"]
TARGETS = ["all", "top", "mid", "bot"]


def simulate(games, seed0=0, window_s=180, game_kw=None):
    rows, taken = [], {"grubs": 0, "herald": 0, "dragon": 0}
    for s in range(seed0, seed0 + games):
        g = Game(seed=s, **(game_kw or {}))
        g.record_frame()
        while g.t < 1200:
            g.step()
        rows += rows_from_sim(g, window_s)
        for k in taken:
            taken[k] += sum(g.obj_taken[k].values())
    return rows, {k: round(v / games, 2) for k, v in taken.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "upper_check.md"))
    ap.add_argument("--calibrate", type=int, default=0, help="보정 반복 수(0이면 검수만)")
    a = ap.parse_args()
    if a.calibrate:
        print(json.dumps(calibrate(a.calibrate, a.games), ensure_ascii=False))
        return
    real = json.load(open(os.path.join(ROOT, "sim", "upper_value.json"), encoding="utf-8"))
    rows, taken = simulate(a.games, a.seed0, real["window_s"])
    sim = fit(rows, real["window_s"])
    lines = [f"# 위쪽 가치 검수 (시뮬 {a.games}판 vs 실제 {real['matches']}판)", "",
             f"판당 오브젝트(시뮬): 유충 묶음 {taken['grubs']} · 전령 {taken['herald']} · 용 {taken['dragon']}", "",
             "팀-분(8~20분) 방패·포탑 수 푸아송 회귀 계수(로그 배율). 판정: 시뮬이 실제 ±2 표준오차 안이면 OK.", "",
             "| 대상 | 항 | 실제 | 시뮬 | 판정 |", "| --- | --- | --- | --- | --- |"]
    bad = 0
    for tg in TARGETS:
        r, s = real["fit"][tg], sim["fit"][tg]
        lines.append(f"| {tg} | 분당 평균 | {r['mean_per_min']} | {s['mean_per_min']} | - |")
        for f in FEATS:
            ok = abs(s[f]["coef"] - r[f]["coef"]) <= 2 * max(r[f]["se"], s[f]["se"])
            bad += not ok
            lines.append(f"| {tg} | {f} | {r[f]['coef']:+.3f} ± {r[f]['se']:.3f} | {s[f]['coef']:+.3f} ± {s[f]['se']:.3f} | {'OK' if ok else '불일치'} |")
    lines += ["", f"불일치 {bad}개"]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- 보정
def _sim_chunk(args):
    seeds, window_s, over = args
    rows = []
    for s in seeds:
        g = Game(seed=s)
        _apply(g, over)
        g.record_frame()
        while g.t < 1200:
            g.step()
        rows += rows_from_sim(g, window_s)
    return rows


def _apply(g, over):
    u = g.p["lanes"]["upper"]
    u["grub_per_stack"] = over["grub"]
    for ln in ("top", "mid", "bot"):
        u["herald"]["own"][ln] = over["own"][ln]
        u["herald"]["opp"][ln] = over["opp"][ln]
    d = g.p["lanes"]["tower_damage"]
    d["minions"], d["per_strength"] = over["minions"], over["per_strength"]


def calibrate(iters, games, gain=1.0):
    """고정점 반복: 계수 += gain × (실제 − 시뮬), 기본 공성 피해는 분당 평균 비율로."""
    import multiprocessing as mp
    import copy
    real = json.load(open(os.path.join(ROOT, "sim", "upper_value.json"), encoding="utf-8"))
    g0 = Game(seed=0)
    u, d = g0.p["lanes"]["upper"], g0.p["lanes"]["tower_damage"]
    over = {"grub": u["grub_per_stack"], "own": dict(u["herald"]["own"]), "opp": dict(u["herald"]["opp"]),
            "minions": d["minions"], "per_strength": d["per_strength"]}
    base_minions, base_ps = d["minions"], d["per_strength"]
    for it in range(iters):
        chunks = [(list(range(i, games, 4)), real["window_s"], over) for i in range(4)]
        with mp.Pool(4) as pool:
            rows = sum(pool.map(_sim_chunk, chunks), [])
        sim = fit(rows, real["window_s"])
        f, r = sim["fit"], real["fit"]
        print(it, json.dumps(over), {tg: (round(f[tg]["g_us"]["coef"], 3), round(f[tg]["h_us"]["coef"], 3), round(f[tg]["h_op"]["coef"], 3), f[tg]["mean_per_min"]) for tg in TARGETS}, flush=True)
        over = copy.deepcopy(over)
        over["grub"] += gain * (r["all"]["g_us"]["coef"] - f["all"]["g_us"]["coef"])
        for ln in ("top", "mid", "bot"):
            over["own"][ln] += gain * (r[ln]["h_us"]["coef"] - f[ln]["h_us"]["coef"])
            over["opp"][ln] += gain * (r[ln]["h_op"]["coef"] - f[ln]["h_op"]["coef"])
        sc = (r["all"]["mean_per_min"] / f["all"]["mean_per_min"]) ** 1.5
        over["minions"] *= sc
        over["per_strength"] *= sc
    return {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in over.items()}


if __name__ == "__main__":
    main()
