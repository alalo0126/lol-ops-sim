"""조언 검증 2: 결정 하나가 아니라 상황 칸마다 범주별 가치(시뮬)와 실제 1,000판 결과를 나란히 낸다.

    python -m sim.advice_cells [--states 40] [--reps 3]          :: → sim/advice_cells.json, reports/advice_cells.md
    python -m sim.advice_cells --report-only

결정 하나마다의 시뮬 예측 차이는 거의 잡음이라(reports/advice_check.md, 신뢰도 0.1) 같은 상황 칸으로 모아 잡음을 줄인다.
칸 = stage6.situation의 시기 × 드래곤 × 팀 골드 × 정글러 상태(4 × 3 × 3 × 3).
시뮬: 칸마다 실제 상황 최대 --states개 × 반복 --reps회. 같은 상태·같은 시드에서 첫 1분을 범주로 묶은 것과
      학습 정책 그대로(free)를 20분까지 돌려 20분 예측 승률 차이(sim_vs_ai)를 낸다. sim_vs_mix는 천상계 선택 비율로
      섞은 평균과 비교한 차이(어느 범주가 그 칸의 평균적인 실제 선택보다 나은가).
실제: 1단계 1,000판 그 칸의 모든 정글러-분. 범주를 고른 경우 − 안 고른 경우(지금 예측 승률 통제 회귀)의
      실제 승패 차이와 3분 뒤 예측 승률 변화 차이.
verified = 시뮬 95% 구간과 실제 3분 뒤 95% 구간이 둘 다 0을 벗어나고 부호가 같음. contradicted = 둘 다 벗어났는데 반대.
실제 차이는 선택 편향(필요해서 고른 귀환 등)이 섞여 있으니 인과 효과가 아니다.
"""
import argparse
import collections
import json
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .ppo import Policy
from .stage6 import CATS, CAT_KO, RealMatch, label, load_compact, rollout_value, situation
from .winprob import WinProb

MINUTES = range(3, 18)
KEYS = ["phase", "dragon", "gold", "need"]
AHEAD = 3
POLICY = os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz")
OUT = os.path.join(ROOT, "sim", "advice_cells.json")
_S = {}


def _init(policy):
    _S["matches"] = load_compact()
    _S["wp"] = WinProb()
    _S["pol"] = Policy.from_file(policy)
    _S["nodes"] = Game(seed=0).map.nodes


def cell_key(sit):
    return "|".join(sit[k] for k in KEYS)


# ---------------------------------------------------------------------- 실제
def _real(mi):
    m = _S["matches"][mi]
    rm = RealMatch(m)
    if not rm.valid:
        return []
    wp, nodes = _S["wp"], _S["nodes"]
    out = []
    for team in (100, 200):
        s = 1 if team == 100 else -1

        def phi(k):
            p = wp.blue_from_diffs(rm.diffs(k), k * 60)
            return p if s == 1 else 1 - p
        for minute in MINUTES:
            if len(rm.frames) <= minute + AHEAD or m["duration"] <= (minute + AHEAD) * 60:
                continue
            out.append({"mi": mi, "team": team, "minute": minute, "cell": cell_key(situation(rm, team, minute)),
                        "real": label(rm.window(team, minute, nodes), team, nodes), "phi": phi(minute),
                        "dphi3": phi(minute + AHEAD) - phi(minute), "win": m["win100"] if team == 100 else 1 - m["win100"]})
    return out


def ols_diff(rows, y, cat):
    """y ~ 1 + logit(phi) + minute + [real == cat] → (계수, 표준오차)."""
    def lg(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))
    X = np.array([[1.0, lg(r["phi"]), r["minute"], float(r["real"] == cat)] for r in rows])
    Y = np.array([r[y] for r in rows])
    if X[:, 3].sum() < 5 or X[:, 3].sum() > len(rows) - 5:
        return None, None
    if len({r["minute"] for r in rows}) == 1:
        X = X[:, [0, 1, 3]]
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ b
    s2 = res @ res / max(1, len(Y) - X.shape[1])
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None, None
    return float(b[-1]), float(math.sqrt(max(cov[-1, -1], 0)))


# ---------------------------------------------------------------------- 시뮬
def _sim(job):
    mi, team, minute, reps, seed0 = job
    rm = RealMatch(_S["matches"][mi])
    wp, pol = _S["wp"], _S["pol"]
    out = {"free": [], "free_lab": []}
    for cat in CATS:
        out[cat], out[cat + "_started"] = [], []
    for r in range(reps):
        seed = seed0 + r
        for cat in [None] + CATS:
            pol.rng = np.random.default_rng(seed)
            v, lab, _ = rollout_value(rm, team, minute, lambda o, m, mw, ms: pol.act(o, m, mw, ms), wp, seed=seed, cat=cat)
            if cat is None:
                out["free"].append(v)
                out["free_lab"].append(lab)
            else:
                out[cat].append(v)
                out[cat + "_started"].append(bool(lab[1]) if lab else False)
    return out


def mean_ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None, [None, None]
    se = v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)]


def excl0(ci):
    return ci[0] is not None and (ci[0] > 0 or ci[1] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-real", type=int, default=60, help="실제 정글러-분이 이보다 적은 칸은 시뮬 안 돌림")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        with open(OUT, encoding="utf-8") as f:
            return report(json.load(f))
    t = time.time()
    _init(args.policy)
    with Pool(args.workers, initializer=_init, initargs=(args.policy,)) as pool:
        real = [r for part in pool.imap(_real, range(len(_S["matches"])), chunksize=8) for r in part]
        print(f"실제 {len(real)}개 ({time.time() - t:.0f}초)", flush=True)
        by = collections.defaultdict(list)
        for r in real:
            by[r["cell"]].append(r)
        rng = random.Random(77)
        jobs, owner = [], []
        for key in sorted(by):
            if len(by[key]) < args.min_real:
                continue
            pick = rng.sample(by[key], min(args.states, len(by[key])))
            for i, r in enumerate(pick):
                jobs.append((r["mi"], r["team"], r["minute"], args.reps, 3_000_000 + len(jobs) * 100))
                owner.append(key)
        print(f"시뮬 칸 {len(set(owner))}개, 상태 {len(jobs)}개 × (1 + {len(CATS)}) × {args.reps}", flush=True)
        sims = collections.defaultdict(list)
        for k, res in zip(owner, pool.imap(_sim, jobs, chunksize=2)):
            sims[k].append(res)
            if len(sims) and sum(len(v) for v in sims.values()) % 200 == 0:
                print(f"  {sum(len(v) for v in sims.values())}/{len(jobs)} ({time.time() - t:.0f}초)", flush=True)
    cells = {}
    for key in sorted(by):
        rows = by[key]
        share = collections.Counter(r["real"] for r in rows)
        c = {"n_real": len(rows), "n_sim_states": len(sims.get(key, [])), "reps": args.reps, "cats": {}}
        S = sims.get(key, [])
        if S:
            labs = collections.Counter(l for s in S for l in s["free_lab"])
            c["ai_choice"] = {k: labs[k] / sum(labs.values()) for k in CATS}
            # 상태·시드 짝마다: 범주 값 − free, 범주 값 − 실제 선택 비율 섞은 평균
            w = {k: share[k] / len(rows) for k in CATS}
            for cat in CATS:
                d_ai = [v - f for s in S for v, f in zip(s[cat], s["free"])]
                d_mix = [s[cat][i] - sum(w[k] * s[k][i] for k in CATS) for s in S for i in range(len(s["free"]))]
                m1, ci1 = mean_ci(d_ai)
                m2, ci2 = mean_ci(d_mix)
                c["cats"][cat] = {"sim_vs_ai": m1, "sim_vs_ai_ci": ci1, "sim_vs_mix": m2, "sim_vs_mix_ci": ci2,
                                  "started": float(np.mean([x for s in S for x in s[cat + "_started"]]))}
            c["best_sim"] = max(CATS, key=lambda k: c["cats"][k]["sim_vs_ai"])
        for cat in CATS:
            e = c["cats"].setdefault(cat, {})
            e["real_share"] = share[cat] / len(rows)
            e["real_n"] = share[cat]
            for y, name in (("win", "real_win"), ("dphi3", "real_dwp3")):
                b, se = ols_diff(rows, y, cat)
                e[name + "_diff"] = b
                e[name + "_ci"] = [b - 1.96 * se, b + 1.96 * se] if b is not None else [None, None]
            sim_ci = e.get("sim_vs_mix_ci", [None, None])
            both = excl0(sim_ci) and excl0(e["real_dwp3_ci"])
            same = both and (e["sim_vs_mix"] > 0) == (e["real_dwp3_diff"] > 0)
            e["verified"] = bool(same)
            e["contradicted"] = bool(both and not same)
        cells[key] = c
    out = {"format": "advice_cells/1", "policy": os.path.relpath(args.policy, ROOT), "cell_keys": KEYS, "key_sep": "|",
           "outcome_ahead_min": AHEAD, "cats": CATS, "cat_ko": CAT_KO, "cells": cells}
    out["validation"] = validation(cells)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    report(out)


def validation(cells):
    """칸 × 범주 단위로 시뮬 가치(sim_vs_mix)와 실제 3분 뒤 차이·승패 차이의 가중 상관(정밀도 가중)."""
    xs, ys, ws, yw = [], [], [], []
    for c in cells.values():
        for cat, e in c["cats"].items():
            if e.get("sim_vs_mix") is None or e.get("real_dwp3_diff") is None:
                continue
            se = (e["real_dwp3_ci"][1] - e["real_dwp3_ci"][0]) / 3.92
            xs.append(e["sim_vs_mix"])
            ys.append(e["real_dwp3_diff"])
            yw.append(e["real_win_diff"] if e["real_win_diff"] is not None else np.nan)
            ws.append(1 / max(se, 1e-4) ** 2)
    xs, ys, ws, yw = map(np.array, (xs, ys, ws, yw))

    def wcorr(a, b, w):
        ok = ~np.isnan(b)
        a, b, w = a[ok], b[ok], w[ok]
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        return float(np.average((a - ma) * (b - mb), weights=w) / math.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(xs), len(xs))
        bs.append(wcorr(xs[i], ys[i], ws[i]))
    n_v = sum(e.get("verified", False) for c in cells.values() for e in c["cats"].values())
    n_c = sum(e.get("contradicted", False) for c in cells.values() for e in c["cats"].values())
    n_sim = sum(excl0(e.get("sim_vs_mix_ci", [None, None])) for c in cells.values() for e in c["cats"].values())
    return {"pairs": int(len(xs)), "corr_sim_vs_real_dwp3": wcorr(xs, ys, ws), "corr_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "corr_sim_vs_real_win": wcorr(xs, yw, ws), "sign_agree": float(np.average(np.sign(xs) == np.sign(ys), weights=ws)),
            "sim_significant": int(n_sim), "verified": int(n_v), "contradicted": int(n_c)}


def report(out):
    cells, v = out["cells"], out["validation"]
    ko = out["cat_ko"]
    simmed = {k: c for k, c in cells.items() if c["n_sim_states"]}
    L = ["# 조언 검증 2: 상황 칸별 범주 가치(시뮬) vs 실제 1,000판", "",
         f"- 정책 `{out['policy']}`. 칸 {len(cells)}개 중 시뮬을 돌린 칸 {len(simmed)}개(실제 정글러-분 60개 이상), "
         f"칸마다 상태 최대 {max(c['n_sim_states'] for c in simmed.values())}개 × 반복 {next(iter(simmed.values()))['reps']}회.",
         "- 시뮬 가치 = 첫 1분을 그 범주로 묶었을 때 20분 예측 승률 − 그 칸 천상계 선택 비율로 섞은 평균(같은 상태·시드).",
         f"- 실제 차이 = 1,000판에서 그 범주를 고른 경우 − 안 고른 경우, {out['outcome_ahead_min']}분 뒤 예측 승률 변화(지금 예측 승률·분 통제). 선택 편향 포함.", "",
         "## 시뮬이 칸별로 좋다고 본 범주가 실제로도 좋았나", "",
         f"- 칸 × 범주 {v['pairs']}쌍. 시뮬 가치와 실제 3분 뒤 차이의 정밀도 가중 상관 **{v['corr_sim_vs_real_dwp3']:+.2f}** "
         f"(95% {v['corr_ci'][0]:+.2f} ~ {v['corr_ci'][1]:+.2f}), 실제 승패 차이와 {v['corr_sim_vs_real_win']:+.2f}. 부호 일치 {v['sign_agree']:.0%}.",
         f"- 시뮬 구간이 0을 벗어난 쌍 {v['sim_significant']}개 → 실제와 같은 방향으로 확인(verified) {v['verified']}개, 반대(contradicted) {v['contradicted']}개.", "",
         "## 범주별 요약(시뮬 칸 평균, 실제 비율 가중)", "",
         "| 범주 | 시뮬 가치 평균 | 시뮬이 그 칸 최고로 본 칸 | AI 자유 선택 비율 | 천상계 선택 비율 | 실제 3분 뒤 차이 평균 | verified | contradicted |",
         "|---|---|---|---|---|---|---|---|"]
    tot = sum(c["n_real"] for c in simmed.values())
    for cat in out["cats"]:
        es = [(c, c["cats"][cat]) for c in simmed.values()]
        wsum = lambda f: sum(c["n_real"] * f(c, e) for c, e in es if f(c, e) is not None) / tot
        L.append(f"| {ko[cat]} | {100 * wsum(lambda c, e: e['sim_vs_mix']):+.1f}%p | {sum(c.get('best_sim') == cat for c, _ in es)} | "
                 f"{wsum(lambda c, e: c['ai_choice'][cat]):.0%} | {wsum(lambda c, e: e['real_share']):.0%} | "
                 f"{100 * wsum(lambda c, e: e['real_dwp3_diff']):+.1f}%p | {sum(e['verified'] for _, e in es)} | {sum(e['contradicted'] for _, e in es)} |")
    L += ["", "## verified / contradicted 쌍", "", "| 칸 | 범주 | 시뮬 가치(95%) | 실제 3분 뒤 차이(95%) | 실제 승패 차이 | 천상계 비율(n) | 판정 |", "|---|---|---|---|---|---|---|"]
    for k, c in sorted(simmed.items()):
        for cat, e in c["cats"].items():
            if e["verified"] or e["contradicted"]:
                L.append(f"| {k} | {ko[cat]} | {100 * e['sim_vs_mix']:+.1f} ({100 * e['sim_vs_mix_ci'][0]:+.1f}~{100 * e['sim_vs_mix_ci'][1]:+.1f}) | "
                         f"{100 * e['real_dwp3_diff']:+.1f} ({100 * e['real_dwp3_ci'][0]:+.1f}~{100 * e['real_dwp3_ci'][1]:+.1f}) | "
                         f"{100 * e['real_win_diff']:+.1f} | {e['real_share']:.0%} ({e['real_n']}) | {'확인' if e['verified'] else '반대'} |")
    dst = os.path.join(ROOT, "reports", "advice_cells.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
