"""조언 검증: 학습 정글러의 추천과 같은 선택을 한 실제 천상계 정글러가 실제로 더 잘 됐나.

    python -m sim.advice_check [--policy models/stage12_a/policy_latest.npz] [--seeds 3]

표본은 sim/imitate.py와 같다(리플레이 경기 × 양 팀 정글러 × 3~17분 시작 시각, 상태는 실제 그대로 시뮬로 옮김).
결정마다
  - 추천 = 학습 정책의 가장 높은 확률 행동, 실제 = 그 초에 실제 정글러가 한 행동(sim/real_actions.py)
  - 시뮬 예측 차이 = 같은 상태·같은 시드에서 첫 결정만 추천 / 실제로 묶어 20분까지 돌린 20분 예측 승률 차이(추천 − 실제)
  - 실제 결과 = 실제 경기의 3분 뒤 예측 승률 변화(골드·경험치·오브젝트, sim/winprob.json)와 실제 승패
검사
  1. 같은 묶음을 고른 결정이 실제로 더 좋았나(지금 예측 승률을 통제한 회귀, 경기 단위 부트스트랩).
  2. 시뮬이 "실제 선택이 손해"라고 본 정도가 클수록 실제 결과가 나빴나(보정 기울기).
결과는 reports/advice_check.md, 결정별 표는 data/advice_rows.json(LLM 코치 묶음 재료).
"""
import argparse
import collections
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from . import imitate
from .agent import ACTION_NAMES
from .data import ROOT
from .ppo import Policy, masked_log_softmax, mlp_forward, HEADS
from .real_actions import GROUP_KO, GROUPS, group
from .stage6 import RealMatch
from .stage6_track import compact_raw
from .winprob import WinProb

AHEAD = 3   # 실제 결과를 보는 분


def main_probs(pol, obs, m):
    if len(obs) > pol.obs_dim:
        pol.grow(len(obs))
    z, _ = mlp_forward(pol.pi, obs[None, :].astype(np.float64))
    a, b = HEADS[0]
    return masked_log_softmax(z[0, a:b], m)[1]


def real_outcomes(path):
    """{(팀, 분): (그 분 예측 승률, AHEAD분 뒤 예측 승률, 승리)}."""
    rm = RealMatch(compact_raw(path))
    wp = WinProb()
    out = {}
    n = len(rm.frames)
    for team in (100, 200):
        win = rm.m["win100"] if team == 100 else 1 - rm.m["win100"]
        for m in range(n):
            p = wp.blue_from_diffs(rm.diffs(m), m * 60)
            out[(team, m)] = p if team == 100 else 1 - p
        out[(team, "win")] = win
    return out


_CF = {}


def _rollout(match, team, minute, forced, seed, horizon):
    """imitate._rollout과 같되 horizon초 뒤(최대 20분) 그 팀 예측 승률을 돌려준다."""
    path, kills = imitate._CF["inputs"][match]
    if match not in imitate._CF["rm"]:
        imitate._CF["rm"][match] = RealMatch(compact_raw(path))
    rm, pol = imitate._CF["rm"][match], imitate._CF["pol"]
    pol.rng = np.random.default_rng(seed)
    state = {"first": True}

    def fn(o, m, mw, ms):
        if state["first"]:
            state["first"] = False
            mm = np.zeros_like(m)
            mm[forced] = True
            return pol.act(o, mm, mw, ms)
        return pol.act(o, m, mw, ms)
    g, _ = imitate._state(rm, team, minute, kills, fn, seed)
    wp = WinProb()
    start = wp.for_team(g, team)
    end = min(1200, g.t + horizon)
    while g.t < end:
        g.step()
    return wp.for_team(g, team), start


def _job(job):
    """(추천 − 실제 20분, 실제 행동 그대로 3분 뒤 시뮬 Δ, 추천 − 실제 3분 뒤)."""
    match, team, minute, rec, real, seed, _ = job
    out = []
    for h in (1200, AHEAD * 60):
        b, s0 = _rollout(match, team, minute, real, seed, h)
        a = b if rec == real else _rollout(match, team, minute, rec, seed, h)[0]
        out.append((a - b, b - s0))
    return out[0][0], out[1][1], out[1][0]


def boot(rows, fn, n=2000, seed=0):
    """경기 단위 부트스트랩 → (값, 2.5%, 97.5%)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["match"]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    est = fn(rows)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        rs = [r for i in pick for r in by[keys[i]]]
        try:
            vals.append(fn(rs))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, lo, hi


def ols(rows, y, xs):
    """y ~ 1 + xs, 마지막 xs 계수."""
    X = np.array([[1.0] + [r[x] for x in xs] for r in rows])
    Y = np.array([r[y] for r in rows])
    return float(np.linalg.lstsq(X, Y, rcond=None)[0][-1])


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fmt(t, pct=True, d=1):
    k = 100 if pct else 1
    return f"{k * t[0]:+.{d}f} ({k * t[1]:+.{d}f} ~ {k * t[2]:+.{d}f})"


def main():
    if "--report-only" in os.sys.argv:   # data/advice_rows.json으로 보고서만 다시
        os.sys.argv.remove("--report-only")
        ap = argparse.ArgumentParser()
        ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz"))
        ap.add_argument("--seeds", type=int, default=8)
        ap.add_argument("--out", default=os.path.join(ROOT, "reports", "advice_check.md"))
        args = ap.parse_args()
        with open(os.path.join(ROOT, "data", "advice_rows.json"), encoding="utf-8") as f:
            return report(json.load(f), args)
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="시험용: 표본 수 제한")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "advice_check.md"))
    args = ap.parse_args()
    imitate.WITHIN = False
    inputs = imitate.match_inputs()
    with Pool(args.workers) as pool:
        S = [s for ss in pool.map(imitate._samples, inputs) for s in ss]
        outs = dict(zip([os.path.basename(p)[:-5] for p, _, _ in inputs], pool.map(real_outcomes, [p for p, _, _ in inputs])))
    if args.limit:
        S = S[:args.limit]
    pol = Policy.from_file(args.policy)
    rows = []
    for s in S:
        p = main_probs(pol, s["obs"], s["m"])
        rec = ACTION_NAMES[int(np.argmax(p))]
        o = outs[s["match"]]
        m = s["minute"]
        if (s["team"], m + AHEAD) not in o:
            continue
        gp = collections.defaultdict(float)
        for i, a in enumerate(ACTION_NAMES[:len(p)]):
            gp[group(a)] += p[i]
        rows.append({"match": s["match"], "team": s["team"], "minute": m, "real": s["action"], "rec": rec,
                     "p_real": float(p[imitate.ACTION_INDEX[s["action"]]]), "p_real_group": float(gp[group(s["action"])]),
                     "same": float(rec == s["action"]), "same_group": float(group(rec) == group(s["action"])),
                     "wp0": o[(s["team"], m)], "wp_ahead": o[(s["team"], m + AHEAD)], "win": o[(s["team"], "win")]})
    for r in rows:
        r["dwp"] = r["wp_ahead"] - r["wp0"]
        r["lg0"] = logit(r["wp0"])
    jobs = [(r["match"], r["team"], r["minute"], imitate.ACTION_INDEX[r["rec"]], imitate.ACTION_INDEX[r["real"]], 7000 * k + i, r["wp0"])
            for i, r in enumerate(rows) for k in range(args.seeds)]
    with Pool(args.workers, initializer=imitate._cf_init, initargs=(args.policy, inputs)) as pool:
        d = pool.map(_job, jobs, chunksize=4)
    for i, r in enumerate(rows):
        v = np.array(d[i * args.seeds:(i + 1) * args.seeds])
        r["sim_gap"] = float(v[:, 0].mean())      # 추천 − 실제, 20분 예측 승률
        r["sim_gap_sd"] = float(v[:, 0].std(ddof=1)) if len(v) > 1 else 0.0
        r["sim_gap_half"] = [float(v[::2, 0].mean()), float(v[1::2, 0].mean())]
        r["sim_dwp_real"] = float(v[:, 1].mean())  # 실제 행동 그대로 둔 시뮬의 AHEAD분 뒤 Δ(실제 Δ와 비교)
        r["sim_gap3"] = float(v[:, 2].mean())      # 추천 − 실제, AHEAD분 뒤
        r["sim_gap3_half"] = [float(v[::2, 2].mean()), float(v[1::2, 2].mean())]
    with open(os.path.join(ROOT, "data", "advice_rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    report(rows, args)


def report(rows, args):
    dis = [r for r in rows if not r["same"]]
    n_m = len({r["match"] for r in rows})
    L = ["# 조언 검증: AI 추천과 같은 선택이 실제로 더 좋았나", "",
         f"- 정책: `{os.path.relpath(args.policy, ROOT)}`. 표본: 리플레이 {n_m}판 정글러 결정 {len(rows)}개(3~17분 시작 시각).",
         f"- 실제 결과: {AHEAD}분 뒤 실제 예측 승률 변화(Δ, 골드·경험치·오브젝트로 계산)와 실제 승패. 괄호는 경기 단위 부트스트랩 95% 구간.",
         f"- 시뮬 예측 차이: 같은 상태·시드 {args.seeds}개에서 첫 결정만 추천/실제로 묶은 20분 예측 승률 차이(추천 − 실제).", "",
         "## 1. 일치율", "",
         f"- 세부 행동 같음 {np.mean([r['same'] for r in rows]):.0%}, 행동 묶음 같음 {np.mean([r['same_group'] for r in rows]):.0%}.",
         f"- 실제 행동에 정책이 준 확률 평균 {np.mean([r['p_real'] for r in rows]):.0%}(묶음 {np.mean([r['p_real_group'] for r in rows]):.0%}).", "",
         "## 2. 추천과 같은 묶음을 골랐을 때 실제 결과", "",
         "| 결과 | 같음 − 다름(통제 없음) | 지금 예측 승률 통제 |", "|---|---|---|"]
    L.append(f"| {AHEAD}분 뒤 Δ예측 승률 | {fmt(boot(rows, lambda rs: ols(rs, 'dwp', ['same_group'])))}%p | "
             f"{fmt(boot(rows, lambda rs: ols(rs, 'dwp', ['lg0', 'minute', 'same_group'])))}%p |")
    L.append(f"| 실제 승리 | {fmt(boot(rows, lambda rs: ols(rs, 'win', ['same_group'])))}%p | "
             f"{fmt(boot(rows, lambda rs: ols(rs, 'win', ['lg0', 'minute', 'same_group'])))}%p |")
    L.append(f"| {AHEAD}분 뒤 Δ, 실제 행동 확률(묶음)이 0→1 | | {fmt(boot(rows, lambda rs: ols(rs, 'dwp', ['lg0', 'minute', 'p_real_group'])))}%p |")
    L.append(f"| 실제 승리, 실제 행동 확률(묶음)이 0→1 | | {fmt(boot(rows, lambda rs: ols(rs, 'win', ['lg0', 'minute', 'p_real_group'])))}%p |")
    L += ["", "실제 행동 묶음별(같음 − 다름, 지금 예측 승률 통제, 3분 뒤 Δ):", "",
          "| 실제 행동 | 결정 | 추천과 같음 | 차이 |", "|---|---|---|---|"]
    for k in GROUPS:
        rs = [r for r in rows if group(r["real"]) == k]
        if len(rs) < 40 or not (5 <= sum(r["same_group"] for r in rs) <= len(rs) - 5):
            continue
        L.append(f"| {GROUP_KO[k]} | {len(rs)} | {np.mean([r['same_group'] for r in rs]):.0%} | "
                 f"{fmt(boot(rs, lambda q: ols(q, 'dwp', ['lg0', 'minute', 'same_group']), n=1000))}%p |")
    L += ["", "## 3. 시뮬이 본 손해 크기 vs 실제 결과(추천과 다른 결정만)", "",
          f"- 추천과 다른 결정 {len(dis)}개. 시뮬 예측 차이(추천 − 실제) 평균 {100 * np.mean([r['sim_gap'] for r in dis]):+.1f}%p, "
          f"같은 결정의 시드 사이 표준편차 평균 {100 * np.mean([r['sim_gap_sd'] for r in dis]):.1f}%p.",
          "- 보정 기울기: 실제 Δ ~ 지금 예측 승률 + 분 + 시뮬 예측 차이. 시뮬이 맞으면 음수(시뮬이 손해라 본 만큼 실제로 나빠짐).", "",
          f"| 결과 | 시뮬 예측 차이 1%p당 |", "|---|---|",
          f"| {AHEAD}분 뒤 Δ예측 승률 | {fmt(boot(dis, lambda rs: ols(rs, 'dwp', ['lg0', 'minute', 'sim_gap'])), pct=False, d=3)} |",
          f"| 실제 승리 | {fmt(boot(dis, lambda rs: ols(rs, 'win', ['lg0', 'minute', 'sim_gap'])), pct=False, d=3)} |", "",
          "시뮬 예측 차이 크기별(추천과 다른 결정):", "",
          "| 시뮬 예측 차이(추천 − 실제) | 결정 | 시뮬 평균 | 실제 3분 뒤 Δ 평균 | 실제 승률 | 지금 예측 승률 평균 |", "|---|---|---|---|---|---|"]
    def rel(key):
        a = np.array([r[key][0] for r in dis]); b = np.array([r[key][1] for r in dis])
        c = float(np.corrcoef(a, b)[0, 1])
        return c, 2 * c / (1 + c)   # 반쪽 상관, 시드 전체 평균의 신뢰도(스피어만-브라운)
    r20, r3 = rel("sim_gap_half"), rel("sim_gap3_half")
    L4 = ["", "## 4. 시뮬 예측 차이는 믿을 만한가", "",
          "- 같은 결정에서 시드 절반끼리 예측 차이 상관(반쪽 상관) → 시드 전체 평균의 신뢰도(0=잡음, 1=시드 바꿔도 같음).", "",
          "| 예측 | 반쪽 상관 | 신뢰도 |", "|---|---|---|",
          f"| 20분 예측 승률 차이 | {r20[0]:.2f} | {r20[1]:.2f} |",
          f"| {AHEAD}분 뒤 예측 승률 차이 | {r3[0]:.2f} | {r3[1]:.2f} |", "",
          f"- 시뮬 재현도: 실제 행동을 그대로 둔 시뮬의 {AHEAD}분 뒤 Δ와 실제 {AHEAD}분 뒤 Δ 상관 "
          f"{float(np.corrcoef([r['sim_dwp_real'] for r in rows], [r['dwp'] for r in rows])[0, 1]):.2f}"
          f"(시뮬 평균 {100 * np.mean([r['sim_dwp_real'] for r in rows]):+.1f}%p, 실제 평균 {100 * np.mean([r['dwp'] for r in rows]):+.1f}%p, "
          f"표준편차 시뮬 {100 * np.std([r['sim_dwp_real'] for r in rows]):.1f} / 실제 {100 * np.std([r['dwp'] for r in rows]):.1f}%p).",
          f"- {AHEAD}분 뒤 예측 차이 보정 기울기(실제 Δ ~ 지금 예측 승률 + 분 + 시뮬 {AHEAD}분 차이): "
          f"{fmt(boot(dis, lambda rs: ols(rs, 'dwp', ['lg0', 'minute', 'sim_gap3'])), pct=False, d=3)}."]
    qs = np.quantile([r["sim_gap"] for r in dis], [0, 1 / 3, 2 / 3, 1]) if dis else []
    for i in range(3 if dis else 0):
        rs = [r for r in dis if qs[i] <= r["sim_gap"] <= qs[i + 1]]
        L.append(f"| {['실제 쪽이 낫다고 봄', '비슷', '추천이 낫다고 봄'][i]} ({100 * qs[i]:+.1f}~{100 * qs[i + 1]:+.1f}%p) | {len(rs)} | "
                 f"{100 * np.mean([r['sim_gap'] for r in rs]):+.1f}%p | {100 * np.mean([r['dwp'] for r in rs]):+.1f}%p | "
                 f"{np.mean([r['win'] for r in rs]):.0%} | {np.mean([r['wp0'] for r in rs]):.0%} |")
    L += L4
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
