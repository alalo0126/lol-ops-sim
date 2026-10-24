"""굳은 행동 점검: 학습 정글러가 거의 고르지 않는 주 행동(고를 수 있을 때 평균 확률 < 0.3%)을 찾아,
그 행동을 한 번 강제로 골랐을 때 20분 예측 승률이 얼마나 바뀌는지 같은 시드로 비교한다.

    python -m sim.stuck_check --policy models/stage9_b/policy_latest.npz --games 200   :: → reports/stuck_check.md

1) 확률 조사: 경기마다 결정 시점의 주 행동 확률을 고를 수 있을 때만 평균.
2) 강제: 행동마다 같은 시드 경기 쌍 — 기준(정책 그대로) vs 3~18분 무작위 시각 이후 고를 수 있는 결정에서 그 행동을 최대 3번(30초 이상 간격) 강제.
   차이(강제 − 기준) 평균과 95% 구간. 구간이 0보다 아래면 '진짜 나쁜 행동', 위면 '굳은 행동'(학습이 가치를 잘못 알고 있음),
   0을 걸치면 '차이 작음'(드물게 해도 손해가 크지 않음 → 바닥으로 계속 시도하게 두면 됨).
   한 번 강제하는 것이라 효과가 작고, 강제 뒤 경기 진행이 갈라져 잡음이 크다(평균은 치우치지 않음).
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from .agent import AgentJungler, ACTION_NAMES, N_MAIN
from .data import ROOT
from .game import Game
from .ppo import Policy, masked_log_softmax, mlp_forward
from .winprob import WinProb

RARE = 0.003
N_FORCE = 3      # 강제 횟수(30초 이상 간격)
_POL = None


def _init(path):
    global _POL
    _POL = Policy.from_file(path)


def _probs(o, m):
    if len(o) > _POL.obs_dim:
        _POL.grow(len(o))
    z, _ = mlp_forward(_POL.pi, o[None].astype(np.float64))
    _, p = masked_log_softmax(z[0, :N_MAIN], m)
    return p


def _game(seed, force=None, t_force=None, survey=False):
    team = 100 if seed % 2 == 0 else 200
    _POL.rng = np.random.default_rng(seed)
    st = {"forced": 0, "last": -999, "sum": np.zeros(N_MAIN), "n": np.zeros(N_MAIN)}
    holder = {}

    def fn(o, m, mw, ms):
        g = holder["g"]
        if survey:
            p = _probs(o, m)
            st["sum"] += np.where(m, p, 0.0)
            st["n"] += m
        a, w, s, rec = _POL.act(o, m, mw, ms)
        if force is not None and st["forced"] < N_FORCE and g.t >= t_force and g.t - st["last"] >= 30 and m[force]:
            st["forced"] += 1
            st["last"] = g.t
            a = force
        return a, w, s, rec

    def make(g, c):
        holder["g"] = g
        holder["bot"] = AgentJungler(g, c, fn, WinProb())
        return holder["bot"]
    g = Game(seed=seed, jungler_override={team: make}).run()
    phi = holder["bot"].finish_episode()
    return phi, st


def _survey(seed):
    _, st = _game(seed, survey=True)
    return st["sum"], st["n"]


def _pair(job):
    seed, a = job
    t_force = 180 + (seed * 7919) % 720
    base, _ = _game(seed)
    forced, st = _game(seed, force=a, t_force=t_force)
    return a, forced - base, st["forced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage9_b", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--survey-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "stuck_check.md"))
    a = ap.parse_args()
    with Pool(a.workers, initializer=_init, initargs=(a.policy,)) as pool:
        res = pool.map(_survey, range(900000, 900000 + a.survey_games))
        tot, cnt = sum(r[0] for r in res), sum(r[1] for r in res)
        mean_p = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        legal = cnt / max(1, cnt.max())
        rare = [i for i in range(N_MAIN) if cnt[i] > 0 and mean_p[i] < RARE and legal[i] > 0.05]
        jobs = [(s, i) for i in rare for s in range(910000, 910000 + a.games)]
        pairs = pool.map(_pair, jobs, chunksize=8)
    rows = []
    for i in rare:
        d = np.array([x for k, x, f in pairs if k == i and f])
        if len(d) < 10:
            rows.append((i, len(d), np.nan, np.nan, np.nan, "강제 기회 부족"))
            continue
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        verdict = "진짜 나쁜 행동" if m + 1.96 * se < 0 else ("굳은 행동(되살릴 후보)" if m - 1.96 * se > 0 else "차이 작음")
        rows.append((i, len(d), m, m - 1.96 * se, m + 1.96 * se, verdict))
    lines = [f"# 굳은 행동 점검", "", f"- 정책 `{os.path.relpath(a.policy, ROOT)}`(바닥 {Policy.from_file(a.policy).floor}), 확률 조사 {a.survey_games}판, 행동마다 강제 {a.games}쌍.",
             f"- 드문 행동 = 고를 수 있는 결정의 5% 이상에서 열려 있는데 평균 확률 < {RARE:.1%}.", "",
             "| 행동 | 열린 비율 | 평균 확률 | 강제 쌍 | 20분 승률 차(강제 − 기준) | 95% 구간 | 판정 |", "|---|---|---|---|---|---|---|"]
    for i, n, m, lo, hi, v in rows:
        lines.append(f"| {ACTION_NAMES[i]} | {legal[i]:.0%} | {mean_p[i]:.4%} | {n} | {m:+.4f} | {lo:+.4f}~{hi:+.4f} | {v} |")
    lines += ["", "## 모든 주 행동 평균 확률(고를 수 있을 때)", "", "| 행동 | 열린 비율 | 평균 확률 |", "|---|---|---|"]
    for i in np.argsort(-np.nan_to_num(mean_p)):
        lines.append(f"| {ACTION_NAMES[i]} | {legal[i]:.0%} | {mean_p[i]:.3%} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(rows)]))


if __name__ == "__main__":
    main()
