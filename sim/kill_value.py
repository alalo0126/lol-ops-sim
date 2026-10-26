"""킬 하나가 3분 동안 만드는 골드·경험치 차이: 실제 1,000판 vs 시뮬.

    python -m sim.kill_value [--policy models/stage15_a/policy_latest.npz] [--games 300]   :: → reports/kill_value.md

팀-분 행(3~13분): 3분 뒤 팀 골드 차 변화 ~ 지금 골드 차 + 그 분 정글러 관여 킬 차 + 정글러 비관여 킬 차.
킬 계수 = 그 킬 하나가 3분 동안 만든 골드(경험치) 차이. 괄호 = 경기(시뮬은 시드) 단위 부트스트랩 95% 구간.
시뮬 승률 모델(sim/winprob.json)은 골드·경험치 차로 승률을 매기므로, 시뮬 킬이 실제보다 골드를 많이 만들면 갱 가치가 부풀려진다.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .evaluate import apply_overrides
from .game import Game

M0, M1, AHEAD = 3, 13, 3
_S = {}


def rows_from(minutes, kills, jg, team_of, gid):
    """minutes[m] = {team: (gold, xp)}; kills = [(t_s, killer, victim, assists)]; jg = {team: pid}."""
    out = []
    for m in range(M0, M1 + 1):
        if m + AHEAD not in minutes or m not in minutes:
            continue
        for T in (100, 200):
            F = 300 - T
            g0 = minutes[m][T][0] - minutes[m][F][0]
            g3 = minutes[m + AHEAD][T][0] - minutes[m + AHEAD][F][0]
            x0 = minutes[m][T][1] - minutes[m][F][1]
            x3 = minutes[m + AHEAD][T][1] - minutes[m + AHEAD][F][1]
            jk = lk = 0
            for t, k, v, a in kills:
                if not (m * 60 <= t < (m + 1) * 60) or team_of.get(v) is None or v in jg.values():
                    continue
                sign = 1 if team_of.get(v) == F else -1
                involved = jg[300 - team_of[v]] in [k] + list(a)
                if involved:
                    jk += sign
                else:
                    lk += sign
            out.append({"g": gid, "dg": g0 / 1000, "d3": (g3 - g0) / 1000, "dx3": (x3 - x0) / 1000, "jk": jk, "lk": lk})
    return out


def real_rows():
    from .macro_roles import load
    out = []
    for m in load():
        parts = {p[0]: p for p in m["parts"]}
        jg = {p[1]: p[0] for p in m["parts"] if p[2] == "JUNGLE"}
        if len(jg) != 2 or len(m["frames"]) <= M1 + AHEAD:
            continue
        team_of = {pid: p[1] for pid, p in parts.items()}
        minutes = {}
        for i, fr in enumerate(m["frames"][:M1 + AHEAD + 1]):
            minutes[i] = {T: (sum(fr[str(q)][6] for q in parts if parts[q][1] == T), sum(fr[str(q)][5] for q in parts if parts[q][1] == T)) for T in (100, 200)}
        kills = [(k[0] / 1000, k[1], k[2], k[3]) for k in m["events"]["kill"]]
        out += rows_from(minutes, kills, jg, team_of, m["match"])
    return out


def _init(policy, overrides):
    from .ppo import Policy
    _S["pol"] = Policy.from_file(policy) if policy else None
    _S["ov"] = overrides


def _one(seed):
    from .agent import AgentJungler
    from .data import Rules
    from .winprob import WinProb
    rules = apply_overrides(Rules(), _S["ov"]) if _S["ov"] else None
    team = 100 if seed % 2 == 0 else 200
    override = {}
    if _S["pol"] is not None:
        pol = _S["pol"]
        pol.rng = np.random.default_rng(seed)
        override = {team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())}
    g = Game(seed=seed, rules=rules, jungler_override=override)
    minutes = {}
    while g.t <= (M1 + AHEAD) * 60:
        if g.t % 60 == 0:
            minutes[g.t // 60] = {T: (sum(500 + c.earned for c in g.champs if c.team == T), sum(c.xp for c in g.champs if c.team == T)) for T in (100, 200)}
        g.step()
    jg = {c.team: c.pid for c in g.champs if c.role == "JUNGLE"}
    team_of = {c.pid: c.team for c in g.champs}
    kills = [(e["t"], e["killer"], e["victim"], e["assists"]) for e in g.events if e["type"] == "kill"]
    return rows_from(minutes, kills, jg, team_of, seed)


def fit(rows, key, n=400, seed=0):
    def ols(sub):
        X = np.array([[1.0, r["dg"], r["jk"], r["lk"]] for r in sub])
        y = np.array([r[key] for r in sub])
        return np.linalg.lstsq(X, y, rcond=None)[0]
    w = ols(rows)
    by = {}
    for r in rows:
        by.setdefault(r["g"], []).append(r)
    ks = list(by)
    rng = np.random.default_rng(seed)
    bs = np.array([ols([r for i in rng.integers(0, len(ks), len(ks)) for r in by[ks[i]]]) for _ in range(n)])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    return w, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage15_a", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--set", action="append", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "kill_value.md"))
    args = ap.parse_args()
    real = real_rows()
    res = {}
    for name, pol in (("rule", None), ("ai", args.policy)):
        with Pool(os.cpu_count(), initializer=_init, initargs=(pol, args.set)) as pool:
            res[name] = [r for rs in pool.map(_one, range(args.games)) for r in rs]
    L = ["# 킬 하나가 3분 동안 만드는 골드·경험치 차이", "",
         f"- 실제: 1단계 1,000판 팀-분 {len(real)}개. 시뮬: {args.games}판씩(규칙 봇끼리 / 한 팀 학습 정글러 `{os.path.relpath(args.policy, ROOT)}`).",
         f"- 행 = 팀-분({M0}~{M1}분). 종속 = {AHEAD}분 뒤 팀 골드(경험치) 차 변화. 설명 = 지금 골드 차(천), 그 분 정글러 관여 킬 차, 정글러 비관여 킬 차(정글러 사망 제외).",
         "- 괄호 = 경기(시드) 단위 부트스트랩 95% 구간. 단위 골드·경험치.", "",
         "| 표본 | 정글러 관여 킬 1개 → 골드 | 비관여 킬 → 골드 | 정글러 관여 킬 → 경험치 | 비관여 킬 → 경험치 | 지금 골드 차 천당 3분 뒤 변화 |",
         "|---|---|---|---|---|---|"]
    for name, rows in (("실제", real), ("규칙 봇끼리", res["rule"]), ("학습 정글러 경기", res["ai"])):
        wg, lg, hg = fit(rows, "d3")
        wx, lx, hx = fit(rows, "dx3")
        c = lambda w, lo, hi, i: f"{w[i] * 1000:+.0f} ({lo[i] * 1000:+.0f}~{hi[i] * 1000:+.0f})"
        L.append(f"| {name} | {c(wg, lg, hg, 2)} | {c(wg, lg, hg, 3)} | {c(wx, lx, hx, 2)} | {c(wx, lx, hx, 3)} | {wg[1] * 1000:+.0f} ({lg[1] * 1000:+.0f}~{hg[1] * 1000:+.0f}) |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
