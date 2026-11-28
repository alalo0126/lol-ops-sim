"""점수판으로 추정한 팀 골드·경험치 차이(맵핵 금지, 공정 비교 설계 2장).

실제 게임에서 상대 골드·경험치는 보이지 않는다. 보이는 건 점수판의 CS·정글 CS·킬·어시스트·레벨이다.
학습 정글러 관측은 정확한 차이 대신 이 값들의 분별 선형 회귀 추정을 쓴다(보상은 그대로 실제 승률).
계수는 규칙 봇 시뮬 경기로 맞춘다: `python -m sim.scoreboard --fit` → sim/scoreboard_fit.json
(시뮬 경기 기준이라 실제 경기 오차와는 다를 수 있음).
"""
import argparse
import json
import os

import numpy as np

from .data import load_json

FEATS = ["lane_cs", "jungle_cs", "kills", "assists", "level"]
PATH = os.path.join(os.path.dirname(__file__), "scoreboard_fit.json")


def _diffs(champs, get):
    d = {k: 0.0 for k in FEATS}
    for c in champs:
        s = 1 if get(c, "team") == 100 else -1
        for k in FEATS:
            d[k] += s * get(c, k)
    return d


class Scoreboard:
    def __init__(self, table=None):
        w = table or load_json("sim", "scoreboard_fit.json")
        self.minutes = {int(m): v for m, v in w["minutes"].items()}
        self.lo, self.hi = min(self.minutes), max(self.minutes)

    def diffs(self, game):
        """WinProb.diffs와 같은 형식(블루 − 레드). 골드·경험치는 점수판 추정, 오브젝트는 전체 알림이라 실제 값."""
        from .winprob import WinProb
        d = WinProb.diffs(game)
        x = _diffs(game.champs, lambda c, k: getattr(c, k))
        m = min(max(game.t / 60.0, self.lo), self.hi)
        a = int(np.floor(m))
        b = min(a + 1, self.hi)
        fr = m - a
        for tgt in ("gold", "xp"):
            ca, cb = self.minutes[a][tgt], self.minutes[b][tgt]
            va = ca["intercept"] + sum(ca["coef"][k] * x[k] for k in FEATS)
            vb = cb["intercept"] + sum(cb["coef"][k] * x[k] for k in FEATS)
            d[tgt] = (1 - fr) * va + fr * vb
        return d


def fit(seeds=200):
    from .game import Game
    rows = {}
    for seed in range(seeds):
        g = Game(seed=1000 + seed).run(1200)
        for fr in g.frames:
            m = fr["t"] // 60
            if m < 1:
                continue
            p = list(fr["p"].values())
            x = _diffs(p, lambda c, k: c[k] if k != "team" else c["team"])
            y = {"gold": sum((1 if c["team"] == 100 else -1) * c["gold"] for c in p),
                 "xp": sum((1 if c["team"] == 100 else -1) * c["xp"] for c in p)}
            rows.setdefault(m, []).append((x, y))
    out = {"_doc": "sim/scoreboard.py fit(): 규칙 봇 시뮬 경기 분별 회귀. 골드·경험치 팀 차이 ~ 점수판 차이(CS·정글 CS·킬·어시스트·레벨)",
           "seeds": seeds, "minutes": {}}
    for m, R in sorted(rows.items()):
        X = np.array([[r[0][k] for k in FEATS] + [1.0] for r in R])
        out["minutes"][m] = {}
        for tgt in ("gold", "xp"):
            y = np.array([r[1][tgt] for r in R])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            out["minutes"][m][tgt] = {"coef": {k: round(float(v), 3) for k, v in zip(FEATS, b[:-1])},
                                      "intercept": round(float(b[-1]), 1), "resid_sd": round(float(res.std()), 1),
                                      "r2": round(float(1 - res.var() / max(y.var(), 1e-9)), 3)}
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    if a.fit:
        o = fit(a.seeds)
        for m in (5, 10, 15, 20):
            print(m, {t: (o["minutes"][m][t]["r2"], o["minutes"][m][t]["resid_sd"]) for t in ("gold", "xp")})
