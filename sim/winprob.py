"""5단계 보상: 20분 예측 승률과 그 잠재 함수.

sim/winprob.json(tools/fit_winprob.py, 1단계 1,000판)의 분별 로지스틱 회귀를 쓴다. 분 사이는 로그오즈를
선형 보간하고 1분 전은 1분 모델을 그대로 쓴다. 입력은 시뮬레이터가 만드는 팀 차이만: 골드·경험치·드래곤·
유충(마리 수)·전령. 시뮬레이터는 유충 3마리를 한 번에 잡으므로 처치 1회 = 3마리.
"""
import math

from .data import load_json


class WinProb:
    def __init__(self, table=None):
        w = table or load_json("sim", "winprob.json")
        self.features = w["features"]
        self.scale = w["scale"]
        self.minutes = {int(m): v for m, v in w["minutes"].items()}
        self.lo, self.hi = min(self.minutes), max(self.minutes)

    @staticmethod
    def diffs(game):
        """블루(100) − 레드(200) 차이."""
        d = {"gold": 0.0, "xp": 0.0}
        for c in game.champs:
            s = 1 if c.team == 100 else -1
            d["gold"] += s * c.total_gold
            d["xp"] += s * c.xp
        ot = game.obj_taken
        d["dragons"] = ot["dragon"][100] - ot["dragon"][200]
        d["grubs"] = 3 * (ot["grubs"][100] - ot["grubs"][200])
        d["heralds"] = ot["herald"][100] - ot["herald"][200]
        return d

    def _logit_at(self, m, d):
        mm = self.minutes[m]
        return mm["intercept"] + sum(mm["coef"][k] * d[k] / self.scale.get(k, 1.0) for k in self.features)

    def blue(self, game, t=None):
        """블루 팀 승리 확률(시각 t초 기준 모델, 기본은 지금)."""
        t = game.t if t is None else t
        d = self.diffs(game)
        m = min(max(t / 60.0, self.lo), self.hi)
        a = int(math.floor(m))
        b = min(a + 1, self.hi)
        frac = m - a
        x = self._logit_at(a, d) if frac == 0 or a == b else (1 - frac) * self._logit_at(a, d) + frac * self._logit_at(b, d)
        return 1.0 / (1.0 + math.exp(-x))

    def for_team(self, game, team, t=None):
        p = self.blue(game, t)
        return p if team == 100 else 1.0 - p
