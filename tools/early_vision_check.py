"""초반 시야 검수: 리플레이 실제 와드 설치(분·포지션) vs 규칙 봇 시뮬 와드·갱 발각(seen).

    python tools/early_vision_check.py [시뮬 판 수=200] [정책 npz]

실제 WARD_PLACED에는 위치가 없어서 횟수·시각만 비교한다. wardType UNDEFINED(챔피언 스킬 설치물로 보임)는 뺀다.
결과 정리는 reports/early_vision_notes.md.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from multiprocessing import Pool  # noqa: E402

import sim.events as E  # noqa: E402
from sim.game import Game  # noqa: E402

ROLES = ["TOP", "MIDDLE", "BOTTOM", "UTILITY"]
POL = None


def real():
    c, n = collections.Counter(), 0
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        role = {q["participantId"]: q["teamPosition"] for q in d["detail"]["info"]["participants"]}
        n += 1
        for fr in d["timeline"]["info"]["frames"]:
            for e in fr["events"]:
                if e["type"] == "WARD_PLACED" and e.get("wardType") != "UNDEFINED" and e["timestamp"] < 600000:
                    c[(role.get(e.get("creatorId")), int(e["timestamp"] // 60000))] += 1
    return c, n


def _init(p):
    global POL
    POL = p


def _one(seed):
    rec, orig = [], E.resolve_gank

    def rg(inp, now, rng, p):
        x = orig(inp, now, rng, p)
        rec.append((now, inp.attackers[0].pid, inp.target.role, inp.seen_prob, x.success, inp.target.flash_ready(now)))
        return x
    E.resolve_gank = rg
    over, team = {}, None
    if POL:
        from sim.agent import AgentJungler
        from sim.ppo import Policy
        from sim.winprob import WinProb
        team = 100 if seed % 2 == 0 else 200
        pol = Policy.from_file(POL)
        pol.rng = np.random.default_rng(seed)
        over = {team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())}
    g = Game(seed=seed, jungler_override=over)
    while g.t < 600:
        g.step()
    E.resolve_gank = orig
    role = {c.pid: c.role for c in g.champs}
    w = collections.Counter((role[e["pid"]], e["t"] // 60) for e in g.events if e["type"] == "ward")
    mine = {c.pid for c in g.champs if c.team == team}
    return w, [x for x in rec if (team is None or x[1] in mine) and x[0] < 300]


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    pol = sys.argv[2] if len(sys.argv) > 2 else None
    c, n = real()
    with Pool(4, initializer=_init, initargs=(pol,)) as p:
        rs = p.map(_one, range(games))
    W, G = collections.Counter(), []
    for w, gk in rs:
        W.update(w)
        G += gk
    print(f"라이너 1명당 분별 와드(실제 {n}판 / 시뮬 {games}판, 0~5분 합)")
    for r in ROLES:
        rv = [c[(r, m)] / n / 2 for m in range(6)]
        sv = [W[(r, m)] / games / 2 for m in range(6)]
        print(f"  {r:8s} 실제 {' '.join(f'{x:.2f}' for x in rv)} (합 {sum(rv[:5]):.2f}) | 시뮬 {' '.join(f'{x:.2f}' for x in sv)} (합 {sum(sv[:5]):.2f})")
    print("시뮬 5분 전 갱(대상 라인별): 횟수 / 평균 발각 / 발각<0.5 비율 / 점멸 없음 / 성공")
    for ln in ("TOP", "MIDDLE", "BOTTOM", "UTILITY"):
        v = [x for x in G if x[2] == ln]
        if v:
            print(f"  {ln:8s} {len(v):4d} {np.mean([x[3] for x in v]):.2f} {np.mean([x[3] < 0.5 for x in v]):.0%} "
                  f"{np.mean([not x[5] for x in v]):.0%} {np.mean([x[4] for x in v]):.0%}")


if __name__ == "__main__":
    main()
