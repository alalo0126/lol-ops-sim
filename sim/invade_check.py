"""카정 검수: 리플레이 1초 동선(실제) vs 시뮬레이터(학습 정글러·규칙 봇).

    python -m sim.invade_check [--policy models/stage10_a/policy_latest.npz] [--games 200]

카정 = 3~18분에 정글러가 적 캠프 600 안에 3초 이상 머문 구간(sim.real_actions.place 기준, 2초 이하 끊김은 이어 봄).
구간마다
  - 머문 시간: 실제는 5초 이상 + 살아 있었음이면 '먹었다'로 봄(실제 선수는 원거리 공격·카이팅으로 캠프 600 밖에 있는 때가 많아
    머문 시간이 시뮬보다 짧다. 시뮬은 처치 기록으로 셈),
  - 캠프가 살아 있었을 가능성: 적 정글러가 그 캠프에 마지막으로 머문 뒤 135초(리젠)가 지났거나 머문 적 없음,
  - 적 정글러와의 최소 거리(구간 ±10초), 1,500 안이면 '마주침',
  - 구간 시작~끝+10초 안 정글러 관여 킬·정글러 사망.
결과는 reports/invade_check.md.
"""
import argparse
import collections
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .lane_visits import T0, T1
from .real_actions import place, real_tracks

SIDE = {100: "B", 200: "R"}
TOOK_S = 5
RESPAWN = 135
MEET = 1500
AFTER = 10


def dwell_runs(pos, team, nodes):
    """pos {초: (x,y)|None} → 캠프 머문 구간 [(캠프 id, 시작, 끝)] (우리·적 모두)."""
    raw = {}
    for t, p in pos.items():
        if p is None:
            continue
        best, bd = None, 1e18
        for nid, n in nodes.items():
            if n["kind"] == "camp":
                d = math.hypot(n["x"] - p[0], n["y"] - p[1])
                if d < 600 and d < bd:
                    best, bd = nid, d
        if best:
            raw[t] = best
    out, cur = [], None
    for t in sorted(raw):
        c = raw[t]
        if cur and cur[0] == c and t - cur[2] <= 3:
            cur[2] = t
        else:
            if cur and cur[2] - cur[1] + 1 >= 3:
                out.append(tuple(cur))
            cur = [c, t, t]
    if cur and cur[2] - cur[1] + 1 >= 3:
        out.append(tuple(cur))
    return out


def invades(pos, team, foe_pos, foe_team, nodes, kills, pid, team_pids):
    """한 정글러의 카정 구간 목록. foe_pos가 없으면 살아 있음·마주침은 None."""
    mine = dwell_runs(pos, team, nodes)
    foe = dwell_runs(foe_pos, foe_team, nodes) if foe_pos is not None else None
    out = []
    for cid, s, e in mine:
        if cid.startswith(SIDE[team]) or not (T0 <= s < T1):
            continue
        alive = meet = None
        if foe is not None:
            last = max((fe for c, fs, fe in foe if c == cid and fe < s), default=None)
            alive = last is None or s - last >= RESPAWN
            ds = [math.hypot(pos[t][0] - foe_pos[t][0], pos[t][1] - foe_pos[t][1])
                  for t in range(s - AFTER, e + AFTER + 1) if pos.get(t) and foe_pos.get(t)]
            meet = bool(ds) and min(ds) <= MEET
        ks = [k for k in kills if s <= k["t"] <= e + AFTER]
        out.append({"camp": cid.split("_", 1)[1], "t": s, "len": e - s + 1, "took": e - s + 1 >= TOOK_S,
                    "alive": alive, "meet": meet,
                    "kill": any(k["victim"] not in team_pids and (k["killer"] == pid or pid in k["assists"]) for k in ks),
                    "death": any(k["victim"] == pid for k in ks)})
    return out


# ---------------------------------------------------------------------- 실제
def real(nodes):
    import glob
    import json
    tracks = real_tracks()
    by = collections.defaultdict(dict)
    for mid, team, pid, pos in tracks:
        by[mid][team] = (pid, pos)
    games = []
    for mid, sides in by.items():
        with open(os.path.join(ROOT, "data", "replay_raw", f"{mid}.json"), encoding="utf-8") as f:
            d = json.load(f)
        parts = d["detail"]["info"]["participants"]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds", [])}
                 for fr in d["timeline"]["info"]["frames"] for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
        for team, (pid, pos) in sides.items():
            if sum(1 for t in range(T0, T1) if t in pos) < 0.8 * (T1 - T0):
                continue
            foe = sides.get(300 - team)
            tp = {q["participantId"] for q in parts if q["teamId"] == team}
            games.append(invades(pos, team, foe[1] if foe else None, 300 - team, nodes, kills, pid, tp))
    return games


# ---------------------------------------------------------------------- 시뮬
_POLICY = None


def _init(path):
    global _POLICY
    _POLICY = path


def _sim_one(seed):
    from .agent import AgentJungler
    from .ppo import Policy
    from .stage6_track import _xy
    from .winprob import WinProb
    team = 100 if seed % 2 == 0 else 200
    pol = Policy.from_file(_POLICY)
    pol.rng = np.random.default_rng(seed)
    holder = {}

    def make(g, c):
        holder["bot"] = AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())
        return holder["bot"]
    g = Game(seed=seed, jungler_override={team: make})
    ai = holder["bot"].c
    rule = next(x for x in g.champs if x.team != team and x.role == "JUNGLE")
    pos = {ai.pid: {}, rule.pid: {}}
    while g.t < T1 + AFTER:
        g.step()
        if g.t >= T0 - 200:
            for c in (ai, rule):
                xy = _xy(g, c)
                pos[c.pid][g.t] = None if xy[0] is None else xy
    kills = [{"t": e["t"], "killer": e["killer"], "victim": e["victim"], "assists": e["assists"]}
             for e in g.events if e["type"] == "kill"]
    res = {}
    for k, c, o in (("ai", ai, rule), ("rule", rule, ai)):
        tp = {x.pid for x in g.champs if x.team == c.team}
        v = invades(pos[c.pid], c.team, pos[o.pid], o.team, g.map.nodes, kills, c.pid, tp)
        # 실제 처치 기록으로 '10초 머묾 = 먹음' 문턱 확인
        eaten = [e for e in g.events if e["type"] == "camp" and e["pid"] == c.pid and not e["camp"].startswith(SIDE[c.team])
                 and T0 <= e["t"] < T1 + AFTER]
        for x in v:
            x["ate"] = any(e["camp"].split("_", 1)[1] == x["camp"] and x["t"] <= e["t"] <= x["t"] + x["len"] + 2 for e in eaten)
        res[k] = v
    return res


def simulate(policy, games, workers=4):
    with Pool(workers, initializer=_init, initargs=(policy,)) as pool:
        r = pool.map(_sim_one, range(games))
    return [x["ai"] for x in r], [x["rule"] for x in r]


def summary(games):
    v = [x for gm in games for x in gm]
    n = max(1, len(games))
    m = lambda k, sub=None: float(np.mean([x[k] for x in (sub or v) if x[k] is not None])) if [x for x in (sub or v) if x[k] is not None] else float("nan")
    took = [x for x in v if (x["ate"] if "ate" in x else x["took"] and x["alive"])]
    out = {"n": len(games), "per_game": len(v) / n, "took_per_game": len(took) / n,
           "len": m("len"), "took": m("took"), "alive": m("alive"), "meet": m("meet"),
           "kill": m("kill"), "death": m("death"),
           "t_med": float(np.median([x["t"] for x in v])) / 60 if v else float("nan"),
           "camps": collections.Counter(x["camp"] for x in took)}
    took_meet = [x for x in v if x["meet"]]
    out["death_if_meet"] = m("death", took_meet) if took_meet else float("nan")
    out["kill_if_meet"] = m("kill", took_meet) if took_meet else float("nan")
    if v and "ate" in v[0]:
        out["ate"] = m("ate")
        out["ate_if_took"] = m("ate", took) if took else float("nan")
        out["ate_if_short"] = m("ate", [x for x in v if not x["took"]]) if len(took) < len(v) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage10_a", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "invade_check.md"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    rg = real(nodes)
    ai, rule = simulate(args.policy, args.games, args.workers)
    rows = [("실제(리플레이)", summary(rg)), ("학습 정글러", summary(ai)), ("규칙 봇(상대)", summary(rule))]
    f = lambda x, p=True: "-" if x != x else (f"{x:.0%}" if p else f"{x:.2f}")
    lines = ["# 카정 검수: 실제 vs 시뮬레이터 (3~18분)", "",
             f"- 실제 정글러 {rows[0][1]['n']}명(리플레이 1초 동선), 시뮬 {args.games}판: 학습 정글러 `{os.path.relpath(args.policy, ROOT)}`, 상대 규칙 봇.",
             f"- 카정 = 적 캠프 600 안 3초 이상. 먹음 = 실제는 {TOOK_S}초 이상 머묾 + 살아 있었음(추정), 시뮬은 처치 기록. 살아 있었음 = 적 정글러가 그 캠프에 마지막으로 머문 뒤 {RESPAWN}초 이상.",
             f"- 마주침 = 구간 ±{AFTER}초에 적 정글러 {MEET} 안. 킬·사망 = 구간 시작~끝+{AFTER}초.", "",
             "| | 판당 카정 | 판당 먹음 | 머문 초 | 살아 있었음 | 마주침 | 킬 | 사망 | 마주쳤을 때 킬 | 마주쳤을 때 사망 | 시각 중앙(분) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['per_game']:.2f} | {s['took_per_game']:.2f} | {s['len']:.1f} | {f(s['alive'])} | {f(s['meet'])} | "
                     f"{f(s['kill'])} | {f(s['death'])} | {f(s['kill_if_meet'])} | {f(s['death_if_meet'])} | {s['t_med']:.1f} |")
    lines += ["", "## 먹은 캠프 종류", "", "| 캠프 | 실제 | 학습 정글러 | 규칙 봇 |", "|---|---|---|---|"]
    for c in sorted(set().union(*[s["camps"] for _, s in rows])):
        lines.append(" | ".join([f"| {c}"] + [f"{s['camps'][c] / max(1, s['n']):.2f}" for _, s in rows]) + " |")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
