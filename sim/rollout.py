"""경기 한 판 돌리기: 한 팀 정글러는 학습 정글러(정책), 나머지 9명은 규칙 봇."""
import numpy as np

from .agent import AgentJungler, N_MAIN
from .game import Game
from .winprob import WinProb

_WP = None


def winprob():
    global _WP
    if _WP is None:
        _WP = WinProb()
    return _WP


def jungler_stats(g, team):
    c = next(x for x in g.champs if x.team == team and x.role == "JUNGLE")
    fr = {f["t"] // 60: f for f in g.frames}
    p10 = fr[10]["p"][c.pid]
    ev = g.events
    ganks = [e for e in ev if e["type"] == "gank" and e["pid"] == c.pid]
    camps = [e for e in ev if e["type"] == "camp" and e["pid"] == c.pid]
    enemy_side = "R_" if team == 100 else "B_"
    return {
        "cs10": p10["jungle_cs"], "gold10": p10["gold"], "level10": p10["level"], "xp10": p10["xp"],
        "k10": p10["kills"], "d10": p10["deaths"], "a10": p10["assists"],
        "cs20": c.jungle_cs, "gold20": round(c.total_gold), "kills": c.kills, "deaths": c.deaths, "assists": c.assists,
        "ganks": len(ganks), "gank_success": sum(e["success"] for e in ganks),
        "invade_camps": sum(1 for e in camps if e["camp"].startswith(enemy_side)),
        "enemy_cs_share": round(sum(1 for e in camps if e["camp"].startswith(enemy_side)) / max(1, len(camps)), 3),
        "lanefarm": sum(1 for e in ev if e["type"] == "lanefarm" and e["pid"] == c.pid),
        "lane_cs20": round(c.lane_cs, 1),
        "dragons": g.obj_taken["dragon"][team], "grubs": g.obj_taken["grubs"][team], "heralds": g.obj_taken["herald"][team],
        "enemy_dragons": g.obj_taken["dragon"][300 - team],
        "first_clear_24_min": next((m for m in sorted(fr) if fr[m]["p"][c.pid]["jungle_cs"] >= 24), None),
        "team_gold_diff20": round(sum((1 if x.team == team else -1) * x.total_gold for x in g.champs)),
    }


def play(seed, agent_team, policy_fn=None, greedy=False, rules=None):
    """policy_fn=None이면 규칙 봇끼리(기준선). 돌려주는 것: 최종 Φ(그 팀 20분 예측 승률), 기록, 통계."""
    wp = winprob()
    holder = {}
    override = None
    if policy_fn is not None:
        def make(g, c):
            holder["bot"] = AgentJungler(g, c, policy_fn, wp)
            return holder["bot"]
        override = {agent_team: make}
    g = Game(seed=seed, jungler_override=override, rules=rules).run()
    out = {"seed": seed, "team": agent_team}
    if "bot" in holder:
        b = holder["bot"]
        out["phi"] = b.finish_episode()
        out["traj"] = b.traj
        out["counts"] = b.counts.copy()
        out["wards"] = b.ward_count
    else:
        out["phi"] = wp.for_team(g, agent_team)
        out["counts"] = np.zeros(N_MAIN, dtype=int)
    out["stats"] = jungler_stats(g, agent_team)
    return out
