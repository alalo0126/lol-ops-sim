"""이른 갱(1.5~5분) 검수: 리플레이 1초 동선(실제) vs 시뮬레이터 같은 정의.

    python -m sim.early_gank_check [--games 400] [--policy models/stage18_c/policy_latest.npz ...]

라인 방문 = 90~300초에 시작한 라인 구역(stage6.zone) 연속 5초 이상 체류(sim.lane_visits와 같음)(3초 이하 끊김은 이어 봄).
결과 = 방문 시작~끝+10초 안 정글러 관여 적 처치 / 정글러 사망(sim.lane_visits.outcome).
방문 시작 직전 분 프레임으로 그 라인의 아군 라이너·적 라이너 레벨·체력 비율을 본다.
시뮬은 같은 위치 정의 외에 갱 이벤트(성공 확률 prob)도 따로 모은다.
결과는 reports/early_gank.md.
"""
import argparse
import csv
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .lane_visits import AFTER, GAP, ROLE_LANE, lane_of, outcome
from .stage6 import zone

S0, S1, MIN_S = 90, 300, 5    # 8초 이상만 보면 실제는 킬이 난 방문이 길어져 킬 비율이 부풀려짐(5초 33% · 8초 56% · 12초 72%)
EARLY_INV, INV_MIN = (55, 130), 12    # 0:55~2:10 적 정글 12초 이상 = 1레벨 카정(첫 캠프 근처)
BUCKETS = [(90, 150, "1:30~2:30"), (150, 210, "2:30~3:30"), (210, 300, "3:30~5:00")]
LANE_ROLE = {"top": ["TOP"], "mid": ["MIDDLE"], "bot": ["BOTTOM", "UTILITY"]}


def visits(zones):
    out, cur, last = [], None, None
    for t in range(S0 - 30, S1 + 60):
        if zones.get(t) != "lane":
            continue
        if cur is None or t - last > GAP:
            if cur is not None and last - cur + 1 >= MIN_S and S0 <= cur < S1:
                out.append((cur, last))
            cur = t
        last = t
    if cur is not None and last - cur + 1 >= MIN_S and S0 <= cur < S1:
        out.append((cur, last))
    return out


def bucket(t):
    for a, b, name in BUCKETS:
        if a <= t < b:
            return name
    return None


# ---------------------------------------------------------------------- 실제
def real(nodes):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        mid = os.path.basename(path)[:-5]
        parts = d["detail"]["info"]["participants"]
        frames = d["timeline"]["info"]["frames"]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds", [])}
                 for fr in frames for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
        for p in parts:
            if p.get("teamPosition") != "JUNGLE":
                continue
            tp = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
            if not os.path.exists(tp):
                continue
            zones, xy = {}, {}
            with open(tp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("valid", "1") == "1":
                        t, x, y = round(float(r["time"])), float(r["x"]), float(r["y"])
                        zones[t], xy[t] = zone(x, y, p["teamId"], nodes), (x, y)
            if sum(1 for t in range(S0, S1) if t in zones) < 0.8 * (S1 - S0):
                continue
            pid = p["participantId"]
            team_pids = {q["participantId"] for q in parts if q["teamId"] == p["teamId"]}
            if sum(1 for t in range(*EARLY_INV) if t in zones) >= 60:
                rows.append({"kind": "early", "match": mid, "enemy_s": sum(1 for t in range(*EARLY_INV) if zones.get(t) == "enemy_jungle")})
            for span in visits(zones):
                lane = lane_of(*xy[span[0]])
                fr = frames[int(span[0] // 60)]["participantFrames"]

                def st(q):
                    pf = fr[str(q["participantId"])]
                    cs = pf["championStats"]
                    return pf["level"], cs["health"] / max(1, cs["healthMax"]), (pf["totalGold"] - pf["currentGold"]) / 1000
                ally = [st(q) for q in parts if q["teamId"] == p["teamId"] and q["teamPosition"] in LANE_ROLE[lane]]
                foe = [st(q) for q in parts if q["teamId"] != p["teamId"] and q["teamPosition"] in LANE_ROLE[lane]]
                rows.append(outcome(span, kills, pid, team_pids) | state(lane, span, st(p), ally, foe) | {"match": mid, "kind": "visit"})
    return rows


def state(lane, span, me, ally, foe):
    m = lambda v, i: float(np.mean([x[i] for x in v])) if v else None
    return {"lane": lane, "t": span[0], "jg_lvl": me[0], "ally_lvl": m(ally, 0), "foe_lvl": m(foe, 0),
            "ally_hp": m(ally, 1), "foe_hp": m(foe, 1), "foe_gold": m(foe, 2), "ally_gold": m(ally, 2)}


# ---------------------------------------------------------------------- 시뮬
_POLICY = None


def _init(path):
    global _POLICY
    _POLICY = path


def _sim_one(seed):
    from .stage6_track import _xy
    over, ai_team = {}, None
    if _POLICY:
        from .agent import AgentJungler
        from .ppo import Policy
        from .winprob import WinProb
        ai_team = 100 if seed % 2 == 0 else 200
        pol = Policy.from_file(_POLICY)
        pol.rng = np.random.default_rng(seed)
        over = {ai_team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())}
    g = Game(seed=seed, jungler_override=over) if over else Game(seed=seed)
    jgs = [c for c in g.champs if c.role == "JUNGLE"]
    zones = {c.pid: {} for c in jgs}
    xys = {c.pid: {} for c in jgs}
    snap = {}
    while g.t < S1 + 80:
        if g.t % 60 == 0:
            snap[g.t // 60] = {c.pid: (c.level, c.hp, c.item_gold / 1000) for c in g.champs}
        g.step()
        for c in jgs:
            x, y = _xy(g, c)
            if x is not None:
                zones[c.pid][g.t] = zone(x, y, c.team, g.map.nodes)
                xys[c.pid][g.t] = (x, y)
    kills = [{"t": e["t"], "killer": e["killer"], "victim": e["victim"], "assists": e["assists"]}
             for e in g.events if e["type"] == "kill"]
    rows = []
    for c in jgs:
        who = "ai" if c.team == ai_team else "rule"
        tp = {x.pid for x in g.champs if x.team == c.team}
        rows.append({"who": who, "kind": "early", "enemy_s": sum(1 for t in range(*EARLY_INV) if zones[c.pid].get(t) == "enemy_jungle"),
                     "jcs5": c.jungle_cs})
        for span in visits(zones[c.pid]):
            lane = lane_of(*xys[c.pid][span[0]])
            s = snap.get(span[0] // 60) or snap[max(snap)]
            ally = [s[x.pid] for x in g.champs if x.team == c.team and x.role in LANE_ROLE[lane]]
            foe = [s[x.pid] for x in g.champs if x.team != c.team and x.role in LANE_ROLE[lane]]
            rows.append(outcome(span, kills, c.pid, tp) | state(lane, span, s[c.pid], ally, foe) | {"who": who, "kind": "visit"})
        for e in g.events:
            if e["type"] == "gank" and e["pid"] == c.pid and S0 <= e["t"] < S1:
                rows.append({"who": who, "kind": "gank", "t": e["t"], "lane": e["lane"], "success": e["success"], "prob": e["prob"]})
    return rows


def simulate(games, policy, workers):
    with Pool(workers, initializer=_init, initargs=(policy,)) as pool:
        return [r for rs in pool.map(_sim_one, range(games)) for r in rs]


# ---------------------------------------------------------------------- 표
def pct(v):
    if not v:
        return "–"
    p = float(np.mean(v))
    return f"{p:.0%} ±{1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(v)):.0%}"


def mean(v, f="{:.2f}"):
    v = [x for x in v if x is not None]
    return f.format(float(np.mean(v))) if v else "–"


def table(groups, n_jg):
    out = ["| | 시기 | 정글러당 방문 | 방문당 킬 | 방문당 사망 | 정글러 레벨 | 아군 라이너 레벨 | 적 라이너 레벨 | 아군 체력 | 적 체력 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rs in groups:
        for _, _, b in BUCKETS + [(0, 0, "전체")]:
            v = [r for r in rs if b == "전체" or bucket(r["t"]) == b]
            out.append(f"| {name} | {b} | {len(v) / max(1, n_jg[name]):.2f} | {pct([r['kill'] for r in v])} | {pct([r['death'] for r in v])} | "
                       f"{mean([r['jg_lvl'] for r in v], '{:.1f}')} | {mean([r['ally_lvl'] for r in v], '{:.1f}')} | "
                       f"{mean([r['foe_lvl'] for r in v], '{:.1f}')} | {mean([r['ally_hp'] for r in v])} | {mean([r['foe_hp'] for r in v])} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--policy", nargs="*", default=[os.path.join(ROOT, "models", "stage18_b", "policy_latest.npz"),
                                                    os.path.join(ROOT, "models", "stage18_c", "policy_latest.npz")])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "early_gank.md"))
    ap.add_argument("--rows", default=os.path.join(ROOT, "data", "early_gank_rows.json"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    rr_all = real(nodes)
    rr = [r for r in rr_all if r["kind"] == "visit"]
    groups, n_jg, ganks, allrows = [("실제", rr)], {}, [], {"real": rr_all}
    n_jg["실제"] = sum(1 for r in rr_all if r["kind"] == "early")   # 1:30~5:00 동선이 있는 정글러 수(근사)
    early = [("실제", [r for r in rr_all if r["kind"] == "early"])]
    sim_rule = simulate(args.games, "", args.workers)
    allrows["rule"] = sim_rule
    groups.append(("규칙 봇", [r for r in sim_rule if r["kind"] == "visit"]))
    early.append(("규칙 봇", [r for r in sim_rule if r["kind"] == "early"]))
    n_jg["규칙 봇"] = 2 * args.games
    ganks.append(("규칙 봇", [r for r in sim_rule if r["kind"] == "gank"], 2 * args.games))
    for pol in args.policy:
        name = os.path.basename(os.path.dirname(pol))
        rs = simulate(args.games, pol, args.workers)
        allrows[name] = rs
        groups.append((name, [r for r in rs if r["kind"] == "visit" and r["who"] == "ai"]))
        early.append((name, [r for r in rs if r["kind"] == "early" and r["who"] == "ai"]))
        n_jg[name] = args.games
        ganks.append((name, [r for r in rs if r["kind"] == "gank" and r["who"] == "ai"], args.games))
    with open(args.rows, "w", encoding="utf-8") as f:
        json.dump(allrows, f)
    lines = ["# 이른 갱(1:30~5:00) 검수: 실제 vs 시뮬", "",
             f"- 실제: 리플레이 1초 동선 {len({r['match'] for r in rr})}판. 시뮬: 규칙 봇 양 팀 {args.games}판, 학습 정글러 각 {args.games}판(상대 규칙 봇).",
             f"- 라인 방문 = {S0}~{S1}초에 시작한 라인 구역 {MIN_S}초 이상 체류. 결과 = 시작~끝+{AFTER}초 안 정글러 관여 킬 / 정글러 사망.",
             "- 레벨·체력은 방문 시작 직전 분 프레임(실제와 같게 1분 단위). 체력은 비율. '±'는 95% 구간(이항 근사).", ""]
    lines += table(groups, n_jg)
    lines += ["", "## 시뮬 갱 이벤트(같은 시기)", "", "| | 시기 | 정글러당 갱 | 성공 | 평균 성공 확률 |", "|---|---|---|---|---|"]
    for name, gs, n in ganks:
        for _, _, b in BUCKETS + [(0, 0, "전체")]:
            v = [r for r in gs if b == "전체" or bucket(r["t"]) == b]
            lines.append(f"| {name} | {b} | {len(v) / n:.2f} | {pct([r['success'] for r in v])} | {mean([r['prob'] for r in v])} |")
    lines += ["", f"## 1레벨 카정(0:55~2:10에 적 정글 {INV_MIN}초 이상)", "", "| | 정글러 | 비율 |", "|---|---|---|"]
    for name, rs in early:
        lines.append(f"| {name} | {len(rs)} | {pct([r['enemy_s'] >= INV_MIN for r in rs])} |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
