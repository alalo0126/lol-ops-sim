"""라인 방문 비교: 리플레이 1초 동선(실제) vs 시뮬레이터 1초 위치(학습 정글러·상대 규칙 봇).

    python -m sim.lane_visits [--policy models/stage6c/policy_latest.npz] [--games 200]

라인 방문 = 3~18분 중 정글러가 라인 구역(stage6.zone)에 연속 5초 이상(3초 이하 끊김은 이어 봄) 있던 구간.
방문마다 시작~끝+10초 안에 정글러가 관여한 적 처치(킬·어시스트)와 정글러 사망이 있었는지 본다.
실제 쪽은 방문 시작 분 프레임으로 '앞선 정도'(레벨 차 + 1.62 × 천 골드 아이템 차, 라인의 아군 정글러·라이너 평균 −
적 라이너 평균)를 재서 킬 확률을 로지스틱 회귀한다(갱 성공식 level_diff·item_gold_diff 근거).
결과는 reports/lane_visits.md.
"""
import argparse
import csv
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .stage6 import zone

T0, T1, MIN_S, GAP, AFTER = 180, 1080, 5, 3, 10
MIN_COVER = 0.8
ITEM_PER_LEVEL = 1.62     # 천 골드 = 레벨 1.62개(6단계 후속, 실제 킬 회귀)
ROLE_LANE = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
MAP_MAX = 14870


def visits(track):
    """{초: 구역} → [(시작, 끝)]. 없는 초는 모르는 것으로 보고 GAP 이하면 이어 붙인다."""
    out, cur, last = [], None, None
    for t in range(T0, T1):
        if track.get(t) != "lane":
            continue
        if cur is None or t - last > GAP:
            if cur is not None and last - cur + 1 >= MIN_S:
                out.append((cur, last))
            cur = t
        last = t
    if cur is not None and last - cur + 1 >= MIN_S:
        out.append((cur, last))
    return out


def outcome(span, kills, pid, team_pids):
    s, e = span
    ks = [k for k in kills if s <= k["t"] <= e + AFTER]
    kill = any(k["victim"] not in team_pids and (k["killer"] == pid or pid in k["assists"]) for k in ks)
    return {"len": e - s + 1, "kill": kill, "death": any(k["victim"] == pid for k in ks)}


def lane_of(x, y):
    rx, ry = MAP_MAX - x, MAP_MAX - y
    if (x < 1900 and y > 4500) or (y > 12900 and x < 10000) or (x < 3300 and y > 11500):
        return "top"
    if (rx < 1900 and ry > 4500) or (ry > 12900 and rx < 10000) or (rx < 3300 and ry > 11500):
        return "bot"
    return "mid"


# ---------------------------------------------------------------------- 실제
def real(nodes):
    """경기·정글러마다 방문 목록(앞선 정도 포함)."""
    games = []
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
            path = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
            if not os.path.exists(path):
                continue
            zones, xy = {}, {}
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("valid", "1") == "1":
                        t, x, y = round(float(r["time"])), float(r["x"]), float(r["y"])
                        zones[t], xy[t] = zone(x, y, p["teamId"], nodes), (x, y)
            if sum(1 for t in range(T0, T1) if t in zones) < MIN_COVER * (T1 - T0):
                continue
            pid = p["participantId"]
            team_pids = {q["participantId"] for q in parts if q["teamId"] == p["teamId"]}
            out = []
            for span in visits(zones):
                lane = lane_of(*xy[span[0]])
                fr = frames[int(span[0] // 60)]["participantFrames"]

                def st(q):
                    pf = fr[str(q["participantId"])]
                    return pf["level"], (pf["totalGold"] - pf["currentGold"]) / 1000

                att = [st(p)] + [st(q) for q in parts if q["teamId"] == p["teamId"] and ROLE_LANE.get(q["teamPosition"]) == lane]
                foe = [st(q) for q in parts if q["teamId"] != p["teamId"] and ROLE_LANE.get(q["teamPosition"]) == lane]
                lead = (np.mean([a[0] for a in att]) - np.mean([a[0] for a in foe])
                        + ITEM_PER_LEVEL * (np.mean([a[1] for a in att]) - np.mean([a[1] for a in foe])))
                out.append(outcome(span, kills, pid, team_pids) | {"lane": lane, "lead": float(lead), "match": mid, "team": p["teamId"]})
            games.append(out)
    return games


def logit_fit(x, y, boot=500, seed=0):
    """y ~ 로지스틱(a + b·x). (a, b, b의 부트스트랩 95% 구간)."""
    X = np.column_stack([np.ones(len(x)), x])
    y = np.asarray(y, float)

    def fit(X, y):
        w = np.zeros(2)
        for _ in range(100):
            p = 1 / (1 + np.exp(-X @ w))
            w += np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2), X.T @ (y - p))
        return w
    w = fit(X, y)
    rng = np.random.default_rng(seed)
    bs = [fit(X[i], y[i])[1] for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
    return w[0], w[1], (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


# ---------------------------------------------------------------------- 시뮬
_POLICY = None


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
    zones = {ai.pid: {}, rule.pid: {}}
    xys = {ai.pid: {}, rule.pid: {}}
    while g.t < 1200:
        g.step()
        if T0 <= g.t < T1:
            for c in (ai, rule):
                x, y = _xy(g, c)
                if x is not None:
                    zones[c.pid][g.t] = zone(x, y, c.team, g.map.nodes)
                    xys[c.pid][g.t] = (x, y)
    kills = [{"t": e["t"], "killer": e["killer"], "victim": e["victim"], "assists": e["assists"]}
             for e in g.events if e["type"] == "kill"]
    res = {}
    for k, c in (("ai", ai), ("rule", rule)):
        tp = {x.pid for x in g.champs if x.team == c.team}
        res[k] = [outcome(s, kills, c.pid, tp) | {"lane": lane_of(*xys[c.pid][s[0]])} for s in visits(zones[c.pid])]
    return res


def _init(path):
    global _POLICY
    _POLICY = path


def simulate(policy, games, workers=4):
    with Pool(workers, initializer=_init, initargs=(policy,)) as pool:
        r = pool.map(_sim_one, range(games))
    return [x["ai"] for x in r], [x["rule"] for x in r]


def summary(games):
    v = [x for gm in games for x in gm]
    n = max(1, len(games))
    if not v:
        return {"games": len(games), "visits": 0.0, "len": 0.0, "kill": 0.0, "death": 0.0, "kills": 0.0}
    return {"games": len(games), "visits": len(v) / n, "len": float(np.mean([x["len"] for x in v])),
            "kill": float(np.mean([x["kill"] for x in v])), "death": float(np.mean([x["death"] for x in v])),
            "kills": sum(x["kill"] for x in v) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage6c", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "lane_visits.md"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    rg = real(nodes)
    ai, rule = simulate(args.policy, args.games, args.workers)
    rows = [("실제(리플레이)", summary(rg)), ("학습 정글러", summary(ai)), ("규칙 봇(상대)", summary(rule))]
    rv = [x for gm in rg for x in gm]
    a, b, ci = logit_fit([x["lead"] for x in rv], [x["kill"] for x in rv])
    lines = [f"# 라인 방문: 실제 vs 시뮬레이터", "",
             f"- 정책: `{os.path.relpath(args.policy, ROOT)}`, 시뮬 {args.games}판(규칙 봇 9명 + 학습 정글러 1명).",
             "- 라인 방문 = 3~18분에 라인 구역 연속 5초 이상. 결과 = 방문 시작~끝+10초 안 정글러 관여 킬 / 정글러 사망.", "",
             "| | 정글러 수 | 경기당 방문 | 평균 길이(초) | 방문당 킬 | 방문당 사망 | 경기당 방문 킬 |", "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['games']} | {s['visits']:.1f} | {s['len']:.1f} | {s['kill']:.0%} | {s['death']:.0%} | {s['kills']:.1f} |")
    lines += ["", "## 라인별", "", "| | 라인 | 경기당 방문 | 방문당 킬 | 방문당 사망 |", "|---|---|---|---|---|"]
    for name, gs in (("실제", rg), ("학습", ai), ("규칙 봇", rule)):
        for ln in ("top", "mid", "bot"):
            s = summary([[x for x in gm if x["lane"] == ln] for gm in gs])
            lines.append(f"| {name} | {ln} | {s['visits']:.1f} | {s['kill']:.0%} | {s['death']:.0%} |")
    lines += ["", f"실제 방문당 킬을 앞선 정도(레벨 차 + {ITEM_PER_LEVEL} × 천 골드 아이템 차)로 로지스틱 회귀: "
              f"절편 {a:+.2f}, 기울기 {b:+.2f} (95% {ci[0]:+.2f}~{ci[1]:+.2f}, 방문 {len(rv)}번)."]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
