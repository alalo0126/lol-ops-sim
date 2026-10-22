"""라이너 1초 동선 비교: 리플레이(실제 천상계) vs 시뮬레이터 규칙 봇 라이너.

    python -m sim.laner_tracks [--games 200]   :: data/replay_tracks + data/replay_compact.json → reports/laner_tracks.md, .json

1,000판 분 프레임으로는 볼 수 없던 것(귀환 횟수·기지 체류·라인 복귀 시간·로밍)을 3~18분 1초 위치로 잰다.
실제와 시뮬레이터에 같은 함수를 쓴다.

- 구역 비율: 자기 라인 / 다른 라인 / 우리 정글 / 적 정글 / 강가(둥지 포함) / 기지 / 사망. 자기 라인은 포지션 기준 고정
  (탑=top, 미드=mid, 원딜·서폿=bot; 14분 이후 미드 전환도 '다른 라인'으로 잡힌다).
- 기지 방문: 살아서 기지 구역에 연속 3초 이상(부활 직후 기지는 '부활'로 따로 셈)과 체류 시간. 라인 띠가 기지 바로 앞에서
  시작해서 '라인 복귀 시간'은 구역으로 잴 수 없다(웨이브 위치가 필요).
- 로밍: 자기 라인을 떠나 기지·사망 없이 다른 라인 띠에 연속 5초 이상 머문 것. 로밍 중~끝+10초 안 킬 관여.
- 강가·정글 나들이: 자기 라인을 떠나 다른 라인에 가지 않고 강가·정글에 연속 10초 이상 머물다 라인으로 돌아온 것.

실제 사망 = <경기>_players.csv(5초 간격 is_dead, 팀·포지션으로 선수 매김)와 킬 이벤트. 무효(카메라 풀림) 초는 모름.
"""
import argparse
import collections
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .stage6 import zone
from .lane_visits import lane_of

T0, T1 = 180, 1080
MIN_COVER = 0.8
BASE_S, ROAM_S, TRIP_S, AFTER = 3, 5, 10, 10
ROLES = ("TOP", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_KO = {"TOP": "탑", "MIDDLE": "미드", "BOTTOM": "원딜", "UTILITY": "서폿"}
ROLE_LANE = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
ZONES = ("own_lane", "other_lane", "own_jungle", "enemy_jungle", "river", "base", "dead")
ZONE_KO = {"own_lane": "자기 라인", "other_lane": "다른 라인", "own_jungle": "우리 정글", "enemy_jungle": "적 정글",
           "river": "강가", "base": "기지", "dead": "사망"}
TEAM = {"ORDER": 100, "CHAOS": 200}


def place(x, y, team, role, nodes):
    """좌표 → ZONES 중 하나(사망 제외)."""
    z = zone(x, y, team, nodes)
    if z == "lane":
        return "own_lane" if lane_of(x, y) == ROLE_LANE[role] else "other_lane"
    return "river" if z == "pit" else z


def segments(track):
    """{초: 구역} → [(구역, 시작, 끝)] (모르는 초는 앞 구역에 붙임)."""
    out = []
    for t in range(T0, T1):
        z = track.get(t)
        if z is None:
            if out:
                out[-1][2] = t
            continue
        if out and out[-1][0] == z:
            out[-1][2] = t
        else:
            out.append([z, t, t])
    return [tuple(s) for s in out]


def behaviour(track, kills, pid, team_pids):
    """한 라이너의 3~18분 {초: 구역} → 지표 dict."""
    known = [t for t in range(T0, T1) if t in track]
    share = collections.Counter(track[t] for t in known)
    segs = segments(track)
    base, respawn, roams, trips = [], 0, [], 0
    for i, (z, s, e) in enumerate(segs):
        if z == "base" and e - s + 1 >= BASE_S:
            if i > 0 and segs[i - 1][0] == "dead":
                respawn += 1
            else:
                base.append(e - s + 1)
    # 로밍·나들이: 자기 라인을 떠난 뒤 다시 자기 라인에 올 때까지의 구간
    i = 0
    while i < len(segs):
        if segs[i][0] != "own_lane":
            i += 1
            continue
        j = i + 1
        while j < len(segs) and segs[j][0] not in ("own_lane", "base", "dead"):
            j += 1
        away = segs[i + 1:j]
        if away:
            other = [s for s in away if s[0] == "other_lane" and s[2] - s[1] + 1 >= ROAM_S]
            if other:
                s0, e0 = away[0][1], away[-1][2]
                kill = any(s0 <= k["t"] <= e0 + AFTER and k["victim"] not in team_pids
                           and (k["killer"] == pid or pid in k["assists"]) for k in kills)
                death = any(s0 <= k["t"] <= e0 + AFTER and k["victim"] == pid for k in kills)
                roams.append({"start": s0, "len": e0 - s0 + 1, "kill": kill, "death": death})
            elif j < len(segs) and segs[j][0] == "own_lane" and any(s[2] - s[1] + 1 >= TRIP_S for s in away
                                                                      if s[0] in ("river", "own_jungle", "enemy_jungle")):
                trips += 1
        i = j
    return {"cover": len(known) / (T1 - T0), "share": {z: share[z] / max(1, len(known)) for z in ZONES},
            "base_visits": len(base), "base_s": base, "respawn": respawn,
            "roams": roams, "trips": trips}


# ---------------------------------------------------------------------- 실제
def real(nodes):
    with open(os.path.join(ROOT, "data", "replay_compact.json"), encoding="utf-8") as f:
        compact = json.load(f)
    out = collections.defaultdict(list)
    tdir = os.path.join(ROOT, "data", "replay_tracks")
    for mid, m in sorted(compact.items()):
        parts = m["parts"]
        by_pos = {(q["teamId"], q["teamPosition"]): q for q in parts}
        dead = collections.defaultdict(set)   # pid → 죽어 있던 초
        ps = os.path.join(tdir, f"{mid}_players.csv")
        if os.path.exists(ps):
            with open(ps, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    q = by_pos.get((TEAM.get(r["team"]), r["position"]))
                    if q and r["is_dead"] == "1":
                        t = round(float(r["time"]))
                        dead[q["participantId"]].update(range(t - 2, t + 3))
        for q in parts:
            role = q["teamPosition"]
            path = os.path.join(tdir, f"{mid}_{q['championName']}.csv")
            if role not in ROLES or not os.path.exists(path):
                continue
            pid = q["participantId"]
            track = {}
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    t = round(float(r["time"]))
                    if T0 <= t < T1 and r.get("valid", "1") == "1":
                        track[t] = place(float(r["x"]), float(r["y"]), q["teamId"], role, nodes)
            for t in dead[pid]:
                if T0 <= t < T1:
                    track[t] = "dead"
            b = behaviour(track, m["kills"], pid, {p["participantId"] for p in parts if p["teamId"] == q["teamId"]})
            if b["cover"] >= MIN_COVER:
                out[role].append(b | {"match": mid, "champion": q["championName"]})
    return out


# ---------------------------------------------------------------------- 시뮬
def _sim_one(seed):
    from .stage6_track import _xy
    g = Game(seed=seed)
    laners = [c for c in g.champs if c.role in ROLES]
    tracks = {c.pid: {} for c in laners}
    while g.t < 1200:
        g.step()
        if T0 <= g.t < T1:
            for c in laners:
                x, y = _xy(g, c)
                tracks[c.pid][g.t] = "dead" if x is None else place(x, y, c.team, c.role, g.map.nodes)
    kills = [{"t": e["t"], "killer": e["killer"], "victim": e["victim"], "assists": e["assists"]}
             for e in g.events if e["type"] == "kill"]
    res = collections.defaultdict(list)
    for c in laners:
        tp = {x.pid for x in g.champs if x.team == c.team}
        res[c.role].append(behaviour(tracks[c.pid], kills, c.pid, tp))
    return dict(res)


def simulate(games, workers=4):
    out = collections.defaultdict(list)
    with Pool(workers) as pool:
        for r in pool.map(_sim_one, range(games)):
            for k, v in r.items():
                out[k] += v
    return out


def summary(rows):
    if not rows:
        return {}
    n = len(rows)
    roams = [r for x in rows for r in x["roams"]]
    flat = lambda k: [v for x in rows for v in x[k]]   # noqa: E731
    med = lambda v: float(np.median(v)) if v else float("nan")   # noqa: E731
    return {"n": n, "share": {z: float(np.mean([x["share"][z] for x in rows])) for z in ZONES},
            "base_visits": float(np.mean([x["base_visits"] for x in rows])), "respawn": float(np.mean([x["respawn"] for x in rows])),
            "base_s": med(flat("base_s")),
            "roams": len(roams) / n, "roam_len": med([r["len"] for r in roams]),
            "roam_kill": float(np.mean([r["kill"] for r in roams])) if roams else float("nan"),
            "roam_death": float(np.mean([r["death"] for r in roams])) if roams else float("nan"),
            "trips": float(np.mean([x["trips"] for x in rows]))}


def verdict(r, s, tol):
    if r != r or s != s:
        return ""
    d = abs(s - r)
    return "OK" if d <= tol else ("주의" if d <= 2 * tol else "불일치")


def report(rs, ss, games):
    lines = ["# 라이너 1초 동선: 실제 vs 시뮬레이터", "",
             f"- 실제: 리플레이 1초 동선(같은 빌드 천상계 {len({x['match'] for v in rs.values() for x in v})}판), "
             f"시뮬: 규칙 봇 라이너 {games}판. 3~18분, 유효 위치 {MIN_COVER:.0%} 이상인 라이너만.",
             "- 기지 방문 = 살아서 기지에 3초 이상(부활 제외). "
             "로밍 = 자기 라인을 떠나 다른 라인에 5초 이상(기지·사망 없이). 나들이 = 다른 라인 없이 강가·정글 10초 이상 갔다 옴.",
             "- 판정: 구역 비율 ±5%p OK, 횟수 ±0.5(로밍·나들이 ±1) OK, 체류 ±10초 OK, 두 배까지 주의.", ""]
    for role in ROLES:
        r, s = summary(rs.get(role, [])), summary(ss.get(role, []))
        if not r:
            continue
        lines += [f"## {ROLE_KO[role]} (실제 {r['n']}명 / 시뮬 {s['n']}명)", "", "| 지표 | 실제 | 시뮬 | 판정 |", "|---|---|---|---|"]
        for z in ZONES:
            lines.append(f"| {ZONE_KO[z]} 비율 | {r['share'][z]:.0%} | {s['share'][z]:.0%} | {verdict(r['share'][z], s['share'][z], 0.05)} |")
        for k, name, tol, fmt in (("base_visits", "기지 방문(판당)", 0.5, "{:.1f}"), ("respawn", "부활(판당)", 0.5, "{:.1f}"),
                                  ("base_s", "기지 체류 중앙(초)", 10, "{:.0f}"),
                                  ("roams", "로밍(판당)", 1, "{:.1f}"), ("roam_len", "로밍 길이 중앙(초)", 15, "{:.0f}"),
                                  ("roam_kill", "로밍당 킬 관여", 0.1, "{:.0%}"), ("roam_death", "로밍당 사망", 0.1, "{:.0%}"),
                                  ("trips", "강가·정글 나들이(판당)", 1, "{:.1f}")):
            lines.append(f"| {name} | {fmt.format(r[k])} | {fmt.format(s[k])} | {verdict(r[k], s[k], tol)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "laner_tracks.md"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    rs = real(nodes)
    ss = simulate(args.games, args.workers)
    md = report(rs, ss, args.games)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(args.out[:-3] + ".json", "w", encoding="utf-8") as f:
        json.dump({"real": {k: summary(v) for k, v in rs.items()}, "sim": {k: summary(v) for k, v in ss.items()},
                   "real_rows": {k: [{kk: vv for kk, vv in x.items() if kk != "roams"} | {"roams": x["roams"]} for x in v]
                                 for k, v in rs.items()}}, f, ensure_ascii=False, indent=1)
    print(md)


if __name__ == "__main__":
    main()
