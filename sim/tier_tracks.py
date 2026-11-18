"""티어별 1초 동선 비교: 골드 리플레이(data/tiers/<TIER>) vs 천상계 리플레이(data/replay_*).

    python -m sim.tier_tracks [--tier GOLD]   :: → reports/tier_tracks_<tier>.md, .json

같은 함수로 두 묶음을 잰다(3~18분, 유효 초 80% 이상인 선수만).
- 정글: sim.real_actions 행동 비율(우리 캠프·카정·바위게·라인·오브젝트·귀환·이동·사망),
  sim.lane_visits 라인 방문(판당 횟수·킬·사망), 첫 라인 방문 시각(1:30 이후).
- 미드·서폿: sim.laner_tracks 구역 비율·기지 방문·로밍(횟수·킬)·나들이.
- 와드: 타임라인 WARD_PLACED(시각·설치자)에 설치자 1초 위치(±2초)를 붙여 구역을 정한다(와드 실제 위치가 아니라
  설치 순간 설치자 위치. 와드는 설치자 600 안에 놓이므로 구역 수준에선 대체로 맞음). 판당 개수·구역 비율.
사망 초는 <경기>_players.csv(5초 간격 is_dead)로 채운다. 신뢰구간은 선수 단위 부트스트랩 95%.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random

from .data import ROOT
from .game import Game
from .stage6 import zone
from . import real_actions as ra
from . import lane_visits as lv
from . import laner_tracks as lt

T0, T1 = 180, 1080
MIN_COVER = 0.8
TEAM = {"ORDER": 100, "CHAOS": 200}
WARD_ZONES = ("own_jungle", "enemy_jungle", "river", "lane", "base")
WARD_KO = {"own_jungle": "우리 정글", "enemy_jungle": "적 정글", "river": "강가·둥지", "lane": "라인", "base": "기지"}


def load_set(raw_dir, track_dir, roles):
    """[(role, dict(mid, team, pid, pos{초:(x,y)|None}, kills, wards))]"""
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        mid = os.path.basename(path)[:-5]
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        parts = d["detail"]["info"]["participants"]
        frames = d["timeline"]["info"]["frames"]
        events = [e for fr in frames for e in fr.get("events", [])]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds") or []} for e in events if e["type"] == "CHAMPION_KILL"]
        by_pos = {(q["teamId"], q.get("teamPosition")): q for q in parts}
        dead = collections.defaultdict(set)
        ps = os.path.join(track_dir, f"{mid}_players.csv")
        if os.path.exists(ps):
            with open(ps, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    q = by_pos.get((TEAM.get(r.get("team")), r.get("position")))
                    if q and r.get("is_dead") == "1":
                        t = round(float(r["time"]))
                        dead[q["participantId"]].update(range(t - 2, t + 3))
        for q in parts:
            role = q.get("teamPosition")
            tp = os.path.join(track_dir, f"{mid}_{q['championName']}.csv")
            if role not in roles or not os.path.exists(tp):
                continue
            pid = q["participantId"]
            pos = {}
            with open(tp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("valid", "1") == "1":
                        pos[round(float(r["time"]))] = (float(r["x"]), float(r["y"]))
            for t in dead[pid]:
                pos[t] = None
            if sum(1 for t in range(T0, T1) if t in pos) < MIN_COVER * (T1 - T0):
                continue
            wards = [(e["timestamp"] / 1000, e.get("wardType", "")) for e in events
                     if e["type"] == "WARD_PLACED" and e.get("creatorId") == pid and e.get("wardType") != "UNDEFINED"]
            out.append((role, {"mid": mid, "team": q["teamId"], "pid": pid, "pos": pos, "kills": kills,
                               "wards": wards, "team_pids": {p["participantId"] for p in parts if p["teamId"] == q["teamId"]}}))
    return out


def ci(vals, boot=1000, seed=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None, 0
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(boot))
    return m, bs[int(0.025 * boot)], bs[int(0.975 * boot)], len(vals)


def jungle_metrics(p, nodes, herald):
    segs = ra.segments(p["pos"], p["team"], nodes, herald)
    cnt = collections.Counter(ra.group(a) for a in segs.values())
    tot = max(1, sum(cnt.values()))
    m = {f"act_{g}": cnt[g] / tot for g in ra.GROUPS}
    zones = {t: (zone(*xy, p["team"], nodes) if xy else "dead") for t, xy in p["pos"].items()}
    vis = lv.visits(zones)
    outs = [lv.outcome(s, p["kills"], p["pid"], p["team_pids"]) for s in vis]
    m["lane_visits"] = len(outs)
    m["lane_visit_kill"] = (sum(o["kill"] for o in outs) / len(outs)) if outs else None
    m["lane_visit_death"] = (sum(o["death"] for o in outs) / len(outs)) if outs else None
    early = [t for t in range(90, 600) if zones.get(t) == "lane"]
    m["first_lane_min"] = early[0] / 60 if early else None
    return m


def laner_metrics(role, p, nodes):
    track = {}
    for t, xy in p["pos"].items():
        if T0 <= t < T1:
            track[t] = "dead" if xy is None else lt.place(*xy, p["team"], role, nodes)
    b = lt.behaviour(track, p["kills"], p["pid"], p["team_pids"])
    m = {f"zone_{z}": b["share"][z] for z in lt.ZONES}
    m["base_visits"] = b["base_visits"]
    m["roams"] = len(b["roams"])
    m["roam_kill"] = (sum(r["kill"] for r in b["roams"]) / len(b["roams"])) if b["roams"] else None
    m["trips"] = b["trips"]
    return m


def ward_metrics(p, nodes):
    zs = []
    for t, _ in p["wards"]:
        if not (T0 <= t < T1):
            continue
        xy = None
        for dt in (0, -1, 1, -2, 2):
            v = p["pos"].get(round(t) + dt)
            if v:
                xy = v
                break
        if xy is None:
            continue
        z = zone(*xy, p["team"], nodes)
        zs.append("river" if z == "pit" else z)
    m = {"wards_3_18": len(zs)}
    for z in WARD_ZONES:
        m[f"ward_{z}"] = (zs.count(z) / len(zs)) if zs else None
    return m


LABELS = {
    "JUNGLE": [(f"act_{g}", f"시간: {ra.GROUP_KO[g]}", "%") for g in ra.GROUPS] + [
        ("lane_visits", "라인 방문 수(3~18분)", ""), ("lane_visit_kill", "라인 방문당 킬 관여", "%"),
        ("lane_visit_death", "라인 방문당 사망", "%"), ("first_lane_min", "첫 라인 방문(분, 1:30~10분)", "")],
    "LANER": [(f"zone_{z}", f"시간: {lt.ZONE_KO[z]}", "%") for z in lt.ZONES] + [
        ("base_visits", "기지 방문(살아서)", ""), ("roams", "로밍(14분 전)", ""), ("roam_kill", "로밍당 킬 관여", "%"),
        ("trips", "강가·정글 나들이", "")],
    "WARD": [("wards_3_18", "와드 설치(3~18분)", "")] + [(f"ward_{z}", f"와드 위치: {WARD_KO[z]}", "%") for z in WARD_ZONES],
}


def fmt(v, unit):
    if v is None:
        return "-"
    return f"{v:.0%}" if unit == "%" else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="GOLD")
    ap.add_argument("--roles", default="JUNGLE,MIDDLE,UTILITY")
    args = ap.parse_args()
    roles = [r.strip().upper() for r in args.roles.split(",")]
    g = Game(seed=0)
    nodes, herald = g.map.nodes, g.objectives["herald"]["spawn_at"]
    tdir = os.path.join(ROOT, "data", "tiers", args.tier.upper())
    sets = {args.tier.upper(): load_set(os.path.join(tdir, "replay_raw"), os.path.join(tdir, "replay_tracks"), roles),
            "천상계": load_set(os.path.join(ROOT, "data", "replay_raw"), os.path.join(ROOT, "data", "replay_tracks"), roles)}
    res = {}
    for name, rows in sets.items():
        per = collections.defaultdict(list)
        for role, p in rows:
            m = jungle_metrics(p, nodes, herald) if role == "JUNGLE" else laner_metrics(role, p, nodes)
            per[role].append(m | ward_metrics(p, nodes))
        res[name] = per
    names = list(sets)
    L = [f"# 1초 동선 티어 비교: {names[0]} vs 천상계 (3~18분)", "",
         "평균 [95% 구간], 괄호 안 n은 선수 수. 와드 위치는 설치 순간 설치자 위치의 구역.", ""]
    out_json = {}
    for role in roles:
        L += [f"## {role}", "", f"| 지표 | {names[0]} | 천상계 | 차이 |", "|---|---|---|---|"]
        keys = LABELS["JUNGLE" if role == "JUNGLE" else "LANER"] + LABELS["WARD"]
        out_json[role] = {}
        for k, label, unit in keys:
            a = ci([m.get(k) for m in res[names[0]][role]])
            b = ci([m.get(k) for m in res[names[1]][role]])
            out_json[role][k] = {names[0]: a, names[1]: b}
            if a[0] is None or b[0] is None:
                continue
            sep = "**" if (a[2] < b[1] or b[2] < a[1]) else ""
            diff = a[0] - b[0]
            L.append(f"| {label} | {fmt(a[0], unit)} [{fmt(a[1], unit)}~{fmt(a[2], unit)}] (n {a[3]}) | "
                     f"{fmt(b[0], unit)} [{fmt(b[1], unit)}~{fmt(b[2], unit)}] (n {b[3]}) | {sep}{'+' if diff >= 0 else ''}"
                     f"{fmt(diff, unit) if unit != '%' else f'{diff * 100:.0f}%p'}{sep} |")
        L.append("")
    L.append("굵은 차이 = 두 구간이 겹치지 않음.")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", f"tier_tracks_{args.tier.lower()}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(out[:-3] + ".json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
