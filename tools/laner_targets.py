"""라이너 검수 목표값: 1단계 1,000판 원본(data/raw)에서 라인 모듈·라이너 행동 모델이 맞춰야 할 분포를 뽑는다.

    python -m tools.laner_targets          :: data/raw → sim/targets_laner.json

뽑는 것(포지션 TOP·MIDDLE·BOTTOM·UTILITY, 양 팀 모두)
- 분별(1~20분) 라인 CS·정글 CS·골드·경험치·레벨·체력 비율·마나 비율·보유 골드
- 분 프레임 위치 구역: 자기 라인 / 다른 라인 / 정글·강가 / 기지
- 구매 묶음(20초 안 연속 구매 = 상점 방문 1번, 되돌리기 제외): 첫 방문 시각, 20분까지 방문 수 → 귀환 대리값
- 사망(10·20분), 와드 설치 수(10·20분)
- 포탑: 방패(TURRET_PLATE_DESTROYED) 포탑별 개수·시각, 포탑 파괴 종류·시각
"""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.stage6 import zone, MAP_MAX   # noqa: E402

LANERS = ["TOP", "MIDDLE", "BOTTOM", "UTILITY"]
LANE_OF = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
SHOP_GAP_MS = 20000
# 방패 이벤트 위치 → 포탑 단계(1단계 1,000판에서 나온 좌표, 양 팀)
TIER = {(981, 10441): "outer", (10504, 1029): "outer", (13866, 4505): "outer", (8955, 8510): "outer", (4318, 13875): "outer",
        (5846, 6396): "outer", (5048, 4812): "inner", (9767, 10113): "inner", (6919, 1483): "inner", (13327, 8226): "inner",
        (1512, 6699): "inner", (7943, 13411): "inner", (3651, 3696): "inhib", (11134, 11207): "inhib", (13624, 10572): "inhib",
        (4281, 1253): "inhib", (1169, 4287): "inhib", (10481, 13650): "inhib"}


def lane_band(x, y):
    """라인 띠 → top/mid/bot (zone()과 같은 경계)."""
    rx, ry = MAP_MAX - x, MAP_MAX - y
    if (x < 1900 and y > 4500) or (y > 12900 and x < 10000) or (x < 3300 and y > 11500):
        return "top"
    if (rx < 1900 and ry > 4500) or (ry > 12900 and rx < 10000) or (rx < 3300 and ry > 11500):
        return "bot"
    if abs(x - y) < 1000 and 7000 < x + y < 2 * MAP_MAX - 7000:
        return "mid"
    return None


def pct(v, qs=(10, 25, 50, 75, 90)):
    v = sorted(v)
    if not v:
        return {}
    out = {f"p{q}": v[min(len(v) - 1, int(q / 100 * len(v)))] for q in qs}
    out["mean"] = round(sum(v) / len(v), 3)
    out["n"] = len(v)
    return out


def main(prefix="16.20."):
    nodes = json.load(open(os.path.join(ROOT, "rules", "map_nodes.json"), encoding="utf-8"))["nodes"]
    by_min = collections.defaultdict(lambda: collections.defaultdict(list))   # (role, minute) → 지표 → 값
    where = collections.defaultdict(collections.Counter)                       # (role, minute) → 구역
    first_shop, shops20, deaths, wards = (collections.defaultdict(list) for _ in range(4))
    plates_turret = collections.defaultdict(list)       # (lane, 포탑 좌표) → 경기별 방패 수
    plate_t = collections.defaultdict(list)             # lane → 방패 시각(초)
    plate_killer = collections.Counter()
    tower_t = collections.defaultdict(list)             # (towerType, lane) → 파괴 시각(초)
    first_tower = []
    by20 = collections.defaultdict(list)                 # 경기당 20분까지 방패·포탑 수(양 팀 합)
    band = collections.defaultdict(collections.Counter)   # (role, minute, 자기 라인 외곽 포탑 상태) → 라인 띠
    plate_tier = collections.defaultdict(list)            # 포탑 단계 → 방패 시각
    n = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        info, tl = d["detail"]["info"], d["timeline"]["info"]
        if info.get("gameDuration", 0) < 20 * 60:
            continue
        n += 1
        role = {p["participantId"]: p["teamPosition"] for p in info["participants"]}
        team = {p["participantId"]: p["teamId"] for p in info["participants"]}
        ev = [e for f in tl["frames"] for e in f["events"]]
        outer_down = {}   # 라인 → 그 라인 외곽 포탑(어느 쪽이든) 첫 파괴 시각(ms)
        for e in ev:
            if e["type"] == "BUILDING_KILL" and e.get("towerType") == "OUTER_TURRET":
                ln = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}[e["laneType"]]
                outer_down.setdefault(ln, e["timestamp"])
        for m, fr in enumerate(tl["frames"][1:21], start=1):
            for pid_s, pf in fr["participantFrames"].items():
                pid = int(pid_s)
                r = role.get(pid)
                if r not in LANERS:
                    continue
                cs = pf["championStats"]
                row = by_min[(r, m)]
                row["lane_cs"].append(pf["minionsKilled"])
                row["jungle_cs"].append(pf["jungleMinionsKilled"])
                row["gold"].append(pf["totalGold"])
                row["current_gold"].append(pf["currentGold"])
                row["xp"].append(pf["xp"])
                row["level"].append(pf["level"])
                row["hp"].append(round(cs["health"] / max(1, cs["healthMax"]), 3))
                if cs.get("powerMax", 0) > 0:
                    row["mana"].append(round(cs["power"] / cs["powerMax"], 3))
                pos = pf.get("position") or {"x": 0, "y": 0}
                z = zone(pos["x"], pos["y"], team[pid], nodes)
                if z == "lane":
                    z = "own_lane" if lane_band(pos["x"], pos["y"]) == LANE_OF[r] else "other_lane"
                where[(r, m)][z] += 1
                down = "down" if outer_down.get(LANE_OF[r], 1e12) <= m * 60000 else "up"
                band[(r, m, down)][lane_band(pos["x"], pos["y"]) or "none"] += 1
        shops = collections.defaultdict(list)
        for e in ev:
            t, ty = e.get("timestamp", 0), e["type"]
            if ty == "ITEM_PURCHASED" and t > 60000:
                s = shops[e["participantId"]]
                if not s or t - s[-1][-1] > SHOP_GAP_MS:
                    s.append([t])
                else:
                    s[-1].append(t)
            elif ty == "TURRET_PLATE_DESTROYED":
                p = e.get("position", {})
                plate_t[e.get("laneType")].append(round(t / 1000))
                plates_turret[(e.get("laneType"), p.get("x"), p.get("y"))].append(n)
                plate_killer[role.get(e.get("killerId"), "none")] += 1
                plate_tier[TIER.get((p.get("x"), p.get("y")), "other")].append(round(t / 1000))
            elif ty == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                tower_t[(e.get("towerType"), e.get("laneType"))].append(round(t / 1000))
        towers = [e["timestamp"] for e in ev if e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING"]
        by20["plates"].append(sum(1 for e in ev if e["type"] == "TURRET_PLATE_DESTROYED" and e["timestamp"] <= 1200000))
        for tt, key in (("OUTER_TURRET", "outer"), ("INNER_TURRET", "inner"), ("BASE_TURRET", "inhib")):
            by20[key].append(sum(1 for e in ev if e["type"] == "BUILDING_KILL" and e.get("towerType") == tt and e["timestamp"] <= 1200000))
        if towers:
            first_tower.append(round(min(towers) / 1000))
        for pid, r in role.items():
            if r not in LANERS:
                continue
            s = shops.get(pid, [])
            first_shop[r].append(round(s[0][0] / 1000) if s else None)
            shops20[r].append(sum(1 for x in s if x[0] <= 1200000))
            deaths[(r, 10)].append(sum(1 for e in ev if e["type"] == "CHAMPION_KILL" and e.get("victimId") == pid and e["timestamp"] <= 600000))
            deaths[(r, 20)].append(sum(1 for e in ev if e["type"] == "CHAMPION_KILL" and e.get("victimId") == pid and e["timestamp"] <= 1200000))
            wards[(r, 10)].append(sum(1 for e in ev if e["type"] == "WARD_PLACED" and e.get("creatorId") == pid and e["timestamp"] <= 600000))
            wards[(r, 20)].append(sum(1 for e in ev if e["type"] == "WARD_PLACED" and e.get("creatorId") == pid and e["timestamp"] <= 1200000))

    # 포탑별 방패 수(경기당 최대가 그 포탑의 방패 개수)
    per_game = collections.defaultdict(collections.Counter)
    for k, games in plates_turret.items():
        for g in games:
            per_game[k][g] += 1
    turret_plates = {f"{k[0]}@{k[1]},{k[2]}": {"games": len(c), "max": max(c.values()), "mean": round(sum(c.values()) / len(c), 2)}
                     for k, c in sorted(per_game.items(), key=lambda kv: -len(kv[1])) if len(c) >= 20}
    out = {
        "_doc": "tools/laner_targets.py: 1단계 1,000판(20분 이상) 라이너 분포. 위치·체력·마나는 분 프레임 순간값",
        "games": n,
        "by_minute": {r: {m: {k: pct(v) for k, v in by_min[(r, m)].items()} for m in range(1, 21)} for r in LANERS},
        "zone_by_minute": {r: {m: {z: round(c / sum(where[(r, m)].values()), 3) for z, c in where[(r, m)].most_common()}
                               for m in range(1, 21)} for r in LANERS},
        "first_shop_s": {r: pct([x for x in v if x is not None]) for r, v in first_shop.items()},
        "shops_by_20": {r: pct(v) for r, v in shops20.items()},
        "deaths": {f"{r}_{m}": pct(v) for (r, m), v in deaths.items()},
        "wards": {f"{r}_{m}": pct(v) for (r, m), v in wards.items()},
        "plate_time_s": {k: pct(v) for k, v in plate_t.items()},
        "plate_killer_role": dict(plate_killer.most_common()),
        "turret_plates": turret_plates,
        "tower_kill_s": {f"{k[0]}_{k[1]}": pct(v) | {"share_games": round(len(v) / max(1, n) / 2, 3)} for k, v in sorted(tower_t.items())},
        "first_tower_s": pct(first_tower),
        "by_20_per_game": {k: pct(v) for k, v in by20.items()},
        "plate_time_by_tier_s": {k: pct(v) for k, v in plate_tier.items()},
        "lane_band_by_minute": {r: {f"{m}_{dn}": {b: round(c / sum(band[(r, m, dn)].values()), 3) for b, c in band[(r, m, dn)].most_common()}
                                    for m in range(1, 21) for dn in ("up", "down") if band[(r, m, dn)]} for r in LANERS},
        "lane_band_n": {r: {f"{m}_{dn}": sum(band[(r, m, dn)].values()) for m in range(1, 21) for dn in ("up", "down")} for r in LANERS},
    }
    dst = os.path.join(ROOT, "sim", "targets_laner.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{n}판 → {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
