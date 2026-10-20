"""4단계 기준값: 시뮬레이터와 비교할 실제 경기 분포와 맵 좌표 검수 자료를 data/stage4_targets.json에 저장한다.

    python tools\stage4_targets.py

담는 것
- 양 팀 정글러 둘 다(시뮬레이터는 두 정글러가 같은 봇이라 이쪽이 맞는 비교 대상): 분 단위 분포, 10분 킬·데스·어시스트, CS 24 첫 도달
- 정글러(시드 선수) 분 단위 정글 CS·골드·레벨·경험치 분포, 10분 킬·데스·어시스트
- 정글 CS가 처음 24 이상이 된 프레임(분) 분포: "캠프 1개 = 4 CS" 가설이 맞다면 첫 풀캠 시각의 1분 해상도 대리값
- 라인별 라이너 10분 골드·CS·레벨 (라이너 봇 수입 보정용)
- 경험치 → 레벨 경계: 레벨 L의 최대 경험치와 L+1의 최소 경험치 (rules/levels.json 검수용)
- 포탑·에픽 몬스터 처치 위치 중앙값 (rules/map_nodes.json 좌표 검수용)
"""
import collections
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(values):
    if not values:
        return None
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "p10": q(0.1), "p25": q(0.25), "median": statistics.median(v),
            "p75": q(0.75), "p90": q(0.9), "mean": round(statistics.fmean(v), 1)}


def med_xy(points):
    return {"x": round(statistics.median(p[0] for p in points)), "y": round(statistics.median(p[1] for p in points)), "n": len(points)}


def main():
    prefix = "16.20."
    by_min = collections.defaultdict(lambda: collections.defaultdict(list))
    first24, kda10 = [], {"kills": [], "deaths": [], "assists": []}
    by_min_all = collections.defaultdict(lambda: collections.defaultdict(list))
    first24_all, kda10_all = [], {"kills": [], "deaths": [], "assists": []}
    jcs_mod4 = collections.Counter()
    laner10 = collections.defaultdict(lambda: collections.defaultdict(list))
    lvl_xp = collections.defaultdict(lambda: [10 ** 9, -1])  # level -> [min xp, max xp]
    towers = collections.defaultdict(list)
    elites = collections.defaultdict(list)
    n = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        det, tl, seed = d.get("detail"), d.get("timeline"), d.get("seed")
        if not det or not tl or not seed or not det["info"]["gameVersion"].startswith(prefix):
            continue
        n += 1
        frames = tl["info"]["frames"]
        jid = str(seed["participantId"])
        pos = {str(p["participantId"]): p["teamPosition"] for p in det["info"]["participants"]}

        jgs = [pid for pid, p in pos.items() if p == "JUNGLE"]
        if len(jgs) != 2:
            jgs = [jid]
        for m, fr in enumerate(frames[:21]):
            for j in jgs:
                pj = fr["participantFrames"][j]
                for k, key in (("jungle_cs", "jungleMinionsKilled"), ("lane_cs", "minionsKilled"), ("gold", "totalGold"),
                               ("level", "level"), ("xp", "xp")):
                    by_min_all[m][k].append(pj[key])
            pf = fr["participantFrames"][jid]
            by_min[m]["jungle_cs"].append(pf["jungleMinionsKilled"])
            by_min[m]["lane_cs"].append(pf["minionsKilled"])
            by_min[m]["gold"].append(pf["totalGold"])
            by_min[m]["level"].append(pf["level"])
            by_min[m]["xp"].append(pf["xp"])
            if 1 <= m <= 6:
                jcs_mod4[pf["jungleMinionsKilled"] % 4] += 1
            for pid, p in fr["participantFrames"].items():
                lo_hi = lvl_xp[p["level"]]
                lo_hi[0], lo_hi[1] = min(lo_hi[0], p["xp"]), max(lo_hi[1], p["xp"])
        hit = next((m for m, fr in enumerate(frames) if fr["participantFrames"][jid]["jungleMinionsKilled"] >= 24), None)
        if hit is not None:
            first24.append(hit)
        for j in jgs:
            h = next((m for m, fr in enumerate(frames) if fr["participantFrames"][j]["jungleMinionsKilled"] >= 24), None)
            if h is not None:
                first24_all.append(h)
        kd = {j: [0, 0, 0] for j in jgs}
        if len(frames) > 10:
            for pid, p in frames[10]["participantFrames"].items():
                if pos[pid] in ("TOP", "MIDDLE", "BOTTOM", "UTILITY"):
                    laner10[pos[pid]]["gold"].append(p["totalGold"])
                    laner10[pos[pid]]["cs"].append(p["minionsKilled"] + p["jungleMinionsKilled"])
                    laner10[pos[pid]]["level"].append(p["level"])

        k = dth = a = 0
        for fr in frames:
            for e in fr["events"]:
                t = e.get("timestamp", 0)
                if e["type"] == "CHAMPION_KILL" and t <= 600000:
                    k += str(e.get("killerId")) == jid
                    dth += str(e.get("victimId")) == jid
                    a += int(jid) in e.get("assistingParticipantIds", [])
                    for j in jgs:
                        kd[j][0] += str(e.get("killerId")) == j
                        kd[j][1] += str(e.get("victimId")) == j
                        kd[j][2] += int(j) in e.get("assistingParticipantIds", [])
                elif e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                    towers[f'{e["teamId"]}_{e["laneType"]}_{e["towerType"]}'].append((e["position"]["x"], e["position"]["y"]))
                elif e["type"] == "ELITE_MONSTER_KILL":
                    elites[e["monsterType"]].append((e["position"]["x"], e["position"]["y"]))
        kda10["kills"].append(k)
        kda10["deaths"].append(dth)
        kda10["assists"].append(a)
        for v in kd.values():
            for key, x in zip(("kills", "deaths", "assists"), v):
                kda10_all[key].append(x)

    out = {
        "matches": n, "patch_prefix": prefix, "jungler": "시드 선수(챌린저·GM 정글러) 한 명",
        "jungler_by_minute": {m: {k: pct(v) for k, v in s.items()} for m, s in sorted(by_min.items())},
        "both_junglers_by_minute": {m: {k: pct(v) for k, v in s.items()} for m, s in sorted(by_min_all.items())},
        "both_junglers_kda_by_10min": {k: pct(v) for k, v in kda10_all.items()},
        "both_first_minute_jungle_cs_ge_24": pct(first24_all),
        "both_first_minute_jungle_cs_ge_24_hist": dict(sorted(collections.Counter(first24_all).items())),
        "first_minute_jungle_cs_ge_24": pct(first24),
        "first_minute_jungle_cs_ge_24_hist": dict(sorted(collections.Counter(first24).items())),
        "jungle_cs_mod4_minutes_1_to_6": dict(sorted(jcs_mod4.items())),
        "jungler_kda_by_10min": {k: pct(v) for k, v in kda10.items()},
        "laners_at_10min": {p: {k: pct(v) for k, v in s.items()} for p, s in laner10.items()},
        "xp_by_level": {lv: {"min": v[0], "max": v[1]} for lv, v in sorted(lvl_xp.items())},
        "tower_kill_positions": {k: med_xy(v) for k, v in sorted(towers.items())},
        "elite_kill_positions": {k: med_xy(v) for k, v in sorted(elites.items())},
    }
    dst = os.path.join(ROOT, "data", "stage4_targets.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"경기 {n}판 → {dst}")
    print("첫 정글 CS 24 도달 분:", out["first_minute_jungle_cs_ge_24_hist"])
    print("10분 정글러:", {k: out["jungler_by_minute"][10][k]["median"] for k in ("gold", "jungle_cs", "lane_cs", "level")})


if __name__ == "__main__":
    main()
