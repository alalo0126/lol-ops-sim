"""1단계 완료 기준 요약: 정글러의 10분 골드·CS·레벨, 초반 정글 CS, 첫 오브젝트 시각 분포.

  python collect/summarize.py                     # data/raw 기준
  python collect/summarize.py --raw ../match_cache --patch any   # 다른 폴더 시험용

타임라인은 1분 단위 프레임이고 일반 캠프 처치 이벤트가 없어서 첫 풀캠 "시각"은 직접 잴 수 없다.
대신 3·4분 시점 정글 CS 분포를 낸다. 캠프별 CS 값은 패치마다 달라 여기서 풀캠 기준선을 정하지 않는다.
"""
import argparse
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OBJECTIVES = {
    "first_blood": lambda e: e["type"] == "CHAMPION_KILL",
    "first_voidgrubs": lambda e: e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") == "HORDE",
    "first_dragon": lambda e: e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") == "DRAGON",
    "first_herald": lambda e: e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") == "RIFTHERALD",
    "first_tower": lambda e: e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING",
}


def pct(values):
    if not values:
        return None
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "p10": q(0.1), "p25": q(0.25), "median": statistics.median(v),
            "p75": q(0.75), "p90": q(0.9), "mean": round(statistics.fmean(v), 1)}


def junglers(detail, seed):
    if seed:
        return [seed["participantId"]]
    # 시드 정보가 없는 예전 캐시는 양 팀 정글러를 모두 쓴다.
    return [p["participantId"] for p in detail["info"]["participants"] if p.get("teamPosition") == "JUNGLE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--patch", default="26.20", help="'any'면 패치 필터를 끈다")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "summary_stage1.json"))
    args = ap.parse_args()

    prefix = None
    if args.patch != "any":
        y, m = args.patch.split(".")
        prefix = f"{int(y) - 10}.{int(m)}."

    m10 = {k: [] for k in ("gold", "lane_cs", "jungle_cs", "total_cs", "level", "xp")}
    early_jcs = {"3min": [], "4min": []}
    obj = {k: [] for k in OBJECTIVES}
    durations, matches, skipped = [], 0, 0

    for path in sorted(glob.glob(os.path.join(args.raw, "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if "detail" not in d or "timeline" not in d:
            skipped += 1
            continue
        info = d["detail"]["info"]
        if prefix and not info["gameVersion"].startswith(prefix):
            skipped += 1
            continue
        frames = d["timeline"]["info"]["frames"]
        matches += 1
        durations.append(info["gameDuration"] / 60)

        for pid in junglers(d["detail"], d.get("seed")):
            if len(frames) > 10:
                pf = frames[10]["participantFrames"][str(pid)]
                m10["gold"].append(pf["totalGold"])
                m10["lane_cs"].append(pf["minionsKilled"])
                m10["jungle_cs"].append(pf["jungleMinionsKilled"])
                m10["total_cs"].append(pf["minionsKilled"] + pf["jungleMinionsKilled"])
                m10["level"].append(pf["level"])
                m10["xp"].append(pf["xp"])
            for minute in (3, 4):
                if len(frames) > minute:
                    early_jcs[f"{minute}min"].append(frames[minute]["participantFrames"][str(pid)]["jungleMinionsKilled"])

        events = [e for fr in frames for e in fr["events"]]
        for name, cond in OBJECTIVES.items():
            t = next((e["timestamp"] for e in events if cond(e)), None)
            if t is not None:
                obj[name].append(round(t / 60000, 2))

    summary = {
        "matches": matches, "skipped_files": skipped, "patch_filter": prefix or "any",
        "game_minutes": pct(durations),
        "jungler_at_10min": {k: pct(v) for k, v in m10.items()},
        "jungler_jungle_cs_early": {k: pct(v) for k, v in early_jcs.items()},
        "first_objective_minute": {k: pct(v) for k, v in obj.items()},
        "notes": [
            "프레임 간격 60초: 10분 값은 10:00 프레임(약간의 지연 포함) 기준",
            "첫 풀캠 시각은 타임라인으로 직접 측정 불가 (일반 캠프 이벤트 없음)",
            "WARD_PLACED 이벤트에 위치가 없음: 와드 위치는 수동 입력 대상",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"경기 {matches}판 (제외 {skipped}) · 패치 필터 {summary['patch_filter']}")
    fmt = lambda s: "-" if not s else f"중앙 {s['median']:>7} | p10 {s['p10']:>7} p90 {s['p90']:>7} (n={s['n']})"
    print("\n[정글러 10분]")
    for k, s in summary["jungler_at_10min"].items():
        print(f"  {k:<10} {fmt(s)}")
    print("\n[정글러 초반 정글 CS]")
    for k, s in summary["jungler_jungle_cs_early"].items():
        print(f"  {k:<10} {fmt(s)}")
    print("\n[첫 오브젝트 시각(분)]")
    for k, s in summary["first_objective_minute"].items():
        print(f"  {k:<16} {fmt(s)}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
