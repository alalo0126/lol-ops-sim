"""6단계 입력: 1단계 원본(data/raw, 약 900MB)을 비교에 필요한 값만 남긴 작은 파일로 줄인다.

    python tools\\stage6_compact.py      :: data\\raw → data\\stage6_compact.json.gz

남기는 것(0~21분): 분 프레임(위치, 체력, 레벨, 경험치, 골드, 정글·라인 CS), 챔피언 킬(위치·관여자),
에픽 몬스터 처치, 구매 시각, 와드 설치(위치 없음), 포탑 파괴. 경기별 승패·참가자 포지션·챔피언.
"""
import glob
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNTIL_MS = 21 * 60000


def compact(d):
    info, tl = d["detail"]["info"], d["timeline"]["info"]
    parts = [[p["participantId"], p["teamId"], p["teamPosition"], p["championName"],
              p.get("totalAllyJungleMinionsKilled"), p.get("totalEnemyJungleMinionsKilled")] for p in info["participants"]]
    frames = []
    for fr in tl["frames"][:22]:
        row = {}
        for pid, pf in fr["participantFrames"].items():
            cs, pos = pf["championStats"], pf.get("position", {"x": 0, "y": 0})
            row[pid] = [pos["x"], pos["y"], cs.get("health", 0), cs.get("healthMax", 1), pf["level"], pf["xp"],
                        pf["totalGold"], pf["currentGold"], pf["jungleMinionsKilled"], pf["minionsKilled"], cs.get("movementSpeed", 0)]
        frames.append(row)
    ev = {"kill": [], "elite": [], "buy": [], "ward": [], "tower": []}
    for fr in tl["frames"]:
        for e in fr["events"]:
            t = e.get("timestamp", 0)
            if t > UNTIL_MS:
                continue
            ty = e["type"]
            if ty == "CHAMPION_KILL":
                p = e.get("position", {})
                ev["kill"].append([t, e.get("killerId", 0), e.get("victimId", 0), e.get("assistingParticipantIds", []), p.get("x"), p.get("y")])
            elif ty == "ELITE_MONSTER_KILL":
                p = e.get("position", {})
                ev["elite"].append([t, e.get("monsterType"), e.get("monsterSubType"), e.get("killerTeamId"), e.get("killerId", 0),
                                    e.get("assistingParticipantIds", []), p.get("x"), p.get("y")])
            elif ty == "ITEM_PURCHASED":
                ev["buy"].append([t, e["participantId"], e.get("itemId")])
            elif ty == "WARD_PLACED":
                ev["ward"].append([t, e.get("creatorId", 0), e.get("wardType")])
            elif ty == "BUILDING_KILL":
                ev["tower"].append([t, e.get("teamId"), e.get("laneType"), e.get("buildingType"), e.get("towerType")])
    win100 = next(t["win"] for t in info["teams"] if t["teamId"] == 100)
    return {"id": info.get("gameId"), "version": info["gameVersion"], "duration": info.get("gameDuration", 0),
            "win100": int(win100), "parts": parts, "frames": frames, "events": ev}


def main(prefix="16.20."):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        m = compact(d)
        m["match"] = os.path.basename(path)[:-5]
        out.append(m)
    dst = os.path.join(ROOT, "data", "stage6_compact.json.gz")
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(out)}판 → {dst} ({os.path.getsize(dst) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
