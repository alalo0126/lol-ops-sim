"""포탑·방패 골드 검수: 1단계 원본에서 팀 분당 총 골드 증가를 그 분의 방패·포탑·킬·에픽 수로 회귀한다.

    python -m tools.tower_gold_fit

위키 값(방패 120 지역, 외곽 포탑 전역 50 등)이 26.20에 맞는지 보는 용도. 계수 = 팀 전체 골드 증가(5명 합).
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["const", "minute", "plate", "outer", "inner", "inhib", "kill", "dragon", "grubs", "herald", "first_tower"]


def main(prefix="16.20."):
    X, y = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("timeline") or not d.get("detail") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        info, fr = d["detail"]["info"], d["timeline"]["info"]["frames"]
        team = {p["participantId"]: p["teamId"] for p in info["participants"]}
        first = True
        for m in range(2, min(21, len(fr))):
            for T in (100, 200):
                g0 = sum(p["totalGold"] for k, p in fr[m - 1]["participantFrames"].items() if team[int(k)] == T)
                g1 = sum(p["totalGold"] for k, p in fr[m]["participantFrames"].items() if team[int(k)] == T)
                row = dict.fromkeys(COLS, 0.0)
                row["const"], row["minute"] = 1.0, m
                for e in fr[m]["events"]:
                    ty = e["type"]
                    # 방패·포탑 이벤트의 teamId는 '잃은 팀'
                    if ty == "TURRET_PLATE_DESTROYED" and e.get("teamId") != T:
                        row["plate"] += 1
                    elif ty == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING" and e.get("teamId") != T:
                        k = {"OUTER_TURRET": "outer", "INNER_TURRET": "inner", "BASE_TURRET": "inhib"}.get(e.get("towerType"))
                        if k:
                            row[k] += 1
                            if first:
                                row["first_tower"] += 1
                    elif ty == "CHAMPION_KILL" and team.get(e.get("killerId")) == T:
                        row["kill"] += 1
                    elif ty == "ELITE_MONSTER_KILL" and e.get("killerTeamId") == T:
                        k = {"DRAGON": "dragon", "HORDE": "grubs", "RIFTHERALD": "herald"}.get(e.get("monsterType"))
                        if k:
                            row[k] += 1
                X.append([row[c] for c in COLS])
                y.append(g1 - g0)
            if any(e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING" for e in fr[m]["events"]):
                first = False
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * resid.var()
    for c, b, s in zip(COLS, beta, np.sqrt(np.diag(cov))):
        print(f"{c:12s} {b:8.1f} ± {1.96 * s:6.1f}")
    print(f"n={len(y)}")


if __name__ == "__main__":
    main()
