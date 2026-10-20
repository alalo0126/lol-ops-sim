"""5단계 버그 악용 점검용 실제 분포: 20분까지 팀별 오브젝트 수와 정글러 20분 지표.

    python tools\\stage5_targets.py     :: data\\raw → sim\\targets_stage5.json

4단계 기준값(targets_stage1.json)은 15분까지만 있어서, 학습 정글러의 20분 행동(드래곤·유충·전령 수, 킬·데스,
정글 CS, 적 정글 캠프 비중)이 실제 범위 안인지 볼 값을 따로 뽑는다. 양 팀 정글러 모두를 표본으로 쓴다.
"""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(values):
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "p10": q(0.1), "p25": q(0.25), "median": statistics.median(v), "p75": q(0.75), "p90": q(0.9),
            "mean": round(statistics.fmean(v), 2)}


def main(prefix="16.20."):
    out = {k: [] for k in ("dragons20", "grubs20", "heralds20", "kills20", "deaths20", "assists20", "cs20", "gold20",
                           "enemy_jungle_cs_share")}
    n = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        det, tl = d.get("detail"), d.get("timeline")
        if not det or not tl or not det["info"]["gameVersion"].startswith(prefix):
            continue
        info = det["info"]
        frames = tl["info"]["frames"]
        if info.get("gameDuration", 0) <= 1200 or len(frames) <= 20:
            continue
        n += 1
        team_of = {p["participantId"]: p["teamId"] for p in info["participants"]}
        jgs = [p["participantId"] for p in info["participants"] if p["teamPosition"] == "JUNGLE"]
        events = [e for fr in frames[:21] for e in fr["events"] if e.get("timestamp", 0) <= 1200000]
        for team in (100, 200):
            elite = [e for e in events if e["type"] == "ELITE_MONSTER_KILL" and e.get("killerTeamId") == team]
            out["dragons20"].append(sum(e.get("monsterType") == "DRAGON" for e in elite))
            out["grubs20"].append(sum(e.get("monsterType") == "HORDE" for e in elite))
            out["heralds20"].append(sum(e.get("monsterType") == "RIFTHERALD" for e in elite))
        for j in jgs:
            pf = frames[20]["participantFrames"][str(j)]
            out["cs20"].append(pf["jungleMinionsKilled"])
            out["gold20"].append(pf["totalGold"])
            ks = [e for e in events if e["type"] == "CHAMPION_KILL"]
            out["kills20"].append(sum(e.get("killerId") == j for e in ks))
            out["deaths20"].append(sum(e.get("victimId") == j for e in ks))
            out["assists20"].append(sum(j in e.get("assistingParticipantIds", []) for e in ks))
        for p in info["participants"]:
            if p["teamPosition"] == "JUNGLE":
                own, enemy = p.get("totalAllyJungleMinionsKilled"), p.get("totalEnemyJungleMinionsKilled")
                if own is not None and enemy is not None and own + enemy > 0:
                    out["enemy_jungle_cs_share"].append(round(enemy / (own + enemy), 3))
    res = {"matches_over_20min": n, "patch_prefix": prefix, **{k: pct(v) for k, v in out.items() if v}}
    res["_doc"] = "20분 넘게 이어진 경기만. 오브젝트는 팀별(양 팀 모두 표본), 정글러 지표는 양 팀 정글러. 적 정글 CS 비중은 경기 끝 기준."
    dst = os.path.join(ROOT, "sim", "targets_stage5.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"{n}판 → {dst}")
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, v)


if __name__ == "__main__":
    main()
