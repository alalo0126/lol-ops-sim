"""에픽 몬스터(용·유충·전령) 실측: 처치 속도·체력 손해·처치 뒤 귀환·둥지 근처 교전.

    python -m tools.objective_fit [--raw data/replay_raw] [--tracks data/replay_tracks] [--out sim/objective_fit.json]

- 인원 = 1 + 처치 기록의 어시스트(피해를 준 아군). 레벨·아이템 골드는 처치 직전 분 프레임의 그 인원 평균.
- 처치 속도(1초 동선이 있는 경기만): 처치한 팀 정글러가 둥지 1,200 안에 들어와 처치까지 계속 머문 시간.
  (오브젝트가 생기기 전에 와 있었으면 생성 시각부터. 유충은 같은 팀이 연속으로 잡은 묶음의 첫 도착→마지막 처치)
  시뮬 공식(HP / (정글러 dps × epic_dps_factor + (인원−1) × helper_dps))과 같은 표본으로 비교해 두 값을 맞춘다.
- 체력 손해: 정글러-분(3~19분) 받은 피해(damageStats.totalDamageTaken 증가) / 최대 체력을
  그 분의 오브젝트 처치(종류별, 1/인원으로 나눈 몫)·정글 CS/4·킬 관여·사망으로 최소제곱 회귀.
- 처치 뒤: 정글러가 처치 후 90초 안에 아이템을 샀나(귀환), 다음 분 프레임 체력 비율.
- 교전: 처치 ±30초에 둥지 3,000 안 챔피언 킬(처치 팀/상대 팀 사망 수).
"""
import argparse
import collections
import csv
import glob
import json
import math
import os

import numpy as np

from sim.data import ROOT, Rules

PIT = {"DRAGON": (9861, 4447), "HORDE": (5007, 10471), "RIFTHERALD": (5007, 10471)}
NAME = {"DRAGON": "dragon", "HORDE": "grubs", "RIFTHERALD": "herald"}
RAD = 1200


def track(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {round(float(r["time"])): (float(r["x"]), float(r["y"])) for r in csv.DictReader(f) if r.get("valid", "1") == "1"}


def arrival(pos, pit, t_kill):
    """t_kill까지 둥지 안에 머문 구간의 시작(3초 이하 끊김 허용). 처치 순간 근처에 없으면 None."""
    near = lambda t: t in pos and math.hypot(pos[t][0] - pit[0], pos[t][1] - pit[1]) < RAD
    t = int(t_kill)
    if not any(near(t - k) for k in range(0, 4)):
        return None
    start, gap = t, 0
    while t > t_kill - 180:
        t -= 1
        if near(t):
            start, gap = t, 0
        else:
            gap += 1
            if gap > 3:
                break
    return start


def game_rows(d, tracks_dir, mid, rules):
    info, fr = d["detail"]["info"], d["timeline"]["info"]["frames"]
    parts = {p["participantId"]: p for p in info["participants"]}
    jg = {p["teamId"]: p["participantId"] for p in parts.values() if p.get("teamPosition") == "JUNGLE"}
    ev = [e for f in fr for e in f["events"]]
    elite = [e for e in ev if e["type"] == "ELITE_MONSTER_KILL" and e["monsterType"] in NAME and e["timestamp"] < 1200000]
    kills = [e for e in ev if e["type"] == "CHAMPION_KILL"]
    buys = collections.defaultdict(list)
    for e in ev:
        if e["type"] == "ITEM_PURCHASED":
            buys[e["participantId"]].append(e["timestamp"] / 1000)

    def st(pid, t):
        pf = fr[min(int(t // 60), len(fr) - 1)]["participantFrames"][str(pid)]
        return pf["level"], (pf["totalGold"] - pf["currentGold"]) / 1000

    tr = {T: track(os.path.join(tracks_dir, f"{mid}_{parts[p]['championName']}.csv")) if tracks_dir else None for T, p in jg.items()}
    o = rules.objectives
    spawn = {"DRAGON": o["drake"]["first_spawn_s"], "HORDE": o["voidgrubs"]["spawn_s"], "RIFTHERALD": o["rift_herald"]["spawn_s"]}
    out = []
    prev_dragon = None
    grub_groups = []
    for e in elite:
        T, t, ty = e["killerTeamId"], e["timestamp"] / 1000, e["monsterType"]
        if ty == "HORDE" and grub_groups and grub_groups[-1]["team"] == T and t - grub_groups[-1]["t"] < 60:
            g = grub_groups[-1]
            g["t"], g["n"] = t, g["n"] + 1
            g["people"] |= {e["killerId"], *e.get("assistingParticipantIds", [])}
            continue
        row = {"type": NAME[ty], "t": t, "team": T, "people": {e["killerId"], *e.get("assistingParticipantIds", [])},
               "spawn": spawn[ty] if ty != "DRAGON" else (prev_dragon + o["drake"]["respawn_s"] if prev_dragon else spawn[ty]),
               "jungler_in": jg.get(T) in {e["killerId"], *e.get("assistingParticipantIds", [])}, "n": 1}
        if ty == "DRAGON":
            prev_dragon = t
        if ty == "HORDE":
            grub_groups.append(row)
        out.append(row)
    rows = []
    for r in out:
        T, t = r["team"], r["t"]
        ppl = [p for p in r["people"] if p in parts and parts[p]["teamId"] == T]
        j = jg.get(T)
        if not ppl or j is None:
            continue
        lv = np.mean([st(p, t)[0] for p in ppl])
        ig = np.mean([st(p, t)[1] for p in ppl])
        pit = PIT[{"dragon": "DRAGON", "grubs": "HORDE", "herald": "RIFTHERALD"}[r["type"]]]
        dur = None
        if tr.get(T) and r["jungler_in"]:
            a = arrival(tr[T], pit, t)
            if a is not None:
                dur = t - max(a, r["spawn"])
        recall = any(t < b <= t + 90 for b in buys.get(j, []))
        nf = fr[min(int(t // 60) + 1, len(fr) - 1)]["participantFrames"][str(j)]["championStats"]
        near = [k for k in kills if abs(k["timestamp"] / 1000 - t) <= 30 and "position" in k
                and math.hypot(k["position"]["x"] - pit[0], k["position"]["y"] - pit[1]) < 3000]
        rows.append({"type": r["type"], "t": t, "first_dragon": r["type"] == "dragon" and r["spawn"] == spawn["DRAGON"],
                     "delay": t - r["spawn"], "people": len(ppl), "level": float(lv), "item_k": float(ig),
                     "grubs": r["n"] if r["type"] == "grubs" else None, "dur": dur, "jungler_in": r["jungler_in"],
                     "recall90": recall, "hp_next": nf["health"] / max(1, nf["healthMax"]),
                     "deaths_own": sum(1 for k in near if parts.get(k["victimId"], {}).get("teamId") == T),
                     "deaths_opp": sum(1 for k in near if parts.get(k["victimId"], {}).get("teamId") not in (None, T))})
    # 정글러-분 피해 회귀 표본
    dmg = []
    for T, j in jg.items():
        for m in range(3, min(19, len(fr) - 1)):
            a, b = fr[m]["participantFrames"][str(j)], fr[m + 1]["participantFrames"][str(j)]
            taken = (b["damageStats"]["totalDamageTaken"] - a["damageStats"]["totalDamageTaken"]) / max(1, a["championStats"]["healthMax"])
            share = {"dragon": 0.0, "grubs": 0.0, "herald": 0.0}
            for r in out:
                if r["team"] == T and m * 60 <= r["t"] < (m + 1) * 60 and j in r["people"]:
                    share[r["type"]] += (r["n"] / 3 if r["type"] == "grubs" else 1) / len(r["people"])
            inv = sum(1 for k in kills if m * 60 <= k["timestamp"] / 1000 < (m + 1) * 60
                      and (k.get("killerId") == j or j in k.get("assistingParticipantIds", []) or k["victimId"] == j))
            died = any(k["victimId"] == j and m * 60 <= k["timestamp"] / 1000 < (m + 1) * 60 for k in kills)
            dmg.append([taken, share["dragon"], share["grubs"], share["herald"], (b["jungleMinionsKilled"] - a["jungleMinionsKilled"]) / 4, inv, float(died), m])
    return rows, dmg


def ols(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ w
    s2 = float(r @ r) / max(1, len(y) - X.shape[1])
    return w, np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * s2))


def summarize(rows, dmg):
    res = {"n_events": len(rows)}
    for ty in ("dragon", "grubs", "herald"):
        R = [r for r in rows if r["type"] == ty]
        d = [r["dur"] for r in R if r["dur"] is not None and r["dur"] >= 0]
        res[ty] = {"n": len(R), "people": collections.Counter(r["people"] for r in R),
                   "people_mean": round(float(np.mean([r["people"] for r in R])), 2) if R else None,
                   "dur_n": len(d), "dur_pct": [round(float(x), 1) for x in np.percentile(d, [25, 50, 75])] if d else None,
                   "delay_median": round(float(np.median([r["delay"] for r in R])), 1) if R else None,
                   "recall90": round(float(np.mean([r["recall90"] for r in R if r["jungler_in"]])), 3) if R else None,
                   "hp_next_median": round(float(np.median([r["hp_next"] for r in R if r["jungler_in"]])), 3) if R else None,
                   "fight_share": round(float(np.mean([(r["deaths_own"] + r["deaths_opp"]) > 0 for r in R])), 3) if R else None,
                   "deaths_own": round(float(np.mean([r["deaths_own"] for r in R])), 2) if R else None,
                   "deaths_opp": round(float(np.mean([r["deaths_opp"] for r in R])), 2) if R else None}
        res[ty]["people"] = {str(k): v for k, v in sorted(res[ty]["people"].items())}
    fd = [r["delay"] for r in rows if r["first_dragon"]]
    res["first_dragon_delay_pct"] = [round(float(x)) for x in np.percentile(fd, [25, 50, 75])] if fd else None
    D = np.array(dmg, float)
    mins = sorted(set(D[:, 7].astype(int)))
    X = np.column_stack([D[:, 1:7]] + [(D[:, 7] == m).astype(float) for m in mins])
    w, se = ols(X, D[:, 0])
    names = ["dragon_share", "grubs_share", "herald_share", "camps", "kill_involvement", "died"]
    res["damage_taken_frac"] = {k: {"coef": round(float(a), 3), "se": round(float(s), 3)} for k, a, s in zip(names, w, se)}
    res["damage_taken_frac"]["n"] = len(D)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "replay_raw"))
    ap.add_argument("--tracks", default=os.path.join(ROOT, "data", "replay_tracks"))
    ap.add_argument("--out", default=os.path.join(ROOT, "sim", "objective_fit.json"))
    a = ap.parse_args()
    rules = Rules()
    rows, dmg, games = [], [], 0
    for p in sorted(glob.glob(os.path.join(a.raw, "*.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if not d.get("timeline"):
            continue
        r, m = game_rows(d, a.tracks if os.path.isdir(a.tracks) else None, os.path.basename(p)[:-5], rules)
        rows += r
        dmg += m
        games += 1
    res = summarize(rows, dmg)
    res["games"] = games
    res["_doc"] = "tools/objective_fit.py: 에픽 몬스터 처치 속도·체력 손해·처치 뒤 귀환·둥지 근처 교전(실제)"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(a.out.replace(".json", "_rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
