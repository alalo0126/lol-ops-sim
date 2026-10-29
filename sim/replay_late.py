"""라인전 이후(14분~경기 끝) 정글·탑·미드 1초 동선: kk 롤 이론 중 20분 이후 주장 확인.

    python -m sim.replay_late [--root data/full]     :: <root>/replay_raw/*.json + <root>/replay_tracks/*.csv → reports/replay_late.md

리플레이 카메라 1초 위치(tools/replay_batch --until 0)로 1,000판 분 프레임 결과(reports/macro_roles.md)를 다시 본다.
- 구역: sim.macro_roles.place (사이드 라인 / 미드 라인 / 우리 정글 / 적 정글 / 강가 / 기지).
- 합류(탑): 같은 초에 우리 정글러나 미드가 2,500 안.
- 라인전 결과: 14분 같은 포지션 상대와 총 골드 차 ±1,000.
- 다음 오브젝트: 드래곤(첫 5:00, 처치 후 5분)·바론(20:00, 처치 후 6분) 생성까지 남은 초. 살아 있으면 0.
같은 경기들의 분 프레임(타임라인 위치)으로 같은 표를 만들어 1분 간격 표본이 치우치는지도 본다.
45판 정도라 행동의 승률 효과는 재지 않는다(구간이 너무 넓음). 비율과 거리만.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from .data import ROOT
from .game import Game
from .macro_roles import place

M0 = 14 * 60
LEAD = 1000
JOIN = 2500
ROLES = ("TOP", "JUNGLE", "MIDDLE")
PLACES = ["side", "mid", "own_jungle", "enemy_jungle", "river", "base"]
PLACE_KO = {"side": "사이드 라인", "mid": "미드 라인", "own_jungle": "우리 정글", "enemy_jungle": "적 정글", "river": "강가·둥지", "base": "기지"}
STATE_KO = {"won": "라인전 이김", "even": "비김", "lost": "라인전 짐"}
DRAKE_FIRST, DRAKE_RE, BARON_FIRST, BARON_RE = 300, 300, 1200, 360


def load_track(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("valid", "1") == "1":
                out[int(float(r["time"]))] = (float(r["x"]), float(r["y"]))
    return out


def spawns(events, end):
    """{'dragon': [(생성초, 처치초)], 'baron': [...]} — 처치 기록으로 다음 생성 시각을 이어 붙인다."""
    kills = defaultdict(list)
    for e in events:
        if e["type"] == "ELITE_MONSTER_KILL" and e["monsterType"] in ("DRAGON", "BARON_NASHOR"):
            kills["dragon" if e["monsterType"] == "DRAGON" else "baron"].append(e["timestamp"] / 1000)
    out = {}
    for name, first, re_ in (("dragon", DRAKE_FIRST, DRAKE_RE), ("baron", BARON_FIRST, BARON_RE)):
        t, lst = first, []
        for k in sorted(kills[name]):
            lst.append((t, k))
            t = k + re_
        lst.append((t, end + 1))
        out[name] = lst
    return out


def until_next(sp, t):
    """다음 드래곤·바론 중 가장 이른 것까지 남은 초(살아 있으면 0)와 그 둥지."""
    best = (1e9, None)
    for name, lst in sp.items():
        for s, k in lst:
            if s <= t < k:
                best = min(best, (0, name))
            elif t < s:
                best = min(best, (s - t, name))
                break
    return best


def game_rows(raw_path, tracks_dir, nodes):
    d = json.load(open(raw_path, encoding="utf-8"))
    mid = os.path.basename(raw_path)[:-5]
    info, frames = d["detail"]["info"], d["timeline"]["info"]["frames"]
    end = info["gameDuration"]
    parts = info["participants"]
    pid = {(p["teamId"], p["teamPosition"]): p for p in parts}
    events = [e for fr in frames for e in fr["events"]]
    sp = spawns(events, end)
    tr = {}
    for (team, role), p in pid.items():
        path = os.path.join(tracks_dir, f"{mid}_{p['championName']}.csv")
        if role in ROLES and os.path.exists(path):
            tr[(team, role)] = load_track(path)
    if len(tr) < 6 or len(frames) <= 14:
        return [], []
    f14 = frames[14]["participantFrames"]
    rows, mrows = [], []
    for (team, role), track in tr.items():
        me, foe = pid[(team, role)], pid[(300 - team, role)]
        diff = f14[str(me["participantId"])]["totalGold"] - f14[str(foe["participantId"])]["totalGold"]
        state = "won" if diff >= LEAD else "lost" if diff <= -LEAD else "even"
        allies = [tr.get((team, r)) for r in ROLES if r != role]
        for t in range(M0, end):
            if t not in track:
                continue
            x, y = track[t]
            near = [a[t] for a in allies if a and t in a]
            nxt, pit = until_next(sp, t)
            rows.append({"match": mid, "team": team, "role": role, "state": state, "t": t, "place": place(x, y, team, nodes),
                         "joined": any(math.dist((x, y), q) < JOIN for q in near) if near else None,
                         "next": nxt, "pit": pit,
                         "pit_dist": math.dist((x, y), (nodes[f"{pit}_pit"]["x"], nodes[f"{pit}_pit"]["y"])) if pit else None,
                         "mid_dist": abs(x - y) / math.sqrt(2)})
        for m in range(14, len(frames)):   # 같은 경기 분 프레임(타임라인 위치)
            pos = frames[m]["participantFrames"][str(me["participantId"])].get("position")
            if pos:
                mrows.append({"role": role, "state": state, "place": place(pos["x"], pos["y"], team, nodes)})
    return rows, mrows


def share(rows, key="place", cats=PLACES):
    c = Counter(r[key] for r in rows)
    n = sum(c.values()) or 1
    return [c[k] / n for k in cats]


def pct(v):
    return f"{v * 100:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "full"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "replay_late.md"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    rows, mrows, games = [], [], 0
    for p in sorted(glob.glob(os.path.join(args.root, "replay_raw", "*.json"))):
        r, m = game_rows(p, os.path.join(args.root, "replay_tracks"), nodes)
        games += bool(r)
        rows += r
        mrows += m
    L = [f"# 라인전 이후 1초 동선 (26.21 천상계 리플레이 {games}판, 14분~경기 끝, 정글·탑·미드)", "",
         "- 1초 위치는 리플레이 카메라 기록(유효 행만). 같은 경기의 분 프레임(타임라인 위치) 결과를 옆에 둔다.",
         "- 라인전 결과 = 14분 같은 포지션 상대와 총 골드 차 ±1,000. 판 수가 적어 승률 효과는 재지 않았다.", ""]
    for role, name in (("TOP", "탑"), ("MIDDLE", "미드"), ("JUNGLE", "정글")):
        L += [f"## {name}", "", "| 라인전 | 초 | " + " | ".join(PLACE_KO[k] for k in PLACES) + " | 분 프레임 사이드 |",
              "|---|---|" + "---|" * (len(PLACES) + 1)]
        for st in ("won", "even", "lost"):
            R = [r for r in rows if r["role"] == role and r["state"] == st]
            M = [r for r in mrows if r["role"] == role and r["state"] == st]
            if not R:
                continue
            L.append(f"| {STATE_KO[st]} | {len(R)} | " + " | ".join(pct(v) for v in share(R)) + f" | {pct(share(M)[0])} ({len(M)}분) |")
        L.append("")
        R = [r for r in rows if r["role"] == role]
        L += ["다음 드래곤·바론까지 남은 시간별:", "", "| 다음 오브젝트 | 초 | 사이드 | 미드 | 우리 정글 | 강가·둥지 | 둥지까지 거리 중앙 | 정글러나 미드(탑)·아군 2,500 안 |", "|---|---|---|---|---|---|---|---|"]
        for lab, lo, hi in (("살아 있음", 0, 0), ("1분 안", 1, 60), ("1~2분", 61, 120), ("2분 넘게", 121, 1e9)):
            S = [r for r in R if lo <= r["next"] <= hi]
            if not S:
                continue
            sh = dict(zip(PLACES, share(S)))
            j = [r["joined"] for r in S if r["joined"] is not None]
            L.append(f"| {lab} | {len(S)} | {pct(sh['side'])} | {pct(sh['mid'])} | {pct(sh['own_jungle'])} | {pct(sh['river'])} | "
                     f"{np.median([r['pit_dist'] for r in S]):.0f} | {pct(np.mean(j)) if j else '-'} |")
        L.append("")
        if role == "JUNGLE":
            S = [r for r in R if r["next"] > 120]
            L += [f"오브젝트가 2분 넘게 없을 때 정글러와 미드 라인(대각선) 거리 중앙 {np.median([r['mid_dist'] for r in S]):.0f}, "
                  f"1,500 안 {pct(np.mean([r['mid_dist'] < 1500 for r in S]))} (오브젝트 1분 안: "
                  f"{np.median([r['mid_dist'] for r in R if 0 < r['next'] <= 60]):.0f}).", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
