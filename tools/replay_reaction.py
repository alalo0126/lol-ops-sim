"""리플레이 1초 동선으로 '적 정글러가 시야에 들어온 뒤 우리 정글러가 반응하기까지' 시간을 본다(공정 비교 설계 3장).

    python tools/replay_reaction.py D:\\AI\\lol\\ops-sim\\data\\replay_tracks

사건: 적 정글러가 우리 정글러 1,350 안에 들어옴(20초 만에 처음). 반응: 이동 중이었으면 방향 90도 넘게 꺾음, 서 있었으면 움직이기 시작.
사건 뒤 L초에 반응한 비율 − 아무 때나 L초 뒤 반응한 비율(기준) = 초과 반응. 우리 정글러 상태(이동·캠프·서 있음·교전)별로 나눈다.
한계: 1초 간격이라 1초보다 짧은 지연은 못 잰다. 수풀·와드는 모름(1,350 안 = 보임으로 침).
"""
import collections
import csv
import glob
import math
import os
import sys

CAMPS = {"B_blue": [3821, 8101], "R_blue": [10999, 6799], "B_gromp": [2288, 8448], "R_gromp": [12532, 6452],
         "B_wolves": [3780, 6443], "R_wolves": [11040, 8457], "B_raptors": [6943, 5422], "R_raptors": [7877, 9478],
         "B_red": [7765, 4020], "R_red": [7055, 10880], "B_krugs": [8394, 2641], "R_krugs": [6426, 12259],
         "river_top_scuttle": [4400, 9640], "river_bot_scuttle": [10420, 5260], "dragon_pit": [9861, 4447], "baron_pit": [5007, 10471]}
LAG = list(range(0, 11))


def heading(p, a, b):
    if a in p and b in p:
        dx, dy = p[b][0] - p[a][0], p[b][1] - p[a][1]
        if dx * dx + dy * dy >= (100 * (b - a)) ** 2:
            return math.atan2(dy, dx)
    return None


def reacted(p, t, L):
    h0 = heading(p, t - 3, t)
    if h0 is None:   # 서 있던 경우: 움직이기 시작하면 반응
        if t not in p:
            return None
        return t + L + 2 in p and heading(p, t + L, t + L + 2) is not None and all(heading(p, t + k, t + k + 2) is None for k in range(L))
    h1 = heading(p, t + L, t + L + 2)
    if h1 is None:
        return None
    return abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi) > math.radians(90)


def activity(p, t, enemies):
    if t not in p:
        return None
    if any(t in e and math.dist(p[t], e[t]) <= 1200 for e in enemies):
        return "fight"
    if heading(p, t - 3, t) is not None:
        return "move"
    if any(math.dist(p[t], c) <= 700 for c in CAMPS.values()):
        return "camp"
    return "still"


def main(root):
    games = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, "*.csv")):
        b = os.path.basename(f)
        if not b.endswith("_players.csv"):
            games["_".join(b.split("_")[:2])].append(f)
    ev, evn = collections.defaultdict(lambda: [0] * len(LAG)), collections.Counter()
    base, basen = collections.defaultdict(lambda: [0] * len(LAG)), collections.Counter()
    n = 0
    for gid, fs in games.items():
        pf = os.path.join(root, gid + "_players.csv")
        if not os.path.exists(pf):
            continue
        pl = list(csv.DictReader(open(pf, encoding="utf-8")))
        tm = {r["champion"]: r["team"] for r in pl}
        role = {r["champion"]: r["position"] for r in pl}
        tr = {}
        for f in fs:
            ch = os.path.basename(f)[len(gid) + 1:-4]
            if ch in tm:
                tr[ch] = {int(float(r["time"])): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(f)) if r["valid"] == "1"}
        n += 1
        for T in ("ORDER", "CHAOS"):
            me = [c for c in tr if tm[c] == T and role[c] == "JUNGLE"]
            ej = [c for c in tr if tm[c] != T and role[c] == "JUNGLE"]
            if not me or not ej:
                continue
            me, ej = tr[me[0]], tr[ej[0]]
            enemies = [tr[c] for c in tr if tm[c] != T and tr[c] is not ej]
            prev, last_seen = None, -99
            for t in range(150, 1200):
                if t not in ej or t not in me:
                    prev = None
                    continue
                s = math.dist(me[t], ej[t]) <= 1350
                act = activity(me, t, enemies)
                if s and prev is False and t - last_seen >= 20 and act:
                    evn[act] += 1
                    for i, L in enumerate(LAG):
                        ev[act][i] += bool(reacted(me, t, L))
                if s:
                    last_seen = t
                prev = s
                if act and t % 7 == 0:
                    basen[act] += 1
                    for i, L in enumerate(LAG):
                        base[act][i] += bool(reacted(me, t, L))
    print("경기", n)
    for act in ("move", "camp", "still", "fight"):
        if evn[act]:
            ex = [round(100 * (ev[act][i] / evn[act] - base[act][i] / max(1, basen[act])), 1) for i in range(len(LAG))]
            print(act, "사건", evn[act], "기준 표본", basen[act], "초과 반응(%p) L=0..10:", ex)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "replay_tracks"))
