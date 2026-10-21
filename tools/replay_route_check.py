"""리플레이로 기록한 실제 1초 정글 동선과, Riot 1분 기록으로 추정한 동선(sim/replay.py estimate_jungle)·단순 직선 보간을 비교한다.

  python -m tools.replay_route_check

sim/replay.py(PR #5)가 필요하다. 분 프레임 ±3초는 둘 다 정답을 알아서 뺀다. 결과: reports/replay_routes.md
"""
import csv, glob, json, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stage6_compact import compact
from sim.stage6 import RealMatch
from sim.replay import estimate_jungle
from sim.game import Game

nodes = Game(seed=0).map.nodes
bad = {r["match"] for r in csv.DictReader(open("data/replay_tracks/summary.csv")) if int(r["median_dist"]) > 1000}

def interp(pts, t):
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if a[0] <= t <= b[0]:
            f = 0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
    return None

buckets = [(0, 180), (180, 360), (360, 600), (600, 900), (900, 1200)]
res = {m: {b: [] for b in buckets} for m in ("est", "lin")}
per = []
for path in sorted(glob.glob("data/replay_raw/*.json")):
    mid = os.path.basename(path)[:-5]
    if mid in bad:
        continue
    d = json.load(open(path, encoding="utf-8"))
    m = compact(d); m["match"] = mid
    rm = RealMatch(m)
    minutes = min(20, len(rm.frames) - 1)
    est, _ = estimate_jungle(rm, nodes, minutes)
    for tm in (100, 200):
        j = rm.jungler(tm)
        champ = rm.champ_of[j]
        f = f"data/replay_tracks/{mid}_{champ}.csv"
        if not os.path.exists(f):
            continue
        rows = [r for r in csv.DictReader(open(f)) if r.get("valid", "1") == "1"]
        real = {round(float(r["time"])): (float(r["x"]), float(r["y"])) for r in rows}
        lin = [[k * 60, rm.pf(k, j)["x"], rm.pf(k, j)["y"]] for k in range(minutes + 1)]
        e_all, l_all = [], []
        for t in range(62, minutes * 60, 2):
            if t % 60 < 3 or t % 60 > 57:
                continue  # 분 프레임 근처는 둘 다 정답을 알아서 뺀다
            if t not in real:
                continue
            rx, ry = real[t]
            pe, pl = interp(est[str(tm)], t), interp(lin, t)
            if not pe or not pl:
                continue
            de, dl = math.dist((rx, ry), pe), math.dist((rx, ry), pl)
            e_all.append(de); l_all.append(dl)
            for b in buckets:
                if b[0] <= t < b[1]:
                    res["est"][b].append(de); res["lin"][b].append(dl)
        if e_all:
            per.append((mid, champ, round(st.median(e_all)), round(st.median(l_all)), len(e_all)))
print("경기·정글러", len(per))
for p in per: print(p)
print("구간  추정 중앙/평균  직선 중앙/평균  n  추정이 더 가까운 비율")
for b in buckets:
    e, l = res["est"][b], res["lin"][b]
    win = sum(x < y for x, y in zip(e, l)) / len(e)
    print(b, round(st.median(e)), round(st.mean(e)), round(st.median(l)), round(st.mean(l)), len(e), round(win, 2))
E = [x for b in buckets for x in res["est"][b]]; L = [x for b in buckets for x in res["lin"][b]]
print("전체", round(st.median(E)), round(st.mean(E)), round(st.median(L)), round(st.mean(L)), len(E), round(sum(x<y for x,y in zip(E,L))/len(E),2))
for th in (500, 1000, 2000):
    print(th, "이내 비율 추정", round(sum(x<=th for x in E)/len(E),2), "직선", round(sum(x<=th for x in L)/len(L),2))
