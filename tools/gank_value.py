"""라인별 갱(라인 방문) 성공률·성공 시 기대치. sim.lane_visits와 같은 방문 정의.
    cd ops-sim && python3 <이 파일>
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from sim.data import ROOT
from sim.game import Game
from sim.stage6 import zone
from sim.lane_visits import visits, lane_of, AFTER

nodes = Game(seed=0).map.nodes
rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
    d = json.load(open(path, encoding="utf-8"))
    mid = os.path.basename(path)[:-5]
    parts = d["detail"]["info"]["participants"]
    frames = d["timeline"]["info"]["frames"]
    kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
              "assists": e.get("assistingParticipantIds", [])}
             for fr in frames for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
    role = {q["participantId"]: q["teamPosition"] for q in parts}
    def gdiff(mi, team):
        mi = min(mi, len(frames) - 1)
        pf = frames[mi]["participantFrames"]
        return sum(pf[str(q["participantId"])]["totalGold"] * (1 if q["teamId"] == team else -1) for q in parts)
    for p in parts:
        if p.get("teamPosition") != "JUNGLE":
            continue
        tp = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
        if not os.path.exists(tp):
            continue
        zones, xy = {}, {}
        for r in csv.DictReader(open(tp, encoding="utf-8")):
            if r.get("valid", "1") == "1":
                t, x, y = round(float(r["time"])), float(r["x"]), float(r["y"])
                zones[t], xy[t] = zone(x, y, p["teamId"], nodes), (x, y)
        if sum(1 for t in range(180, 1080) if t in zones) < 0.8 * 900:
            continue
        pid, team = p["participantId"], p["teamId"]
        mine = {q["participantId"] for q in parts if q["teamId"] == team}
        for s, e in visits(zones):
            lane = lane_of(*xy[s])
            ks = [k for k in kills if s <= k["t"] <= e + AFTER]
            ek = [k for k in ks if k["victim"] not in mine]
            involved = [k for k in ek if k["killer"] == pid or pid in k["assists"]]
            ours = [k for k in ks if k["victim"] in mine]
            m0, m1 = int(s // 60), int((e + AFTER) // 60) + 2
            rows.append(dict(lane=lane, kill=bool(involved), n_enemy=len(ek), n_ours=len(ours),
                             jdeath=any(k["victim"] == pid for k in ks),
                             nonlaner=sum(1 for k in ek if role.get(k["victim"]) in ("JUNGLE", "MIDDLE") and lane == "bot"
                                          or role.get(k["victim"]) in ("JUNGLE",) and lane == "top"),
                             dg=gdiff(m1, team) - gdiff(m0, team), base_dg=gdiff(m0, team), match=mid))

rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return f"{v.mean():+.0f} ({np.percentile(bs,2.5):+.0f}~{np.percentile(bs,97.5):+.0f})"
n_games = len({(r['match']) for r in rows})
print("visits", len(rows), "games", n_games)
for lane in ("top", "mid", "bot"):
    R = [r for r in rows if r["lane"] == lane]
    S = [r for r in R if r["kill"]]
    F = [r for r in R if not r["kill"]]
    print(f"\n[{lane}] 방문 {len(R)}  성공 {len(S)/len(R):.0%}  정글러 사망 {np.mean([r['jdeath'] for r in R]):.0%}")
    print("  성공 시: 적 처치 평균", round(np.mean([r['n_enemy'] for r in S]), 2), " 2명 이상", f"{np.mean([r['n_enemy']>=2 for r in S]):.0%}",
          " 우리 사망", round(np.mean([r['n_ours'] for r in S]), 2), " 정글·미드 등 추가", f"{np.mean([r['nonlaner']>0 for r in S]):.0%}")
    print("  팀 골드차 변화(방문 전 분→끝+10초 뒤 2분): 성공", ci([r['dg'] for r in S]), " 실패", ci([r['dg'] for r in F]), " 전체", ci([r['dg'] for r in R]))
    print("  성공-실패 차", round(np.mean([r['dg'] for r in S]) - np.mean([r['dg'] for r in F])))
json.dump(rows, open(os.path.join(ROOT, "data", "gank_value_rows.json"), "w"))
