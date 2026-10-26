"""커뮤니티·가이드에서 말하는 정글 판단 기준을 실제 천상계 데이터로 검수한다.

    python -m sim.guide_check            :: → reports/guide_check.md

A. 갱(라인 방문, sim/lane_visits.py 정의) 성공 조건 — 리플레이 70판(양 팀 정글러 1초 동선), 20판은 라이너 1초 동선까지.
   방문 시작 시각의 특징 → 방문 중 킬 관여 / 정글러 사망(로지스틱, 경기 단위 부트스트랩):
   - 적 정글러가 가까이 있음(1초 동선 거리 < 4,000): "적 정글 위치 모르면/가까우면 갱 가지 마라"
   - 적 라이너가 우리 쪽으로 밀고 와 있음(라이너 1초 동선, 20판): "적이 강 너머로 과하게 나와 있으면 갱"
   - 라인 아군 라이너 체력 비율, 적 라이너 체력 비율(분 프레임)
   - 라이너끼리 앞선 정도(레벨 차 + 1.62 × 천 골드 아이템 차, 정글러 제외): "이기는 라인을 갱하라"
   - 정글러 마지막 기지 방문 뒤 경과(1초 동선): "귀환해서 아이템 산 직후 갱"
   - 3~4레벨 창(첫 풀캠 직후): 방문 시작 분
B. 귀환 — 1단계 1,000판(분 프레임 + 구매 기록):
   - "오브젝트 생성 전에 귀환해서 준비": 용 생성 90~20초 전에 정글러가 구매했나 → 그 용을 우리 팀이 잡았나(골드 차 통제)
   - "부품 살 골드가 됐을 때 귀환": 정글러 첫 구매 시 쓸 골드(구매 직전 분 프레임) 분포
"""
import collections
import csv
import glob
import json
import math
import os

import numpy as np

from .data import ROOT, Rules
from .game import Game
from .lane_visits import ITEM_PER_LEVEL, ROLE_LANE, T0, T1, MIN_COVER, lane_of, outcome, visits
from .stage6 import ELITE_KEY, RealMatch, load_compact, zone

NEAR = 4000


def read_track(path):
    xy = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("valid", "1") == "1":
                xy[round(float(r["time"]))] = (float(r["x"]), float(r["y"]))
    return xy


def near_xy(xy, t, tol=3):
    for d in range(tol + 1):
        for s in (t - d, t + d):
            if s in xy:
                return xy[s]
    return None


# ---------------------------------------------------------------------- A. 갱
def gank_rows(nodes):
    fount = {100: (nodes["B_fountain"]["x"], nodes["B_fountain"]["y"]), 200: (nodes["R_fountain"]["x"], nodes["R_fountain"]["y"])}
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        mid = os.path.basename(path)[:-5]
        parts = d["detail"]["info"]["participants"]
        frames = d["timeline"]["info"]["frames"]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds", [])}
                 for fr in frames for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
        tracks = {}
        for q in parts:
            f = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{q['championName']}.csv")
            if os.path.exists(f):
                tracks[q["participantId"]] = read_track(f)
        for p in parts:
            if p.get("teamPosition") != "JUNGLE" or p["participantId"] not in tracks:
                continue
            pid, team = p["participantId"], p["teamId"]
            xy = tracks[pid]
            zones = {t: zone(x, y, team, nodes) for t, (x, y) in xy.items()}
            if sum(1 for t in range(T0, T1) if t in zones) < MIN_COVER * (T1 - T0):
                continue
            team_pids = {q["participantId"] for q in parts if q["teamId"] == team}
            foe_j = next((q["participantId"] for q in parts if q["teamId"] != team and q.get("teamPosition") == "JUNGLE"), None)
            base_ts = sorted(t for t, z in zones.items() if z == "base")
            for span in visits(zones):
                s = span[0]
                lane = lane_of(*xy[s])
                fr = frames[min(int(s // 60), len(frames) - 1)]["participantFrames"]

                def st(q):
                    pf = fr[str(q["participantId"])]
                    cs = pf["championStats"]
                    return pf["level"], (pf["totalGold"] - pf["currentGold"]) / 1000, cs["health"] / max(1, cs["healthMax"])
                ally = [q for q in parts if q["teamId"] == team and ROLE_LANE.get(q["teamPosition"]) == lane]
                foe = [q for q in parts if q["teamId"] != team and ROLE_LANE.get(q["teamPosition"]) == lane]
                if not ally or not foe:
                    continue
                a, b = [st(q) for q in ally], [st(q) for q in foe]
                row = outcome(span, kills, pid, team_pids) | {
                    "match": mid, "team": team, "lane": lane, "minute": s / 60,
                    "laner_lead": float(np.mean([x[0] for x in a]) - np.mean([x[0] for x in b])
                                        + ITEM_PER_LEVEL * (np.mean([x[1] for x in a]) - np.mean([x[1] for x in b]))),
                    "ally_hp": float(np.mean([x[2] for x in a])), "foe_hp": float(np.mean([x[2] for x in b])),
                    "since_base": float(s - max([t for t in base_ts if t < s], default=0)),
                }
                ej = near_xy(tracks.get(foe_j, {}), s)
                row["enemy_jg_dist"] = math.dist(xy[s], ej) if ej else None
                # 적 라이너가 우리 쪽으로 얼마나 나와 있나: (그 라이너 본진까지 − 우리 본진까지) 거리 / 1,000. 양수 = 우리 쪽 절반
                ov = []
                for q in foe:
                    e = near_xy(tracks.get(q["participantId"], {}), s)
                    if e:
                        ov.append((math.dist(e, fount[300 - team]) - math.dist(e, fount[team])) / 1000)
                row["foe_over"] = float(np.mean(ov)) if ov else None
                rows.append(row)
    return rows


def logit(X, y, iters=60, ridge=1e-4):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w += np.linalg.solve(H, X.T @ (y - p))
    return w


def fit(rows, ys, xs, boot=1000, seed=0):
    """y ~ 로지스틱(1 + xs). xs는 표준화. 계수(표준편차 1당 로그오즈)와 경기 단위 부트스트랩 95% 구간."""
    rows = [r for r in rows if all(r[x] is not None for x in xs)]
    mu = {x: np.mean([r[x] for r in rows]) for x in xs}
    sd = {x: np.std([r[x] for r in rows]) or 1.0 for x in xs}

    def mats(rs):
        X = np.array([[1.0] + [(r[x] - mu[x]) / sd[x] for x in xs] for r in rs])
        return X, np.array([float(r[ys]) for r in rs])
    w = logit(*mats(rows))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["match"]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        rs = [r for i in rng.integers(0, len(keys), len(keys)) for r in by[keys[i]]]
        bs.append(logit(*mats(rs)))
    bs = np.array(bs)
    return len(rows), {x: (float(w[i + 1]), float(np.percentile(bs[:, i + 1], 2.5)), float(np.percentile(bs[:, i + 1], 97.5)), float(sd[x]))
                       for i, x in enumerate(xs)}


def rate_by(rows, key, edges, y):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rs = [r for r in rows if r[key] is not None and lo <= r[key] < hi]
        if rs:
            out.append((lo, hi, len(rs), float(np.mean([r[y] for r in rs]))))
    return out


# ---------------------------------------------------------------------- B. 귀환
def recall_rows():
    dr = Rules().objectives["drake"]
    drag_first, drag_resp = dr["first_spawn_s"], dr["respawn_s"]
    M = load_compact()
    drows, first_buy = [], []
    for m in M:
        rm = RealMatch(m)
        if not rm.valid:
            continue
        kills = sorted(e for e in rm.elite if ELITE_KEY.get(e[1]) == "dragon")
        spawn = drag_first
        for e in kills:
            t_kill = e[0] / 1000
            if spawn > 19 * 60:
                break
            minute = int(spawn // 60)
            if minute >= len(rm.frames):
                break
            d = rm.diffs(minute)
            for team in (100, 200):
                j = rm.jungler(team)
                buys = [b[0] / 1000 for b in rm.buys if b[1] == j]
                drows.append({"match": m["id"], "team": team, "spawn": spawn,
                              "prep": float(any(spawn - 90 <= b <= spawn - 20 for b in buys)),
                              "gold_diff": d["gold"] / 1000 * (1 if team == 100 else -1),
                              "took": float(e[3] == team), "wait": t_kill - spawn})
            spawn = t_kill + drag_resp
        for team in (100, 200):
            j = rm.jungler(team)
            bt = sorted(b[0] / 1000 for b in rm.buys if b[1] == j and b[0] > 90000)
            if bt and int(bt[0] // 60) < len(rm.frames):
                first_buy.append({"t": bt[0], "gold": rm.pf(int(bt[0] // 60), j)["current_gold"]})
    return drows, first_buy


def main():
    nodes = Game(seed=0).map.nodes
    G = gank_rows(nodes)
    L = ["# 커뮤니티 정글 판단 기준 검수", "",
         "출처는 가이드·커뮤니티 글(mobatrainer 갱 가이드, dodge.gg 2026 정글 가이드, metabot 귀환 가이드, r/JarvanIVmains 'powerfarm or spamgank' 스레드).",
         "계수는 표준편차 1당 로그오즈(양수 = 그 특징이 클수록 확률이 높음). 괄호는 경기 단위 부트스트랩 95% 구간.", "",
         f"## A. 갱(라인 방문) — 리플레이 {len({r['match'] for r in G})}판, 방문 {len(G)}번", "",
         f"- 평균: 킬 관여 {np.mean([r['kill'] for r in G]):.0%}, 정글러 사망 {np.mean([r['death'] for r in G]):.0%}.", ""]
    base = ["laner_lead", "ally_hp", "foe_hp", "since_base", "minute"]
    for xs, note in ((base, "전체 방문, 다른 특징 함께 통제"),
                     (base + ["enemy_jg_dist"], "적 정글러 1초 동선이 있는 방문"),
                     (base + ["enemy_jg_dist", "foe_over"], "라이너 1초 동선까지 있는 20판")):
        L += [f"### {note}", "", "| 특징 | 1 표준편차 | 킬 관여 | 정글러 사망 |", "|---|---|---|---|"]
        n, k = fit(G, "kill", xs)
        _, dd = fit(G, "death", xs)
        for x in xs:
            L.append(f"| {x} | {k[x][3]:.2f} | {k[x][0]:+.2f} ({k[x][1]:+.2f}~{k[x][2]:+.2f}) | {dd[x][0]:+.2f} ({dd[x][1]:+.2f}~{dd[x][2]:+.2f}) |")
        L += ["", f"표본 {n}번.", ""]
    L += ["### 구간별 비율", "", "| 특징 | 구간 | 방문 | 킬 관여 | 정글러 사망 |", "|---|---|---|---|---|"]
    for key, edges in (("enemy_jg_dist", [0, 2000, 4000, 7000, 30000]), ("foe_over", [-20, -2, 0, 2, 20]),
                       ("ally_hp", [0, 0.4, 0.7, 1.01]), ("foe_hp", [0, 0.4, 0.7, 1.01]),
                       ("laner_lead", [-20, -1.5, -0.5, 0.5, 1.5, 20]), ("since_base", [0, 60, 120, 240, 2000])):
        for (lo, hi, n, kr), (_, _, _, dr) in zip(rate_by(G, key, edges, "kill"), rate_by(G, key, edges, "death")):
            L.append(f"| {key} | {lo:g}~{hi:g} | {n} | {kr:.0%} | {dr:.0%} |")
    D, FB = recall_rows()
    n_drag = len(D) // 2
    L += ["", f"## B. 귀환 — 1,000판, 용 {n_drag}마리(생성 시각은 규칙 기준, 첫 용 이후는 처치 + 리젠)", ""]
    for prep in (1.0, 0.0):
        rs = [r for r in D if r["prep"] == prep]
        L.append(f"- 생성 90~20초 전 정글러 구매 {'함' if prep else '안 함'}: {len(rs)}건, 그 용 우리 팀 처치 {np.mean([r['took'] for r in rs]):.0%}")
    n, k = fit(D, "took", ["prep", "gold_diff"])
    L += [f"- 로지스틱(골드 차 통제, {n}건): 준비 귀환 {k['prep'][0] / k['prep'][3]:+.2f} 로그오즈({k['prep'][1] / k['prep'][3]:+.2f}~{k['prep'][2] / k['prep'][3]:+.2f}), "
          f"천 골드 앞섬당 {k['gold_diff'][0] / k['gold_diff'][3]:+.2f}.", ""]
    g = np.array([r["gold"] for r in FB])
    t = np.array([r["t"] for r in FB])
    L += [f"- 정글러 첫 귀환(1:30 이후 첫 구매) {len(FB)}번: 시각 중앙 {np.median(t) / 60:.1f}분, 구매 직전 분 프레임 쓸 골드 중앙 {np.median(g):.0f}"
          f"(사분위 {np.percentile(g, 25):.0f}~{np.percentile(g, 75):.0f}). 분 프레임이라 구매 순간 골드는 아님."]
    dst = os.path.join(ROOT, "reports", "guide_check.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
