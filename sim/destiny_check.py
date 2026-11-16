"""데스티니 롤재판 원칙 카드 검수: 리플레이 1초 동선(26.21, 70판)으로 카드 D01·D02·D03·D07을 확인한다.

    python -m sim.destiny_check

카드 원문은 kk PC D:\\AI\\lol\\coach_knowledge\\destiny\\cards.json (자막 원문은 저장소에 넣지 않음).
- D01/D02: 첫 정글링에서 우리 캠프 3개를 먹고 곧장 라인(5초 이상)에 간 정글러 → 3번째 캠프가 끝난 시각, 첫 라인 도착 시각·라인.
- D03: 3~8분 적 블루·레드 카정(적 캠프 600 안 3초 이상). 구간 ±10초에 적 라이너(라이너 동선 있는 20판)·적 정글러가
  2,000 안에 있었는지별로 정글러 사망·킬(구간 시작~끝+10초).
- D07: 용 처치 전 '준비' — 처치 팀 정글러가 둥지(1,200 안)에 처음 들어가기 전 60초에 먹은 우리 캠프 수(5초 이상 머문 캠프),
  그 수별로 그 용을 그 팀이 가져갔는지(둥지에 들어간 정글러 기준, 양 팀 모두).
결과는 reports/destiny_check.md.
"""
import collections
import csv
import glob
import json
import math
import os

from .data import ROOT
from .game import Game
from .invade_check import dwell_runs
from .lane_visits import lane_of
from .real_actions import real_tracks
from .stage6 import zone

SIDE = {100: "B", 200: "R"}
NEAR = 2000
AFTER = 10
PREP = 60


def load(mid):
    with open(os.path.join(ROOT, "data", "replay_raw", f"{mid}.json"), encoding="utf-8") as f:
        return json.load(f)


def laner_tracks(mid, parts):
    out = {}
    for p in parts:
        if p.get("teamPosition") == "JUNGLE":
            continue
        f = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
        if not os.path.exists(f):
            return None
        with open(f, encoding="utf-8") as fh:
            out[p["participantId"]] = {round(float(r["time"])): (float(r["x"]), float(r["y"]))
                                       for r in csv.DictReader(fh) if r.get("valid", "1") == "1"}
    return out


def first_lane(pos, team, nodes, t0, t1=300):
    run = 0
    for t in range(t0, t1):
        p = pos.get(t)
        if p and zone(p[0], p[1], team, nodes) == "lane":
            run += 1
            if run >= 5:
                s = t - 4
                return s, lane_of(*pos[s])
        else:
            run = 0
    return None


def main():
    nodes = Game(seed=0).map.nodes
    tracks = real_tracks()
    by = collections.defaultdict(dict)
    for mid, team, pid, pos in tracks:
        by[mid][team] = (pid, pos)

    d1 = []            # (3번째 캠프 끝, 첫 라인 도착, 라인)
    d3 = []            # 카정 행
    d7 = []            # (캠프 수, 가져감)
    for mid, sides in sorted(by.items()):
        d = load(mid)
        parts = d["detail"]["info"]["participants"]
        frames = d["timeline"]["info"]["frames"]
        evs = [e for fr in frames for e in fr["events"]]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds", [])} for e in evs if e["type"] == "CHAMPION_KILL"]
        dragons = [(e["timestamp"] / 1000, e.get("killerTeamId")) for e in evs
                   if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") == "DRAGON"]
        lt = laner_tracks(mid, parts)
        team_of = {p["participantId"]: p["teamId"] for p in parts}
        for team, (pid, pos) in sides.items():
            runs = dwell_runs(pos, team, nodes)
            own = [(c, s, e) for c, s, e in runs if c.startswith(SIDE[team]) and e - s + 1 >= 5]
            # D01/D02: 첫 정글링
            early = [r for r in own if r[1] < 240]
            if len(early) >= 3:
                third_end = early[2][2]
                fl = first_lane(pos, team, nodes, 60)
                fourth = early[3][1] if len(early) >= 4 else 10 ** 9
                if fl and third_end <= fl[0] < fourth:
                    d1.append((third_end, fl[0], fl[1]))
            # D03: 3~8분 적 버프 카정
            foe = sides.get(300 - team)
            for c, s, e in runs:
                if c.startswith(SIDE[team]) or not (180 <= s < 480) or c.split("_", 1)[1] not in ("blue", "red"):
                    continue
                cx, cy = nodes[c]["x"], nodes[c]["y"]
                win = range(s - AFTER, e + AFTER + 1)
                lan = None
                if lt is not None:
                    lan = any(team_of[q] != team and lt[q].get(t) and math.hypot(lt[q][t][0] - cx, lt[q][t][1] - cy) <= NEAR
                              for q in lt for t in win)
                jg = None
                if foe:
                    jg = any(foe[1].get(t) and math.hypot(foe[1][t][0] - cx, foe[1][t][1] - cy) <= NEAR for t in win)
                ks = [k for k in kills if s <= k["t"] <= e + AFTER]
                tp = {q for q, tm in team_of.items() if tm == team}
                d3.append({"laner": lan, "jg": jg, "len": e - s + 1,
                           "death": any(k["victim"] == pid for k in ks),
                           "kill": any(k["victim"] not in tp and (k["killer"] == pid or pid in k["assists"]) for k in ks)})
            # D07: 용 전 캠프
            pit = nodes["dragon_pit"]
            prev = 0
            for t_kill, kt in dragons:
                enter = next((t for t in range(int(prev), int(t_kill) + 1) if pos.get(t)
                              and math.hypot(pos[t][0] - pit["x"], pos[t][1] - pit["y"]) <= 1200
                              and t_kill - t <= 90), None)
                prev = t_kill + 1
                if enter is None:
                    continue
                n = sum(1 for c, s, e in own if enter - PREP <= e < enter)
                d7.append((n, kt == team))

    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    mmss = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
    pct = lambda a, b: f"{a / b:.0%} ({a}/{b})" if b else "-"
    L = ["# 데스티니 롤재판 카드 검수 (리플레이 70판, 26.21)", ""]
    L += ["## D01·D02: 3캠 후 바로 라인", "",
          f"- 첫 정글링에서 우리 캠프 3개 먹고 4번째 캠프 전에 라인(5초 이상)에 간 정글러 {len(d1)}명 / 정글러 {len(tracks)}명.",
          f"- 3번째 캠프 끝 중앙 {mmss(med([a for a, _, _ in d1]))} (25~75%: {mmss(sorted(a for a,_,_ in d1)[len(d1)//4])}~{mmss(sorted(a for a,_,_ in d1)[3*len(d1)//4])}).",
          f"- 첫 라인 도착 중앙 {mmss(med([b for _, b, _ in d1]))}.",
          "- 라인: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(l for _, _, l in d1).most_common()), ""]
    L += ["## D03: 3~8분 적 블루·레드 카정", "", "| 조건 | 구간 | 정글러 사망 | 정글러 킬 관여 |", "|---|---|---|---|"]
    for name, sel in [("전체", lambda r: True),
                      ("적 라이너 2,000 안(라이너 동선 있는 판)", lambda r: r["laner"] is True),
                      ("적 라이너 없음(라이너 동선 있는 판)", lambda r: r["laner"] is False),
                      ("적 정글러 2,000 안", lambda r: r["jg"] is True),
                      ("적 정글러 없음", lambda r: r["jg"] is False),
                      ("머문 시간 15초 이상", lambda r: r["len"] >= 15),
                      ("머문 시간 15초 미만", lambda r: r["len"] < 15)]:
        rs = [r for r in d3 if sel(r)]
        L.append(f"| {name} | {len(rs)} | {pct(sum(r['death'] for r in rs), len(rs))} | {pct(sum(r['kill'] for r in rs), len(rs))} |")
    L += ["", "## D07: 용 둥지 들어가기 전 60초에 먹은 우리 캠프 수", "", "| 캠프 수 | 정글러 | 그 용을 가져감 |", "|---|---|---|"]
    for n in sorted(set(a for a, _ in d7)):
        rs = [w for a, w in d7 if a == n]
        L.append(f"| {n} | {len(rs)} | {pct(sum(rs), len(rs))} |")
    out = os.path.join(ROOT, "reports", "destiny_check.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
