"""초반 동선 유형 → 주도권: 실제(리플레이 1초 동선) vs 시뮬레이터(규칙 봇·학습 정글러).

    python -m sim.route_types [--games 400] [--policy models/stage12_a/policy_latest.npz]

정글러마다 0~7분 1초 위치를 sim.real_actions.segments로 행동(캠프·바위게·라인·귀환·카정)으로 바꾼 뒤
- 시작 쪽: 첫 우리 캠프가 맵 대각선 위(탑 쪽)인가 아래(바텀 쪽)인가
- 첫 행동 전환: 첫 캠프 뒤 처음으로 캠프가 아닌 행동(갱 / 바위게 / 귀환 / 카정)과 그때까지 돈 우리 캠프 수
- 유형: 풀캠(6개 이상) / 3~5캠프 / 2개 이하, 그리고 전환 행동
- 첫 바위게(4:30 전 체류 6초 이상): 쪽·시각
- 첫 갱(7분 전 라인 구역 8초 이상, 미드 가로지르기 제외): 라인·시각·성공(방문~끝+10초 안 정글러 관여 킬)·정글러 사망
경기 결과: 첫 유충·첫 용(15분 전)·첫 킬을 가져간 팀, 7분 전 정글러 사망.
같은 함수를 시뮬 위치에도 써서 비교한다. 결과는 reports/route_types.md.
"""
import argparse
import collections
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .real_actions import real_tracks, segments

T_END = 420            # 동선은 0~7분
SCUTTLE_BY = 270       # 첫 바위게는 4:30까지
OUT_END = 900          # 첫 용·유충은 15분까지
AFTER = 10
SCUTTLE_MIN = 6        # 바위게 900 안 3~5초는 강가를 지나가는 길(시뮬은 바위게 노드를 지나 이동) → 6초 이상만
GANK_MIN = 8           # 라인 5~7초는 대부분 미드 라인을 가로지르는 길(실제 동선 확인) → 8초 이상만 갱으로 본다


def side_of(x, y):
    return "top" if y - x > 0 else "bot"


def visits(seg, prefix):
    """seg: {초: 행동} → [(시작, 끝, 행동)] (prefix로 시작하는 행동의 연속 구간)."""
    out, cur = [], None
    for t in sorted(seg):
        a = seg[t]
        if a.startswith(prefix):
            if cur and cur[2] == a and t - cur[1] <= 3:
                cur[1] = t
            else:
                if cur:
                    out.append(tuple(cur))
                cur = [t, t, a]
        elif cur and t - cur[1] > 3:
            out.append(tuple(cur))
            cur = None
    if cur:
        out.append(tuple(cur))
    return out


def describe(seg, team, nodes, kills, pid, pos=None):
    """한 정글러의 초반 동선 요약. kills: [(초, 킬 팀, 관여 pid 집합, 피해자 pid)]."""
    short = {t for s, e, a in visits(seg, "gank_") if e - s + 1 < GANK_MIN for t in range(s, e + 1)}
    short |= {t for s, e, a in visits(seg, "scuttle_") if e - s + 1 < SCUTTLE_MIN for t in range(s, e + 1)}
    seg = {t: ("move" if t in short else a) for t, a in seg.items()}
    ts = sorted(t for t in seg if t < T_END)
    acts = [seg[t] for t in ts]
    first_camp = next((a for a in acts if a.startswith("camp_")), None)
    if first_camp is None:
        return None
    cn = nodes[("B_" if team == 100 else "R_") + first_camp[5:]]
    start = side_of(cn["x"], cn["y"])
    i0 = acts.index(first_camp)
    camps, switch, switch_t = [], "none", None
    for t, a in zip(ts[i0:], acts[i0:]):
        if a.startswith("camp_"):
            if a not in camps:
                camps.append(a)
            continue
        if a in ("move", "dead"):
            continue
        switch, switch_t = a.split("_")[0], t
        break
    n = len(camps)
    kind = "풀캠" if n >= 6 else "3~5캠프" if n >= 3 else "2캠프 이하"
    t0 = ts[i0]   # 첫 캠프로 출발한 초. 그 전(우물에서 나와 라인을 지나는 길)은 빼고 본다
    sc = next(((s, a) for s, e, a in visits(seg, "scuttle_") if t0 <= s < SCUTTLE_BY), None)
    gk = next(((s, e, a) for s, e, a in visits(seg, "gank_") if s >= t0 and e - s + 1 >= GANK_MIN), None)
    g_ok = g_die = None
    if gk:
        g_ok = any(gk[0] <= t <= gk[1] + AFTER and pid in inv and kt == team for t, kt, inv, v in kills)
        g_die = any(gk[0] <= t <= gk[1] + AFTER and v == pid for t, kt, inv, v in kills)
    p3 = next((pos[t] for t in range(180, 190) if pos and pos.get(t)), None)
    return {"start": start, "side3": side_of(*p3) if p3 else "none", "switch_t": switch_t,
            "first_camp": first_camp[5:], "t_first_camp": t0, "camps": n, "kind": kind, "switch": switch,
            "scuttle": sc[1][8:] if sc else "none", "scuttle_t": sc[0] if sc else None,
            "gank_lane": gk[2][5:] if gk else "none", "gank_t": gk[0] if gk else None, "gank_kill": g_ok, "gank_death": g_die,
            "death7": any(v == pid and t < T_END for t, kt, inv, v in kills)}


def outcomes(row, team, firsts):
    row.update({k: (firsts.get(k) == team) if firsts.get(k) else None for k in ("grubs", "dragon", "kill")})
    return row


# ---------------------------------------------------------------------- 실제
NAME = {"DRAGON": "dragon", "HORDE": "grubs"}


def real_rows(nodes):
    rows = []
    for mid, team, pid, pos in real_tracks():
        if sum(1 for t in range(15, T_END) if t in pos) < 0.8 * (T_END - 15):
            continue
        with open(os.path.join(ROOT, "data", "replay_raw", f"{mid}.json"), encoding="utf-8") as f:
            d = json.load(f)
        tm = {p["participantId"]: p["teamId"] for p in d["detail"]["info"]["participants"]}
        ev = [e for fr in d["timeline"]["info"]["frames"] for e in fr["events"]]
        kills = [(e["timestamp"] / 1000, tm.get(e.get("killerId"), 300 - tm[e["victimId"]]),
                  {e.get("killerId")} | set(e.get("assistingParticipantIds", [])), e["victimId"])
                 for e in ev if e["type"] == "CHAMPION_KILL"]
        firsts = {}
        for e in ev:
            if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in NAME and e["timestamp"] < OUT_END * 1000:
                firsts.setdefault(NAME[e["monsterType"]], e.get("killerTeamId"))
        if kills:
            firsts["kill"] = kills[0][1]
        seg = segments(pos, team, nodes, 900, t0=0, t1=T_END)
        r = describe(seg, team, nodes, kills, pid, pos)
        if r:
            r["match"] = mid
            fr = d["timeline"]["info"]["frames"]
            r["cs3"] = fr[3]["participantFrames"][str(pid)]["jungleMinionsKilled"] if len(fr) > 3 else None
            rows.append(outcomes(r, team, firsts))
    return rows


# ---------------------------------------------------------------------- 시뮬
_POLICY = None


def _init(path):
    global _POLICY
    _POLICY = path


def _sim_one(seed):
    from .stage6_track import _xy
    over, ai_team = {}, None
    if _POLICY:
        from .agent import AgentJungler
        from .ppo import Policy
        from .winprob import WinProb
        ai_team = 100 if seed % 2 == 0 else 200
        pol = Policy.from_file(_POLICY)
        pol.rng = np.random.default_rng(seed)
        over = {ai_team: lambda g, c: AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())}
    g = Game(seed=seed, jungler_override=over) if over else Game(seed=seed)
    jgs = [c for c in g.champs if c.role == "JUNGLE"]
    pos = {c.pid: {} for c in jgs}
    while g.t < OUT_END:
        g.step()
        if g.t < T_END:
            for c in jgs:
                xy = _xy(g, c)
                pos[c.pid][g.t] = None if xy[0] is None else xy
    team_of = {c.pid: c.team for c in g.champs}
    kills = [(e["t"], team_of[e["killer"]], {e["killer"], *e["assists"]}, e["victim"]) for e in g.events if e["type"] == "kill"]
    firsts = {}
    for e in g.events:
        if e["type"] == "objective" and e["name"] in ("grubs", "dragon"):
            firsts.setdefault(e["name"], e["team"])
    if kills:
        firsts["kill"] = kills[0][1]
    out = []
    for c in jgs:
        seg = segments(pos[c.pid], c.team, g.map.nodes, 900, t0=0, t1=T_END)
        r = describe(seg, c.team, g.map.nodes, kills, c.pid, pos[c.pid])
        if r:
            r["who"] = "ai" if c.team == ai_team else "rule"
            # 실제 프레임과 같은 단위: 3:00까지 처치한 캠프 × 4(캠프당 정글 CS 4, 4단계 확인)
            r["cs3"] = 4 * sum(1 for e in g.events if e["type"] == "camp" and e["pid"] == c.pid and e["t"] <= 180
                               and "scuttle" not in e["camp"])
            out.append(outcomes(r, c.team, firsts))
    return out


def simulate(games, policy, workers):
    with Pool(workers, initializer=_init, initargs=(policy,)) as pool:
        return [r for rs in pool.map(_sim_one, range(games)) for r in rs]


# ---------------------------------------------------------------------- 표
def rate(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    if not v:
        return "–"
    p = float(np.mean(v))
    return f"{p:.0%} ±{1.96 * math.sqrt(p * (1 - p) / len(v)):.0%}"


def mean_t(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    return f"{np.median(v) / 60:.1f}분" if v else "–"


def dist(rs, k, order=None):
    c = collections.Counter(r[k] for r in rs)
    n = len(rs) or 1
    keys = order or [a for a, _ in c.most_common()]
    return " · ".join(f"{a} {c[a] / n:.0%}" for a in keys if c[a])


def block(name, rs):
    return [f"| {name} | {len(rs)} | {dist(rs, 'start', ['top', 'bot'])} | {dist(rs, 'side3', ['top', 'bot'])} | "
            f"{dist(rs, 'kind', ['풀캠', '3~5캠프', '2캠프 이하'])} | {dist(rs, 'switch')} | {mean_t(rs, 'switch_t')} | {dist(rs, 'scuttle', ['top', 'bot', 'none'])} | {mean_t(rs, 'scuttle_t')} | "
            f"{dist(rs, 'gank_lane', ['top', 'mid', 'bot', 'none'])} | {mean_t(rs, 'gank_t')} |"]


def outcome_block(name, rs):
    return [f"| {name} | {len(rs)} | {rate(rs, 'gank_kill')} | {rate(rs, 'gank_death')} | {rate(rs, 'death7')} | "
            f"{rate(rs, 'grubs')} | {rate(rs, 'dragon')} | {rate(rs, 'kill')} |"]


def report(groups, games, policy):
    L = ["# 초반 동선 유형 → 주도권 (실제 vs 시뮬)", "",
         f"- 실제: 리플레이 1초 동선 정글러 {len(groups['실제'])}명(26.21 한국 챌린저). 시뮬: {games}판, 규칙 봇 양 팀"
         + (f" + 학습 정글러 `{os.path.relpath(policy, ROOT)}`(상대 규칙 봇) {games}판" if policy else "") + ".",
         "- 0~7분 위치를 `sim.real_actions.segments`로 행동으로 바꿔 같은 규칙으로 잰다(`sim/route_types.py`).",
         "- 전환 = 첫 캠프 뒤 처음 한 캠프 아닌 행동. 첫 바위게 = 4:30 전 6초 이상(3~5초는 강가를 지나가는 길). 첫 갱 = 7분 전 라인 구역 8초 이상(5~7초는 대개 미드 라인을 가로지르는 길), 시각은 라인으로 출발한 초, 성공 = 그 동안~10초 뒤 정글러 관여 킬.",
         "- ± 는 95% 구간(이항 근사).", "",
         "## 동선 분포", "",
         "| 집단 | 정글러 | 시작 쪽 | 3분 위치 | 첫 전환 전 캠프 수 | 첫 전환 | 첫 전환 시각 | 첫 바위게 | 첫 바위게 시각 | 첫 갱 라인 | 첫 갱 시각 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, rs in groups.items():
        L += block(k, rs)
    L += ["", "## 결과", "",
          "| 집단 | 정글러 | 첫 갱 킬 | 첫 갱 정글러 사망 | 7분 전 정글러 사망 | 첫 유충 | 첫 용 | 첫 킬 |", "|---|---|---|---|---|---|---|---|"]
    for k, rs in groups.items():
        L += outcome_block(k, rs)
    L += ["", "## 3:00 정글 CS (캠프 수 교차 확인)", "",
          "위치 기준 '풀캠'은 지나가며 바위게·라인에 3~8초 머문 것도 전환으로 잡아 적게 나온다. 정글 CS로 다시 센다.", "",
          "| 집단 | 3:00 정글 CS 중앙값 | 24 이상(6캠프) |", "|---|---|---|"]
    for k, rs in groups.items():
        v = np.array([r["cs3"] for r in rs if r.get("cs3") is not None])
        if len(v):
            p = float((v >= 24).mean())
            L.append(f"| {k} | {np.median(v):.0f} | {p:.0%} ±{1.96 * math.sqrt(p * (1 - p) / len(v)):.0%} |")
    L += ["", "## 유형별 결과", ""]
    for k, rs in groups.items():
        L += [f"### {k}", "", "| 유형 | 정글러 | 첫 갱 킬 | 첫 갱 정글러 사망 | 7분 전 정글러 사망 | 첫 유충 | 첫 용 | 첫 킬 |", "|---|---|---|---|---|---|---|---|"]
        for key, label in (("start", "시작 {}"), ("side3", "3분 위치 {}"), ("kind", "{}"), ("switch", "전환 {}"), ("scuttle", "첫 바위게 {}"), ("gank_lane", "첫 갱 {}")):
            for v, n in collections.Counter(r[key] for r in rs).most_common():
                sub = [r for r in rs if r[key] == v]
                if len(sub) >= 8:
                    L += outcome_block(label.format(v), sub)
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "route_types.md"))
    a = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    groups = {"실제": real_rows(nodes)}
    rule = simulate(a.games, None, a.workers)
    groups["시뮬 규칙 봇"] = rule
    if a.policy and os.path.exists(a.policy):
        mix = simulate(a.games, a.policy, a.workers)
        groups["시뮬 학습 정글러"] = [r for r in mix if r["who"] == "ai"]
    with open(os.path.join(ROOT, "data", "route_types_rows.json"), "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False)
    txt = report(groups, a.games, a.policy)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
