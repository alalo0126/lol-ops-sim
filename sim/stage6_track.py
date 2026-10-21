"""6단계 다시 보기: 리플레이 1초 동선으로 "정글러가 그 1분 동안 한 일"을 매기고 학습 정글러와 비교한다.

    python -m sim.stage6_track      :: data/replay_raw + data/replay_tracks → reports/stage6_track.md, .json

6단계 label()은 1분 기록 중 사건(에픽·교전·구매)이 없으면 "1분 뒤 위치" 한 점으로 범주를 정했다. 그 1분 사이 어디를
다녀왔는지 모르니 갱을 갔다 정글로 돌아오면 '우리 정글'로, 1분 뒤 마침 라인에 서 있으면 '라인 개입'으로 잡혔다.
여기서는 사건 규칙은 그대로 두고, 나머지를 1분 동안 구역별로 머문 시간으로 정한다(label_track). 실제는 리플레이 1초
동선, 학습 정글러는 시뮬레이터 1초 위치에 같은 함수를 쓴다.
"""
import collections
import csv
import glob
import json
import math
import os

import numpy as np

from .data import ROOT
from .game import Game
from .ppo import Policy
from .stage6 import CATS, CAT_KO, RealMatch, label, reconstruct, sim_window, zone
from .winprob import WinProb

MINUTES = range(3, 18)
POLICY = os.environ.get("STAGE6_POLICY") or os.path.join(ROOT, "models", "stage6c", "policy_latest.npz")
LANE_S = 15      # 1분 중 라인에 이만큼 있으면 라인 개입(갱·합류·라인 CS)
INVADE_S = 15    # 적 정글에 이만큼 있으면 카정
PIT_S = 10       # 에픽 몬스터 둥지에 이만큼 있고 살아 있었으면 오브젝트(처치 못 해도 다툼·견제)
RECALL_S = 10    # 1분 중간에 우리 기지에 이만큼 있으면 귀환
MIN_COVER = 0.8  # 1분 중 유효한 1초 위치가 이만큼 있어야 비교에 쓴다
SIDE = {100: "blue", 200: "red"}


def zone_seconds(track, team, nodes):
    """[(t, x, y) 또는 (t, None, None)=사망] → 구역별 초, 처음 위치의 구역."""
    sec = collections.Counter()
    first = None
    for _, x, y in track:
        z = "dead" if x is None else zone(x, y, team, nodes)
        sec[z] += 1
        first = first or z
    return sec, first


def label_track(window, track, team, nodes):
    """label()과 같은 사건 우선순위(에픽 > 교전 위치 > 구매 귀환) 뒤, 1분 동안 머문 시간으로 범주를 정한다."""
    if window["elite"] or window["fights"]:
        return label(window, team, nodes)
    sec, first = zone_seconds(track, team, nodes)
    if window["recalled"] or (first != "base" and sec["base"] >= RECALL_S):
        return "recall"
    picks = [(sec["lane"], "lane")] if sec["lane"] >= LANE_S else []
    picks += [(sec["enemy_jungle"], "invade")] if sec["enemy_jungle"] >= INVADE_S else []
    if picks:
        return max(picks)[1]
    if sec["pit"] >= PIT_S and window["obj_live"]:
        return "objective"
    if sec["base"] > max(sec["own_jungle"], sec["river"] + sec["pit"]):
        return "recall"
    if sec["river"] + sec["pit"] > sec["own_jungle"]:
        return "farm" if window["cs_gain"] >= 4 and window.get("scuttle_near") else "river"
    return "farm"


# ---------------------------------------------------------------------- 실제
def load_real(match_id, champ, rm, team, minute):
    """리플레이 CSV에서 [minute, minute+1)분의 1초 위치. 유효 행이 MIN_COVER 미만이면 None.
    정글러가 죽어 있던 초(Riot 킬 기록 기준)는 사망으로 채운다."""
    path = os.path.join(ROOT, "data", "replay_tracks", f"{match_id}_{champ}.csv")
    if not os.path.exists(path):
        return None
    if path not in _TRACKS:
        with open(path, encoding="utf-8") as f:
            _TRACKS[path] = {round(float(r["time"])): (float(r["x"]), float(r["y"]))
                             for r in csv.DictReader(f) if r.get("valid", "1") == "1"}
    rows = _TRACKS[path]
    t0 = minute * 60
    pid = rm.jungler(team)
    dead = set()
    for k in rm.kills:
        if k[2] == pid:
            t = k[0] / 1000
            lv = rm.pf(min(int(t // 60), len(rm.frames) - 1), pid)["level"]
            dead.update(range(math.ceil(t), math.ceil(t + _death_timer()[lv - 1])))
    out = [(t, None, None) if t in dead else (t, *rows[t]) for t in range(t0, t0 + 60) if t in dead or t in rows]
    if len(out) < MIN_COVER * 60:
        return None
    return out


_TRACKS = {}


def _death_timer():
    from .data import Rules
    global _DT
    if _DT is None:
        _DT = Rules().combat["death_timer_s"]
    return _DT


_DT = None


def compact_raw(path):
    from tools.stage6_compact import compact
    with open(path, encoding="utf-8") as f:
        m = compact(json.load(f))
    m["match"] = os.path.basename(path)[:-5]
    return m


# ---------------------------------------------------------------------- 시뮬
def sim_minute_track(rm, team, minute, policy_fn, wp, seed):
    """실제 상태에서 학습 정글러를 1분 돌려 (6단계 범주, 1초 동선 범주, 구역별 초)를 돌려준다."""
    g, bot = reconstruct(rm, team, minute, policy_fn, wp, seed=seed)
    j = bot.c
    nodes = g.map.nodes
    cs0, t0 = j.jungle_cs, g.t
    track = []
    while g.t < t0 + 60:
        track.append((g.t, *_xy(g, j)))
        g.step()
    w = sim_window(g, team, t0, g.t, cs0)
    w["obj_live"] = any(rm.objective_live(k, t0 * 1000, (t0 + 60) * 1000) for k in ("dragon", "grubs", "herald"))
    return label(w, team, nodes), label_track(w, track, team, nodes), dict(zone_seconds(track, team, nodes)[0])


def _xy(g, c):
    """시뮬레이터 챔피언의 지금 좌표(엣지 위면 보간). 죽어 있으면 None."""
    if not c.alive(g.t):
        return None, None
    n = g.map.nodes
    a = n[c.node]
    if not c.path:
        return a["x"], a["y"]
    b = n[c.path[0]]
    d = g.map.dist(c.node, c.path[0]) or 1.0
    frac = min(1.0, max(0.0, 1 - c.edge_left / d))
    return a["x"] + (b["x"] - a["x"]) * frac, a["y"] + (b["y"] - a["y"]) * frac


def _fn(pol, seed):
    pol.rng = np.random.default_rng(seed)
    return lambda o, m, mw, ms: pol.act(o, m, mw, ms)


# ---------------------------------------------------------------------- 실행
def collect():
    nodes = Game(seed=0).map.nodes
    wp, pol = WinProb(), Policy.from_file(POLICY)
    rows = []
    for mi, path in enumerate(sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json")))):
        m = compact_raw(path)
        rm = RealMatch(m)
        if not rm.valid:
            continue
        for team in (100, 200):
            j = rm.jungler(team)
            for minute in MINUTES:
                if len(rm.frames) <= minute + 1 or m["duration"] <= (minute + 1) * 60:
                    continue
                track = load_real(m["match"], rm.champ_of[j], rm, team, minute)
                if track is None:
                    continue
                w = rm.window(team, minute, nodes)
                seed = mi * 1000 + team + minute
                ai = [sim_minute_track(rm, team, minute, _fn(pol, seed + 7919 * r), wp, seed + 7919 * r) for r in range(2)]
                rows.append({"match": m["match"], "team": team, "minute": minute, "champion": rm.champ_of[j],
                             "real_old": label(w, team, nodes), "real_new": label_track(w, track, team, nodes),
                             "ai_old": [a[0] for a in ai], "ai_new": [a[1] for a in ai],
                             "real_sec": dict(zone_seconds(track, team, nodes)[0]), "real_n": len(track),
                             "ai_sec": [a[2] for a in ai]})
    return rows


def dist(labels):
    c = collections.Counter(labels)
    n = sum(c.values()) or 1
    return {k: c[k] / n for k in CATS}


def report(rows):
    real_old = dist(r["real_old"] for r in rows)
    real_new = dist(r["real_new"] for r in rows)
    ai_old = dist(a for r in rows for a in r["ai_old"])
    ai_new = dist(a for r in rows for a in r["ai_new"])
    agree = sum(r["real_old"] == r["real_new"] for r in rows) / len(rows)
    conf = collections.Counter((r["real_old"], r["real_new"]) for r in rows)
    match_old = sum(a == r["real_old"] for r in rows for a in r["ai_old"]) / (2 * len(rows))
    match_new = sum(a == r["real_new"] for r in rows for a in r["ai_new"]) / (2 * len(rows))
    games = len({r["match"] for r in rows})
    pct = lambda v: f"{100 * v:.0f}%"
    lines = [
        "# 6단계 다시 보기: 리플레이 1초 동선으로 매긴 실제 정글러 vs 학습 정글러",
        "",
        f"- 데이터: 26.21 챌린저·그랜드마스터 {games}판, 양 팀 정글러 × 3~17분 중 1초 위치가 {pct(MIN_COVER)} 이상 있는 "
        f"{len(rows)}개 1분 구간. 학습 정글러는 `{os.path.relpath(POLICY, ROOT)}`, 같은 실제 상태에서 2번씩 1분.",
        f"- 1초 동선 범주(`label_track`): 에픽·교전·구매 귀환은 6단계 그대로, 나머지는 1분 중 라인 {LANE_S}초↑ = 라인 개입, "
        f"적 정글 {INVADE_S}초↑ = 카정, 둥지 {PIT_S}초↑(살아 있을 때) = 오브젝트, 중간에 기지 {RECALL_S}초↑ = 귀환, 그 밖은 "
        "우리 정글과 강가 중 더 오래 있던 쪽. 학습 정글러에도 시뮬레이터 1초 위치로 같은 함수를 썼다.",
        "",
        "## 범주 비율",
        "",
        "| 범주 | 실제 (6단계 방식) | 실제 (1초 동선) | AI (6단계 방식) | AI (1초 동선) |",
        "|---|---|---|---|---|",
    ]
    for k in CATS:
        lines.append(f"| {CAT_KO[k]} | {pct(real_old[k])} | {pct(real_new[k])} | {pct(ai_old[k])} | {pct(ai_new[k])} |")
    zones = ["own_jungle", "enemy_jungle", "lane", "river", "pit", "base", "dead"]
    zone_ko = {"own_jungle": "우리 정글", "enemy_jungle": "적 정글", "lane": "라인", "river": "강가", "pit": "에픽 둥지",
               "base": "기지", "dead": "사망"}
    # 실제는 유효한 초만 있으므로 60초로 맞춰 비율을 낸다
    real_z = {z: 60 * sum(r["real_sec"].get(z, 0) / r["real_n"] for r in rows) / len(rows) for z in zones}
    ai_z = {z: sum(a.get(z, 0) for r in rows for a in r["ai_sec"]) / (2 * len(rows)) for z in zones}
    lines += ["", "## 1분 동안 구역별로 머문 시간(초, 평균)", "", "| 구역 | 실제 | AI |", "|---|---|---|"]
    for z in zones:
        lines.append(f"| {zone_ko[z]} | {real_z[z]:.1f} | {ai_z[z]:.1f} |")
    lines += [
        "",
        f"- 같은 실제 구간에서 6단계 방식과 1초 동선 범주가 같은 비율: {pct(agree)}.",
        f"- AI가 실제와 같은 범주를 고른 비율: 6단계 방식 {pct(match_old)}, 1초 동선 {pct(match_new)}.",
        "",
        "## 6단계 방식 → 1초 동선 (실제, 많은 순)",
        "",
        "| 6단계 방식 | 1초 동선 | 구간 수 |",
        "|---|---|---|",
    ]
    for (a, b), n in conf.most_common():
        if a != b:
            lines.append(f"| {CAT_KO[a]} | {CAT_KO[b]} | {n} |")
    return "\n".join(lines) + "\n", {"rows": rows, "real_old": real_old, "real_new": real_new, "ai_old": ai_old,
                                    "ai_new": ai_new, "real_zone_s": real_z, "ai_zone_s": ai_z, "agree": agree, "match_old": match_old, "match_new": match_new}


def main():
    rows = collect()
    md, js = report(rows)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "stage6_track.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(ROOT, "reports", "stage6_track.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False)
    print(md)


if __name__ == "__main__":
    main()
