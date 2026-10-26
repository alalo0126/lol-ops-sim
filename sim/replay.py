"""시각화용 재생 기록: 시뮬 경기(학습 정글러 한 명 + 규칙 봇 9명)와 실제 천상계 경기를 같은 형식의 JSON으로 뽑는다.

    python -m sim.replay                         :: → viz/replay_data.js (viz/index.html이 읽음) (시뮬 6판 + 실제 6판)
    python -m sim.replay --sim 10 --real 10

시뮬은 2초마다 10명의 위치(노드 사이는 이동한 거리 비율로 보간)·체력·하는 일, 캠프·오브젝트 상태, 블루 예측 승률을 남긴다.
실제 경기는 Riot 타임라인이 1분 간격이라 1분마다 위치·골드·경험치·정글 CS만 있고, 킬·에픽 몬스터·구매(귀환 대리)는 이벤트 시각 그대로.
stats 열: 시뮬 [골드, 경험치, 레벨, 정글 CS, 라인 CS, 킬, 데스, 어시스트], 실제 [골드, 경험치, 레벨, 정글 CS, 라인 CS].
실제 경기의 캠프 처치 시각은 기록에 없어서 보여 주지 않는다. 입력: models/stage6b/policy_latest.npz, data/stage6_compact.json.gz.
"""
import argparse
import csv
import json
import math
import os
import random

import numpy as np

from .agent import ACTION_NAMES, AgentJungler
from .data import ROOT
from .game import Game, ROLES, SIDE
from .ppo import Policy
from .stage6 import CAT_KO, ELITE_KEY, RealMatch, label, load_compact, sim_window
from .winprob import WinProb

POLICY = os.path.join(ROOT, "models", "stage6b", "policy_latest.npz")
STEP_S = 2
_CACHE = {}
TASK_KO = {"camp": "캠프", "clearing": "캠프 처치", "gank": "갱 가는 중", "ganking": "갱", "cover": "라인 지원",
           "objective": "오브젝트", "recalling": "귀환", "lanefarm": "라인 미니언", "wait": "대기", "move": "이동",
           "helping": "오브젝트 지원", "retreat": "후퇴"}
ACTION_KO = {
    "camp": "우리 정글", "invade": "카정", "scuttle": "바위게", "gank": "갱", "cover": "라인 지원",
    "obj": "오브젝트", "recall": "귀환", "lanefarm": "라인 미니언", "wait": "대기", "move": "이동",
}


def action_ko(name):
    return ACTION_KO[name.split("_")[0]]


def champ_xy(g, c):
    """노드 c.node에서 c.path[0]로 가는 중이면 남은 거리로 위치를 보간한다."""
    n = g.map.nodes
    if not c.alive(g.t):
        f = n[f"{SIDE[c.team]}_fountain"]
        return f["x"], f["y"]
    a = n[c.node]
    if not c.path:
        return a["x"], a["y"]
    b = n[c.path[0]]
    d = g.map.dist(c.node, c.path[0]) or 1.0
    frac = min(1.0, max(0.0, 1 - c.edge_left / d))
    return a["x"] + (b["x"] - a["x"]) * frac, a["y"] + (b["y"] - a["y"]) * frac


def task_label(g, c, bot):
    if not c.alive(g.t):
        return "사망"
    if isinstance(bot, AgentJungler) and bot.cur_action is not None:
        if c.task and c.task["kind"] in ("clearing", "recalling", "ganking", "objective"):
            return TASK_KO[c.task["kind"]] + (" (카정)" if c.task["kind"] == "clearing" and not c.task["camp"].startswith(SIDE[c.team]) and not c.task["camp"].startswith("river") else "")
        return action_ko(ACTION_NAMES[bot.cur_action])
    if c.role != "JUNGLE" and g.t < c.retreat_until:
        return "후퇴"
    if c.task:
        return TASK_KO.get(c.task["kind"], c.task["kind"])
    return "라인" if c.role != "JUNGLE" else "대기"


def record_sim(seed, team, pol, wp):
    pol.rng = np.random.default_rng(seed)
    holder = {}

    def make(gm, ch):
        holder["bot"] = AgentJungler(gm, ch, lambda o, m, mw, ms: pol.act(o, m, mw, ms), wp)
        return holder["bot"]

    g = Game(seed=seed, jungler_override={team: make})
    camp_ids = sorted(g.camps)
    labels, label_idx = [], {}

    def lab(s):
        if s not in label_idx:
            label_idx[s] = len(labels)
            labels.append(s)
        return label_idx[s]

    out = {"seed": seed, "agent_team": team, "step_s": STEP_S, "camp_ids": camp_ids,
           "champs": [{"pid": c.pid, "team": c.team, "role": c.role} for c in g.champs],
           "pos": [], "hp": [], "task": [], "camps": [], "obj": [], "wp": [], "stats": [], "wards": []}
    agent = None
    actions = []
    last_action = None

    def snap():
        out["pos"].append([v for c in g.champs for v in map(round, champ_xy(g, c))])
        out["hp"].append([round(c.hp * 100) if c.alive(g.t) else 0 for c in g.champs])
        out["task"].append([lab(task_label(g, c, g.bots[c.pid])) for c in g.champs])
        out["camps"].append("".join("1" if g.camps[k].alive else "0" for k in camp_ids))
        out["obj"].append("".join("1" if g.objectives[k]["alive"] else "0" for k in ("dragon", "grubs", "herald")))
        out["wp"].append(round(wp.blue(g), 4))
        out["stats"].append([[round(c.total_gold), round(c.xp), c.level, c.jungle_cs, int(c.lane_cs), c.kills, c.deaths, c.assists]
                             for c in g.champs])
        out["wards"].append([[w["team"], w["node"]] for w in g.wards])

    snap()
    cats = {100: [], 200: []}
    cs0 = {100: 0, 200: 0}
    obj_seen = False
    while g.t < 1200:
        g.step()
        obj_seen = obj_seen or any(o["alive"] for o in g.objectives.values())
        if g.t % 60 == 0:
            # 6단계와 같은 범주 함수(위치·이벤트만 보고 붙임)로 지난 1분을 분류한다
            for tm in (100, 200):
                w = sim_window(g, tm, g.t - 60, g.t, cs0[tm])
                w["obj_live"] = obj_seen
                cats[tm].append(label(w, tm, g.map.nodes))
                cs0[tm] = next(c for c in g.champs if c.team == tm and c.role == "JUNGLE").jungle_cs
            obj_seen = any(o["alive"] for o in g.objectives.values())
        agent = holder["bot"]
        if agent.cur_action is not None and agent.cur_action != last_action:
            last_action = agent.cur_action
            actions.append([g.t, ACTION_NAMES[last_action], action_ko(ACTION_NAMES[last_action])])
        if g.t % STEP_S == 0:
            snap()
    out["labels"] = labels
    out["agent_actions"] = actions
    out["cats"] = {str(k): v for k, v in cats.items()}
    out["events"] = [e for e in g.events if e["type"] in ("kill", "objective", "gank", "skirmish", "recall", "lanefarm")
                     or (e["type"] == "camp" and e["pid"] in (2, 7))]
    out["phi_end"] = round(agent.finish_episode(), 4)
    return out


def record_real(m, wp):
    rm = RealMatch(m)
    n = min(21, len(rm.frames))
    pids = [rm.pid[(t, r)] for t in (100, 200) for r in ROLES]
    out = {"match": m["match"], "win100": int(m["win100"]), "minutes": n - 1,
           "champs": [{"pid": i + 1, "team": rm.team_of[p], "role": rm.role_of[p], "champion": rm.champ_of[p]} for i, p in enumerate(pids)],
           "pos": [], "hp": [], "stats": [], "wp": []}
    remap = {p: i + 1 for i, p in enumerate(pids)}
    for mi in range(n):
        fs = [rm.pf(mi, p) for p in pids]
        out["pos"].append([v for f in fs for v in (f["x"], f["y"])])
        out["hp"].append([round(f["hp"] * 100) for f in fs])
        out["stats"].append([[f["total_gold"], f["xp"], f["level"], f["jcs"], f["mcs"]] for f in fs])
        out["wp"].append(round(wp.blue_from_diffs(rm.diffs(mi), mi * 60), 4))
    if "nodes" not in _CACHE:
        _CACHE["nodes"] = Game(seed=0).map.nodes
    nodes = _CACHE["nodes"]
    out["cats"] = {str(tm): [label(rm.window(tm, mi, nodes), tm, nodes) for mi in range(n - 1)] for tm in (100, 200)}
    until = (n - 1) * 60000
    ev = []
    for t, killer, victim, assists, x, y in rm.kills:
        if t <= until:
            ev.append({"t": round(t / 1000), "type": "kill", "killer": remap.get(killer, 0), "victim": remap.get(victim, 0),
                       "assists": [remap[a] for a in assists if a in remap], "x": x, "y": y})
    for e in rm.elite:
        if e[0] <= until and ELITE_KEY.get(e[1]):
            ev.append({"t": round(e[0] / 1000), "type": "objective", "name": ELITE_KEY[e[1]], "team": e[3],
                       "pid": remap.get(e[4], 0), "x": e[6], "y": e[7]})
    # 구매: 같은 사람의 30초 안 연속 구매는 한 번(귀환·사망 한 번)으로 묶는다. 시작 아이템(1분 전)은 뺀다.
    last = {}
    for t, pid, item in sorted(rm.buys):
        if t <= until and t > 60000 and t - last.get(pid, -1e9) > 30000:
            ev.append({"t": round(t / 1000), "type": "buy", "pid": remap.get(pid, 0)})
        last[pid] = t
    out["events"] = sorted(ev, key=lambda e: e["t"])
    out["jg_track"], out["camp_kills"] = estimate_jungle(rm, nodes, n - 1)
    out["info"] = match_index().get(m["match"], {})
    out["camp_timing"] = {cid: _camp_timing(cid) for cid in out["camp_kills"]}
    return out


def match_index():
    """tools/match_index.py가 만든 data/match_index.csv(누구의 어떤 경기인지). 없으면 빈 dict."""
    if "index" not in _CACHE:
        path = os.path.join(ROOT, "data", "match_index.csv")
        rows = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    rows[r["match"]] = {k: r[k] for k in ("date_kst", "seed_riot_id", "seed_team", "seed_champion",
                                                          "enemy_jungler_riot_id", "opgg_seed")}
        _CACHE["index"] = rows
    return _CACHE["index"]


def _camp_timing(cid):
    if "camps" not in _CACHE:
        _CACHE["camps"] = Game(seed=0).rules.camps
    spec = _CACHE["camps"]["scuttle" if cid.startswith("river") else cid.split("_", 1)[1]]
    return [spec["first_spawn_s"], spec["respawn_s"]]


def estimate_jungle(rm, nodes, minutes):
    """실제 정글러의 1분 사이 동선을 추정한다(Riot 기록은 위치가 1분 간격이라 그대로 이으면 순간 이동처럼 보임).

    1분 동안 정글 CS가 4 늘 때마다 캠프 하나를 처치한 것으로 보고, 그 1분의 앞뒤 위치에 가까운(그때 살아 있다고
    추정되는) 캠프를 고른다(6단계 _assign_camps와 같은 방식). 그 1분 안의 정확한 기록(정글러가 관여한 킬·사망,
    에픽 몬스터 처치 위치, 구매 = 우물)은 시각이 고정된 지점으로 넣고, 캠프는 그 지점들 사이에서 가장 덜 돌아가는
    곳에 끼운다. 구매·사망 직전에는 이동하지 않고 우물로 바로 옮긴다(귀환·부활). 캠프마다 처치에 최대 14초를 준다.
    돌려주는 것: {팀: [[초, x, y], ...]}, {캠프: [추정 처치 초, ...]}.
    """
    camp_ids = sorted(cid for cid, n in nodes.items() if n["kind"] in ("camp", "scuttle"))
    timing = {cid: _camp_timing(cid) for cid in camp_ids}
    kills = {cid: [] for cid in camp_ids}
    tracks = {}
    xy = lambda cid: (nodes[cid]["x"], nodes[cid]["y"])
    d = math.dist

    def alive_at(cid, t):
        first, respawn = timing[cid]
        return t >= first - 10 and (not kills[cid] or t >= kills[cid][-1] + respawn)

    for tm in (100, 200):
        tracks[tm] = []
    for k in range(1, minutes + 1):
        t0, t1 = (k - 1) * 60, k * 60
        for tm in (100, 200):
            j = rm.jungler(tm)
            fountain = (14426, 14170) if tm == 200 else (662, 285)
            a, b = rm.pf(k - 1, j), rm.pf(k, j)
            # 시각이 고정된 지점: (초, x, y, 앞에서 순간 이동인가)
            fixed = [(t0, a["x"], a["y"], False)]
            for t, killer, victim, assists, x, y in rm.kills:
                if t0 * 1000 < t <= t1 * 1000 and x is not None and j in (killer, victim, *assists):
                    fixed.append((t / 1000, x, y, False))
                    if victim == j:
                        fixed.append((t / 1000 + 1, *fountain, True))
            for e in rm.elite:
                if t0 * 1000 < e[0] <= t1 * 1000 and e[6] is not None and (e[4] == j or j in e[5]):
                    fixed.append((e[0] / 1000, e[6], e[7], False))
            for t, pid, _ in rm.buys:
                if pid == j and t0 * 1000 < t <= t1 * 1000 and t > 60000:
                    fixed.append((t / 1000, *fountain, True))
                    break
            fixed.append((t1, b["x"], b["y"], False))
            fixed.sort(key=lambda p: p[0])
            # 처치한 캠프 추정
            n = max(0, round((b["jcs"] - a["jcs"]) / 4))
            cands = [cid for cid in camp_ids if alive_at(cid, t0 + 45)]
            cands.sort(key=lambda cid: min(d((a["x"], a["y"]), xy(cid)), d((b["x"], b["y"]), xy(cid))))
            chosen = cands[:n]
            # 구간마다 끼울 캠프: 덜 돌아가는 구간으로
            segs = [[] for _ in range(len(fixed) - 1)]
            for cid in chosen:
                best, bi = None, 0
                for i in range(len(fixed) - 1):
                    P, Q = fixed[i][1:3], fixed[i + 1][1:3]
                    if fixed[i + 1][3]:
                        Q = P
                    last = segs[i][-1] if segs[i] else None
                    start = xy(last) if last else P
                    cost = d(start, xy(cid)) + d(xy(cid), Q) - d(start, Q) + 300 * len(segs[i])
                    if fixed[i + 1][0] - fixed[i][0] < 8:
                        cost += 1e6
                    if best is None or cost < best:
                        best, bi = cost, i
                segs[bi].append(cid)
            pts = tracks[tm]
            for i in range(len(fixed) - 1):
                ta, xa, ya, _ = fixed[i]
                tb, xb, yb, jump = fixed[i + 1]
                if not pts or pts[-1][0] < ta:
                    pts.append([ta, xa, ya])
                # 가까운 캠프부터 차례로
                order, cur, rest = [], (xa, ya), list(segs[i])
                while rest:
                    nxt = min(rest, key=lambda c: d(cur, xy(c)))
                    order.append(nxt)
                    rest.remove(nxt)
                    cur = xy(nxt)
                span = tb - ta
                dwell = min(14.0, 0.6 * span / len(order)) if order else 0.0
                way = [(xa, ya)] + [xy(c) for c in order] + ([] if jump else [(xb, yb)])
                dist = [d(way[q], way[q + 1]) for q in range(len(way) - 1)]
                move = max(0.0, span - dwell * len(order) - (0.5 if jump else 0.0))
                total = sum(dist) or 1.0
                t = ta
                for q, cid in enumerate(order):
                    t += move * dist[q] / total
                    pts.append([t, *xy(cid)])
                    t += dwell
                    pts.append([t, *xy(cid)])
                    kills[cid].append(t)
                if jump:
                    pts.append([tb - 0.5, *pts[-1][1:]])
                pts.append([tb, xb, yb])
    for tm in tracks:
        tracks[tm] = [[round(t, 1), round(x), round(y)] for t, x, y in tracks[tm]]
    return {str(tm): v for tm, v in tracks.items()}, {cid: [round(t) for t in v] for cid, v in kills.items()}


def pick_real(matches, k, seed=0):
    """정글러 10분 정글 CS가 실제 p25~p75인 경기 중에서 고른다(전형적인 경기). 블루·레드 승리를 섞는다."""
    rng = random.Random(seed)
    good = []
    for m in matches:
        rm = RealMatch(m)
        if not rm.valid or len(rm.frames) < 21:
            continue
        cs = [rm.pf(10, rm.jungler(t))["jcs"] for t in (100, 200)]
        if all(56 <= c <= 84 for c in cs):
            good.append(m)
    rng.shuffle(good)
    wins = [m for m in good if int(m["win100"]) == 1][: (k + 1) // 2]
    losses = [m for m in good if int(m["win100"]) == 0][: k // 2]
    return wins + losses


def bands():
    with open(os.path.join(ROOT, "sim", "targets_stage1.json"), encoding="utf-8") as f:
        t = json.load(f)
    out = {}
    for mi, row in t["both_junglers_by_minute"].items():
        out[mi] = {k: [row[k][q] for q in ("p10", "p25", "median", "p75", "p90")] for k in ("gold", "jungle_cs", "xp", "level") if k in row}
    return out


def map_layout(g):
    return {"nodes": {k: [round(v["x"]), round(v["y"]), v["kind"]] for k, v in g.map.nodes.items()},
            "edges": sorted({tuple(sorted((a, b))) for a in g.map.nodes for b in g.map.adj[a]})}


# ---------------------------------------------------------------------- 리플레이 1초 기록 경기
TRACKS = os.path.join(ROOT, "data", "replay_tracks")
REPLAY_RAW = os.path.join(ROOT, "data", "replay_raw")
CAMP_RADIUS = 900     # 정글러가 이 거리 안에 머문 캠프를 처치 후보로 본다(맵 노드 좌표는 캠프 중심의 근사)


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replay_matches():
    """10명 모두 1초 기록이 있고 위치 검증(summary.csv 거리 중앙값 1,000 이하)을 통과한 경기."""
    if not os.path.isdir(TRACKS):
        return []
    bad = set()
    summary = os.path.join(TRACKS, "summary.csv")
    if os.path.exists(summary):
        bad = {r["match"] for r in _read_csv(summary) if int(r["median_dist"]) > 1000}
    out = []
    for name in sorted(os.listdir(REPLAY_RAW)) if os.path.isdir(REPLAY_RAW) else []:
        mid = name[:-5]
        tracks = [f for f in os.listdir(TRACKS) if f.startswith(mid + "_") and not f.endswith(("_players.csv",))]
        if mid not in bad and len(tracks) == 10:
            out.append(mid)
    return out


def _track_xy(rows, grid, fallback):
    """유효한 1초 기록을 grid 시각으로 보간. 앞뒤 유효 기록이 8초 넘게 비면 fallback(1분 위치 직선 보간)을 쓴다."""
    ts = np.array([float(r["time"]) for r in rows if r.get("valid", "1") == "1"])
    xs = np.array([float(r["x"]) for r in rows if r.get("valid", "1") == "1"])
    ys = np.array([float(r["y"]) for r in rows if r.get("valid", "1") == "1"])
    out = []
    for i, t in enumerate(grid):
        j = np.searchsorted(ts, t)
        near = min(abs(ts[j - 1] - t) if j > 0 else 1e9, abs(ts[j] - t) if j < len(ts) else 1e9)
        if len(ts) and near <= 8:
            out.append((float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))))
        else:
            out.append(fallback[i])
    return out


def _camps_from_tracks(rm, nodes, jg_xy, grid, minutes):
    """1분 동안 늘어난 정글 CS(4 = 캠프 1개)만큼, 정글러가 그 1분에 가장 오래 머문 캠프를 처치한 것으로 본다."""
    camp_ids = sorted(cid for cid, n in nodes.items() if n["kind"] in ("camp", "scuttle"))
    timing = {cid: _camp_timing(cid) for cid in camp_ids}
    kills = {cid: [] for cid in camp_ids}
    for k in range(1, minutes + 1):
        t0, t1 = (k - 1) * 60, k * 60
        for tm in (100, 200):
            j = rm.jungler(tm)
            n = max(0, round((rm.pf(k, j)["jcs"] - rm.pf(k - 1, j)["jcs"]) / 4))
            if not n:
                continue
            stay = {}
            for t, (x, y) in zip(grid, jg_xy[tm]):
                if not t0 <= t <= t1:
                    continue
                for cid in camp_ids:
                    first, respawn = timing[cid]
                    if t < first - 5 or (kills[cid] and t < kills[cid][-1] + respawn):
                        continue
                    if math.dist((x, y), (nodes[cid]["x"], nodes[cid]["y"])) <= CAMP_RADIUS:
                        s = stay.setdefault(cid, [0, t])
                        s[0] += 1
                        s[1] = t
            for cid, (cnt, last) in sorted(stay.items(), key=lambda kv: -kv[1][0])[:n]:
                if cnt >= 2:
                    kills[cid].append(last)
    for v in kills.values():
        v.sort()
    return {cid: [round(t) for t in v] for cid, v in kills.items()}


def record_replay(mid, wp):
    """리플레이에서 1초마다 기록한 10명 위치(tools/replay_track.py)로 실제 경기를 만든다. 골드·경험치·이벤트는 Riot 타임라인."""
    from tools.stage6_compact import compact
    with open(os.path.join(REPLAY_RAW, mid + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    m = compact(d)
    m["match"] = mid
    out = record_real(m, wp)
    rm = RealMatch(m)
    nodes = _CACHE["nodes"]
    grid = list(range(0, out["minutes"] * 60 + 1, STEP_S))
    minute_pos = out["pos"]

    def fallback(k):
        res = []
        for t in grid:
            mi = min(out["minutes"] - 1, int(t // 60))
            fr = min(1.0, t / 60 - mi)
            a, b = minute_pos[mi], minute_pos[mi + 1]
            res.append((a[2 * k] + (b[2 * k] - a[2 * k]) * fr, a[2 * k + 1] + (b[2 * k + 1] - a[2 * k + 1]) * fr))
        return res

    xy, dead = [], []
    states = _read_csv(os.path.join(TRACKS, mid + "_players.csv"))
    for k, ch in enumerate(out["champs"]):
        path = os.path.join(TRACKS, f"{mid}_{ch['champion']}.csv")
        xy.append(_track_xy(_read_csv(path), grid, fallback(k)) if os.path.exists(path) else fallback(k))
        st_rows = sorted((float(r["time"]), r["is_dead"] in ("1", "True", "true")) for r in states if r["champion"] == ch["champion"])
        ts = [t for t, _ in st_rows]
        dead.append([bool(st_rows) and st_rows[max(0, np.searchsorted(ts, t, side="right") - 1)][1] for t in grid])
    out["step_s"] = STEP_S
    out["source"] = "replay"
    out["pos"] = [[round(v) for k in range(10) for v in xy[k][i]] for i in range(len(grid))]
    out["dead"] = ["".join("1" if dead[k][i] else "0" for k in range(10)) for i in range(len(grid))]
    jg = {tm: xy[next(k for k, c in enumerate(out["champs"]) if c["team"] == tm and c["role"] == "JUNGLE")] for tm in (100, 200)}
    out["camp_kills"] = _camps_from_tracks(rm, nodes, jg, grid, out["minutes"])
    out["camp_timing"] = {cid: _camp_timing(cid) for cid in out["camp_kills"]}
    del out["jg_track"]
    info = d["detail"]["info"]
    parts = {p["participantId"]: p for p in info["participants"]}
    seed = parts.get((d.get("seed") or {}).get("participantId"), {})
    import datetime
    when = datetime.datetime.fromtimestamp(info["gameCreation"] / 1000, datetime.timezone(datetime.timedelta(hours=9)))
    jname = {p["teamId"]: f"{p.get('riotIdGameName', '?')}#{p.get('riotIdTagline', '')}" for p in parts.values() if p.get("teamPosition") == "JUNGLE"}
    out["info"] = {"date_kst": when.strftime("%Y-%m-%d %H:%M"), "patch": ".".join(info["gameVersion"].split(".")[:2]),
                   "seed_team": "블루" if seed.get("teamId") == 100 else "레드" if seed else "",
                   "seed_tier": (d.get("seed") or {}).get("tier", ""), "jungler_ids": {str(k): v for k, v in jname.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=int, default=6)
    ap.add_argument("--real", type=int, default=6, help="리플레이 1초 기록 경기가 없을 때만: 1,000판에서 고를 판 수")
    ap.add_argument("--replays", type=int, default=12, help="리플레이 1초 기록 경기 수(있으면 이것을 실제 경기로 씀)")
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "replay_data.js"))
    a = ap.parse_args()
    wp = WinProb()
    pol = Policy.from_file(POLICY)
    sims = [record_sim(s, 100 if s % 2 == 0 else 200, pol, wp) for s in range(a.sim)]
    if "nodes" not in _CACHE:
        _CACHE["nodes"] = Game(seed=0).map.nodes
    replays = replay_matches()[: a.replays]
    reals = [record_replay(mid, wp) for mid in replays] if replays else [record_real(m, wp) for m in pick_real(load_compact(), a.real)]
    data = {"map": map_layout(Game(seed=0)), "bands": bands(), "sim": sims, "real": reals,
            "policy": os.path.relpath(POLICY, ROOT).replace("\\", "/"), "cat_ko": CAT_KO}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        # file://로 연 index.html에서도 읽히도록 JSON이 아니라 스크립트로 쓴다
        f.write("window.REPLAY_DATA=")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"시뮬 {len(sims)}판, 실제 {len(reals)}판 → {a.out} ({os.path.getsize(a.out) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
