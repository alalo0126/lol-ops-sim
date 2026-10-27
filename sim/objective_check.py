"""오브젝트·바위게 시간 검수: 리플레이 1초 동선(실제) vs 시뮬레이터(학습 정글러·규칙 봇), 3~18분.

    python -m sim.objective_check [--policy models/stage11_a/policy_latest.npz] [--games 200]

1) 시간 나누기: sim.real_actions.segments로 정한 행동(obj_*, scuttle_*)의 초를 '가는 중'(이동)과 '머묾'(둥지 1,200·바위게 900 안)으로 나눈다.
2) 둥지 머문 초를 그때 오브젝트 상태로 나눈다: 생성 전(대기) / 살아 있음 / 이미 잡힘.
3) 방문(머문 구간)마다 결과: 우리 팀이 처치(구간 시작~끝+10초), 상대 팀이 처치, 정글러 사망.
   바위게는 처치 기록이 없어 '두 정글러가 함께 있었나'(900 안 동시)와 사망만 본다.
결과는 reports/objective_check.md.
"""
import argparse
import collections
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT, Rules
from .game import Game
from .lane_visits import T0, T1
from .real_actions import RADIUS, real_tracks, segments

AFTER = 10
OBJ_OF = {"obj_dragon": "dragon", "obj_grubs": "grubs", "obj_herald": "herald"}


def windows(kills, rules):
    """kills: [(초, 이름, 팀)] → 이름별 살아 있던 구간 [(시작, 끝)] (3~18분 안에서 쓰기에 충분하게)."""
    o = rules.objectives
    out = {}
    # 드래곤: 5:00 생성, 처치 후 300초
    t, w = o["drake"]["first_spawn_s"], []
    for k in sorted(x[0] for x in kills if x[1] == "dragon"):
        w.append((t, k))
        t = k + o["drake"]["respawn_s"]
    w.append((t, 10 ** 9))
    out["dragon"] = w
    g = sorted(x[0] for x in kills if x[1] == "grubs")
    out["grubs"] = [(o["voidgrubs"]["spawn_s"], g[-1] if len(g) >= 3 else o["voidgrubs"]["despawn_s"])]
    h = sorted(x[0] for x in kills if x[1] == "herald")
    out["herald"] = [(o["rift_herald"]["spawn_s"], h[0] if h else o["rift_herald"]["despawn_s"])]
    return out


def state(name, t, win, first):
    if t < first:
        return "before"
    return "alive" if any(a <= t < b for a, b in win[name]) else "after"


def analyse(pos, foe_pos, team, nodes, kills, deaths, rules):
    """한 정글러: 시간 나누기·둥지 상태별 초·방문 결과."""
    herald_spawn = rules.objectives["rift_herald"]["spawn_s"]
    seg = segments(pos, team, nodes, herald_spawn)
    win = windows(kills, rules)
    first = {"dragon": rules.objectives["drake"]["first_spawn_s"],
             "grubs": rules.objectives["voidgrubs"]["spawn_s"], "herald": herald_spawn}
    pit = {"dragon": nodes["dragon_pit"], "grubs": nodes["baron_pit"], "herald": nodes["baron_pit"]}
    scut = {"scuttle_top": nodes["river_top_scuttle"], "scuttle_bot": nodes["river_bot_scuttle"]}
    split = collections.Counter()      # (행동 묶음, 가는 중/머묾)
    pit_state = collections.Counter()  # (이름, 상태)
    visits = []
    raw = collections.Counter()   # 행동 라벨과 무관하게 둥지 반경 안 초
    for t, p in pos.items():
        if not p or not (T0 <= t < T1):
            continue
        for nm, n in pit.items():
            if math.hypot(n["x"] - p[0], n["y"] - p[1]) < RADIUS["epic"] and (nm != "herald" or t >= herald_spawn) and (nm != "grubs" or t < herald_spawn):
                raw[(nm, state(nm, t, win, first[nm]))] += 1
    cur = None
    for t in sorted(seg):
        a = seg[t]
        p = pos.get(t)
        here = False
        if a in OBJ_OF and p:
            n = pit[OBJ_OF[a]]
            here = math.hypot(n["x"] - p[0], n["y"] - p[1]) < RADIUS["epic"]
        elif a in scut and p:
            n = scut[a]
            here = math.hypot(n["x"] - p[0], n["y"] - p[1]) < RADIUS["scuttle"]
        if a in OBJ_OF or a in scut:
            split[("obj" if a in OBJ_OF else "scuttle", "here" if here else "going")] += 1
        if here and a in OBJ_OF:
            pit_state[(OBJ_OF[a], state(OBJ_OF[a], t, win, first[OBJ_OF[a]]))] += 1
        if here:
            if cur and cur["a"] == a and t - cur["e"] <= 3:
                cur["e"] = t
                if a in scut and foe_pos and foe_pos.get(t) and math.hypot(foe_pos[t][0] - scut[a]["x"], foe_pos[t][1] - scut[a]["y"]) < RADIUS["scuttle"]:
                    cur["contest"] = True
            else:
                if cur:
                    visits.append(cur)
                cur = {"a": a, "s": t, "e": t, "contest": False}
    if cur:
        visits.append(cur)
    out = []
    for v in visits:
        if v["e"] - v["s"] + 1 < 3:
            continue
        r = {"a": v["a"], "len": v["e"] - v["s"] + 1, "death": any(v["s"] <= d <= v["e"] + AFTER for d in deaths)}
        if v["a"] in OBJ_OF:
            nm = OBJ_OF[v["a"]]
            r["start"] = state(nm, v["s"], win, first[nm])
            ks = [k for k in kills if k[1] == nm and v["s"] <= k[0] <= v["e"] + AFTER]
            r["own"] = any(k[2] == team for k in ks)
            r["to_kill"] = min((k[0] - v["s"] for k in ks if k[2] == team), default=None)
            r["opp"] = any(k[2] != team for k in ks)
        else:
            r["contest"] = v["contest"] if foe_pos is not None else None
        out.append(r)
    return split, pit_state, out, raw


# ---------------------------------------------------------------------- 실제
NAME = {"DRAGON": "dragon", "HORDE": "grubs", "RIFTHERALD": "herald"}


def real(nodes, rules):
    by = collections.defaultdict(dict)
    for mid, team, pid, pos in real_tracks():
        if sum(1 for t in range(T0, T1) if t in pos) >= 0.8 * (T1 - T0):
            by[mid][team] = (pid, pos)
    res = []
    for mid, sides in by.items():
        with open(os.path.join(ROOT, "data", "replay_raw", f"{mid}.json"), encoding="utf-8") as f:
            d = json.load(f)
        ev = [e for fr in d["timeline"]["info"]["frames"] for e in fr["events"]]
        kills = [(e["timestamp"] / 1000, NAME[e["monsterType"]], e.get("killerTeamId") or (100 if e["killerId"] <= 5 else 200))
                 for e in ev if e["type"] == "ELITE_MONSTER_KILL" and e["monsterType"] in NAME]
        for team, (pid, pos) in sides.items():
            deaths = [e["timestamp"] / 1000 for e in ev if e["type"] == "CHAMPION_KILL" and e["victimId"] == pid]
            foe = sides.get(300 - team)
            res.append(analyse(pos, foe[1] if foe else None, team, nodes, kills, deaths, rules))
    return res


# ---------------------------------------------------------------------- 시뮬
_POLICY = None


def _init(path):
    global _POLICY
    _POLICY = path


def _sim_one(seed):
    from .agent import AgentJungler
    from .ppo import Policy
    from .stage6_track import _xy
    from .winprob import WinProb
    team = 100 if seed % 2 == 0 else 200
    pol = Policy.from_file(_POLICY)
    pol.rng = np.random.default_rng(seed)
    holder = {}

    def make(g, c):
        holder["bot"] = AgentJungler(g, c, lambda o, m, mw, ms: pol.act(o, m, mw, ms), WinProb())
        return holder["bot"]
    g = Game(seed=seed, jungler_override={team: make})
    ai = holder["bot"].c
    rule = next(x for x in g.champs if x.team != team and x.role == "JUNGLE")
    pos = {ai.pid: {}, rule.pid: {}}
    while g.t < T1 + AFTER:
        g.step()
        if g.t >= T0:
            for c in (ai, rule):
                xy = _xy(g, c)
                pos[c.pid][g.t] = None if xy[0] is None else xy
    kills = [(e["t"], e["name"], e["team"]) for e in g.events if e["type"] == "objective"]
    rules = g.rules
    out = {"events": event_rows(g, rules)}
    for k, c, o in (("ai", ai, rule), ("rule", rule, ai)):
        deaths = [e["t"] for e in g.events if e["type"] == "kill" and e["victim"] == c.pid]
        p = {t: v for t, v in pos[c.pid].items() if t < T1}
        out[k] = analyse(p, pos[o.pid], c.team, g.map.nodes, kills, deaths, rules)
    return out


def event_rows(g, rules):
    """시뮬 처치 사건 → tools.objective_fit과 같은 열(인원·생성 뒤 지연·90초 안 귀환·±30초 둥지 근처 사망)."""
    o = rules.objectives
    nodes = g.map.nodes
    pit = {"dragon": nodes["dragon_pit"], "grubs": nodes["baron_pit"], "herald": nodes["baron_pit"]}
    spawn = {"grubs": o["voidgrubs"]["spawn_s"], "herald": o["rift_herald"]["spawn_s"]}
    jg = {c.team: c.pid for c in g.champs if c.role == "JUNGLE"}
    team_of = {c.pid: c.team for c in g.champs}
    rows, prev = [], None
    for e in g.events:
        if e["type"] != "objective" or e["t"] >= 1200:
            continue
        nm, t, T = e["name"], e["t"], e["team"]
        if nm == "dragon":
            sp = prev + o["drake"]["respawn_s"] if prev else o["drake"]["first_spawn_s"]
            prev = t
        else:
            sp = spawn[nm]
        near = [k for k in g.events if k["type"] == "kill" and abs(k["t"] - t) <= 30
                and math.hypot(nodes[k["node"]]["x"] - pit[nm]["x"], nodes[k["node"]]["y"] - pit[nm]["y"]) < 3000]
        rows.append({"type": nm, "first_dragon": nm == "dragon" and sp == o["drake"]["first_spawn_s"], "delay": t - sp,
                     "people": e.get("people", 1),
                     "recall90": any(x["type"] == "shop" and x["pid"] == jg[T] and t < x["t"] <= t + 90 for x in g.events),
                     "deaths_own": sum(1 for k in near if team_of[k["victim"]] == T),
                     "deaths_opp": sum(1 for k in near if team_of[k["victim"]] != T)})
    return rows


def event_table(cols):
    """cols: [(이름, 사건 행 목록, 판 수)] → 표 줄."""
    L = ["", "## 4. 처치 사건(20분까지) — 실제 `tools/objective_fit.py`", "",
         "| 오브젝트 | | 판당 처치 | 인원 평균 | 혼자 | 생성 뒤 지연(중앙, 초) | 첫 용 지연(중앙) | 90초 안 귀환 | ±30초 둥지 근처 사망(처치 팀/상대) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nm in ("dragon", "grubs", "herald"):
        for name, rows, n in cols:
            R = [r for r in rows if r["type"] == nm]
            if not R:
                continue
            fd = [r["delay"] for r in R if r["first_dragon"]]
            L.append(f"| {nm} | {name} | {len(R) / n:.2f} | {np.mean([r['people'] for r in R]):.2f} | {np.mean([r['people'] == 1 for r in R]):.0%} | "
                     f"{np.median([r['delay'] for r in R]):.0f} | {np.median(fd) if fd else float('nan'):.0f} | {np.mean([r['recall90'] for r in R]):.0%} | "
                     f"{np.mean([r['deaths_own'] for r in R]):.2f} / {np.mean([r['deaths_opp'] for r in R]):.2f} |")
    return L


def summarize(rows):
    n = max(1, len(rows))
    split = sum((r[0] for r in rows), collections.Counter())
    pit = sum((r[1] for r in rows), collections.Counter())
    vis = [v for r in rows for v in r[2]]
    total = n * (T1 - T0)
    s = {"n": len(rows), "split": {k: v / total for k, v in split.items()}, "pit": {k: v / n for k, v in pit.items()}}
    for a in ("obj_dragon", "obj_grubs", "obj_herald", "scuttle_top", "scuttle_bot"):
        vs = [v for v in vis if v["a"] == a]
        d = {"per_game": len(vs) / n, "len": float(np.median([v["len"] for v in vs])) if vs else float("nan"),
             "death": float(np.mean([v["death"] for v in vs])) if vs else float("nan")}
        if a in OBJ_OF:
            al = [v for v in vs if v["start"] == "alive"]
            d["start_before"] = float(np.mean([v["start"] == "before" for v in vs])) if vs else float("nan")
            d["start_after"] = float(np.mean([v["start"] == "after" for v in vs])) if vs else float("nan")
            d["own"] = float(np.mean([v["own"] for v in vs])) if vs else float("nan")
            d["opp"] = float(np.mean([v["opp"] for v in vs])) if vs else float("nan")
            tk = [v["to_kill"] for v in vs if v["to_kill"] is not None]
            d["to_kill"] = float(np.median(tk)) if tk else float("nan")
        else:
            cs = [v["contest"] for v in vs if v["contest"] is not None]
            d["contest"] = float(np.mean(cs)) if cs else float("nan")
        s[a] = d
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage11_a", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "objective_check.md"))
    args = ap.parse_args()
    g0 = Game(seed=0)
    R = summarize(real(g0.map.nodes, g0.rules))
    with Pool(args.workers, initializer=_init, initargs=(args.policy,)) as pool:
        sims = pool.map(_sim_one, range(args.games))
    A, B = summarize([s["ai"] for s in sims]), summarize([s["rule"] for s in sims])
    cols = [("실제", R), ("학습 정글러", A), ("규칙 봇", B)]
    pc = lambda x: "-" if x != x else f"{x:.0%}"
    L = ["# 오브젝트·바위게 시간 검수 (3~18분)", "",
         f"- 실제 정글러 {R['n']}명(리플레이 1초 동선), 시뮬 {args.games}판: 학습 정글러 `{os.path.relpath(args.policy, ROOT)}`, 상대 규칙 봇.",
         "- 행동은 `sim.real_actions.segments`(이동 중인 초는 다음에 머문 곳의 행동). 머묾 = 둥지 1,200·바위게 900 안.", "",
         "## 1. 시간 비율(3~18분 전체 대비)", "", "| | 실제 | 학습 정글러 | 규칙 봇 |", "|---|---|---|---|"]
    for k, lab in ((("obj", "going"), "오브젝트: 가는 중"), (("obj", "here"), "오브젝트: 둥지에 머묾"),
                   (("scuttle", "going"), "바위게: 가는 중"), (("scuttle", "here"), "바위게: 머묾")):
        L.append(f"| {lab} | " + " | ".join(pc(s["split"].get(k, 0)) for _, s in cols) + " |")
    L += ["", "## 2. 둥지에 머문 초(판당) — 그때 오브젝트 상태", "", "| | 실제 | 학습 정글러 | 규칙 봇 |", "|---|---|---|---|"]
    for nm in ("dragon", "grubs", "herald"):
        for st, lab in (("before", "생성 전 대기"), ("alive", "살아 있음"), ("after", "이미 잡힘")):
            L.append(f"| {nm} {lab} | " + " | ".join(f"{s['pit'].get((nm, st), 0):.1f}" for _, s in cols) + " |")
    L += ["", "## 3. 방문(둥지·바위게 3초 이상 머묾)", "",
          "| 방문 | | 판당 | 머문 초(중앙) | 생성 전 도착 | 잡힌 뒤 도착 | 우리 처치 | 도착→처치 초(중앙) | 상대 처치 | 바위게 다툼 | 사망 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ("obj_dragon", "obj_grubs", "obj_herald", "scuttle_top", "scuttle_bot"):
        for name, s in cols:
            d = s[a]
            L.append(f"| {a} | {name} | {d['per_game']:.2f} | {d['len']:.0f} | {pc(d.get('start_before', float('nan')))} | "
                     f"{pc(d.get('start_after', float('nan')))} | {pc(d.get('own', float('nan')))} | {d.get('to_kill', float('nan')):.0f} | {pc(d.get('opp', float('nan')))} | "
                     f"{pc(d.get('contest', float('nan')))} | {pc(d['death'])} |")
    fit_rows = os.path.join(ROOT, "sim", "objective_fit_rows.json")
    if os.path.exists(fit_rows):
        with open(fit_rows, encoding="utf-8") as f:
            real_rows = json.load(f)
        with open(os.path.join(ROOT, "sim", "objective_fit.json"), encoding="utf-8") as f:
            n_real = json.load(f)["games"]
        L += event_table([("실제", real_rows, n_real), ("학습 정글러 판", [r for s in sims for r in s["events"]], args.games)])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
