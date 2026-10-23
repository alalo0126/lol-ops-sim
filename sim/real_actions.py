"""1초 동선 → 정글러 행동(학습 정글러 행동 이름) 구간. 실제(리플레이)와 시뮬레이터에 같은 함수를 쓴다.

    python -m sim.real_actions [--policy models/stage6c/policy_latest.npz] [--games 200]

머문 곳으로 행동을 정한다: 캠프 600 안 3초 이상 = camp_*/invade_*(우리/적 캠프), 바위게 900 안 = scuttle_*,
용 둥지 1,200 안 = obj_dragon, 바론 둥지 = obj_grubs(전령 생성 전)/obj_herald, 기지 = recall,
라인 구역 5초 이상 = gank_<라인>. 이동 중인 초는 다음에 머문 곳의 행동으로 채운다(그곳에 가려던 것으로 봄).
죽은 초는 dead, 끝까지 아무 데도 안 머물면 move.
결과는 reports/real_actions.md: 3~18분 행동별 시간 비율(실제 / 학습 정글러 / 규칙 봇).
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from .data import ROOT, Rules
from .game import Game
from .lane_visits import T0, T1, lane_of
from .stage6 import zone

RADIUS = {"camp": 600, "scuttle": 900, "epic": 1200}
MIN_DWELL = {"camp": 3, "scuttle": 3, "epic": 3, "base": 3, "lane": 5}
SIDE = {100: "B", 200: "R"}
GROUPS = ["camp", "invade", "scuttle", "gank", "obj", "recall", "move", "dead"]
GROUP_KO = {"camp": "우리 캠프", "invade": "카정", "scuttle": "바위게", "gank": "라인(갱·합류)", "obj": "오브젝트",
            "recall": "귀환·기지", "move": "이동(목적지 없음)", "dead": "사망"}


def place(x, y, team, nodes, herald_at):
    """좌표 하나 → (종류, 행동 이름) 또는 None."""
    z = zone(x, y, team, nodes)
    if z == "base":
        return "base", "recall"
    if z == "lane":
        return "lane", f"gank_{lane_of(x, y)}"
    best, bd = None, 1e18
    for nid, n in nodes.items():
        if n["kind"] in RADIUS:
            d = math.hypot(n["x"] - x, n["y"] - y)
            if d < RADIUS[n["kind"]] and d < bd:
                best, bd = nid, d
    if best is None:
        return None
    kind = nodes[best]["kind"]
    if kind == "camp":
        side, k = best.split("_", 1)
        return kind, ("camp_" if side == SIDE[team] else "invade_") + k
    if kind == "scuttle":
        return kind, "scuttle_top" if "top" in best else "scuttle_bot"
    return kind, best  # epic: 시각을 보고 아래에서 정함


def segments(pos, team, nodes, herald_at, t0=T0, t1=T1):
    """pos: {초: (x, y) 또는 None=사망} → {초: 행동}. 위치를 모르는 초는 빠진다."""
    raw = {}
    for t in range(t0, t1):
        if t not in pos:
            continue
        if pos[t] is None:
            raw[t] = ("dead", "dead")
            continue
        p = place(*pos[t], team, nodes, herald_at)
        if p and p[0] == "epic":
            p = ("epic", "obj_dragon" if p[1] == "dragon_pit" else "obj_herald" if t >= herald_at else "obj_grubs")
        raw[t] = p
    # 머문 구간(같은 행동 연속, 2초 이하 끊김 허용)
    acts = {}
    ts = sorted(raw)
    i = 0
    while i < len(ts):
        p = raw[ts[i]]
        if p is None or p[0] == "dead":
            i += 1
            continue
        j, last = i, ts[i]
        while j + 1 < len(ts) and ts[j + 1] - last <= 3 and (raw[ts[j + 1]] == p or raw[ts[j + 1]] is None):
            j += 1
            if raw[ts[j]] == p:
                last = ts[j]
        if last - ts[i] + 1 >= MIN_DWELL[p[0]]:
            for t in range(ts[i], last + 1):
                acts[t] = p[1]
        i = j + 1 if last == ts[j] else ts.index(last) + 1
    out, nxt = {}, "move"
    for t in reversed(ts):
        if raw[t] and raw[t][0] == "dead":
            out[t] = "dead"
            continue
        if t in acts:
            nxt = acts[t]
        out[t] = nxt
    return out


def group(a):
    return a.split("_")[0] if a.split("_")[0] in GROUPS else a


# ---------------------------------------------------------------------- 실제
def real_tracks():
    """[(경기, 팀, 참가자 번호, {초: (x, y) 또는 None})]. 죽어 있던 초는 킬 기록 + 레벨별 부활 시간으로 채운다."""
    dt = Rules().combat["death_timer_s"]
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        mid = os.path.basename(path)[:-5]
        frames = d["timeline"]["info"]["frames"]
        kills = [e for fr in frames for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
        for p in d["detail"]["info"]["participants"]:
            if p.get("teamPosition") != "JUNGLE":
                continue
            f = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
            if not os.path.exists(f):
                continue
            with open(f, encoding="utf-8") as fh:
                pos = {round(float(r["time"])): (float(r["x"]), float(r["y"]))
                       for r in csv.DictReader(fh) if r.get("valid", "1") == "1"}
            pid = p["participantId"]
            for k in kills:
                if k["victimId"] == pid:
                    t = k["timestamp"] / 1000
                    lv = frames[min(int(t // 60), len(frames) - 1)]["participantFrames"][str(pid)]["level"]
                    for s in range(math.ceil(t), math.ceil(t + dt[lv - 1])):
                        pos[s] = None
            out.append((mid, p["teamId"], pid, pos))
    return out


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
    while g.t < T1:
        g.step()
        if g.t >= T0:
            for c in (ai, rule):
                xy = _xy(g, c)
                pos[c.pid][g.t] = None if xy[0] is None else xy
    herald = g.objectives["herald"]["spawn_at"] if g.objectives["herald"]["spawn_at"] is not None else 900
    return {"ai": segments(pos[ai.pid], ai.team, g.map.nodes, herald),
            "rule": segments(pos[rule.pid], rule.team, g.map.nodes, herald)}


def share(seg_list):
    c = collections.Counter(group(a) for s in seg_list for a in s.values())
    n = sum(c.values()) or 1
    return {k: c[k] / n for k in GROUPS}, collections.Counter(a for s in seg_list for a in s.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage6c", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "real_actions.md"))
    args = ap.parse_args()
    nodes = Game(seed=0).map.nodes
    herald = Game(seed=0).objectives["herald"]["spawn_at"]
    real = []
    for mid, team, pid, pos in real_tracks():
        if sum(1 for t in range(T0, T1) if t in pos) >= 0.8 * (T1 - T0):
            real.append(segments(pos, team, nodes, herald))
    with Pool(args.workers, initializer=_init, initargs=(args.policy,)) as pool:
        sims = pool.map(_sim_one, range(args.games))
    rs, rc = share(real)
    ais, ac = share([s["ai"] for s in sims])
    bs, bc = share([s["rule"] for s in sims])
    lines = ["# 1초 동선으로 본 정글러 행동 시간 비율 (3~18분)", "",
             f"- 실제 정글러 {len(real)}명(리플레이), 시뮬 {args.games}판: 학습 정글러 `{os.path.relpath(args.policy, ROOT)}`, 상대 규칙 봇.",
             "- 머문 곳으로 행동을 정하고, 이동 중인 초는 다음에 머문 곳의 행동으로 채움(`sim/real_actions.py`).", "",
             "| 행동 | 실제 | 학습 정글러 | 규칙 봇 |", "|---|---|---|---|"]
    for k in GROUPS:
        lines.append(f"| {GROUP_KO[k]} | {rs[k]:.0%} | {ais[k]:.0%} | {bs[k]:.0%} |")
    lines += ["", "## 세부 행동 (실제 상위 20)", "", "| 행동 | 실제 | 학습 정글러 | 규칙 봇 |", "|---|---|---|---|"]
    tr, ta, tb = sum(rc.values()), sum(ac.values()), sum(bc.values())
    for a, v in rc.most_common(20):
        lines.append(f"| {a} | {v / tr:.1%} | {ac[a] / ta:.1%} | {bc[a] / tb:.1%} |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
