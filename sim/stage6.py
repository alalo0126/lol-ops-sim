"""6단계: 같은 상황에서 학습 정글러와 천상계 정글러의 선택 비교.

흐름
1. 실제 경기(1단계 1,000판, tools/stage6_compact.py로 줄인 파일)의 분 프레임마다 정글러가 '다음 1분 동안 무엇을
   했는지'를 운영 범주 하나로 붙인다(label_real).
2. 같은 시각의 실제 상태(위치·체력·레벨·경험치·골드·CS·킬·오브젝트)를 시뮬레이터 상태로 옮기고(reconstruct),
   학습 정글러를 1분 돌려 같은 함수로 범주를 붙인다(label_sim). 두 쪽 모두 '위치와 이벤트'만 보고 범주를 정하므로
   정의 차이가 비교에 섞이지 않는다.
3. 범주 분포가 다른 상황에서 반사실 실험: 같은 상태에서 첫 1분을 각 범주로 묶은 학습 정글러를 20분까지 돌려
   20분 예측 승률(Φ)을 비교한다(counterfactual). 범주 밖 행동은 그 1분 동안만 가린다.

실제 프레임은 60초 간격이고 일반 캠프 처치·와드 위치·점멸·웨이브가 없다. 그래서 상태 옮기기에는 추정이 들어간다
(캠프 타이머는 정글 CS 증가와 위치로 추정, 점멸은 모두 있음, 웨이브 0.5, 와드 없음). 이 추정은 '조건 차이'로 다룬다.
"""
import gzip
import json
import math
import os

import numpy as np

from .bots import JunglerBot
from .agent import AgentJungler, A_CAMP, A_INVADE, A_SCUTTLE, A_GANK, A_COVER, A_OBJ, A_RECALL, A_FARM, A_WAIT, A_MOVE, N_MAIN
from .data import ROOT, Rules
from .game import Game, SIDE, ROLES
from .winprob import WinProb

MAP_MAX = 14870
CATS = ["farm", "invade", "lane", "objective", "river", "recall"]
CAT_KO = {"farm": "우리 정글", "invade": "카정", "lane": "라인 개입(갱·합류)", "objective": "오브젝트",
          "river": "강가·이동", "recall": "귀환"}

# 범주 → 학습 정글러 주 행동(반사실 실험에서 첫 1분 동안 허용하는 행동)
CAT_ACTIONS = {
    "farm": list(range(A_CAMP, A_CAMP + 6)) + [A_SCUTTLE, A_SCUTTLE + 1] + [A_MOVE, A_MOVE + 1],
    "invade": list(range(A_INVADE, A_INVADE + 6)) + [A_MOVE + 2, A_MOVE + 3],
    "lane": list(range(A_GANK, A_GANK + 3)) + list(range(A_COVER, A_COVER + 3)) + list(range(A_FARM, A_FARM + 3)),
    "objective": list(range(A_OBJ, A_OBJ + 3)) + [A_MOVE + 6, A_MOVE + 7],
    "river": [A_MOVE + 4, A_MOVE + 5, A_WAIT],
    "recall": [A_RECALL],
}
ELITE_KEY = {"DRAGON": "dragon", "HORDE": "grubs", "RIFTHERALD": "herald"}
ROLE_OF = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM", "UTILITY": "UTILITY"}


# ---------------------------------------------------------------------- 구역
def zone(x, y, team, nodes):
    """맵 좌표 → 이 팀 기준 구역: base / lane / pit / own_jungle / enemy_jungle / river.

    라인은 좌표 띠로, 나머지는 가장 가까운 비라인 노드의 종류와 진영으로 정한다. 실제 위치와 시뮬레이터 노드 좌표에
    같은 함수를 쓴다(시뮬레이터 라인·포탑 노드는 모두 라인 띠 안에 들어가도록 경계를 잡음).
    """
    rx, ry = MAP_MAX - x, MAP_MAX - y     # 점대칭: 아래 라인 조건 = 위 라인 조건의 반사
    if (x < 4200 and y < 4200) or (rx < 4200 and ry < 4200):
        return "base"
    top = (x < 1900 and y > 4500) or (y > 12900 and x < 10000) or (x < 3300 and y > 11500)
    bot = (rx < 1900 and ry > 4500) or (ry > 12900 and rx < 10000) or (rx < 3300 and ry > 11500)
    mid = abs(x - y) < 1000 and 7000 < x + y < 2 * MAP_MAX - 7000
    if top or bot or mid:
        return "lane"
    best, bd = None, 1e18
    for nid, n in nodes.items():
        if n["kind"] in ("lane", "tower", "fountain"):
            continue
        d = (n["x"] - x) ** 2 + (n["y"] - y) ** 2
        if d < bd:
            best, bd = n, d
    if best["kind"] == "epic":
        return "pit" if bd < 1500 ** 2 else "river"
    if best["kind"] in ("river", "scuttle"):
        return "river"
    return "own_jungle" if best["side"] == team else "enemy_jungle"


def nearest_node(x, y, nodes, kinds=None):
    return min((nid for nid, n in nodes.items() if kinds is None or n["kind"] in kinds),
               key=lambda nid: (nodes[nid]["x"] - x) ** 2 + (nodes[nid]["y"] - y) ** 2)


def label(window, team, nodes):
    """다음 1분의 기록 → 운영 범주. window(실제·시뮬 공통 형식):

    end_xy: 1분 뒤 위치(죽어 있으면 None), cs_gain: 정글 CS 증가, elite: 정글러가 관여한 에픽 몬스터 처치 수,
    fights: 정글러가 관여한(킬·어시스트·사망) 챔피언 처치의 위치 목록, recalled: 살아서 우물에서 구매했는지,
    obj_live: 1분 동안 살아 있던(또는 생성된) 에픽 몬스터가 있었는지.
    우선순위: 오브젝트 > 교전 위치 > 귀환 > 1분 뒤 위치.
    """
    if window["elite"]:
        return "objective"
    for x, y in window["fights"]:
        z = zone(x, y, team, nodes)
        return {"lane": "lane", "enemy_jungle": "invade", "own_jungle": "farm", "base": "lane",
                "pit": "objective" if window["obj_live"] else "river", "river": "river"}[z]
    if window["recalled"]:
        return "recall"
    if window["end_xy"] is None:
        return "river"
    z = zone(*window["end_xy"], team, nodes)
    if z == "base":
        return "recall"
    if z == "pit":
        return "objective" if window["obj_live"] else "river"
    if z == "river":
        return "farm" if window["cs_gain"] >= 4 and window.get("scuttle_near") else "river"
    return {"lane": "lane", "own_jungle": "farm", "enemy_jungle": "invade"}[z]


# ---------------------------------------------------------------------- 실제 경기
def load_compact(path=None):
    path = path or os.path.join(ROOT, "data", "stage6_compact.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class RealMatch:
    """줄인 경기 한 판을 다루기 쉽게 감싼다."""

    def __init__(self, m):
        self.m = m
        self.team_of = {p[0]: p[1] for p in m["parts"]}
        self.role_of = {p[0]: p[2] for p in m["parts"]}
        self.champ_of = {p[0]: p[3] for p in m["parts"]}
        self.pid = {(p[1], p[2]): p[0] for p in m["parts"]}
        self.frames = m["frames"]
        ev = m["events"]
        self.kills = ev["kill"]
        self.elite = ev["elite"]
        self.buys = ev["buy"]
        self.valid = len(self.pid) == 10 and all((t, r) in self.pid for t in (100, 200) for r in ROLES)

    def pf(self, minute, pid):
        x, y, hp, hpmax, lv, xp, tg, cg, jcs, mcs, ms = self.frames[minute][str(pid)]
        return {"x": x, "y": y, "hp": hp / max(1, hpmax), "level": lv, "xp": xp, "total_gold": tg, "current_gold": cg,
                "jcs": jcs, "mcs": mcs, "ms": ms}

    def jungler(self, team):
        return self.pid[(team, "JUNGLE")]

    def diffs(self, minute):
        t = minute * 60000
        d = {"gold": 0.0, "xp": 0.0, "dragons": 0, "grubs": 0, "heralds": 0}
        for pid in self.team_of:
            s = 1 if self.team_of[pid] == 100 else -1
            f = self.pf(minute, pid)
            d["gold"] += s * f["total_gold"]
            d["xp"] += s * f["xp"]
        for e in self.elite:
            k = ELITE_KEY.get(e[1])
            if k and e[0] <= t:
                d[{"dragon": "dragons", "grubs": "grubs", "herald": "heralds"}[k]] += 1 if e[3] == 100 else -1
        return d

    def window(self, team, minute, nodes):
        """실제 정글러의 minute분 → minute+1분 기록(label 입력 형식)."""
        j = self.jungler(team)
        t0, t1 = minute * 60000, (minute + 1) * 60000
        a, b = self.pf(minute, j), self.pf(minute + 1, j)
        fights, died, kills_for = [], False, 0
        for t, killer, victim, assists, x, y in self.kills:
            if t0 < t <= t1 and (j == killer or j == victim or j in assists) and x is not None:
                fights.append((x, y))
                died = died or j == victim
                kills_for += j != victim
        elite = sum(1 for e in self.elite if t0 < e[0] <= t1 and e[3] == team and (e[4] == j or j in e[5]))
        recalled = not died and any(t0 < t <= t1 and pid == j for t, pid, _ in self.buys)
        obj_live = any(self.objective_live(k, t0, t1) for k in ("dragon", "grubs", "herald"))
        scuttle_near = min(math.dist((b["x"], b["y"]), (n["x"], n["y"])) for n in
                           (nodes["river_top_scuttle"], nodes["river_bot_scuttle"])) < 1500
        return {"end_xy": (b["x"], b["y"]), "cs_gain": b["jcs"] - a["jcs"], "elite": elite, "fights": fights,
                "recalled": recalled, "obj_live": obj_live, "scuttle_near": scuttle_near, "died": died, "kills_for": kills_for}

    def objective_live(self, name, t0, t1):
        """실제 경기에서 그 오브젝트가 [t0, t1] 사이에 살아 있었는지(규칙 데이터의 생성 시각 + 실제 처치 기록)."""
        o = OBJ_RULES[name]
        kills = sorted(e[0] for e in self.elite if ELITE_KEY.get(e[1]) == name)
        if name == "dragon":
            spawn = o["first"] * 1000
            for k in kills:
                if k <= t0:
                    spawn = k + o["respawn"] * 1000
            return spawn <= t1
        n_needed = 3 if name == "grubs" else 1
        if len([k for k in kills if k <= t0]) >= n_needed:
            return False
        return o["first"] * 1000 <= t1 and t0 < o["until"] * 1000


OBJ_RULES = {}


def _obj_rules(rules):
    """규칙 데이터의 오브젝트 생성·리스폰 시각(실제 경기 쪽 판정에도 같은 값을 씀)."""
    o = rules.objectives
    OBJ_RULES.update({
        "dragon": {"first": o["drake"]["first_spawn_s"], "respawn": o["drake"]["respawn_s"]},
        "grubs": {"first": o["voidgrubs"]["spawn_s"], "until": o["voidgrubs"]["despawn_s"]},
        "herald": {"first": o["rift_herald"]["spawn_s"], "until": o["rift_herald"]["despawn_s"]},
    })


_obj_rules(Rules())


# ---------------------------------------------------------------------- 상황 구분
def situation(rm, team, minute):
    """비교 단위가 되는 상황: 시기 × 드래곤 상태 × 팀 골드 차 × 정글러 체력·쓸 골드."""
    j = rm.jungler(team)
    f = rm.pf(minute, j)
    t0, t1 = minute * 60000, (minute + 1) * 60000
    phase = "3-5분" if minute <= 5 else "6-9분" if minute <= 9 else "10-13분" if minute <= 13 else "14-17분"
    drag = "드래곤 있음" if rm.objective_live("dragon", t0, t0) else ("드래곤 1분 안 생성" if rm.objective_live("dragon", t0, t1) else "드래곤 없음")
    top = rm.objective_live("grubs", t0, t1) or rm.objective_live("herald", t0, t1)
    d = rm.diffs(minute)["gold"] * (1 if team == 100 else -1)
    gold = "골드 앞섬" if d > 1500 else "골드 뒤짐" if d < -1500 else "골드 비슷"
    need = "체력 낮음" if f["hp"] < 0.4 else "골드 쌓임" if f["current_gold"] >= 1300 else "보통"
    return {"phase": phase, "dragon": drag, "top_obj": "윗쪽 오브젝트 있음" if top else "윗쪽 오브젝트 없음",
            "gold": gold, "need": need}


# ---------------------------------------------------------------------- 상태 옮기기
def _assign_camps(rm, team, minute, g, shift_s=0.0):
    """두 팀 정글러의 정글 CS 증가(1분 단위)와 위치로 각 캠프의 마지막 처치 시각을 추정해 캠프 상태를 정한다.

    1분 동안 CS가 4 늘 때마다 캠프 하나를 그 1분의 두 위치에 가장 가까운(지금 살아 있다고 추정되는) 캠프에 붙인다.
    """
    nodes = g.map.nodes
    camps = g.camps
    last_kill = {cid: None for cid in camps}
    respawn = {cid: g.rules.camps[c.kind]["respawn_s"] for cid, c in camps.items()}
    first = {cid: g.rules.camps[c.kind]["first_spawn_s"] for cid, c in camps.items()}
    for k in range(1, minute + 1):
        t_mid = (k - 0.5) * 60
        for tm in (100, 200):
            j = rm.jungler(tm)
            a, b = rm.pf(k - 1, j), rm.pf(k, j)
            n = max(0, round((b["jcs"] - a["jcs"]) / 4))
            if n == 0:
                continue

            def alive_at(cid, t):
                lk = last_kill[cid]
                return t >= first[cid] - 10 and (lk is None or t >= lk + respawn[cid])
            cands = [cid for cid in camps if alive_at(cid, k * 60)]
            cands.sort(key=lambda cid: min(math.dist((a["x"], a["y"]), (nodes[cid]["x"], nodes[cid]["y"])),
                                           math.dist((b["x"], b["y"]), (nodes[cid]["x"], nodes[cid]["y"]))))
            for i, cid in enumerate(cands[:n]):
                last_kill[cid] = (k - 1) * 60 + 60 * (i + 1) / (n + 1)
    t = minute * 60
    for cid, camp in camps.items():
        lk = last_kill[cid]
        if lk is not None:
            lk -= shift_s          # 민감도: 처치 시각을 이만큼 이르게(캠프가 더 빨리 다시 생김)
        if lk is None:
            camp.alive = t >= first[cid]
            camp.spawn_at = first[cid]
            camp.first = True
        else:
            camp.first = False
            camp.spawn_at = lk + respawn[cid]
            camp.alive = t >= camp.spawn_at
        if camp.alive:
            camp.level = max(1, round(g.avg_level()))
    return last_kill


def reconstruct(rm, team, minute, policy_fn, wp, seed=0, rules=None, constrain=None, camp_shift_s=0.0):
    """실제 경기 minute분 상태를 시뮬레이터로 옮긴 Game과 그 팀 학습 정글러를 돌려준다.

    constrain = (범주, 끝 시각): 끝 시각 전까지, 그 범주 행동을 한 번 끝낼 때까지 학습 정글러의 주 행동을 그 범주로 묶는다(반사실 실험).
    """
    holder = {}

    def make(g, c):
        if policy_fn is None:                 # 규칙 봇 정글러(기준선)
            holder["bot"] = JunglerBot(g, c)
            return holder["bot"]
        fn = policy_fn
        if constrain is not None:
            allowed = np.zeros(N_MAIN, dtype=bool)
            allowed[CAT_ACTIONS[constrain[0]]] = True

            state = {"started": False, "free": False}

            def fn(o, m, mw, ms, _base=policy_fn):
                # 범주 행동을 한 번 끝내면(작업이 비었을 때 다시 고르면) 풀어 준다. 1분 동안 같은 범주를 반복하게
                # 묶으면 귀환 뒤 우물에서 귀환만 되풀이하는 식의 낭비가 생긴다.
                if state["started"] and holder["bot"].c.task is None:
                    state["free"] = True
                if not state["free"] and g.t < constrain[1] and (m & allowed).any():
                    m = m & allowed
                a, w, s, rec = _base(o, m, mw, ms)
                state["started"] = state["started"] or bool(allowed[a])
                return a, w, s, rec
        holder["bot"] = AgentJungler(g, c, fn, wp)
        holder["bot"].cstate = state if constrain is not None else None
        return holder["bot"]

    g = Game(seed=seed, rules=rules, jungler_override={team: make})
    t = minute * 60
    g.t = t
    t_ms = t * 1000
    nodes = g.map.nodes
    cb = g.rules.combat
    g.first_blood = any(k[0] <= t_ms for k in rm.kills)
    for c in g.champs:
        pid = rm.pid[(c.team, c.role)]
        f = rm.pf(minute, pid)
        c.level, c.xp = f["level"], float(f["xp"])
        c.earned = float(f["total_gold"] - 500)
        c.spent = float(f["total_gold"] - f["current_gold"])
        c.item_gold = c.spent - (g.rules.positions["jungle_item"]["cost"] if c.role == "JUNGLE" else 0)
        c.item_gold = max(0.0, c.item_gold)
        c.hp = max(0.05, min(1.0, f["hp"]))
        c.lane_cs = float(f["mcs"])
        c.jungle_cs = int(f["jcs"])
        c.kills = sum(1 for k in rm.kills if k[0] <= t_ms and k[1] == pid)
        c.deaths = sum(1 for k in rm.kills if k[0] <= t_ms and k[2] == pid)
        c.assists = sum(1 for k in rm.kills if k[0] <= t_ms and pid in k[3])
        c.path, c.task, c.edge_left, c.busy_until = [], None, 0.0, 0.0
        c.exposures, c.pull_credit, c.homeguard_left = [], 0.0, 0.0
        c.flash_at = 0.0
        last_death = max((k[0] for k in rm.kills if k[0] <= t_ms and k[2] == pid), default=None)
        dead_for = 0.0
        if last_death is not None:
            lv_then = rm.pf(min(minute, int(last_death // 60000)), pid)["level"]
            dead_for = last_death / 1000 + cb["death_timer_s"][lv_then - 1] - t
        if dead_for > 0:
            c.dead_until = t + dead_for
            c.node = f"{SIDE[c.team]}_fountain"
            c.respawned = True
        else:
            c.dead_until = -1.0
            z = zone(f["x"], f["y"], c.team, nodes)
            if z == "base" and math.dist((f["x"], f["y"]), (nodes[f"{SIDE[c.team]}_fountain"]["x"], nodes[f"{SIDE[c.team]}_fountain"]["y"])) < 4000:
                c.node = f"{SIDE[c.team]}_fountain"
            elif c.role == "JUNGLE":
                c.node = nearest_node(f["x"], f["y"], nodes)
            else:
                # 라이너는 자기 라인의 중앙 또는 자기 포탑 중 가까운 쪽(라이너 봇이 움직이는 두 노드)
                lane = c.lane
                cands = [g.lane_node(lane), g.tower_node(lane, c.team)]
                c.node = min(cands, key=lambda n: math.dist((f["x"], f["y"]), (nodes[n]["x"], nodes[n]["y"])))
        if c.role == "JUNGLE":
            champ = rm.champ_of[pid]
            if champ in g.rules.champions:
                c.champion = champ
            c.ms = max(300.0, float(f["ms"]) or c.ms)
            c.route = []
            c.first_clear_done_s = c.first_clear_done_s or 0.0
            buys = [b[0] / 1000 for b in rm.buys if b[1] == pid and 90000 < b[0] <= t_ms]
            c.first_recall_s = min(buys) if buys else (None if minute < 4 else t)
            n_large = c.jungle_cs // 4
            pj = g.p["jungler"]
            tr = g.rules.positions["jungle_item"]["treats"]
            bonus = max(0, int((t - pj["first_bonus_treat_s"]) // tr["bonus_treat_every_s"]) + 1) if t >= pj["first_bonus_treat_s"] else 0
            c.treats = n_large + bonus
            c.big_kills = n_large
            c.next_bonus_treat = t + tr["bonus_treat_every_s"] / 2
            c.monster_gold = 30.0 * c.jungle_cs
            c.minion_gold = 0.0
            # 강타: 1충전 + 다음 충전 45초 뒤로 가정(실제 사용 기록 없음)
            c.smite_charges, c.smite_next = 1, t + 45
    _assign_camps(rm, team, minute, g, camp_shift_s)
    # 오브젝트
    t0 = t_ms
    for name, ob in g.objectives.items():
        kills = [e for e in rm.elite if ELITE_KEY.get(e[1]) == name and e[0] <= t0]
        for tm in (100, 200):
            n = sum(1 for e in kills if e[3] == tm)
            g.obj_taken[name][tm] = n / 3 if name == "grubs" else n
        o = OBJ_RULES[name]
        if name == "dragon":
            last = max((e[0] for e in kills), default=None)
            spawn = o["first"] if last is None else last / 1000 + o["respawn"]
            ob["alive"], ob["spawn_at"] = (True, None) if spawn <= t else (False, spawn)
        else:
            done = len(kills) >= (3 if name == "grubs" else 1)
            if done or t >= o["until"]:
                ob["alive"], ob["spawn_at"] = False, None
            elif t >= o["first"]:
                ob["alive"], ob["spawn_at"] = True, None
            else:
                ob["alive"], ob["spawn_at"] = False, o["first"]
    # 적 정글러를 마지막으로 본 곳: 지금 또는 1분 전 실제 위치가 그 팀 시야 노드였으면
    for tm in (100, 200):
        ej = g.enemy(next(x for x in g.champs if x.team == tm), "JUNGLE")
        vis = g.vision(tm)
        g.enemy_jgl_seen[tm] = (None, -999.0)
        if ej.alive(t) and ej.node in vis:
            g.enemy_jgl_seen[tm] = (ej.node, float(t))
        else:
            prev = rm.pf(minute - 1, rm.jungler(300 - tm))
            pn = nearest_node(prev["x"], prev["y"], nodes)
            if pn in vis:
                g.enemy_jgl_seen[tm] = (pn, float(t - 60))
    # 첫 결정에서 시작 승률을 잡도록 비워 둠
    bot = holder["bot"]
    return g, bot


def sim_window(g, team, t0, t1, cs0):
    """시뮬레이터에서 [t0, t1] 동안의 정글러 기록(label 입력 형식)."""
    j = next(c for c in g.champs if c.team == team and c.role == "JUNGLE")
    nodes = g.map.nodes
    fights, died, kills_for = [], False, 0
    for e in g.events:
        if not (t0 < e["t"] <= t1):
            continue
        if e["type"] == "kill" and (j.pid in (e["killer"], e["victim"]) or j.pid in e["assists"]):
            n = nodes[e["node"]]
            fights.append((n["x"], n["y"]))
            died = died or e["victim"] == j.pid
            kills_for += e["victim"] != j.pid
    elite = sum(1 for e in g.events if t0 < e["t"] <= t1 and e["type"] == "objective" and e["team"] == team
                and (e["pid"] == j.pid or nodes[j.node]["kind"] == "epic"))
    recalled = not died and any(t0 < e["t"] <= t1 and e["type"] == "recall" and e["pid"] == j.pid for e in g.events)
    end = None if not j.alive(g.t) else (nodes[j.node]["x"], nodes[j.node]["y"])
    return {"end_xy": end, "cs_gain": j.jungle_cs - cs0, "elite": elite, "fights": fights, "recalled": recalled,
            "obj_live": None, "scuttle_near": j.node in ("river_top_scuttle", "river_bot_scuttle"),
            "died": died, "kills_for": kills_for}


def run_sim_minute(rm, team, minute, policy_fn, wp, seed, rules=None, with_window=False, camp_shift_s=0.0):
    """실제 상태에서 학습 정글러를 1분 돌려 범주를 붙인다. 오브젝트 생존 여부는 실제 경기 기준(같은 상황)."""
    g, bot = reconstruct(rm, team, minute, policy_fn, wp, seed=seed, rules=rules, camp_shift_s=camp_shift_s)
    j = bot.c
    cs0, t0, g0, xp0 = j.jungle_cs, g.t, j.total_gold, j.xp
    while g.t < t0 + 60:
        g.step()
    w = sim_window(g, team, t0, g.t, cs0)
    w["obj_live"] = any(rm.objective_live(k, t0 * 1000, (t0 + 60) * 1000) for k in ("dragon", "grubs", "herald"))
    gains = {"gold": j.total_gold - g0, "jcs": j.jungle_cs - cs0, "xp": j.xp - xp0}
    if with_window:
        return label(w, team, g.map.nodes), gains, w
    return label(w, team, g.map.nodes), gains


def rollout_value(rm, team, minute, policy_fn, wp, seed, cat=None, rules=None, camp_shift_s=0.0):
    """실제 상태에서 20분까지 돌린 그 팀의 20분 예측 승률. cat이면 첫 1분을 그 범주로 묶는다.

    돌려주는 것: (Φ20, 첫 1분 범주(cat이면 [범주, 묶은 행동을 시작했나]), 시작 Φ)
    """
    g, bot = reconstruct(rm, team, minute, policy_fn, wp, seed=seed, rules=rules,
                         constrain=(cat, minute * 60 + 60) if cat else None, camp_shift_s=camp_shift_s)
    j = bot.c
    phi0 = wp.for_team(g, team)
    cs0, t0 = j.jungle_cs, g.t
    lab = None
    while g.t < 1200:
        g.step()
        if g.t == t0 + 60:
            w = sim_window(g, team, t0, g.t, cs0)
            w["obj_live"] = any(rm.objective_live(k, t0 * 1000, (t0 + 60) * 1000) for k in ("dragon", "grubs", "herald"))
            lab = label(w, team, g.map.nodes)
            if cat:
                lab = [lab, bool(bot.cstate["started"])]   # [1분 범주, 묶은 범주 행동을 실제로 시작했나]
    return wp.for_team(g, team), lab, phi0
