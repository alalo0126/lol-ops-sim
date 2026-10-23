"""5단계 학습 정글러: 3단계 행동 목록을 그대로 행동 공간으로 쓰는 봇.

결정 시점(3단계 확정): 행동이 끝났을 때, 이동·대기 중 5초마다(같은 행동을 고르면 계속), 새 정보가 들어왔을 때
(적 정글러 발견·킬·오브젝트 생성, 반응 지연 반영). 결정마다 (주 행동, 와드, 강타) 세 가지를 함께 고른다.

주 행동(36개, 팀 기준 '아군/적' + 라인은 맵 기준 위/중/아래)
  0~5   캠프 처치: 아군 블루·두꺼비·늑대·칼날부리·레드·돌거북
  6~11  카정: 같은 순서의 적 캠프(살아 있는지는 모르는 채로 감)
  12~13 바위게 위/아래
  14~16 갱킹 탑/미드/봇
  17~19 카운터 갱·합류: 라인에 가서 cover_hold_s 동안 머무름(적 갱이 오면 방어에 낌)
  20~22 오브젝트 드래곤/유충/전령
  23    귀환(우물에서 고르면 채널링 후 구매만 함: 부활 뒤 구매 수단)
  24~26 라인 미니언 먹기 탑/미드/봇
  27    대기(wait_s)
  28~35 이동: 아군 위/아래 정글 입구, 적 위/아래 정글 입구, 위/아래 강가 수풀, 용 둥지, 바론 둥지
부가 행동(와드): 없음 / 지금 노드 / 다음 도착 노드.  강타: 사용 / 아낌(캠프·카정·바위게일 때만 의미).
마스크는 '불가능한 행동'만 가린다(3단계 원칙: 규칙 대신 비용).
"""
import numpy as np

from .bots import JunglerBot
from .game import SIDE, ROLES

CAMP_KINDS = ["blue", "gromp", "wolves", "raptors", "red", "krugs"]
LANES = ["top", "mid", "bot"]
LANE_ROLE = {"top": "TOP", "mid": "MIDDLE", "bot": "BOTTOM"}
OBJS = ["dragon", "grubs", "herald"]
TASK_KINDS = ["none", "camp", "gank", "cover", "objective", "recalling", "lanefarm", "wait", "move", "clearing", "ganking", "other"]

A_CAMP, A_INVADE, A_SCUTTLE, A_GANK, A_COVER, A_OBJ, A_RECALL, A_FARM, A_WAIT, A_MOVE = 0, 6, 12, 14, 17, 20, 23, 24, 27, 28
N_MAIN, N_WARD, N_SMITE = 36, 3, 2
ACTION_NAMES = ([f"camp_{k}" for k in CAMP_KINDS] + [f"invade_{k}" for k in CAMP_KINDS] + ["scuttle_top", "scuttle_bot"]
                + [f"gank_{l}" for l in LANES] + [f"cover_{l}" for l in LANES] + [f"obj_{o}" for o in OBJS] + ["recall"]
                + [f"lanefarm_{l}" for l in LANES] + ["wait"]
                + ["move_own_top_entrance", "move_own_bot_entrance", "move_enemy_top_entrance", "move_enemy_bot_entrance",
                   "move_river_top_brush", "move_river_bot_brush", "move_dragon_pit", "move_baron_pit"])
assert len(ACTION_NAMES) == N_MAIN


def move_nodes(team):
    me, en = SIDE[team], SIDE[300 - team]
    return [f"{me}_top_entrance", f"{me}_bot_entrance", f"{en}_top_entrance", f"{en}_bot_entrance",
            "river_top_brush", "river_bot_brush", "dragon_pit", "baron_pit"]


class AgentJungler(JunglerBot):
    """policy(obs, mask_main, mask_ward, mask_smite) → (주 행동, 와드, 강타, 기록용 dict 또는 None)."""

    def __init__(self, game, champ, policy, winprob, keep_route=False):
        super().__init__(game, champ)
        self.policy, self.wp = policy, winprob
        c = champ
        if not keep_route:
            c.route = []                  # 정해진 동선 없이 처음부터 스스로 고른다(모방 교사만 규칙 봇 동선을 씀)
        self.extra_kw = {}                # 교사가 넘기는 작업 정보(동선 캠프 표시 등)
        pw = game.p["agent"]["ward"]
        c.ward_charges, c.ward_next = pw["start_charges"], pw["recharge_s"]
        self.team = c.team
        self.moves = move_nodes(c.team)
        self.cur_action = None
        self.smite_choice = 0
        self.ward_on_node = None
        self.seen_info = 0
        self.known = {}                   # 적 캠프·바위게 cid → (살아 있었나, 본 시각)
        self.traj = []                    # 결정마다 policy가 돌려준 기록 + 보상
        self.last_phi = None              # 첫 결정에서 잼(봇은 캠프·오브젝트보다 먼저 만들어짐)
        self.counts = np.zeros(N_MAIN, dtype=int)
        self.ward_count = 0
        self._node_index = {n: i for i, n in enumerate(sorted(game.map.nodes))}

    # -------------------------------------------------------------- 매 틱
    def tick(self):
        g, c = self.g, self.c
        pw = g.p["agent"]["ward"]
        if c.ward_charges < pw["max_charges"] and g.t >= c.ward_next:
            c.ward_charges += 1
            c.ward_next = g.t + pw["recharge_s"]
        if self.ward_on_node and c.node == self.ward_on_node:
            self.place_ward(c.node)
            self.ward_on_node = None
        task = c.task
        if task and task["kind"] == "lanefarm" and task.get("started") and g.t >= c.busy_until:
            self.finish_lanefarm(task["lane"])
            c.task = None
        if task and task["kind"] in ("clearing", "recalling") or g.t < c.busy_until:
            return
        task = c.task
        new_info = g.info_version[c.team] != self.seen_info
        if task and (c.path or task["kind"] in ("cover", "wait")):
            if task["kind"] in ("cover", "wait") and not c.path:
                if "until" not in task:
                    task["until"] = g.t + (g.p["agent"]["cover_hold_s"] if task["kind"] == "cover" else g.p["agent"]["wait_s"])
                if g.t >= task["until"]:
                    c.task = None
                    self.decide()
                    return
            if g.t - c.last_decision >= 5 or new_info:
                self.decide(midway=True)
            return
        if task:
            self.arrive(task)
            return
        self.decide()

    def arrive(self, task):
        g, c = self.g, self.c
        kind = task["kind"]
        if kind == "move":
            c.task = None
            self.decide()
        elif kind in ("cover", "wait"):
            task["until"] = g.t + (g.p["agent"]["cover_hold_s"] if kind == "cover" else g.p["agent"]["wait_s"])
        elif kind == "lanefarm":
            lane = task["lane"]
            dur = g.p["agent"]["lanefarm_s"]
            task["started"] = True
            c.busy_until = g.t + dur
            g.lane_taken_until[lane] = max(g.lane_taken_until[lane], g.t + dur)
        else:
            super().arrive(task)
            if c.task is None:
                self.decide()

    def go(self, kind, node, **kw):
        g, c = self.g, self.c
        c.task = {"kind": kind, "node": node, **kw, **self.extra_kw}
        self.extra_kw = {}
        c.last_decision = g.t
        if c.node != node or c.path:      # 엣지 중간이면 목적지가 지나온 노드여도 경로를 다시 잡아야 함
            g.move_to(c, node)

    def use_smite(self, camp):
        return self.c.smite_charges > 0 and self.smite_choice == 0

    # -------------------------------------------------------------- 결과
    def place_ward(self, node):
        g, c = self.g, self.c
        if c.ward_charges <= 0:
            return
        if c.ward_charges == g.p["agent"]["ward"]["max_charges"]:
            c.ward_next = g.t + g.p["agent"]["ward"]["recharge_s"]
        c.ward_charges -= 1
        g.wards.append({"team": c.team, "node": node, "until": g.t + g.p["agent"]["ward"]["duration_s"], "pid": c.pid})
        self.ward_count += 1

    def finish_lanefarm(self, lane):
        g, c, pa = self.g, self.c, self.g.p["agent"]
        role = LANE_ROLE[lane]
        inc = g.p["laner"]["income_per_s"][role]
        secs = pa["lanefarm_income_s"]
        gold, xp = inc["gold"] * secs, inc["xp"] * secs
        cs = g.p["laner"]["cs_per_s"][role] * secs
        pen = pa["pet_lane_xp_penalty"]
        xp *= 1 - pen["start"] * max(0.0, 1 - g.t / pen["zero_at_s"])
        mh = pa["monster_hunter"]
        if g.t < mh["until_s"] and c.minion_gold + gold > mh["minion_to_monster_ratio"] * c.monster_gold:
            gold = max(0.0, gold + mh["gold_per_minion"] * cs)
            xp *= mh["xp_mult"]
        c.minion_gold += gold
        c.lane_cs += cs
        g.give_gold(c, gold, gv=True)
        g.give_xp(c, xp)
        g.events.append({"t": g.t, "type": "lanefarm", "pid": c.pid, "lane": lane, "gold": round(gold)})

    # -------------------------------------------------------------- 결정
    def action_target(self, a):
        g, c = self.g, self.c
        me, en = SIDE[c.team], SIDE[300 - c.team]
        if a < A_INVADE:
            return f"{me}_{CAMP_KINDS[a - A_CAMP]}"
        if a < A_SCUTTLE:
            return f"{en}_{CAMP_KINDS[a - A_INVADE]}"
        if a < A_GANK:
            return ["river_top_scuttle", "river_bot_scuttle"][a - A_SCUTTLE]
        if a < A_OBJ:
            return g.lane_node(LANES[(a - A_GANK) % 3])
        if a < A_RECALL:
            return g.objectives[OBJS[a - A_OBJ]]["node"]
        if a == A_RECALL:
            return f"{me}_fountain"
        if a < A_WAIT:
            return g.lane_node(LANES[a - A_FARM])
        if a == A_WAIT:
            return c.node
        return self.moves[a - A_MOVE]

    def masks(self):
        g, c = self.g, self.c
        m = np.ones(N_MAIN, dtype=bool)
        for i, lane in enumerate(LANES):
            foes = [e for e in g.champs if e.team != c.team and e.role != "JUNGLE" and e.lane == lane and e.alive(g.t)]
            if not foes:
                m[A_GANK + i] = False
            if g.t < g.p["laner"]["lane_start_s"]:
                m[A_FARM + i] = False
                m[A_GANK + i] = False
                m[A_COVER + i] = False
        for i, o in enumerate(OBJS):
            if not g.objectives[o]["alive"]:
                m[A_OBJ + i] = False
        for i, n in enumerate(self.moves):
            if n == c.node:
                m[A_MOVE + i] = False
        mw = np.array([True, c.ward_charges > 0, c.ward_charges > 0])
        ms = np.array([c.smite_charges > 0, True])
        return m, mw, ms

    def decide(self, midway=False):
        g, c = self.g, self.c
        self.seen_info = g.info_version[c.team]
        c.last_decision = g.t
        phi = self.wp.for_team(g, c.team)
        if self.traj:
            self.traj[-1]["r"] += phi - self.last_phi
        if self.last_phi is None:
            self.start_phi = phi
        self.last_phi = phi
        obs = self.observe()
        m, mw, ms = self.masks()
        if midway and self.cur_action is not None and not m[self.cur_action]:
            m_cont = m
        else:
            m_cont = m.copy()
            if midway and self.cur_action is not None:
                m_cont[self.cur_action] = True       # 지금 하던 행동을 계속하는 선택은 항상 가능
        a, w, s, rec = self.policy(obs, m_cont, mw, ms)
        if rec is not None:
            rec["r"] = 0.0
            self.traj.append(rec)
        self.counts[a] += 1
        if w == 1:
            self.place_ward(c.node)
        self.smite_choice = s
        if midway and a == self.cur_action:
            if w == 2 and c.path:
                self.ward_on_node = c.path[0]
            return
        self.ward_on_node = None
        self.cur_action = a
        self.execute(a)
        if w == 2 and c.path:
            self.ward_on_node = c.path[0]

    def execute(self, a):
        g, c = self.g, self.c
        node = self.action_target(a)
        if a < A_GANK:
            self.go("camp", node, camp=node, invade=A_INVADE <= a < A_SCUTTLE)
        elif a < A_COVER:
            self.go("gank", node, lane=LANES[a - A_GANK])
        elif a < A_OBJ:
            self.go("cover", node, lane=LANES[a - A_COVER])
        elif a < A_RECALL:
            self.go("objective", node, name=OBJS[a - A_OBJ])
        elif a == A_RECALL:
            g.start_recall(c)
        elif a < A_WAIT:
            self.go("lanefarm", node, lane=LANES[a - A_FARM])
        elif a == A_WAIT:
            c.path = []      # 엣지 중간이면 지나온 노드에서 기다린 것으로 봄
            c.task = {"kind": "wait", "node": c.node, "until": g.t + g.p["agent"]["wait_s"]}
        else:
            self.go("move", node)

    def finish_episode(self):
        """경기 끝(20:00): 마지막 결정 뒤 승률 변화를 보상에 더한다. 보상 합 = Φ(20분) − Φ(시작)."""
        phi = self.wp.for_team(self.g, self.c.team)
        if self.traj:
            self.traj[-1]["r"] += phi - self.last_phi
        self.last_phi = phi
        return phi

    # -------------------------------------------------------------- 관측
    def observe(self):
        g, c = self.g, self.c
        T = c.team
        vis = g.vision(T)
        f = [g.t / 1200, 1.0 if T == 100 else 0.0]
        # 자신
        f += [c.hp, c.level / 18, c.unspent / 3000, c.item_gold / 5000, c.smite_charges / 2,
              max(0.0, c.smite_next - g.t) / 90, min(c.treats, 40) / 40, c.ward_charges / 2,
              1.0 if c.flash_ready(g.t) else 0.0, c.ms / 400, g.map.nodes[c.node]["x"] / 15000, g.map.nodes[c.node]["y"] / 15000,
              (c.path and 1.0) or 0.0, c.jungle_cs / 100]
        node_oh = np.zeros(len(self._node_index))
        node_oh[self._node_index[c.node]] = 1
        dest = c.path[-1] if c.path else c.node
        dest_oh = np.zeros(len(self._node_index))
        dest_oh[self._node_index[dest]] = 1
        task_oh = np.zeros(len(TASK_KINDS))
        kind = c.task["kind"] if c.task else "none"
        task_oh[TASK_KINDS.index(kind) if kind in TASK_KINDS else TASK_KINDS.index("other")] = 1
        cur_oh = np.zeros(N_MAIN)
        if self.cur_action is not None and c.task:
            cur_oh[self.cur_action] = 1
        # 아군 캠프(타이머는 안다고 봄)
        me, en = SIDE[T], SIDE[300 - T]
        for k in CAMP_KINDS:
            camp = g.camps[f"{me}_{k}"]
            f += [1.0 if camp.alive else 0.0, 0.0 if camp.alive else min(1.0, max(0.0, camp.spawn_at - g.t) / 150),
                  min(1.0, self.travel(camp.node) / 60)]
        # 적 캠프·바위게: 지금 보이면 실제 상태, 아니면 마지막으로 본 상태와 지난 시간
        for cid in [f"{en}_{k}" for k in CAMP_KINDS] + ["river_top_scuttle", "river_bot_scuttle"]:
            camp = g.camps[cid]
            if camp.node in vis:
                self.known[cid] = (camp.alive, g.t)
            alive, seen_t = self.known.get(cid, (None, None))
            f += [1.0 if camp.node in vis else 0.0, 1.0 if alive else 0.0, 1.0 if alive is None else 0.0,
                  min(1.0, (g.t - seen_t) / 300) if seen_t is not None else 1.0, min(1.0, self.travel(camp.node) / 60)]
        # 오브젝트
        for o in OBJS:
            ob = g.objectives[o]
            f += [1.0 if ob["alive"] else 0.0,
                  min(1.0, max(0.0, ob["spawn_at"] - g.t) / 300) if ob["spawn_at"] is not None else 1.0,
                  min(1.0, self.travel(ob["node"]) / 60), g.obj_taken[o][T] / 3, g.obj_taken[o][300 - T] / 3]
        # 라인(라인 중앙은 양 팀 미니언 시야라 보임)
        for lane in LANES:
            mine = [x for x in g.champs if x.team == T and x.role != "JUNGLE" and x.lane == lane]
            foes = [x for x in g.champs if x.team != T and x.role != "JUNGLE" and x.lane == lane]
            for grp in (mine, foes):
                alive = [x for x in grp if x.alive(g.t)]
                f += [len(alive) / 2, min((x.hp for x in alive), default=0.0), sum(x.level for x in grp) / max(1, len(grp)) / 18,
                      sum(1 for x in alive if x.node == g.lane_node(lane)) / 2,
                      sum(1 for x in alive if x.flash_ready(g.t)) / 2]
            f += [g.wave_for(T, lane), min(1.0, self.travel(g.lane_node(lane)) / 60),
                  1.0 if g.t < g.lane_taken_until[lane] else 0.0]
        # 적 정글러: 마지막으로 본 곳(인지 제약 반영), 레벨·생존은 점수판으로 앎
        ej = g.enemy(c, "JUNGLE")
        node, seen_t = g.enemy_jgl_seen[T]
        if node is not None and seen_t <= g.t:
            n = g.map.nodes[node]
            f += [1.0, n["x"] / 15000, n["y"] / 15000, min(1.0, (g.t - seen_t) / 60), min(1.0, self.travel(node) / 60)]
        else:
            f += [0.0, 0.0, 0.0, 1.0, 1.0]
        f += [1.0 if ej.alive(g.t) else 0.0, (ej.level - c.level) / 5, ej.kills / 5, ej.deaths / 5, c.kills / 5, c.deaths / 5]
        # 팀 차이와 지금 예측 승률
        d = self.wp.diffs(g)
        s = 1 if T == 100 else -1
        f += [s * d["gold"] / 5000, s * d["xp"] / 5000, s * d["dragons"] / 3, s * d["grubs"] / 6, s * d["heralds"],
              self.wp.for_team(g, T), sum(1 for w in g.wards if w["team"] == T) / 5]
        return np.concatenate([np.array(f, dtype=np.float32), node_oh, dest_oh, task_oh, cur_oh]).astype(np.float32)


def obs_size():
    from .game import Game
    holder = {}

    def pol(obs, m, mw, ms):
        holder["n"] = len(obs)
        return int(np.flatnonzero(m)[0]), 0, 0, None

    from .winprob import WinProb
    wp = WinProb()
    g = Game(seed=0, jungler_override={100: lambda gm, ch: AgentJungler(gm, ch, pol, wp)})
    g.step()
    return holder["n"]
