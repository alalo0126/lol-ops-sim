"""4단계 규칙 봇.

JunglerBot: 정해진 동선으로 첫 풀캠 → 귀환 → 이후 '갱 기회 > 살아 있는 캠프 > 오브젝트 > 다음 리스폰 대기' 순서.
  학습 대상이 될 정글러의 기준선(5단계 PPO가 이겨야 할 상대)이자 적 정글 봇이다.
LanerBot: 3단계 라이너 판단 기준(kk 확정) 중 v1 최소 시뮬레이터에 입력이 있는 것만 쓴다.
  후퇴, 귀환, 와드, 기본(받아먹기). 다이브·프리징·푸시·로밍은 웨이브 모듈이 생기면 붙인다.
"""
import math

from . import events
from .game import SIDE
from .vision import detect_prob

RIVER_WARD_NODE = {"top": ["river_top_brush"], "bot": ["river_bot_brush"], "mid": ["river_top_scuttle", "river_bot_scuttle"]}
OBJ_LANE = {"dragon": "bot", "grubs": "top", "herald": "top"}


class JunglerBot:
    def __init__(self, game, champ):
        self.g, self.c = game, champ
        self.last_gank = -999
        self.checked_empty = {}   # 카정 갔다가 비어 있던 적 캠프 → 확인 시각. 일반 캠프 리스폰(135초) 동안 제외
        pj = game.p["jungler"]
        c = champ
        c.ms = pj["base_ms"]
        c.smite_charges = game.rules.positions["smite"]["charges"]["start"]
        c.smite_next = game.rules.positions["smite"]["charges"]["second_charge_recharge_starts_at_s"] + game.rules.positions["smite"]["recharge_s"]
        c.next_bonus_treat = pj["first_bonus_treat_s"]
        c.clear_mult = math.exp(game.rng.gauss(0, pj["clear_speed_sigma"]))
        weights = pj["start_camp_weights"]
        start = game.rng.choices(list(weights), weights=list(weights.values()))[0]
        c.route = [f"{SIDE[c.team]}_{k}" for k in pj["routes"][start]]
        game.give_gold(c, 0)
        c.spent = game.rules.positions["jungle_item"]["cost"]   # 시작 정글 펫 구매

    # -------------------------------------------------------------- 공용
    def travel(self, node):
        return self.g.map.travel_s(self.c.node, node, self.c.ms)

    def own_camps(self):
        g, c = self.g, self.c
        return [k for k in g.camps.values() if k.side in (c.team, 0)]

    def use_smite(self, camp):
        c, g = self.c, self.g
        if c.smite_charges <= 0:
            return False
        if c.smite_charges >= 2:
            return True
        keep = g.p["jungler"]["smite_keep_before_objective_s"]
        for ob in g.objectives.values():
            if ob["alive"] or (ob["spawn_at"] is not None and ob["spawn_at"] - g.t <= keep):
                return False
        return True

    def gank_target(self, hp_threshold, max_travel):
        g, c = self.g, self.c
        best = None
        for role in ("TOP", "MIDDLE", "BOTTOM"):
            lane = g.enemy(c, role).lane
            node = g.lane_node(lane)
            enemies = [e for e in g.champs if e.team != c.team and e.role != "JUNGLE" and e.lane == lane and e.alive(g.t) and e.node in (node, g.tower_node(lane, e.team))]
            if not enemies:
                continue
            t = min(enemies, key=lambda e: e.hp)
            tt = self.travel(node)
            if t.hp < hp_threshold and tt <= max_travel and (best is None or t.hp < best[1]):
                best = (lane, t.hp)
        return best[0] if best else None

    def invade_target(self, own_best_travel):
        """카정: 적 정글러를 최근 반대편에서 봤을 때만, 가까운 적 캠프를 확률적으로 노린다.

        캠프가 살아 있는지는 모르는 채로 간다(전장의 안개). 도착해서 없으면 다음 결정으로 넘어간다.
        """
        g, c = self.g, self.c
        pj = g.p["jungler"]
        node, seen_t = g.enemy_jgl_seen[c.team]
        if node is None or g.t - seen_t > pj["invade_seen_within_s"]:
            return None
        enemy_camps = [k for k in g.camps.values() if k.side == 300 - c.team and g.t - self.checked_empty.get(k.cid, -999) > 135]
        far = [k for k in enemy_camps if g.map.dist(node, k.node) > pj["invade_min_enemy_dist"]]
        if not far:
            return None
        camp = min(far, key=lambda k: self.travel(k.node))
        limit = pj["invade_max_travel_s"] if own_best_travel is None else min(pj["invade_max_travel_s"], own_best_travel + 5)
        if self.travel(camp.node) > limit or g.rng.random() >= pj["invade_prob"]:
            return None
        return camp

    def gank_ready(self):
        return self.g.t - self.last_gank >= self.g.p["jungler"]["gank_cooldown_s"]

    # -------------------------------------------------------------- 결정
    def tick(self):
        g, c = self.g, self.c
        task = c.task
        if task and task["kind"] in ("clearing", "recalling") or g.t < c.busy_until:
            return
        if task and c.path:
            # 이동 중 5초마다 다시 고름. 기본은 계속, 목표가 사라졌으면 취소.
            if g.t - c.last_decision >= 5:
                c.last_decision = g.t
                if not self.still_valid(task):
                    c.task, c.path = None, []
            return
        if task:
            self.arrive(task)
            return
        self.decide()

    def still_valid(self, task):
        g = self.g
        if task["kind"] == "camp":
            camp = g.camps[task["camp"]]
            return camp.alive or camp.spawn_at - g.t <= self.travel(camp.node) + 20
        if task["kind"] == "objective":
            return g.objectives[task["name"]]["alive"]
        return True

    def go(self, kind, node, **kw):
        g, c = self.g, self.c
        c.task = {"kind": kind, "node": node, **kw}
        c.last_decision = g.t
        if c.node != node:
            g.move_to(c, node)

    def decide(self):
        g, c = self.g, self.c
        pj = g.p["jungler"]
        if c.route:
            camp = g.camps[c.route[0]]
            if camp.alive or camp.spawn_at > g.t:
                self.go("camp", camp.node, camp=camp.cid, route=True)
                return
            c.route.pop(0)    # 적이 먼저 먹었다
            return
        if c.first_clear_done_s is None:
            c.first_clear_done_s = g.t
        if c.first_recall_s is None:
            lane = self.gank_target(0.4, 20)
            if lane:
                self.go("gank", g.lane_node(lane), lane=lane)
            else:
                g.start_recall(c)
            return
        alive_near = [k for k in self.own_camps() if k.alive and self.travel(k.node) < 15]
        if c.hp < pj["recall_hp"] or (c.unspent >= pj["recall_gold"] and not alive_near):
            g.start_recall(c)
            return
        lane = self.gank_target(pj["gank_hp_threshold"], pj["gank_max_travel_s"]) if self.gank_ready() else None
        if lane:
            self.go("gank", g.lane_node(lane), lane=lane)
            return
        live_obj = [n for n, ob in g.objectives.items() if ob["alive"]]
        if live_obj and g.rng.random() < pj["objective_go_prob"]:
            name = min(live_obj, key=lambda n: self.travel(g.objectives[n]["node"]))
            self.go("objective", g.objectives[name]["node"], name=name)
            return
        alive = [k for k in self.own_camps() if k.alive]
        target = self.invade_target(min((self.travel(k.node) for k in alive), default=None))
        if target:
            self.go("camp", target.node, camp=target.cid, invade=True)
            return
        if alive:
            camp = min(alive, key=lambda k: self.travel(k.node))
            self.go("camp", camp.node, camp=camp.cid)
            return
        lane = self.gank_target(0.7, 60) if self.gank_ready() else None
        if lane:
            self.go("gank", g.lane_node(lane), lane=lane)
            return
        camp = min(self.own_camps(), key=lambda k: max(k.spawn_at - g.t, self.travel(k.node)))
        self.go("camp", camp.node, camp=camp.cid)

    # -------------------------------------------------------------- 도착
    def arrive(self, task):
        g, c = self.g, self.c
        kind = task["kind"]
        if kind == "camp":
            camp = g.camps[task["camp"]]
            if camp.alive:
                g.start_camp(c, camp)
                if task.get("route"):
                    c.route.pop(0)
            elif camp.spawn_at - g.t <= 25:
                c.busy_until = camp.spawn_at       # 생성까지 기다림(작업은 그대로)
            else:
                if task.get("route"):
                    c.route.pop(0)
                if task.get("invade"):
                    self.checked_empty[camp.cid] = g.t
                c.task = None
        elif kind == "objective":
            ob = g.objectives[task["name"]]
            if not ob["alive"]:
                c.task = None
                return
            helpers = []
            lane = OBJ_LANE[task["name"]]
            for a in g.champs:
                if a.team == c.team and a.role != "JUNGLE" and a.lane == lane and a.alive(g.t) and a.node == g.lane_node(lane) and not a.task:
                    helpers.append(a)
            dur = g.objective_time(c, task["name"], helpers)
            for h in helpers:
                h.task = {"kind": "helping", "until": g.t + dur}
                g.move_to(h, ob["node"])
            task.update(started=True, helpers=helpers)
            c.busy_until = g.t + dur
        elif kind == "gank":
            self.last_gank = g.t
            self.resolve_gank(task)
        else:
            c.task = None

    def resolve_gank(self, task):
        g, c = self.g, self.c
        lane = task["lane"]
        center, enemy_tower = g.lane_node(lane), g.tower_node(lane, 300 - c.team)
        defenders = [e for e in g.champs if e.team != c.team and e.role != "JUNGLE" and e.lane == lane and e.alive(g.t) and e.node in (center, enemy_tower)]
        if not defenders:
            c.task = None
            return
        target = min(defenders, key=lambda e: e.hp)
        at_tower = target.node == enemy_tower
        if at_tower and c.node != enemy_tower:
            c.path = g.map.path(c.node, enemy_tower)
            c.edge_left = g.map.dist(c.node, c.path[0])
            return
        defenders = [d for d in defenders if d.node == target.node]
        attackers = [c] + [a for a in g.champs if a.team == c.team and a.role != "JUNGLE" and a.lane == lane and a.alive(g.t) and a.node == center and not a.task]
        seen = detect_prob(c.exposures, g.cog)
        inp = events.GankInput(attackers=attackers, defenders=defenders, target=target, seen_prob=seen,
                               target_push=g.wave_for(target.team, lane), at_tower=at_tower)
        res = events.resolve_gank(inp, g.t, g.rng, g.p["gank"])
        for pid, hp in res.hp_after.items():
            next(x for x in g.champs if x.pid == pid).hp = hp
        for f in res.flash_used:
            f.flash_at = g.t + g.p["gank"]["flash_cooldown_s"]
        for v in res.deaths:
            g.kill(res.killer, v, res.assists)
        g.events.append({"t": g.t, "type": "gank", "pid": c.pid, "lane": lane, "success": res.success, "prob": round(res.prob, 3), "seen": round(seen, 3)})
        if c.alive(g.t):
            c.busy_until = g.t + res.duration_s
            c.task = {"kind": "ganking", "resolved": True}


class LanerBot:
    def __init__(self, game, champ):
        self.g, self.c = game, champ
        champ.ms = 335
        champ.ward_charges = 1
        champ.ward_next = game.p["laner"]["ward_recharge_s"]
        lo, hi = game.p["laner"]["trade_every_s"]
        champ.next_trade = game.p["laner"]["lane_start_s"] + game.rng.uniform(lo, hi)

    def tick(self):
        g, c, pl = self.g, self.c, self.g.p["laner"]
        lane = c.lane
        center, tower = g.lane_node(lane), g.tower_node(lane, c.team)
        if c.task:
            k = c.task["kind"]
            if k == "helping" and g.t >= c.task["until"]:
                c.task = None
                g.move_to(c, center)
            elif k == "recalling":
                pass
            return
        if c.ward_charges < pl["ward_max_charges"] and g.t >= c.ward_next:
            c.ward_charges += 1
            c.ward_next = g.t + pl["ward_recharge_s"]
        if c.path:
            return
        if c.node not in (center, tower):
            g.move_to(c, center)
            return
        if c.node == tower and g.t >= c.retreat_until:
            g.move_to(c, center)
            return
        if c.team == 100 and c.role in ("TOP", "MIDDLE", "BOTTOM"):   # 라인마다 한 명만 웨이브를 움직임
            w = g.wave[lane] + g.rng.gauss(0, pl["wave_drift_sigma"]) + 0.01 * (0.5 - g.wave[lane])
            g.wave[lane] = min(1.0, max(0.0, w))

        # kk 기준 1. 후퇴
        if c.node == center and self.should_retreat(center):
            c.retreat_until = g.t + 15
            g.move_to(c, tower)
            g.events.append({"t": g.t, "type": "retreat", "pid": c.pid})
            return
        # kk 기준 2. 귀환 (마나 모델이 없어 체력만 봄)
        pushed = g.wave_for(c.team, lane) >= pl["pushed"]
        if c.hp < pl["recall_hp_always"] or ((c.hp < pl["recall_hp"] or c.unspent >= pl["core_component_gold"]) and pushed):
            g.start_recall(c)
            return
        # kk 기준 9. 와드
        seen_node, seen_t = g.enemy_jgl_seen[c.team]
        if c.ward_charges > 0 and g.t - seen_t >= pl["unseen_ward_s"] and g.t >= pl["lane_start_s"]:
            spots = [n for n in RIVER_WARD_NODE[lane] if not any(w["team"] == c.team and w["node"] == n for w in g.wards)]
            if spots:
                g.wards.append({"team": c.team, "node": spots[0], "until": g.t + pl["ward_duration_s"], "pid": c.pid})
                c.ward_charges -= 1
        # 기본: 받아먹기. 웨이브 모듈 대신 고정 수입
        if g.t >= pl["lane_start_s"]:
            share = 1.0 if c.node == center else 0.5
            inc = pl["income_per_s"][c.role]
            g.give_gold(c, inc["gold"] * share)
            g.give_xp(c, inc["xp"] * share)
            c.lane_cs += pl["cs_per_s"][c.role] * share
        c.hp = min(1.0, c.hp + pl["hp_regen_per_s"])
        if c.node == center and g.t >= c.next_trade:
            self.trade(center)

    def should_retreat(self, center):
        g, c, pl = self.g, self.c, self.g.p["laner"]
        seen_node, seen_t = g.enemy_jgl_seen[c.team]
        if seen_node is None or not (0 <= g.t - seen_t <= pl["retreat_seen_within_s"]):
            return False
        if g.map.hops(seen_node, center) > pl["retreat_hops"]:
            return False
        ej = g.enemy(c, "JUNGLE")
        attackers = [ej] + [e for e in g.champs if e.team != c.team and e.role != "JUNGLE" and e.lane == c.lane and e.alive(g.t) and e.node == center]
        defenders = [a for a in g.champs if a.team == c.team and a.role != "JUNGLE" and a.lane == c.lane and a.alive(g.t) and a.node == center]
        inp = events.GankInput(attackers=attackers, defenders=defenders, target=c, seen_prob=0.0,
                               target_push=g.wave_for(c.team, c.lane), at_tower=False)
        return events.gank_success_prob(inp, g.t, g.p["gank"]["coef"]) > pl["retreat_gank_prob"]

    def trade(self, center):
        g, c, pl = self.g, self.c, self.g.p["laner"]
        lo, hi = pl["trade_every_s"]
        c.next_trade = g.t + g.rng.uniform(lo, hi)
        foes = [e for e in g.champs if e.team != c.team and e.role == c.role and e.alive(g.t) and e.node == center and not e.task]
        if not foes:
            return
        e = foes[0]
        dlo, dhi = pl["trade_damage"]
        c.hp -= g.rng.uniform(dlo, dhi)
        e.hp -= g.rng.uniform(dlo, dhi)
        for victim, killer in ((c, e), (e, c)):
            if victim.hp <= 0.02 and victim.alive(g.t) and killer.alive(g.t):
                g.kill(killer, victim, [])
