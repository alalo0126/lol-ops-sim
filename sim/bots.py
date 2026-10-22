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

# 로밍 와드 자리: 라인 → (강가·바위게 노드, 정글 입구 방향 'top'/'bot'). 입구는 우리/상대 쪽을 그때 고른다
ROAM_WARD = {"top": (["river_top_brush", "river_top_scuttle"], "top"), "bot": (["river_bot_brush", "river_bot_scuttle"], "bot"),
             "mid": (["river_top_scuttle", "river_bot_scuttle", "river_top_brush", "river_bot_brush"], None)}
# (팀, 맵 위/아래) → 그쪽 정글 캠프. 블루는 위쪽이 블루 버프, 레드는 위쪽이 레드 버프(점대칭)
JUNGLE_SIDE = {(100, "top"): ["blue", "gromp", "wolves"], (100, "bot"): ["red", "krugs", "raptors"],
               (200, "top"): ["red", "krugs", "raptors"], (200, "bot"): ["blue", "gromp", "wolves"]}
RIVER_WARD_NODE = {"top": ["river_top_brush"], "bot": ["river_bot_brush"], "mid": ["river_top_scuttle", "river_bot_scuttle"]}
OBJ_LANE = {"dragon": "bot", "grubs": "top", "herald": "top"}


class JunglerBot:
    def __init__(self, game, champ):
        self.g, self.c = game, champ
        self.last_gank = -999
        self.checked_empty = {}   # 카정 갔다가 비어 있던 적 캠프 → 확인 시각. 일반 캠프 리스폰(135초) 동안 제외
        pj = game.p["jungler"]
        c = champ
        weights = pj["champion_weights"]
        c.champion = game.rng.choices(list(weights), weights=list(weights.values()))[0]
        c.ms = game.rules.champions[c.champion]["stats"]["movespeed"]
        c.smite_charges = game.rules.positions["smite"]["charges"]["start"]
        c.smite_next = game.rules.positions["smite"]["charges"]["second_charge_at_s"]   # kk 실측 2:20
        c.next_bonus_treat = pj["first_bonus_treat_s"]
        c.clear_mult = math.exp(game.rng.gauss(0, pj["clear_speed_sigma"]))
        weights = pj["start_camp_weights"]
        start = game.rng.choices(list(weights), weights=list(weights.values()))[0]
        c.route = [self.route_node(k) for k in pj["routes"][start]]
        game.give_gold(c, 0)
        c.spent = game.rules.positions["jungle_item"]["cost"]   # 시작 정글 펫 구매

    # -------------------------------------------------------------- 공용
    def route_node(self, token):
        """동선 표기 → 노드. 'recall'은 그대로, 바위게는 우리 블루 쪽/레드 쪽(점대칭이라 팀마다 위아래가 바뀜)."""
        if token == "recall":
            return token
        if token.startswith("scuttle_"):
            top = (token == "scuttle_blue_side") == (self.c.team == 100)
            return "river_top_scuttle" if top else "river_bot_scuttle"
        return f"{SIDE[self.c.team]}_{token}"

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

    def contest_target(self):
        """살아 있는 오브젝트 근처에서 적 정글러를 최근에 봤으면 확률적으로 다투러 간다(6단계 이후 추가)."""
        g, c = self.g, self.c
        pj = g.p["jungler"]
        ej = g.enemy(c, "JUNGLE")
        if ej.alive(g.t) and ej.task and ej.task.get("kind") == "objective" and ej.task.get("started") \
                and self.travel(g.objectives[ej.task["name"]]["node"]) <= ej.busy_until - g.t and g.rng.random() < pj["contest_prob"]:
            return ej.task["name"]     # 적이 오브젝트를 치는 중(시야·라이너 핑으로 안다고 봄)
        node, seen_t = g.enemy_jgl_seen[c.team]
        if node is None or g.t - seen_t > pj["contest_seen_within_s"]:
            return self.setup_target()
        for name, ob in g.objectives.items():
            if ob["alive"] and g.map.hops(node, ob["node"]) <= pj["contest_hops"] \
                    and self.travel(ob["node"]) <= pj["contest_max_travel_s"] and g.rng.random() < pj["contest_prob"]:
                return name
        return None

    def setup_target(self):
        """드래곤 생성 직전이면 확률적으로 둥지 앞에 미리 가서 기다린다(6단계 이후 추가)."""
        g = self.g
        pj = g.p["jungler"]
        ob = g.objectives["dragon"]
        if ob["alive"] or ob["spawn_at"] is None:
            return None
        left = ob["spawn_at"] - g.t
        tt = self.travel(ob["node"])
        if tt <= left <= tt + pj["setup_before_s"] and g.rng.random() < pj["setup_prob"]:
            return "dragon"
        return None

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
        if task["kind"] == "objective" and not g.objectives[task["name"]]["alive"]:
            ob = g.objectives[task["name"]]
            return ob["spawn_at"] is not None and ob["spawn_at"] - g.t <= g.p["jungler"]["setup_before_s"] + 30
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
        g = self.g
        choice = self.choose()
        kind = choice[0]
        if kind == "recall":
            g.start_recall(self.c)
        elif kind in ("camp", "gank", "objective"):
            self.go(kind, choice[1], **choice[2])

    def choose(self):
        """규칙 봇의 선택만 계산해 돌려준다(동선 진행 같은 봇 내부 상태는 여기서 바뀜).

        ("recall",) / ("camp"|"gank"|"objective", 노드, 작업 정보) / ("skip",) = 이번 틱은 아무것도 안 함.
        5단계 학습 정글러의 모방 학습 교사(sim/teacher.py)도 이 함수를 쓴다.
        """
        g, c = self.g, self.c
        pj = g.p["jungler"]
        if c.route and c.route[0] == "recall":
            c.route.pop(0)
            return ("recall",)
        if c.route:
            camp = g.camps[c.route[0]]
            if camp.alive or camp.spawn_at > g.t:
                return ("camp", camp.node, {"camp": camp.cid, "route": True})
            c.route.pop(0)    # 적이 먼저 먹었다
            return ("skip",)
        if c.first_clear_done_s is None:
            c.first_clear_done_s = g.t
        if c.first_recall_s is None:
            lane = self.gank_target(0.4, 20)
            if lane:
                return ("gank", g.lane_node(lane), {"lane": lane})
            return ("recall",)
        alive_near = [k for k in self.own_camps() if k.alive and self.travel(k.node) < 15]
        if c.hp < pj["recall_hp"] or (c.unspent >= pj["recall_gold"] and not alive_near):
            return ("recall",)
        contest = self.contest_target()
        if contest:
            return ("objective", g.objectives[contest]["node"], {"name": contest})
        lane = self.gank_target(pj["gank_hp_threshold"], pj["gank_max_travel_s"]) if self.gank_ready() else None
        if lane:
            return ("gank", g.lane_node(lane), {"lane": lane})
        live_obj = [n for n, ob in g.objectives.items() if ob["alive"]]
        if live_obj and g.rng.random() < pj["objective_go_prob"]:
            name = min(live_obj, key=lambda n: self.travel(g.objectives[n]["node"]))
            return ("objective", g.objectives[name]["node"], {"name": name})
        alive = [k for k in self.own_camps() if k.alive]
        target = self.invade_target(min((self.travel(k.node) for k in alive), default=None))
        if target:
            return ("camp", target.node, {"camp": target.cid, "invade": True})
        if alive:
            camp = min(alive, key=lambda k: self.travel(k.node))
            return ("camp", camp.node, {"camp": camp.cid})
        lane = self.gank_target(0.7, 60) if self.gank_ready() else None
        if lane:
            return ("gank", g.lane_node(lane), {"lane": lane})
        camp = min(self.own_camps(), key=lambda k: max(k.spawn_at - g.t, self.travel(k.node)))
        return ("camp", camp.node, {"camp": camp.cid})

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
            elif camp.spawn_at - g.t <= 25 or (task.get("route") and camp.first):
                c.busy_until = camp.spawn_at       # 생성까지 기다림(작업은 그대로). 첫 동선의 시작 캠프는 일찍 와도 기다림
            else:
                if task.get("route"):
                    c.route.pop(0)
                if task.get("invade"):
                    self.checked_empty[camp.cid] = g.t
                c.task = None
        elif kind == "objective":
            ob = g.objectives[task["name"]]
            if not ob["alive"]:
                if ob["spawn_at"] is not None and ob["spawn_at"] - g.t <= g.p["jungler"]["setup_before_s"] + 5:
                    c.busy_until = ob["spawn_at"]     # 생성 직전이면 둥지 앞에서 기다림(작업은 그대로)
                    return
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
        # 카운터 갱: 그 자리에서 라인을 지키던(합류·라인 미니언) 적 정글러도 방어에 낀다
        defenders += [j for j in g.champs if j.team != c.team and j.role == "JUNGLE" and j.alive(g.t) and j.node == target.node
                      and j.task and j.task.get("kind") in ("cover", "lanefarm")]
        attackers = [c] + [a for a in g.champs if a.team == c.team and a.role != "JUNGLE" and a.lane == lane and a.alive(g.t) and a.node == center and not a.task]
        win = g.p["gank"]["seen_window_s"]
        seen = detect_prob([d for t, d in c.exposures if g.t - t <= win], g.cog)
        # 최근 이 라인에 갱이 왔으면 라이너가 경계한다(6단계 이후 추가, 연속 갱 악용 대응)
        last = g.gank_alert.get((300 - c.team, lane))
        if last is not None:
            pg = g.p["gank"]
            seen = max(seen, pg["alert_seen"] * max(0.0, 1 - (g.t - last) / pg["alert_window_s"]))
        g.gank_alert[(300 - c.team, lane)] = g.t
        inp = events.GankInput(attackers=attackers, defenders=defenders, target=target, seen_prob=seen,
                               target_push=g.wave_for(target.team, lane), at_tower=at_tower)
        res = events.resolve_gank(inp, g.t, g.rng, g.p["gank"])
        for pid, hp in res.hp_after.items():
            next(x for x in g.champs if x.pid == pid).hp = hp
        for f in res.flash_used:
            f.flash_at = g.t + g.p["gank"]["flash_cooldown_s"]
        for v in res.deaths:
            g.kill(res.killer, v, res.assists)
        for v, k, a in res.trades:
            if v.alive(g.t):
                g.kill(k, v, a)
        g.events.append({"t": g.t, "type": "gank", "pid": c.pid, "lane": lane, "success": res.success, "prob": round(res.prob, 3), "seen": round(seen, 3)})
        for d in defenders:
            if d.role != "JUNGLE" and d.alive(g.t):
                g.bots[d.pid].after_gank()
        if c.alive(g.t):
            c.busy_until = g.t + res.duration_s
            c.task = {"kind": "ganking", "resolved": True}


class LanerBot:
    def __init__(self, game, champ):
        self.g, self.c = game, champ
        champ.ms = 335
        champ.spent = champ.item_gold = 500.0   # 시작 아이템(1단계: 1분 보유 골드 중앙 0)
        champ.ward_charges = 1
        champ.ward_next = game.p["laner"]["ward_recharge_s"]
        lo, hi = game.p["laner"]["trade_every_s"]
        champ.next_trade = game.p["laner"]["lane_start_s"] + game.rng.uniform(lo, hi)

    def tick(self):
        g, c, pl = self.g, self.c, self.g.p["laner"]
        lane = c.lane
        center, tower = g.lane_node(lane), g.tower_node(lane, c.team)
        self.maybe_rotate()
        if c.task:
            k = c.task["kind"]
            if k == "helping" and g.t >= c.task["until"]:
                c.task = None
                g.move_to(c, center)
            elif k == "roaming":
                self.roam_tick(center)
            elif k == "recalling":
                pass
            return
        if c.role == "UTILITY":
            pr = pl["support_ward_per_min"]
            c.support_wards = min(pr["max_charges"], c.support_wards + (pr["early"] if g.t < 600 else pr["late"]) / 60)
        if c.ward_charges < pl["ward_max_charges"] and g.t >= c.ward_next:
            c.ward_charges += 1
            c.ward_next = g.t + pl["ward_recharge_s"]
        if c.path:
            return
        if c.node not in (center, tower):
            g.move_to(c, center)
            return
        if c.node == tower and g.t >= c.retreat_until and not self.behind():
            g.move_to(c, center)
            return

        # kk 기준 1. 후퇴
        if c.node == center and self.behind():
            g.move_to(c, tower)     # 크게 밀리면 포탑 근처에서 받아먹기(6단계 이후 추가)
            return
        if c.node == center and self.should_retreat(center):
            c.retreat_until = g.t + 15
            g.move_to(c, tower)
            g.events.append({"t": g.t, "type": "retreat", "pid": c.pid})
            return
        # kk 기준 2. 귀환: 체력 15% 미만은 무조건, 나머지는 1,000판 구매 기록으로 맞춘 확률(laner_recall.json)
        if c.hp < pl["recall_hp_always"] or (g.wave_for(c.team, lane) > 1 - pl["pushed"] and g.rng.random() < self.recall_hazard()):
            g.start_recall(c)
            return
        # 로밍(와드·다른 라인). 1단계 분 프레임 위치 비율에 맞춘 확률(sim/targets_laner.json zone_by_minute)
        if c.node == center and g.t >= pl["lane_start_s"] and self.start_roam():
            return
        # kk 기준 9. 와드
        seen_node, seen_t = g.enemy_jgl_seen[c.team]
        if c.ward_charges > 0 and g.t - seen_t >= pl["unseen_ward_s"] and g.t >= pl["lane_start_s"]:
            spots = [n for n in RIVER_WARD_NODE[lane] if not any(w["team"] == c.team and w["node"] == n for w in g.wards)]
            if spots:
                g.wards.append({"team": c.team, "node": spots[0], "until": g.t + pl["ward_duration_s"], "pid": c.pid})
                g.events.append({"t": g.t, "type": "ward", "pid": c.pid})
                c.ward_charges -= 1
        # 기본: 받아먹기. CS·경험치는 웨이브가 도착할 때 라인에 서 있으면 sim/lanes.py가 준다
        c.hp = min(1.0, c.hp + pl["hp_regen_per_s"])
        if c.node == center and g.t >= c.next_trade:
            self.trade(center)

    def maybe_rotate(self):
        """원딜·서폿이 미드로, 미드가 바텀으로 라인을 바꿈. 1단계: 원딜 14분 미드 띠 34%(바텀 외곽 포탑 남음) / 56%(깨짐)."""
        g, c, pr = self.g, self.c, self.g.p["laner"]["rotate"]
        if c.role != "BOTTOM" or c.lane_override or not c.alive(g.t):
            return
        down = any(g.lanes.towers[T]["bot"][0] <= 0 for T in (100, 200))
        if down:   # 14분 전에는 바텀 외곽이 깨져도 잘 안 바꿈(1단계 12분 깨짐: 원딜 미드 띠 28%, 미드 바텀 띠 8%)
            rate = pr["after_bot_outer_down"] if g.t >= pr["after_s"] else pr["after_bot_outer_down_early"]
        else:
            rate = pr["after_s_rate"] if g.t >= pr["after_s"] else 0.0
        if g.rng.random() >= rate / 60:
            return
        for m in g.champs:
            if m.team == c.team and m.role in ("BOTTOM", "UTILITY"):
                m.lane_override = "mid"
            elif m.team == c.team and m.role == "MIDDLE":
                m.lane_override = "bot"
        g.events.append({"t": g.t, "type": "rotate", "team": c.team})

    def start_roam(self):
        g, c, pr = self.g, self.c, self.g.p["laner"]["roam"]
        rate = pr["rate_per_min"][c.role][0 if g.t < 840 else 1]
        if c.role == "UTILITY" and c.support_wards >= 1:
            rate *= pr["support_ward_boost"]
        if g.rng.random() >= rate / 60:
            return False
        lane = c.lane
        kind = g.rng.choices(["other_lane", "river", "jungle"], weights=pr["dest"][c.role][0 if g.t < 840 else 1])[0]
        if kind == "other_lane":
            others = [ln for ln in ("top", "mid", "bot") if ln != lane and (lane == "mid" or ln == "mid" or c.role == "MIDDLE")]
            # 실제 로밍은 갱처럼 싸울 상대가 있는 라인으로 간다(리플레이 미드 로밍당 킬 관여 64%). 적 라이너가 서 있는 라인만,
            # 아군이 많은 쪽을 더 자주. 없으면 강가로
            score = {ln: len(g.lanes.present(300 - c.team, ln)) and 1 + len(g.lanes.present(c.team, ln)) for ln in others}
            targets = [ln for ln in others if score[ln]]
            if targets:
                dest = g.lane_node(g.rng.choices(targets, weights=[score[ln] for ln in targets])[0])
            else:
                dest = g.rng.choice(ROAM_WARD[lane][0])
        elif kind == "river":
            dest = g.rng.choice(ROAM_WARD[lane][0])
        else:
            side = ROAM_WARD[lane][1] or g.rng.choice(["top", "bot"])
            team = c.team if g.rng.random() < pr["own_entrance"] else 300 - c.team
            camps = JUNGLE_SIDE[(team, side)]
            dest = g.rng.choice([f"{SIDE[team]}_{side}_entrance"] * 2 + [f"{SIDE[team]}_{k}" for k in camps])
        lo, hi = pr["linger_s"]
        c.task = {"kind": "roaming", "node": dest, "linger": g.rng.uniform(lo, hi), "until": None}
        g.move_to(c, dest)
        g.events.append({"t": g.t, "type": "roam", "pid": c.pid, "node": dest})
        return True

    def roam_tick(self, center):
        g, c, pl = self.g, self.c, self.g.p["laner"]
        task = c.task
        if c.path:
            return
        node = g.map.nodes[c.node]
        if node["kind"] == "lane" and task.get("fight") is None:
            task["fight"] = g.rng.random() < pl["roam"]["fight_on_arrival"][c.role]
        if task.get("fight"):
            # 다른 라인 = 갱처럼 그 라인 적과 붙는다(실제 미드 로밍당 킬 관여 64%, sim/laner_tracks.py).
            # 도착했을 때 적이 없으면 머무는 동안 적이 오면 붙는다
            ln = next(l for l in ("top", "mid", "bot") if g.lane_node(l) == c.node)
            here = {T: g.lanes.present(T, ln) for T in (100, 200)}
            if here[300 - c.team]:
                task["fight"] = False
                g.lanes.fight(ln, here, kind="roam_fight", resolve=pl["roam"]["resolve"])
        if task["until"] is None:
            task["until"] = g.t + task["linger"]
            if node["kind"] != "lane":
                n = int(c.support_wards) if c.role == "UTILITY" else 0
                n += 1 if c.ward_charges > 0 else 0
                for _ in range(max(0, n)):
                    g.wards.append({"team": c.team, "node": c.node, "until": g.t + pl["ward_duration_s"], "pid": c.pid})
                    g.events.append({"t": g.t, "type": "ward", "pid": c.pid})
                if c.ward_charges > 0:
                    c.ward_charges -= 1
                if c.role == "UTILITY":
                    c.support_wards -= int(c.support_wards)
        if g.t >= task["until"]:
            c.task = None
            g.move_to(c, center)

    def recall_hazard(self):
        g, c = self.g, self.c
        w = g.recall_model["roles"][c.role]["w"]
        z = w[0] + w[1] * c.unspent / 1000 + w[2] * c.hp + w[3] * g.t / 600
        p = 1 / (1 + math.exp(-z))
        return (1 - (1 - p) ** (1 / 60)) * g.p["laner"]["recall_mult"][c.role]

    def after_gank(self):
        """갱을 버틴 뒤: 체력이 낮으면 바로 귀환, 아니면 잠시 포탑으로 물러남(5단계에서 찾은 연속 갱 악용 대응, Claude 기본값).

        kk 귀환 기준(체력 35% 미만 + 웨이브 밀림)은 웨이브 모듈이 없어 거의 발동하지 않아서, 갱 직후에는 웨이브 조건을 뺀다.
        """
        g, c, pl = self.g, self.c, self.g.p["laner"]
        if c.task:
            return
        if c.hp < pl["post_gank_recall_hp"]:
            g.start_recall(c)
            return
        c.retreat_until = g.t + pl["post_gank_retreat_s"]
        tower = g.tower_node(c.lane, c.team)
        if c.node != tower:
            g.move_to(c, tower)

    def behind(self):
        """맞라이너 + 적 정글러 평균보다 레벨이나 아이템 골드가 크게 뒤지면 True. 최근 그 라인에 갱이 왔을 때만 본다."""
        g, c, pl = self.g, self.c, self.g.p["laner"]
        last = g.gank_alert.get((c.team, c.lane))
        if last is None or g.t - last > pl["behind_alert_s"]:
            return False
        foes = [e for e in g.champs if e.team != c.team and (e.role == c.role or e.role == "JUNGLE")]
        lv = sum(e.level for e in foes) / len(foes) - c.level
        items = (sum(e.item_gold for e in foes) / len(foes) - c.item_gold) / 1000
        return lv >= pl["behind_level"] or items >= pl["behind_item_k"]

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
