"""4단계 최소 시뮬레이터: 1초 틱, 0~20분, 10명 모두 규칙 봇.

정글러는 bots.JunglerBot, 라이너는 bots.LanerBot이 결정한다. 이 파일은 결정의 '결과'(이동, 처치 시간,
보상, 사망, 시야)만 계산한다. 결정 시점은 3단계에서 정한 대로 행동이 끝났을 때 + 이동 중 5초마다 + 새 정보.
"""
import random
from types import SimpleNamespace
from dataclasses import dataclass, field

from . import events
from .data import Rules, lerp_level
from .mapgraph import MapGraph
from .vision import detect_prob, team_vision

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
LANE_OF = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
SIDE = {100: "B", 200: "R"}
GAME_END_S = 1200


@dataclass
class Champ:
    pid: int
    team: int
    role: str
    node: str
    ms: float = 345.0
    hp: float = 1.0
    xp: float = 0.0
    level: int = 1
    earned: float = 0.0          # 시작 골드 500 제외 누적 획득
    spent: float = 0.0
    item_gold: float = 0.0
    lane_cs: float = 0.0
    monster_gold: float = 0.0    # 정글러: 캠프에서 번 골드(몬스터 사냥꾼 패널티 판정용)
    minion_gold: float = 0.0     # 정글러: 라인 미니언에서 번 골드
    jungle_cs: int = 0
    bounty: float = 0.0          # 현상금 골드(+면 셧다운 쌓임, −면 연속 사망으로 깎임)
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    path: list = field(default_factory=list)
    edge_left: float = 0.0
    task: dict = None
    busy_until: float = 0.0
    dead_until: float = -1.0
    flash_at: float = 0.0        # 이 시각부터 점멸 사용 가능
    exposures: list = field(default_factory=list)   # 적 시야 노출 (시각, 머문 초). 갱 발각은 최근 seen_window_s만 봄
    champion: str = ""
    homeguard_left: float = 0.0  # 우물 이속 효과가 남은 이동 거리
    homeguard_t: float = 0.0     # 우물을 나선 뒤 지난 초
    respawned: bool = False      # 마지막으로 우물에 온 이유가 부활이면 True
    pull_credit: float = 0.0     # 직전 캠프를 다음 캠프 쪽으로 당겨 잡아 줄어드는 이동 시간(초)
    last_decision: float = -99.0
    # 정글러 전용
    smite_charges: int = 0
    smite_next: float = 0.0
    treats: int = 0
    next_bonus_treat: float = 0.0
    clear_mult: float = 1.0
    route: list = field(default_factory=list)
    first_clear_done_s: float = None
    first_recall_s: float = None
    big_kills: int = 0
    # 라이너 전용
    retreat_until: float = 0.0
    ward_charges: int = 0
    ward_next: float = 0.0
    next_trade: float = 0.0

    def alive(self, now):
        return now >= self.dead_until

    def flash_ready(self, now):
        return now >= self.flash_at

    @property
    def total_gold(self):
        return 500 + self.earned

    @property
    def unspent(self):
        return 500 + self.earned - self.spent

    @property
    def lane(self):
        return LANE_OF.get(self.role)


@dataclass
class Camp:
    cid: str
    kind: str        # camps.json 키
    node: str
    side: int
    alive: bool = False
    spawn_at: float = 0.0
    level: int = 1
    first: bool = True


class Game:
    def __init__(self, seed=0, cognition="challenger", rules=None, bot_classes=None, jungler_override=None):
        """jungler_override = {팀: (game, champ) → 봇}. 5단계 학습 정글러를 한 팀에만 넣을 때 쓴다."""
        from .bots import JunglerBot, LanerBot
        self.rules = rules or Rules()
        self.p = self.rules.params
        self.map = MapGraph(self.rules.map)
        self.rng = random.Random(seed)
        self.cog = self.p["cognition"]["profiles"][cognition]
        self.t = 0
        self.frames = []
        self.events = []
        self.wards = []
        self.first_blood = False
        self.wave = {"top": 0.5, "mid": 0.5, "bot": 0.5}   # 블루 기준 웨이브 위치(0 = 블루 포탑)
        # 팀별로 적 정글러를 마지막으로 본 (노드, 인지된 시각)
        self.enemy_jgl_seen = {100: (None, -999.0), 200: (None, -999.0)}
        # 팀별 '새 정보' 카운터(적 정글러 발견, 킬, 오브젝트 생성). 3단계 결정 주기의 세 번째 조건
        self.info_version = {100: 0, 200: 0}
        # 정글러가 라인 미니언을 먹는 동안 그 라인 라이너는 수입이 없다(같은 웨이브를 나눠 먹지 않음)
        self.lane_taken_until = {"top": -1.0, "mid": -1.0, "bot": -1.0}

        self.champs = []
        for team in (100, 200):
            for i, role in enumerate(ROLES):
                c = Champ(pid=i + 1 + (0 if team == 100 else 5), team=team, role=role, node=f"{SIDE[team]}_fountain")
                self.champs.append(c)
        jb, lb = (bot_classes or (JunglerBot, LanerBot))
        override = jungler_override or {}
        self.bots = {c.pid: ((override.get(c.team) or jb)(self, c) if c.role == "JUNGLE" else lb(self, c)) for c in self.champs}

        self.camps = {}
        for team in (100, 200):
            for kind in ("blue", "red", "gromp", "wolves", "raptors", "krugs"):
                node = f"{SIDE[team]}_{kind}"
                self.camps[node] = Camp(node, kind, node, team, spawn_at=self.rules.camps[kind]["first_spawn_s"])
        for node in ("river_top_scuttle", "river_bot_scuttle"):
            self.camps[node] = Camp(node, "scuttle", node, 0, spawn_at=self.rules.camps["scuttle"]["first_spawn_s"])

        o = self.rules.objectives
        self.objectives = {
            "dragon": {"node": "dragon_pit", "alive": False, "spawn_at": o["drake"]["first_spawn_s"], "respawn": o["drake"]["respawn_s"], "until": None},
            "grubs": {"node": "baron_pit", "alive": False, "spawn_at": o["voidgrubs"]["spawn_s"], "respawn": None, "until": o["voidgrubs"]["despawn_s"]},
            "herald": {"node": "baron_pit", "alive": False, "spawn_at": o["rift_herald"]["spawn_s"], "respawn": None, "until": o["rift_herald"]["despawn_s"]},
        }
        self.gank_alert = {}   # (지키는 팀, 라인) → 마지막 갱 시각
        self.obj_taken = {"dragon": {100: 0, 200: 0}, "grubs": {100: 0, 200: 0}, "herald": {100: 0, 200: 0}}

    # ------------------------------------------------------------------ 조회
    def ally(self, c, role):
        return next(x for x in self.champs if x.team == c.team and x.role == role)

    def enemy(self, c, role):
        return next(x for x in self.champs if x.team != c.team and x.role == role)

    def avg_level(self):
        return sum(c.level for c in self.champs) / len(self.champs)

    def lane_node(self, lane):
        return f"{lane}_center"

    def tower_node(self, lane, team):
        return f"{lane}_{SIDE[team]}_outer"

    def wave_for(self, team, lane):
        w = self.wave[lane]
        return w if team == 100 else 1 - w

    def vision(self, team):
        return team_vision(team, self.champs, self.wards, self.map.nodes, self.t)

    # ------------------------------------------------------------------ 보상
    def give_xp(self, c, xp):
        c.xp += xp
        c.level = self.rules.level_for_xp(c.xp)

    def give_gold(self, c, g, gv=False):
        """gv=True: 미니언·몬스터 수입(현상금 누적 비율이 다름, rules/combat.json bounty)."""
        c.earned += g
        if gv and g > 0:
            b = self.rules.combat["bounty"]
            if c.bounty < 0:
                c.bounty = min(0.0, c.bounty + g / b["gv_gold_per_bounty_negative"])
            else:
                c.bounty += g / b["gv_gold_per_bounty"]

    def kill_value(self, victim):
        """처치 골드 = 레벨별 기본 + 현상금(셧다운은 기본+100 이상일 때만), 최소값 아래로는 안 내려감."""
        b = self.rules.combat["bounty"]
        base = b["base_by_level"][victim.level - 1]
        extra = victim.bounty if victim.bounty < 0 or victim.bounty >= b["shutdown_min"] else 0.0
        return max(b["min"], base + extra)

    def kill(self, killer, victim, assists):
        cb = self.rules.combat
        b = cb["bounty"]
        value = self.kill_value(victim)
        gold = value + (cb["first_blood_bonus"] if not self.first_blood else 0)
        self.first_blood = True
        self.give_gold(killer, gold)
        killer.bounty += value / b["kill_gold_per_bounty"]
        killer.kills += 1
        kxp = cb["kill_xp"]["base"] + cb["kill_xp"]["per_victim_level"] * (victim.level - 1)
        self.give_xp(killer, kxp)
        for a in assists:
            a.assists += 1
            share = value * b["assist_share"] / len(assists)
            self.give_gold(a, share)
            a.bounty += share / b["kill_gold_per_bounty"]
            self.give_xp(a, kxp * 0.5 / len(assists))
        victim.deaths += 1
        dist = value * (1 + (b["assist_share"] if assists else 0))
        victim.bounty = 0.0 if victim.bounty > 0 else victim.bounty - dist / b["death_gold_per_bounty"]
        died_at = victim.node
        victim.dead_until = self.t + cb["death_timer_s"][victim.level - 1]
        victim.path, victim.task, victim.hp = [], None, 1.0
        victim.node = f"{SIDE[victim.team]}_fountain"
        victim.exposures = []
        victim.respawned = True
        self.shop(victim)   # 죽어 있는 동안 상점 이용
        self.bump_info()
        self.events.append({"t": self.t, "type": "kill", "killer": killer.pid, "victim": victim.pid, "assists": [a.pid for a in assists],
                            "node": died_at})

    def bump_info(self, team=None):
        for tm in ((team,) if team else (100, 200)):
            self.info_version[tm] += 1

    # ------------------------------------------------------------------ 이동
    def move_to(self, c, dest):
        if c.path and c.edge_left > 0:
            self._reroute(c, dest)
            return
        c.path = self.map.path(c.node, dest)
        c.edge_left = self.map.dist(c.node, c.path[0]) if c.path else 0.0
        if c.pull_credit and c.path and self.map.nodes[c.node]["kind"] in ("camp", "scuttle"):
            c.edge_left -= c.pull_credit * c.ms     # 직전 캠프를 이쪽으로 당기며 잡았으니 그만큼 이미 와 있음
        c.pull_credit = 0.0
        if self.map.nodes[c.node]["kind"] == "fountain" and c.path and self.t >= 20:   # 0:20부터 받음
            self._start_homeguard(c)

    def _reroute(self, c, dest):
        """엣지 중간에서 목적지를 바꿀 때 지금까지 간 거리를 버리지 않는다(5단계: 이동 중 5초마다 다시 고름).

        c.node와 c.path[0] 사이에 있으므로, 앞 노드로 계속 가는 길과 뒤 노드로 돌아가는 길 중 짧은 쪽을 고른다.
        """
        ahead = c.path[0]
        traveled = max(0.0, self.map.dist(c.node, ahead) - c.edge_left)
        via_ahead = c.edge_left + self.map.dist(ahead, dest)
        via_back = traveled + self.map.dist(c.node, dest)
        if via_ahead <= via_back:
            c.path = [ahead] + self.map.path(ahead, dest)
        else:
            c.path = [c.node] + self.map.path(c.node, dest)
            c.edge_left = traveled
        c.pull_credit = 0.0

    def _start_homeguard(self, c):
        """우물 이속 효과(wiki:Homeguard, V26.01). 정글에 들어가면 사라지고, 라인에서는 외곽 포탑 500 뒤까지 유지."""
        hg = self.rules.combat["homeguard"]
        if c.role == "JUNGLE":
            c.homeguard_left = self.map.dist(c.node, c.path[0])   # 정글에 들어가면 사라지므로 우물에서 나가는 첫 엣지까지
        else:
            c.homeguard_left = max(0.0, self.map.dist(c.node, self.tower_node(c.lane, c.team)) - 500)
        c.homeguard_t = 0.0

    def homeguard_bonus(self, c):
        hg = self.rules.combat["homeguard"]
        if c.respawned and self.t < 840:
            return hg["respawn_before_14"]
        start, end = hg["spawn_before_14"] if self.t < 840 else hg["spawn_after_14"]
        return max(end, start - (start - end) * c.homeguard_t / hg["decay_s"])

    def _advance(self, c):
        if not c.path or self.t < c.busy_until:
            return
        step = c.ms
        if c.homeguard_left > 0:
            step = c.ms * (1 + self.homeguard_bonus(c))
            c.homeguard_left -= step
            c.homeguard_t += 1
        c.edge_left -= step
        while c.path and c.edge_left <= 0:
            c.node = c.path.pop(0)
            self._on_node(c)
            if c.path:
                c.edge_left += self.map.dist(c.node, c.path[0])

    def _on_node(self, c):
        """노드에 닿을 때 적 시야 노출을 기록하고, 적이 알아채면 그 팀의 '마지막으로 본 정글러'를 갱신한다."""
        enemy_team = 300 - c.team
        if c.node not in self.vision(enemy_team):
            return
        dwell = self.p["cognition"]["node_dwell_radius"] / c.ms
        if c.path:   # 목적지는 노출로 치지 않는다(도착하면 어차피 보임)
            c.exposures.append((self.t, dwell))
            c.exposures = c.exposures[-8:]
        if c.role == "JUNGLE" and self.rng.random() < detect_prob([dwell], self.cog):
            self.enemy_jgl_seen[enemy_team] = (c.node, self.t + self.cog["reaction_s"])
            self.bump_info(enemy_team)

    # ------------------------------------------------------------------ 정글
    def jungler_dps(self, c, epic=False):
        pj = self.p["jungler"]
        dps = pj["dps_level1"] * c.clear_mult * (1 + pj["dps_per_level"] * (c.level - 1)) \
            * (1 + pj["dps_per_1000_item_gold"] * c.item_gold / 1000)
        if epic:
            return dps * self.p["objective"]["epic_dps_factor"]
        pet = 20 + (150 - 20) * (c.level - 1) / 17          # positions.json pet_damage 기본 20~150
        bonus = self.rules.positions["jungle_item"]["monster_damage"]["bonus_damage_to_non_epic"]
        return dps * (1 + bonus) + pet

    def smite_damage(self, c):
        dmg = self.rules.positions["smite"]["damage_true"]
        th = self.rules.positions["smite"]["upgrade_thresholds_treats"]
        if c.treats >= th["primal"]:
            return dmg["primal"]
        if c.treats >= th["unleashed"]:
            return dmg["unleashed"]
        return dmg["smite"]

    def _formula_clear_time(self, c, kind, lv, use_smite, first=False):
        """체력 ÷ 초당 피해 모델. 절대값이 아니라 실측 기준 시간과의 비율로만 쓴다."""
        spec = self.rules.camps[kind]
        aoe = self.p["jungler"]["aoe_small_unit_factor"]
        large = small = 0.0
        for u in spec["units"]:
            hp_rng = u.get("hp_first_spawn") if (kind == "scuttle" and first) else u["hp"]
            hp = lerp_level(hp_rng, lv) * u.get("count", 1)
            if u.get("large"):
                large += hp
            else:
                small += hp
        if use_smite:
            large = max(large * 0.1, large - self.smite_damage(c))
        return 2.0 + (large + aoe * small) / self.jungler_dps(c)

    def _ref_ratio(self, kind):
        """실측 처치 시간 ÷ 같은 조건의 공식 시간. 실측이 없는 캠프(바위게)는 실측 캠프들의 평균."""
        cache = self.__dict__.setdefault("_ref_cache", {})
        if kind not in cache:
            ref = self.p["jungler"]["reference_clear"]
            def ratio(k):
                r = ref["camps"][k]
                me = SimpleNamespace(level=r["level"], clear_mult=1.0, item_gold=0.0, treats=0)
                return r["seconds"] / self._formula_clear_time(me, k, 1, r["smite"])
            if kind in ref["camps"]:
                cache[kind] = ratio(kind)
            else:
                cache[kind] = sum(ratio(k) for k in ref["camps"]) / len(ref["camps"])
        return cache[kind]

    def camp_clear_time(self, c, camp, use_smite):
        """kk 실측(마스터 이, 연습 모드) 처치 시간을 기준으로, 레벨·아이템·챔피언 차이·몬스터 레벨은 공식 비율로 반영."""
        t = self._formula_clear_time(c, camp.kind, camp.level, use_smite, camp.first)
        return t * self._ref_ratio(camp.kind)

    def camp_rewards(self, c, camp):
        spec = self.rules.camps[camp.kind]
        lv = camp.level
        gold = spec.get("camp_gold")
        if gold is None:
            gold = sum(lerp_level(u["gold"], lv) * u.get("count", 1) for u in spec["units"])
        xp = sum(lerp_level(u["xp"], lv) * u.get("count", 1) for u in spec["units"])
        n_large = sum(1 for u in spec["units"] if u.get("large"))
        bx = self.rules.positions["jungle_item"]["large_monster_bonus_xp"]
        xp += bx["per_kill"] * n_large
        if c.big_kills == 0:
            xp += bx["first_kill_extra"]
        if c.treats >= self.rules.positions["role_quests"]["jungle"]["required_points"]:
            gold += 10
            xp += 10
        c.big_kills += n_large
        c.treats += n_large
        return gold, xp

    def start_camp(self, c, camp):
        use = self.bots[c.pid].use_smite(camp)
        if use:
            c.smite_charges -= 1
            if c.smite_charges == 1 and c.smite_next <= self.t:
                c.smite_next = self.t + self.rules.positions["smite"]["recharge_s"]
        dur = self.camp_clear_time(c, camp, use)
        c.busy_until = self.t + dur
        c.task = {"kind": "clearing", "camp": camp.cid, "until": c.busy_until, "dur": dur}

    def finish_camp(self, c, camp):
        pj = self.p["jungler"]
        if camp.alive:
            gold, xp = self.camp_rewards(c, camp)
            c.monster_gold += gold
            self.give_gold(c, gold, gv=True)
            self.give_xp(c, xp)
            c.jungle_cs += 4
            camp.alive, camp.first = False, False
            respawn = self.rules.camps[camp.kind]["respawn_s"]
            camp.spawn_at = self.t + respawn
            c.hp = min(1.0, c.hp - pj["hp_loss_per_clear_s"] * c.task["dur"] + pj["kill_heal"])
            c.pull_credit = pj["pull_overlap_s"]
            self.events.append({"t": self.t, "type": "camp", "pid": c.pid, "camp": camp.cid})
        c.task = None

    # ------------------------------------------------------------------ 오브젝트
    def objective_time(self, c, name, helpers):
        lv = max(self.avg_level(), {"dragon": 6, "grubs": 7, "herald": 9}[name])
        o = self.rules.objectives
        if name == "dragon":
            hp = 3625 + 375 * (lv - 1)
        elif name == "grubs":
            hp = lerp_level(o["voidgrubs"]["stats_v26_1"]["hp_by_level"], lv) * 3
        else:
            hp = lerp_level(o["rift_herald"]["stats_v26_1"]["hp_by_level"], lv)
        dps = self.jungler_dps(c, epic=True) + len(helpers) * self.p["objective"]["helper_dps"]
        return hp / dps

    def finish_objective(self, c, name, helpers):
        ob = self.objectives[name]
        if not ob["alive"]:
            c.task = None
            return
        rival = self.enemy(c, "JUNGLE")
        if rival.task and rival.task.get("kind") == "objective" and rival.task.get("name") == name and rival.alive(self.t) and rival.node == ob["node"]:
            # 둘 다 오브젝트 앞: 레벨 차로 승자를 뽑고 진 쪽은 일정 확률로 사망
            p_win = 0.5 + 0.1 * (c.level - rival.level)
            if self.rng.random() > p_win:
                c, rival, helpers = rival, c, rival.task.get("helpers", [])
            if self.rng.random() < self.p["objective"]["contest_loser_death"]:
                self.kill(c, rival, [])
            rival.task = None
        o = self.rules.objectives
        lv = self.avg_level()
        group = [c] + [h for h in helpers if h.alive(self.t)]
        if name == "dragon":
            for m in group:
                self.give_gold(m, o["drake"]["rewards"]["kill_gold"])
                self.give_xp(m, lerp_level(o["drake"]["rewards"]["local_xp"], lv))
        elif name == "grubs":
            r = o["voidgrubs"]["rewards_each"]
            for m in group:
                self.give_gold(m, r["local_gold"] * 3)
                self.give_xp(m, r["xp"] * 3)
        else:
            r = o["rift_herald"]["rewards"]
            self.give_gold(c, r["gold_to_killer"])
            for m in group:
                self.give_xp(m, r["xp"])
        self.obj_taken[name][c.team] += 1
        ob["alive"] = False
        ob["spawn_at"] = self.t + ob["respawn"] if ob["respawn"] else None
        self.events.append({"t": self.t, "type": "objective", "name": name, "team": c.team, "pid": c.pid})
        c.task = None

    # ------------------------------------------------------------------ 귀환
    def start_recall(self, c):
        c.path = []
        c.busy_until = self.t + self.rules.combat["recall_channel_s"]
        c.task = {"kind": "recalling"}

    def shop(self, c):
        """남은 골드를 모두 아이템으로. 첫 300골드 넘는 구매에 신발(정글러 이속)을 넣는다."""
        buy = c.unspent
        if c.item_gold < 300 <= c.item_gold + buy and c.role == "JUNGLE":
            c.ms += self.p["jungler"]["boots_ms"]
        c.item_gold += buy
        c.spent += buy

    def finish_recall(self, c):
        c.node = f"{SIDE[c.team]}_fountain"
        c.respawned = False
        c.hp = 1.0
        self.shop(c)
        if c.role == "JUNGLE" and c.first_recall_s is None:
            c.first_recall_s = self.t
        self.events.append({"t": self.t, "type": "recall", "pid": c.pid})
        c.task = None

    # ------------------------------------------------------------------ 틱
    def _spawns(self):
        for camp in self.camps.values():
            if not camp.alive and self.t >= camp.spawn_at:
                camp.alive, camp.level = True, max(1, round(self.avg_level()))
        for ob in self.objectives.values():
            if ob["until"] is not None and self.t >= ob["until"]:
                ob["alive"] = False
                continue
            if not ob["alive"] and ob["spawn_at"] is not None and self.t >= ob["spawn_at"]:
                ob["alive"], ob["spawn_at"] = True, None
                self.bump_info()
        self.wards = [w for w in self.wards if w["until"] > self.t]

    def _economy(self, c):
        if self.t >= self.rules.passive_start_s:
            self.give_gold(c, self.rules.combat["passive_gold_per_s"])
        if c.role == "JUNGLE":
            pj = self.p["jungler"]
            if self.t >= c.next_bonus_treat:
                c.treats += 1
                self.give_gold(c, self.rules.positions["jungle_item"]["treats"]["bonus_treat_gold"])
                done = c.treats >= self.rules.positions["role_quests"]["jungle"]["required_points"]
                tr = self.rules.positions["jungle_item"]["treats"]
                c.next_bonus_treat = self.t + (tr["bonus_treat_every_s_after_final"] if done else tr["bonus_treat_every_s"])
            if c.smite_charges < 2 and self.t >= c.smite_next:
                c.smite_charges += 1
                c.smite_next = self.t + self.rules.positions["smite"]["recharge_s"]
            if not c.task or c.task.get("kind") not in ("clearing", "objective"):
                c.hp = min(1.0, c.hp + pj["hp_regen_per_s"])

    def _finish_tasks(self, c):
        if not c.task or self.t < c.busy_until:
            return
        kind = c.task["kind"]
        if kind == "clearing":
            self.finish_camp(c, self.camps[c.task["camp"]])
        elif kind == "recalling":
            self.finish_recall(c)
        elif kind == "objective" and c.task.get("started"):
            self.finish_objective(c, c.task["name"], c.task.get("helpers", []))
        elif kind == "ganking" and c.task.get("resolved"):
            c.task = None

    def _skirmish(self):
        """정글·강가 조우 교전: 두 정글러가 가까이 있으면 확률적으로 교전이 난다(3단계 '적 정글러 조우 시 교전 모듈')."""
        ps = self.p["skirmish"]
        j1, j2 = (next(c for c in self.champs if c.team == t and c.role == "JUNGLE") for t in (100, 200))
        if not (j1.alive(self.t) and j2.alive(self.t)) or self.t - getattr(self, "_last_skirmish", -999) < ps["cooldown_s"]:
            return
        if any(self.map.nodes[j.node]["kind"] == "fountain" or (j.task and j.task.get("kind") == "recalling") for j in (j1, j2)):
            return
        n1, n2 = self.map.nodes[j1.node], self.map.nodes[j2.node]
        if ((n1["x"] - n2["x"]) ** 2 + (n1["y"] - n2["y"]) ** 2) ** 0.5 > ps["encounter_range"]:
            return
        if self.rng.random() >= ps["engage_prob_per_s"]:
            return
        self._last_skirmish = self.t
        sides = {}
        for j in (j1, j2):
            side = [j]
            for c in self.champs:
                if c.team == j.team and c.role != "JUNGLE" and c.alive(self.t) and not c.task \
                        and self.map.travel_s(c.node, j.node, c.ms) <= ps["join_travel_s"] and self.rng.random() < ps["join_prob"]:
                    side.append(c)
            sides[j.team] = side
        res = events.resolve_skirmish(sides, self.t, self.rng, ps)
        for pid, hp in res.hp_after.items():
            next(x for x in self.champs if x.pid == pid).hp = hp
        for m in sides[100] + sides[200]:
            m.busy_until = max(m.busy_until, self.t) + res.duration_s
            if m.task and m.task.get("kind") == "clearing":
                m.task["until"] = m.busy_until
        for victim, killer, assists in res.deaths:
            self.kill(killer, victim, assists)
        self.events.append({"t": self.t, "type": "skirmish", "node": j1.node, "winner": res.winner,
                            "sizes": [len(sides[100]), len(sides[200])], "deaths": len(res.deaths)})

    def record_frame(self):
        self.frames.append({
            "t": self.t,
            "p": {c.pid: {"role": c.role, "team": c.team, "gold": round(c.total_gold), "jungle_cs": c.jungle_cs,
                          "lane_cs": int(c.lane_cs), "level": c.level, "xp": round(c.xp),
                          "kills": c.kills, "deaths": c.deaths, "assists": c.assists} for c in self.champs},
        })

    def step(self):
        self.t += 1
        self._spawns()
        for c in self.champs:
            if not c.alive(self.t):
                continue
            self._economy(c)
            self._finish_tasks(c)
            self._advance(c)
        self._skirmish()
        for c in self.champs:
            if c.alive(self.t):
                self.bots[c.pid].tick()
        if self.t % 60 == 0:
            self.record_frame()

    def run(self, until=GAME_END_S):
        self.record_frame()
        while self.t < until:
            self.step()
        return self
