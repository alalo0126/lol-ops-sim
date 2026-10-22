"""라인 모듈: 웨이브 도착·라인 CS·웨이브 밀기·포탑(외곽·2차·억제기 포탑, 방패 5개씩).

v1 라이너는 초당 고정 수입을 받았다(params laner.income_per_s). 여기서는 웨이브가 도착할 때 라인에 있는 챔피언만
미니언을 먹는다. 그래서 귀환·로밍·사망·갱 대응으로 라인을 비우면 CS와 경험치를 잃고, 비운 쪽으로 웨이브가 밀려
포탑과 방패를 잃는다.

상태
- Game.wave[라인]: 블루 기준 웨이브 위치(0 = 블루 포탑 앞, 1 = 레드 포탑 앞). 라인에 선 양 팀 힘 차이로 움직인다.
- towers[팀][라인]: [외곽, 2차, 억제기] 남은 방패(0~5, 실수). 맨 앞 살아 있는 포탑이 공격받는다.

'라인에 있음' = 그 라인 중앙 노드나 그 라인 포탑 노드에 살아 있고 귀환·교전 중이 아님. 맵에는 외곽 포탑 노드만 있어서
2차·억제기 포탑 공성도 같은 노드에서 한다(0~20분 범위의 단순화).

검수: tools/laner_targets.py(1,000판 라이너 분포), tools/tower_gold_fit.py(방패·포탑 골드 회귀), sim/laner_eval.py.
"""
import math

from . import events

LANES = ("top", "mid", "bot")
TIERS = ("outer", "inner", "inhib")


class Lanes:
    def __init__(self, game):
        self.g = game
        self.p = game.p["lanes"]
        self.towers = {T: {ln: [float(self.p["plates_per_turret"])] * len(TIERS) for ln in LANES} for T in (100, 200)}
        self.first_tower = False
        self.wave_no = 0
        self.next_wave = self.p["first_arrival_s"]
        self.last_fight = {ln: -999.0 for ln in LANES}
        self.backlog = {(T, ln): 0.0 for T in (100, 200) for ln in LANES}   # 아무도 없을 때 쌓인 상대 미니언(웨이브 수)

    # ------------------------------------------------------------ 조회
    def present(self, team, lane):
        g = self.g
        nodes = (g.lane_node(lane), g.tower_node(lane, 100), g.tower_node(lane, 200))
        return [c for c in g.champs if c.team == team and c.alive(g.t) and c.node in nodes and not c.path
                and not (c.task and c.task.get("kind") in ("recalling", "helping", "ganking", "clearing", "objective"))]

    def front(self, team, lane):
        """그 팀 라인의 맨 앞 살아 있는 포탑 단계 번호(없으면 None)."""
        return next((i for i, hp in enumerate(self.towers[team][lane]) if hp > 0), None)

    def towers_lost(self, team):
        return sum(1 for ln in LANES for hp in self.towers[team][ln] if hp <= 0)

    def strength(self, c):
        p = self.p["push"]
        return p["role"][c.role] * (1 + p["per_level"] * (c.level - 1) + p["per_1000_item_gold"] * c.item_gold / 1000)

    # ------------------------------------------------------------ 틱
    def tick(self):
        g = self.g
        if g.t >= self.next_wave:
            self.wave_arrival()
            self.wave_no += 1
            self.next_wave += self.interval(g.t)
        if g.t < self.p["first_arrival_s"]:
            return
        p = self.p["push"]
        for ln in LANES:
            here = {T: self.present(T, ln) for T in (100, 200)}
            s = {T: sum(self.strength(c) for c in here[T]) for T in (100, 200)}
            for T in (100, 200):
                for c in here[T]:
                    if c.role == "UTILITY":
                        g.give_gold(c, self.p["support_item_gold_per_s"], gv=True)
            w = g.wave[ln] + p["rate"] * (s[100] - s[200]) + p["revert"] * (0.5 - g.wave[ln]) + g.rng.gauss(0, p["sigma"])
            g.wave[ln] = min(1.0, max(0.0, w))
            self.lane_fight(ln, here)
            for att, w_att in ((100, g.wave[ln]), (200, 1 - g.wave[ln])):
                if w_att >= self.p["siege_at"]:
                    self.siege(att, ln, s[att])

    def lane_fight(self, ln, here):
        """정글러 없는 라인 교전(솔로킬·2대2). 양 팀 라이너가 라인에 있을 때 초당 확률로 시작한다.

        1단계 1,000판에서 14분 전 라이너 사망 중 정글러가 관여하지 않은 것: 탑 55%, 미드 46%, 원딜 60%, 서폿 48%.
        승패·사망은 정글 조우 교전과 같은 식(events.resolve_skirmish), 시작 확률만 라인별로 맞춘다.
        """
        g, pf = self.g, self.p["lane_fight"]
        sides = {T: [c for c in here[T] if c.role != "JUNGLE"] for T in (100, 200)}
        if not sides[100] or not sides[200] or g.t - self.last_fight[ln] < pf["cooldown_s"]:
            return
        if g.rng.random() >= pf["rate_per_s"][ln][0 if g.t < 840 else 1]:
            return
        self.fight(ln, here)

    def fight(self, ln, here, kind="lane_fight", resolve=None):
        """그 라인에 선 양 팀 라이너끼리 교전(정글러 제외). 한쪽이 비어 있으면 안 함. 로밍 도착(bots.LanerBot.roam_tick)도
        씀(resolve = 로밍 교전용 사망 확률)."""
        g, pf = self.g, self.p["lane_fight"]
        sides = {T: [c for c in here[T] if c.role != "JUNGLE"] for T in (100, 200)}
        if not sides[100] or not sides[200]:
            return
        self.last_fight[ln] = g.t
        res = events.resolve_skirmish(sides, g.t, g.rng, {**g.p["skirmish"], **(resolve or pf["resolve"])})
        for pid, hp in res.hp_after.items():
            next(x for x in g.champs if x.pid == pid).hp = hp
        for victim, killer, assists in res.deaths:
            g.kill(killer, victim, assists)
        g.events.append({"t": g.t, "type": kind, "lane": ln, "winner": res.winner, "deaths": len(res.deaths)})

    def interval(self, t):
        for step in self.g.rules.minions["wave_interval_s"]:
            if step["until_s"] is None or t < step["until_s"]:
                return step["interval"]

    def wave_arrival(self):
        """양 팀 웨이브가 라인에 도착: 팀마다 상대 웨이브 미니언을 라인에 선 챔피언이 먹는다."""
        g, pv = self.g, self.p
        n_melee, n_caster = 3, 3
        cannon = self.has_cannon(self.wave_no, g.t)
        if cannon and g.t >= 840:
            n_melee -= 1
        mg = pv["minion_gold"]
        mx = pv["minion_xp"]
        gold_cannon = mg["cannon_base"] + int(g.t // mg["cannon_every_s"])
        wave_gold = n_melee * mg["melee"] + n_caster * mg["caster"] + (gold_cannon if cannon else 0)
        wave_xp = n_melee * mx["melee"] + n_caster * mx["caster"] + (mx["cannon"] if cannon else 0)
        wave_n = n_melee + n_caster + (1 if cannon else 0)
        eff = pv["last_hit"]
        for ln in LANES:
            for T in (100, 200):
                if g.t < g.lane_taken_until[ln] and any(c.team == T and c.role == "JUNGLE" and c.task and c.task.get("kind") == "lanefarm"
                                                       for c in g.champs):
                    continue   # 정글러가 이 웨이브를 먹음(agent.finish_lanefarm이 보상)
                here = self.present(T, ln)
                if not here:
                    # 비운 라인의 상대 웨이브는 다 사라지지 않고 일부가 남아 돌아왔을 때 먹을 수 있다
                    self.backlog[(T, ln)] = min(pv["backlog"]["max_waves"], self.backlog[(T, ln)] + pv["backlog"]["keep"])
                    continue
                waves = 1.0 + self.backlog[(T, ln)]
                self.backlog[(T, ln)] = 0.0
                # 라인에 너무 밀려 포탑 앞에서 받아먹으면 일부를 포탑이 먹는다
                w_own = g.wave[ln] if T == 100 else 1 - g.wave[ln]
                loss = pv["under_tower_loss"] if w_own <= 1 - pv["siege_at"] else 0.0
                # 막타는 비율이 높은 사람부터 가져가고 남은 미니언을 다음 사람이(원딜이 먹고 남은 것을 서폿이)
                left = 1.0 - loss
                xp_each = wave_xp * waves * pv["xp_share_mult"][min(len(here), 3) - 1] / len(here)
                for c in sorted(here, key=lambda m: -eff.get(m.role, eff["JUNGLE"])):
                    got = min(left, eff.get(c.role, eff["JUNGLE"]) * (1 - loss))
                    left -= got
                    share = got * waves
                    c.lane_cs += wave_n * share
                    g.give_gold(c, wave_gold * share, gv=True)
                    if c.role == "JUNGLE":
                        c.minion_gold += wave_gold * share
                    g.give_xp(c, xp_each)   # 경험치는 근처면 나눠 받음(막타와 무관)

    def has_cannon(self, k, t):
        """대포: 3번째 웨이브부터, 14:00 전 3웨이브마다, 이후 2웨이브마다(minions.json)."""
        if k < 2:
            return False
        return (k - 2) % (3 if t < 840 else 2) == 0

    def siege(self, att, ln, champ_strength):
        g, pv = self.g, self.p
        dfn = 300 - att
        i = self.front(dfn, ln)
        if i is None:
            return
        d = pv["tower_damage"]
        dmg = (d["minions"] + d["per_strength"] * champ_strength) * d["tier_mult"][i]
        dmg *= min(1.0, g.t / d["full_at_s"]) ** d["early_power"]   # 초반 포탑은 잘 안 깨짐(방패 시각 분포에 맞춘 증가 곡선)
        before = self.towers[dfn][ln][i]
        after = max(0.0, before - dmg)
        self.towers[dfn][ln][i] = after
        here = self.present(att, ln)
        broken = math.ceil(before) - math.ceil(after) - (1 if after <= 0 else 0)   # 마지막 칸은 포탑 파괴로 계산
        for _ in range(broken):
            self.plate(att, dfn, ln, i, here)
        if after <= 0 < before:
            self.destroy(att, dfn, ln, i, here)

    def plate(self, att, dfn, ln, i, here):
        g = self.g
        for c in here:
            g.give_gold(c, self.p["plate_gold_local"] / len(here))
        g.events.append({"t": g.t, "type": "plate", "team": att, "lane": ln, "tier": TIERS[i], "pids": [c.pid for c in here]})

    def destroy(self, att, dfn, ln, i, here):
        g, pv = self.g, self.p
        for c in g.champs:
            if c.team == att:
                g.give_gold(c, pv["tower_gold_global"][TIERS[i]])
        if here:
            local = pv["plate_gold_local"]   # 마지막 방패 몫
            if not self.first_tower:
                local += g.rules.objectives["first_blood_and_turret_gold"]["first_turret_bonus"]
            for c in here:
                g.give_gold(c, local / len(here))
        self.first_tower = True
        g.wave[ln] = 0.5   # 포탑을 깨면 웨이브가 녹고 다시 가운데에서 맞붙는다
        g.bump_info()
        g.events.append({"t": g.t, "type": "tower", "team": att, "lane": ln, "tier": TIERS[i], "pids": [c.pid for c in here]})

