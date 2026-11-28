"""미드 라인전 0~10분 1:1 (웨이브·CS·레벨·귀환·사망 포함). sim.duel의 교전 규칙을 그대로 쓰고 위에 라인을 얹는다.

    python -m sim.lane_duel --pair Syndra Yasuo --n 50
좌표: 1차원. 미드 웨이브는 0에서 맞부딪힌다고 둔다(밀고 당기기는 아직 없음). 블루 포탑 −1500, 레드 +1500(미검증 거리).
CS: 상대 웨이브 미니언이 죽을 때마다, 웨이브 근처(사거리+150)에 있고 바쁘지 않으면 HUMAN_LANE.last_hit 확률로 먹음.
미니언 골드·경험치는 sim/params.json, 레벨 경험치·부활 시간은 rules(검증값).
"""
import argparse
import json
import os
import random

from sim import duel

ROOT = duel.ROOT
COMBAT = json.load(open(os.path.join(ROOT, "rules", "combat.json"), encoding="utf-8"))
DEATH = COMBAT["death_timer_s"]["value"]
XP_TABLE = [0, 280, 660, 1140, 1720, 2400, 3180, 4060, 5040, 6120, 7300, 8580, 9960, 11440, 13020, 14700, 16480, 18360]
# sim/params.json의 minion_gold·minion_xp와 같은 값(근접 20·대포 50+90초마다 1·대포 경험치 75는 검증, 나머지 미검증)
MG = {"melee": 20, "caster": 14, "cannon_base": 50, "cannon_every_s": 90}
MX = {"melee": 61, "caster": 30, "cannon": 75}
KITS16 = json.load(open(os.path.join(ROOT, "rules", "champ_kits_16.23.1.json"), encoding="utf-8"))
KITS16 = KITS16.get("champions", KITS16)

# 사람 값(미검증, 실제 10분 CS 평균에 맞출 것)
HUMAN_LANE = {"last_hit": 0.97, "wave_first_s": 65.0, "wave_every_s": 30.0, "minion_gap_s": 3.0,
              "tower": 1500.0, "walk_s": 22.0, "recall_s": 8.0, "recall_hp": 0.3, "recall_gold": 1250.0,
              "kill_gold": 300.0, "first_blood": 400.0, "farm_back": 120.0, "start_gold_items": 450.0,
              # 포탑 사거리(750, 미검증 기억값) 안으로는 쫓지 않음. 점멸(300초, 400 거리) 도망. 시작 포션 2개(15초 120)
              "tower_range": 750.0, "flash_cd": 300.0, "flash_dist": 400.0, "flash_hp": 0.3,
              "potions": 2, "potion_hp": 120.0, "potion_s": 15.0, "potion_at": 0.65,
              # 콤보를 다 쏟는 건 '죽일 수 있을 때'만, 아니면 스킬 하나 교환 후 빠짐(초안)
              "all_in_frac": 0.9, "farm_reach": 550.0,
              # 시작 회복(미검증 기억값): 근접 미드 = 도란 방패류+두 번째 바람으로 5초당 체력 회복 추가, 원거리 = 도란 반지 마나
              "melee_hp5": 12.0, "ranged_mp5": 6.0}


def regen(name):
    s = KITS16.get(name, {}).get("stats", {})
    return s.get("hpregen", 5.0), s.get("hpregenperlevel", 0.5), s.get("mpregen", 7.0), s.get("mpregenperlevel", 0.7)


class Laner:
    def __init__(self, name, kits, side, rng):
        self.name, self.side, self.kits, self.rng = name, side, kits, rng
        self.lv, self.xp, self.gold, self.items = 1, 0.0, 500.0, HUMAN_LANE["start_gold_items"]
        self.cs = self.kills = self.deaths = 0
        self.spent_mark = 500.0
        self.flash_ready = 0.0
        self.potions = HUMAN_LANE["potions"]
        self.potion_until = 0.0
        self.disengage_until = 0.0
        self.f = duel.Fighter(name, kits, 1, self.items, side)
        self.f.x = self.home_spot()
        self.away_until = 0.0      # 사망·귀환으로 라인에 없음
        self.recall_end = None
        self.hr, self.hrl, self.mr, self.mrl = regen(name)
        if self.f.range <= 300:
            self.hr += HUMAN_LANE["melee_hp5"]
        else:
            self.mr += HUMAN_LANE["ranged_mp5"]

    def home_spot(self):
        # 웨이브(0) 뒤 자기 쪽 막타 위치: 원거리는 사거리 0.8, 근접은 120
        back = max(self.f.range * 0.8, HUMAN_LANE["farm_back"])
        return -back if self.side == 0 else back

    def relevel(self):
        while self.lv < 18 and self.xp >= XP_TABLE[self.lv]:
            self.lv += 1
        hp_frac = self.f.hp / self.f.max_hp
        mana_frac = self.f.mana / self.f.max_mana if self.f.max_mana < 1e8 else 1
        ready = {k: v["ready"] for k, v in self.f.spells.items()}
        self._old_spells = dict(self.f.spells)
        keep = {k: getattr(self.f, k) for k in ("x", "cc_until", "busy_until", "aa_ready", "dealt", "shield", "magic_shield",
                                                "last_hurt", "empower_ready", "empowered", "proc_hits", "proc_lock",
                                                "wall_ready", "wall_until")}
        self.f = duel.Fighter(self.name, self.kits, self.lv, self.items, self.side)
        self.f.hp, self.f.mana = hp_frac * self.f.max_hp, mana_frac * self.f.max_mana
        for k, v in self.f.spells.items():
            v["ready"] = ready.get(k, 0.0)
            old = self._old_spells.get(k)
            if old and v["max_charges"] > 1:
                v["charges"], v["recharge_at"] = old["charges"], old["recharge_at"]
        for k, v in keep.items():
            setattr(self.f, k, v)

    def present(self, t):
        return t >= self.away_until and self.recall_end is None


def lane_game(a_name, b_name, kits, seconds=600.0, seed=0):
    rng = random.Random(seed)
    L = [Laner(a_name, kits, 0, rng), Laner(b_name, kits, 1, rng)]
    t, first_blood = 0.0, True
    # 미니언 죽는 시각(웨이브마다 6마리 + 90초마다 대포). 양쪽 같은 시각(웨이브 대칭 가정)
    deaths = []
    w = 0
    while True:
        t0 = HUMAN_LANE["wave_first_s"] + w * HUMAN_LANE["wave_every_s"] + 8.0
        if t0 > seconds:
            break
        kinds = ["melee"] * 3 + ["caster"] * 3 + (["cannon"] if (w % 3 == 2) else [])
        for i, k in enumerate(kinds):
            deaths.append((t0 + i * HUMAN_LANE["minion_gap_s"], k))
        w += 1
    di = 0
    while t < seconds:
        # 미니언 처치 기회
        while di < len(deaths) and deaths[di][0] <= t:
            _, kind = deaths[di]
            di += 1
            for me in L:
                if not me.present(t):
                    continue
                gold = MG["melee"] if kind == "melee" else MG["caster"] if kind == "caster" else MG["cannon_base"] + int(t // MG["cannon_every_s"])
                # 막타는 평타 또는 스킬로(웨이브 정리 스킬). 닿는 거리 = max(평타 사거리, farm_reach)
                near = abs(me.f.x) <= max(me.f.range, HUMAN_LANE["farm_reach"]) + 150
                if near:
                    me.xp += MX[kind]
                    if t >= me.f.busy_until and t >= me.f.cc_until and rng.random() < HUMAN_LANE["last_hit"]:
                        me.cs += 1
                        me.gold += gold
                        me.f.busy_until = max(me.f.busy_until, t + 0.25 / me.f.as_)
                elif abs(me.f.x) <= 1600:
                    me.xp += MX[kind]
            for me in L:
                me.relevel()
        a, b = L
        both = a.present(t) and b.present(t)
        for me, op in ((a, b), (b, a)):
            f = me.f
            f.refill(t)
            f.tick_charges(t)
            # 재생(초당). 마나는 자원형만
            f.hp = min(f.max_hp, f.hp + (me.hr + me.hrl * (me.lv - 1)) / 5 * duel.DT)
            if me.potions and f.hp / f.max_hp < HUMAN_LANE["potion_at"] and t >= me.potion_until:
                me.potions -= 1
                me.potion_until = t + HUMAN_LANE["potion_s"]
            if t < me.potion_until:
                f.hp = min(f.max_hp, f.hp + HUMAN_LANE["potion_hp"] / HUMAN_LANE["potion_s"] * duel.DT)
            if f.max_mana < 1e8:
                f.mana = min(f.max_mana, f.mana + (me.mr + me.mrl * (me.lv - 1)) / 5 * duel.DT)
            if me.recall_end is not None and t >= me.recall_end:
                me.recall_end = None
                me.items += me.gold - me.spent_mark   # 그때까지 번 골드를 전부 아이템으로
                me.spent_mark = me.gold
                me.away_until = t + HUMAN_LANE["walk_s"]
                me.f.hp, me.f.mana = me.f.max_hp, me.f.max_mana
                me.potions = max(me.potions, 1)   # 귀환 때 포션 1개 산다고 둠
                me.relevel()
                me.f.x = me.home_spot()
            if not me.present(t) or not op.present(t):
                if me.present(t):
                    f.x = me.home_spot()
                continue
            # 귀환 판단: 체력 낮음, 또는 돈이 모였고 상대 근처가 아님
            if (f.hp / f.max_hp < HUMAN_LANE["recall_hp"] or (me.gold - me.spent_mark >= HUMAN_LANE["recall_gold"] and t > 240)) \
                    and duel.dist(f, op.f) > 900:
                me.recall_end = t + HUMAN_LANE["recall_s"]
                me.away_until = t + HUMAN_LANE["recall_s"]
                continue
        if both:
            # 교전 한 틱(duel 규칙). 웨이브 근처에 서 있으려는 힘: 교전 생각이 없으면 막타 자리로
            step_duel(a, b, t, rng)
            for me in L:
                if me.f.hp <= 0:
                    killer = b if me is a else a
                    killer.kills += 1
                    killer.gold += HUMAN_LANE["first_blood"] if first_blood else HUMAN_LANE["kill_gold"]
                    first_blood = False
                    me.deaths += 1
                    me.away_until = t + DEATH[me.lv - 1] + HUMAN_LANE["walk_s"]
                    me.f.hp, me.f.mana = me.f.max_hp, me.f.max_mana
                    me.f.x = me.home_spot()
        t += duel.DT
    a, b = L
    return {"cs": [a.cs, b.cs], "gold": [a.gold, b.gold], "kills": [a.kills, b.kills], "lv": [a.lv, b.lv]}


def step_duel(a, b, t, rng):
    """sim.duel 루프 한 틱을 라인 위에서. 싸울 생각이 없으면 막타 자리 쪽으로 돌아감."""
    for me, op in ((a, b), (b, a)):
        f, o = me.f, op.f
        if f.hp <= 0 or o.hp <= 0 or t < f.busy_until or t < f.cc_until:
            continue
        # 점멸 도망: 체력 낮고 상대 콤보에 죽을 수 있으면
        if t >= me.flash_ready and f.hp / f.max_hp < HUMAN_LANE["flash_hp"]:
            danger = sum(o.spell_damage(s, f) for s in o.spells.values() if s["ready"] <= t)
            if danger >= f.hp and duel.dist(f, o) < 800:
                f.x += HUMAN_LANE["flash_dist"] * (1 if f.x > o.x else -1)
                me.flash_ready = t + HUMAN_LANE["flash_cd"]
                continue
        f.self_heal(t)
        ready = [s for s in f.spells.values() if s["ready"] <= t and f.mana >= s["cost"] and not s["self_cast"]
                 and (not s["needs_cc"] or o.cc_until > t)]
        op_ready = [s for s in o.spells.values() if s["ready"] <= t]
        my_burst = sum(f.spell_damage(s, o) for s in ready)
        op_burst = sum(o.spell_damage(s, f) for s in op_ready)
        low = f.hp / f.max_hp < duel.HUMAN["retreat_hp"] and f.hp < o.hp
        kill_threat = my_burst >= o.hp * HUMAN_LANE["all_in_frac"]
        want = ready and not low and (my_burst >= op_burst * 0.8 or not op_ready or kill_threat)
        d = duel.dist(f, o)
        # 상대 포탑 사거리 안으로는 안 들어감
        enemy_tower = HUMAN_LANE["tower"] if me.side == 0 else -HUMAN_LANE["tower"]
        limit = enemy_tower - HUMAN_LANE["tower_range"] * (1 if me.side == 0 else -1)
        toward = -1 if f.x > o.x else 1
        step = f.ms * duel.DT * toward
        if want:
            best = max(ready, key=lambda s: (f.spell_damage(s, o) / max(s["cd"], 1), s["range"]))
            dashes = [s for s in ready if s["dash"]]
            if d > best["range"] and dashes:
                dsp = max(dashes, key=lambda s: s["range"])
                if d - dsp["range"] <= max(best["range"], f.range) + 50 or d <= dsp["range"]:
                    f.x += min(dsp["range"], max(d - f.range, 0.0)) * toward
                    best = dsp
                    d = duel.dist(f, o)
            if d <= best["range"]:
                if not kill_threat:
                    # 견제 한 번 뒤엔 다른 스킬로 이어가지 않고 빠짐: 남은 준비 스킬을 짧게 잠금
                    for s2 in ready:
                        if s2 is not best:
                            s2["ready"] = max(s2["ready"], t + 1.5)
                f.busy_until = t + best["cast"]
                if not kill_threat:
                    me.disengage_until = t + best["cast"] + 1.5
                f.mana -= best["cost"]
                f.use(best, t)
                f.empowered = True
                if not o.blocks(best, t) and duel.resolve_hit(best, f, o, rng, t):
                    dmg = f.spell_damage(best, o) + f.proc(o, t)
                    f.dealt += o.take(dmg, best["magic"], t)
                    if best.get("heal"):
                        f.hp = min(f.max_hp, f.hp + f.heal_amount(best))
                    if best["cc"]:
                        o.cc_until = max(o.cc_until, t + best["cast"] + best["cc"])
            else:
                nx = f.x + step
                if (me.side == 0 and nx < limit) or (me.side == 1 and nx > limit):
                    f.x = nx
        else:
            spot = me.home_spot()
            keep = max([s["range"] for s in op_ready] + [o.range]) + duel.HUMAN["spacing"]
            tower = -HUMAN_LANE["tower"] if me.side == 0 else HUMAN_LANE["tower"]
            # 기본은 막타 자리. 물러나는 건 (a) 체력 낮음 (b) 상대 콤보에 죽을 수 있음 (c) 방금 견제하고 빠지는 중일 때만
            op_threat = op_burst >= f.hp * HUMAN_LANE["all_in_frac"]
            backing = t < me.disengage_until
            if (low or op_threat or backing) and d < keep and abs(f.x - tower) > 50:
                f.x -= step
            elif abs(f.x - spot) > 30 and not (low or op_threat or backing):
                f.x += f.ms * duel.DT * (1 if spot > f.x else -1)
        if d <= f.range + 35 and t >= f.aa_ready and t >= f.busy_until:
            dmg = f.ad * 100 / (100 + o.armor)
            f.dealt += o.take(dmg, False, t) + o.take(f.aa_bonus(o, t, rng), True, t)
            f.aa_ready = t + 1 / f.as_
            f.busy_until = t + 0.25 / f.as_


def matchup(a, b, kits, n=40, seconds=600.0):
    acc = {"cs_diff": 0.0, "gold_diff": 0.0, "kill_diff": 0.0, "cs_a": 0.0}
    for i in range(n):
        if i % 2 == 0:
            r = lane_game(a, b, kits, seconds, seed=i)
            sa, sb = 0, 1
        else:
            r = lane_game(b, a, kits, seconds, seed=i)
            sa, sb = 1, 0
        acc["cs_diff"] += (r["cs"][sa] - r["cs"][sb]) / n
        acc["gold_diff"] += (r["gold"][sa] - r["gold"][sb]) / n
        acc["kill_diff"] += (r["kills"][sa] - r["kills"][sb]) / n
        acc["cs_a"] += r["cs"][sa] / n
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    kits = duel.load()
    print(lane_game(*a.pair, kits))
    print(matchup(*a.pair, kits, n=a.n))


if __name__ == "__main__":
    main()
