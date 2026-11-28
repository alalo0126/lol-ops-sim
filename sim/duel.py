"""라인전 1:1 교전 층 초안(0.1초, 1차원 거리). 스킬·평타로 '턴 교환'을 흉내 낸다.

    python -m sim.duel --level 6 --gold 1500 --n 200          # 미드 상위 챔피언 맞대결 표
    python -m sim.duel --pair Syndra Yone --level 3 --n 500

입력: rules/champ_skills_16.23.json(CommunityDragon 16.23) + 아래 KIT_OVERRIDES(Claude가 고른 피해 계산식·투사체).
모델(전부 초안, 실제 경기로 맞출 값은 HUMAN에 모음):
  - 거리 d 하나만 본다(옆 피하기는 확률로). 미니언 어그로·웨이브·포탑은 아직 없음.
  - 대상 지정: 사거리 안이면 맞음. 스킬샷: 도착 시간(선딜+거리/속도)이 반응 시간+옆으로 빠지는 시간보다 길면
    HUMAN.dodge 확률로 피함. 범위 지정: 선딜+지연(area_delay) 뒤 같은 식.
  - CC 태그(Trait_ImmobilizingCCSpell) 스킬이 맞으면 cc_s 동안 상대는 못 피하고 못 움직임.
  - 턴: 쓸 수 있는 피해 스킬이 있고 상대 주력 스킬이 쿨이거나 내 콤보가 더 세면 들어가고,
    다 쓰면 상대 사거리 밖(+여유)으로 빠진다. 체력이 낮으면 물러남.
  - 능력치: 기본+성장(위키 성장식 g*(L-1)*(0.7025+0.0175*(L-1)), 미검증), 아이템은 골드→능력치 비율(PROFILE, 미검증).
"""
import argparse
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.1

# 사람 실행력 값(실제 경기로 맞출 대상). 반응 0.4초는 설계 기본값(리플레이 1초 간격으론 못 잰 값).
HUMAN = {"reaction_s": 0.4, "dodge": 0.85, "dodge_base": 0.5, "area_delay": 0.5, "cc_s": 1.0, "retreat_hp": 0.3,
         "spacing": 75.0, "aa_trade": 1.0, "minion_aggro_dps": 0.0,
         # 웨이브 뒤로 물러날 수 있는 거리(포탑·웨이브 때문에 무한히 카이팅 못 함), 미니언에 막히는 확률
         "retreat_room": 500.0, "minion_block": 0.35}
DASH_TAGS = {"PositiveEffect_MoveBlock", "Trait_PlayerSelectedDashDirection", "PositiveEffect_Teleport"}
CC_TAGS = {"Trait_ImmobilizingCCSpell", "Trait_ImmobilizingCCAbility"}

# 아이템 골드 → 능력치(초안. 미드 첫 아이템 평균 구성 가정, item_vectors 단가로 환산 예정)
PROFILE = {"ap": {"ap": 0.042, "hp": 0.06, "mr": 0.0, "haste": 0.004},
           "ad": {"ad": 0.022, "hp": 0.06, "armor": 0.0, "haste": 0.004}}

# 스킬마다 쓸 피해 계산식(합)과 판정 보정. 게임 파일에 이름이 여러 개라 Claude가 골랐다(미검증).
KIT_OVERRIDES = {
    "Ahri": {"Q": {"dmg": ["TotalDamage"], "mult": 2.0, "missile": "AhriQMissile"},  # 갈 때+올 때(올 때 고정 피해)
             "W": {"dmg": ["SingleFireDamage"], "mult": 2.0, "hit": "targeted"},
             "E": {"dmg": ["TotalDamage"], "missile": "AhriEMissile", "cc": 1.4},
             "R": {"dmg": ["RCalculatedDamage"], "mult": 1.0, "hit": "targeted", "range": 500}},
    "Syndra": {"Q": {"dmg": ["TotalDamage"], "delay": 0.6, "radius": 210},
               "W": {"dmg": ["ThrowDamage"], "hit": "area", "delay": 0.4, "radius": 220},
               "E": {"dmg": ["TotalDamage"], "hit": "area", "delay": 0.25, "radius": 200, "cc": 1.25},
               "R": {"dmg": ["DamageCalc"], "mult": 4.0}},   # 구체 기본 3+α개
    "Sylas": {"Q": {"dmg": ["Damage", "ExplosionDamage"], "hit": "area", "delay": 0.6, "radius": 180},
              "W": {"dmg": ["MinDamage"], "range": 400, "heal": "MinHealing"},
              "E": {"dmg": ["Damage"], "missile": "SylasE2", "cc": 0.5, "range": 800},
              "R": {"dmg": []}},
    "Galio": {"Q": {"dmg": ["QMissileDamage"], "missile": "GalioQMissile"},
              "W": {"dmg": ["MinTotalDamage"], "hit": "self_area", "radius": 275, "cc": 0.5},
              "E": {"dmg": ["TotalDamage"], "hit": "skillshot", "speed": 2300, "width": 160, "cc": 0.75},
              "R": {"dmg": []}},
    "Akali": {"Q": {"dmg": ["Damage"], "hit": "area", "delay": 0.25, "radius": 300},
              "W": {"dmg": []},
              "E": {"dmg": ["E1Damage", "E2DamageCalc"], "missile": "AkaliEMis"},
              "R": {"dmg": ["Cast1Damage", "Cast2DamageMin"]}},
    "Yone": {"Q": {"dmg": ["QDamage"], "hit": "area", "delay": 0.35, "radius": 120},
             "W": {"dmg": ["WDamage"], "hit": "area", "delay": 0.5, "radius": 300, "pct_maxhp": 0.06},
             "E": {"dmg": []},
             "R": {"dmg": ["Damage"], "cc": 0.75}},
    "Viktor": {"Q": {"dmg": ["TotalMissileDamage", "AttackTotalDMG"]},
               "W": {"dmg": []},
               "E": {"dmg": ["LaserDamage"], "missile": "ViktorEMissile"},
               "R": {"dmg": ["InitialBurstDamage", "SubsequentBurstDamage"], "hit": "area", "delay": 0.25, "radius": 325}},
    "TwistedFate": {"Q": {"dmg": ["TotalDamage"], "range": 1450, "width": 120},
                    "W": {"dmg": ["BlueDamage"], "hit": "targeted", "range": 600},  # 근접 땐 노란 카드(기절)로 바꿈
                    "E": {"dmg": ["BonusDamage"], "mult": 0.25, "hit": "targeted", "range": 550},
                    "R": {"dmg": []}},
    "Ekko": {"Q": {"dmg": ["InitialDamage", "RecallDamage"], "missile": "EkkoQMis"},
             "W": {"dmg": []},
             "E": {"dmg": ["TotalDamage"], "hit": "targeted", "range": 475},
             "R": {"dmg": ["TotalDamage"], "hit": "self_area", "radius": 375}},
    "Yasuo": {"Q": {"dmg": ["TotalDamage"], "hit": "area", "delay": 0.35, "radius": 60},
              "W": {"dmg": []},
              "E": {"dmg": ["TotalDamage"], "cd": 6.0},  # 같은 대상 재사용 대기(미니언 타고 다가가는 건 아직 없음)
              "R": {"dmg": ["Damage"], "hit": "targeted", "needs_cc": True}},
    "Orianna": {"Q": {"dmg": ["TotalDamage"], "missile": "OrianaIzuna"},
                "W": {"dmg": ["TotalDamage"], "hit": "area", "delay": 0.6, "radius": 225, "range": 815},
                "E": {"dmg": ["TotalDamageTooltip"], "mult": 0.5},
                "R": {"dmg": ["TotalDamage"], "hit": "area", "delay": 0.75, "radius": 415, "range": 815}},
    "Locke": {"Q": {"dmg": ["MissileDamage", "NailDamage"], "missile": "LockeQMissile", "charges": 3},  # Ammo 3(게임 파일)
              "W": {"dmg": [], "heal": "DamageRestoreAmount", "self": True},  # 받은 피해 일부 회복(최대치로 단순화)
              "E": {"dmg": ["DashDamage", "OnHitDamage"], "hit": "targeted"},
              "R": {"dmg": ["Damage"], "missile": "LockeRNailMissile"}},
}
MAGE_AD = {"Yone", "Yasuo", "Talon", "Zed", "Qiyana", "Naafiri"}

# 패시브·특수 기술(미드 10명). 수치는 게임 파일(패시브 계산식·스킬 데이터), 발동 방식은 Claude가 단순화(미검증).
#   shield: 보호막(by_level 또는 스킬 계산식), 다시 차는 시간(recharge_s, 피해 안 받은 시간 기준)
#   magic_shield: 마법 피해만 막는 보호막
#   empower_aa: 스킬 뒤(after_spell) 또는 주기(every_s)마다 다음 평타 추가 피해, hit_rate=실제로 이어 칠 확률
#   proc3: 같은 대상 3타(스킬·평타)마다 추가 피해, onhit: 평타마다 추가 피해(레벨 보간)
#   windwall: 투사체 스킬 막기(쿨·지속은 W 데이터)
SPECIALS = {
    "Yasuo": {"shield": {"calc": "ShieldValue", "recharge_s": 18.0}, "windwall": {"duration": 4.0}},
    "Galio": {"empower_aa": {"calc": "TotalDamage", "every_s": 5.0, "magic": True, "hit_rate": 0.8},
              "magic_shield": {"spell": "W", "calc": "TotalPassiveShield", "recharge_s": 12.0}},
    "Sylas": {"empower_aa": {"calc": "PassiveDamage", "after_spell": True, "magic": True, "hit_rate": 0.6}},
    "Akali": {"empower_aa": {"calc": "Damage", "after_spell": True, "magic": True, "hit_rate": 0.5}},
    "Ekko": {"proc3": {"calc": "ThreeHitDamage", "lockout_s": 4.0, "magic": True}},
    "Locke": {"onhit": {"min": [5.0, 40.0], "max": [10.0, 80.0], "ap": [0.1, 0.2], "magic": True}},
    "TwistedFate": {"spell_cc": {"W": 1.25}},   # 금색 카드 기절(초, 미검증)
}


def level_value(calc, lv, f):
    """패시브 계산식(by_level + 계수) → 값"""
    v = (calc.get("by_level") or [0.0] * 18)[lv - 1]
    for r in calc.get("ratios", []):
        coef = r.get("coef")
        if coef is None:
            coef = (r.get("coef_by_rank") or [0])[0] or 0
        if isinstance(r["stat"], str):
            v += coef * f.stat(r)
    return v


def load(path=None):
    path = path or os.path.join(ROOT, "rules", "champ_skills_16.23.json")
    return json.load(open(path, encoding="utf-8"))["champions"]


def grow(base, g, lv):
    return base + g * (lv - 1) * (0.7025 + 0.0175 * (lv - 1))


def skill_ranks(lv):
    """표준 선마: Q→E→W, 궁 6/11/16. 레벨별 [Q,W,E,R] 단계."""
    order = ["Q", "W", "E", "Q", "Q", "R", "Q", "E", "Q", "E", "R", "E", "E", "W", "W", "R", "W", "W"]
    r = {"Q": 0, "W": 0, "E": 0, "R": 0}
    for s in order[:lv]:
        r[s] += 1
    return r


class Fighter:
    def __init__(self, name, kits, lv, gold, side):
        k = kits[name]
        self.kit = k
        st = k["stats"]
        self.name, self.side, self.lv = name, side, lv
        prof = PROFILE["ad" if name in MAGE_AD else "ap"]
        self.ap = gold * prof.get("ap", 0)
        self.bonus_ad = gold * prof.get("ad", 0)
        self.ad = grow(st["ad"], st["ad_lv"], lv) + self.bonus_ad
        self.base_hp = grow(st["hp"], st["hp_lv"], lv)
        self.max_hp = self.base_hp + gold * prof["hp"]
        self.hp = self.max_hp
        self.armor = grow(st["armor"], st["armor_lv"], lv)
        self.mr = grow(st["mr"], st["mr_lv"], lv)
        self.ms = st["ms"]
        self.range = st["range"]
        self.as_ = st["as"] * (1 + grow(0, st["as_lv"], lv) / 100)
        self.max_mana = grow(st["mana"], st["mana_lv"], lv) if st["resource_type"] == 0 else 1e9
        self.mana = self.max_mana
        self.haste = gold * prof.get("haste", 0)
        self.ranks = skill_ranks(lv)
        self.spells = {}
        ov = KIT_OVERRIDES.get(name, {})
        for slot, s in k["spells"].items():
            o = ov.get(slot, {})
            rk = self.ranks[slot]
            names = o.get("dmg") if "dmg" in o else [n for n, d in s["damage"].items() if d.get("base")][:1]
            is_dash = o.get("dash", bool(DASH_TAGS & set(s["tags"])))
            if rk == 0 or not (names or is_dash or o.get("heal")):
                continue
            m = s["missiles"].get(o.get("missile")) if o.get("missile") else None
            if m is None and s["missiles"]:
                cand = [x for x in s["missiles"].values() if (x.get("speed") or 0) > 300 and x.get("width")]
                m = max(cand, key=lambda x: x["speed"]) if cand else None
            hit = o.get("hit") or {"targeted_or_ground": "targeted", "self": "self_area"}.get(s["kind"], s["kind"])
            self.spells[slot] = {
                "slot": slot, "rank": rk, "hit": hit, "names": names, "mult": o.get("mult", 1.0),
                "dmg_parts": [s["damage"][n] for n in names if n in s["damage"]],
                "range": o.get("range") or s["range"] or o.get("radius") or 300,
                "cast": s["cast_time"] if s["cast_time"] is not None else 0.25,
                "speed": o.get("speed") or (m or {}).get("speed"), "width": o.get("width") or (m or {}).get("width") or 70,
                "delay": o.get("delay", HUMAN["area_delay"]), "radius": o.get("radius") or s.get("radius") or 150,
                "cd": o.get("cd") or s["cooldown"][min(rk, len(s["cooldown"])) - 1],
                "cost": (s["cost"] or [0] * 5)[min(rk, len(s["cost"] or [0])) - 1] if s["cost"] else 0,
                "cc": o.get("cc", HUMAN["cc_s"] if CC_TAGS & set(s["tags"]) else 0.0),
                "dash": o.get("dash", bool(DASH_TAGS & set(s["tags"]))),
                "blockable": "Trait_Ranged_StopsFirstHit" in s["tags"],
                "pct_maxhp": o.get("pct_maxhp", 0.0), "needs_cc": o.get("needs_cc", False),
                "magic": not any(r["stat"] == "ad" for p in [s["damage"].get(n, {}) for n in names] for r in p.get("ratios", []))
                         or name not in MAGE_AD,
                "ready": 0.0, "heal": s["damage"].get(o["heal"]) if o.get("heal") else None, "self_cast": o.get("self", False),
                "max_charges": o.get("charges", 1), "charges": o.get("charges", 1), "recharge_at": None,
            }
        self.passive = k.get("passive") or {}
        self.sp = SPECIALS.get(name, {})
        for slot, cc in self.sp.get("spell_cc", {}).items():
            if slot in self.spells:
                self.spells[slot]["cc"] = cc
        self.shield = self.magic_shield = 0.0
        self.last_hurt = -99.0
        self.empower_ready = 0.0
        self.empowered = False
        self.proc_hits, self.proc_lock = 0, 0.0
        self.wall_ready, self.wall_until = 0.0, -1.0
        w = k["spells"].get("W", {})
        self.wall_cd = (w.get("cooldown") or [25])[max(self.ranks["W"], 1) - 1] if "windwall" in self.sp else None
        self.refill(0.0, force=True)
        self.x = 0.0
        self.cc_until = 0.0
        self.busy_until = 0.0
        self.aa_ready = 0.0
        self.dealt = 0.0

    def stat(self, r):
        bonus = r["formula"] == "bonus"
        v = {"ap": self.ap, "ad": self.bonus_ad if bonus else self.ad,
             "max_hp": (self.max_hp - self.base_hp) if bonus else self.max_hp,
             "armor": 0.0 if bonus else self.armor, "mr": 0.0 if bonus else self.mr}.get(r["stat"], 0.0)
        return v

    def heal_amount(self, sp):
        h = sp.get("heal")
        if not h:
            return 0.0
        base = (h.get("base") or [0])[min(sp["rank"], len(h.get("base") or [0])) - 1] if h.get("base") else 0.0
        for r in h.get("ratios", []):
            coef = r.get("coef")
            if coef is None:
                cb = r.get("coef_by_rank") or [0]
                coef = cb[min(sp["rank"], len(cb)) - 1] or 0
            if isinstance(r["stat"], str):
                base += coef * self.stat(r)
        return base

    def self_heal(self, t):
        """체력 60% 아래면 자기 회복 스킬(로크 W) 사용"""
        if self.hp / self.max_hp >= 0.6 or t < self.busy_until or t < self.cc_until:
            return
        for sp in self.spells.values():
            if sp["self_cast"] and sp["ready"] <= t and self.mana >= sp["cost"]:
                self.use(sp, t)
                self.mana -= sp["cost"]
                self.hp = min(self.max_hp, self.hp + self.heal_amount(sp))
                return

    def spell_damage(self, sp, target):
        if sp.get("self_cast"):
            return 0.0
        tot = 0.0
        for p in sp["dmg_parts"]:
            base = (p.get("base") or [0])[min(sp["rank"], len(p.get("base") or [0])) - 1] if p.get("base") else 0.0
            add = 0.0
            for r in p.get("ratios", []):
                coef = r.get("coef")
                if coef is None:
                    cb = r.get("coef_by_rank") or [0]
                    coef = cb[min(sp["rank"], len(cb)) - 1] or 0
                if isinstance(r["stat"], str):
                    add += coef * self.stat(r)
            tot += base + add
        tot = tot * sp["mult"] + sp["pct_maxhp"] * target.max_hp
        res = target.mr if sp["magic"] else target.armor
        return tot * 100 / (100 + res)

    def use(self, sp, t):
        """스킬 사용: 충전형이면 한 충전 쓰고 0.5초 뒤 다시 쓸 수 있음, 충전은 쿨마다 하나씩"""
        cd = self.cdr(sp["cd"])
        if sp["max_charges"] <= 1:
            sp["ready"] = t + cd
            return
        sp["charges"] -= 1
        if sp["recharge_at"] is None:
            sp["recharge_at"] = t + cd
        sp["ready"] = t + 0.5 if sp["charges"] > 0 else sp["recharge_at"]

    def tick_charges(self, t):
        for sp in self.spells.values():
            if sp["max_charges"] > 1 and sp["recharge_at"] is not None and t >= sp["recharge_at"]:
                sp["charges"] += 1
                sp["recharge_at"] = t + self.cdr(sp["cd"]) if sp["charges"] < sp["max_charges"] else None
                sp["ready"] = min(sp["ready"], t)

    def cdr(self, cd):
        return cd * 100 / (100 + self.haste)

    def refill(self, t, force=False):
        """피해를 recharge_s 동안 안 받으면 보호막이 다시 참"""
        sh = self.sp.get("shield")
        if sh and (force or t - self.last_hurt >= sh["recharge_s"]):
            self.shield = level_value(self.passive.get("damage", {}).get(sh["calc"], {}), self.lv, self)
        ms = self.sp.get("magic_shield")
        if ms and self.ranks.get(ms["spell"], 0) and (force or t - self.last_hurt >= ms["recharge_s"]):
            calc = self.kit["spells"][ms["spell"]]["damage"].get(ms["calc"], {})
            rk = self.ranks[ms["spell"]]
            coef = (calc.get("ratios") or [{}])[0].get("coef_by_rank") or [0]
            self.magic_shield = coef[min(rk, len(coef)) - 1] * self.max_hp

    def take(self, dmg, magic, t):
        """보호막 → 체력 순으로 피해"""
        self.last_hurt = t
        if magic and self.magic_shield > 0:
            a = min(self.magic_shield, dmg)
            self.magic_shield -= a
            dmg -= a
        if self.shield > 0:
            a = min(self.shield, dmg)
            self.shield -= a
            dmg -= a
        self.hp -= dmg
        return dmg

    def blocks(self, sp, t):
        """바람 장막: 투사체 스킬(속도가 있는 스킬샷·대상 지정 투사체)을 막음. 쿨이면 지금 펼침."""
        if self.wall_cd is None or not self.ranks.get("W") or not sp.get("speed"):
            return False
        if t < self.wall_until:
            return True
        if t >= self.wall_ready and t >= self.cc_until:
            self.wall_until = t + self.sp["windwall"]["duration"]
            self.wall_ready = t + self.cdr(self.wall_cd)
            return True
        return False

    def aa_bonus(self, target, t, rng):
        """평타 추가 피해(패시브)"""
        add = 0.0
        e = self.sp.get("empower_aa")
        if e:
            ok = self.empowered if e.get("after_spell") else t >= self.empower_ready
            if ok and rng.random() < e["hit_rate"]:
                v = level_value(self.passive.get("damage", {}).get(e["calc"], {}), self.lv, self)
                add += v * 100 / (100 + (target.mr if e["magic"] else target.armor))
                self.empowered = False
                self.empower_ready = t + e.get("every_s", 0.0)
        oh = self.sp.get("onhit")
        if oh:
            frac = (self.lv - 1) / 17
            lo = oh["min"][0] + (oh["min"][1] - oh["min"][0]) * frac + oh["ap"][0] * self.ap
            hi = oh["max"][0] + (oh["max"][1] - oh["max"][0]) * frac + oh["ap"][1] * self.ap
            add += (lo + hi) / 2 * 100 / (100 + target.mr)
        return add + self.proc(target, t)

    def proc(self, target, t):
        """3타 표식(에코 패시브)"""
        p3 = self.sp.get("proc3")
        if not p3 or t < self.proc_lock:
            return 0.0
        self.proc_hits += 1
        if self.proc_hits < 3:
            return 0.0
        self.proc_hits, self.proc_lock = 0, t + p3["lockout_s"]
        v = level_value(self.passive.get("damage", {}).get(p3["calc"], {}), self.lv, self)
        return v * 100 / (100 + target.mr)


def dist(a, b):
    return abs(a.x - b.x)


def resolve_hit(sp, caster, target, rng, t):
    d = dist(caster, target)
    if target.cc_until > t + sp["cast"]:
        return True
    if sp["blockable"] and rng.random() < HUMAN["minion_block"]:
        return False
    if sp["hit"] == "targeted" or sp["dash"]:
        return d <= sp["range"] + 30
    if sp["hit"] == "self_area":
        return d <= sp["radius"] + 35
    if sp["hit"] == "skillshot":
        travel = sp["cast"] + (d / sp["speed"] if sp["speed"] else 0.0)
        need = (sp["width"] / 2 + 35) / max(target.ms, 1)
    else:  # area
        travel = sp["cast"] + sp["delay"]
        need = (sp["radius"] + 35) / max(target.ms, 1)
    # 피할 여유(초) = 도착 시간 − (반응 + 옆으로 빠지는 시간). 여유가 클수록 dodge에 가깝게, 여유가 없어도
    # 미리 움직임·예측 실패로 dodge_base만큼은 빗나감(사람 값, 실제 킬 수로 맞춤)
    slack = travel - HUMAN["reaction_s"] - need
    p_dodge = HUMAN["dodge_base"] + (HUMAN["dodge"] - HUMAN["dodge_base"]) / (1 + math.exp(-slack / 0.1))
    return rng.random() > p_dodge


def combo_value(f, other):
    return sum(f.spell_damage(s, other) for s in f.spells.values())


def duel(a_name, b_name, kits, lv=6, gold=1000, seconds=60.0, seed=0, start_gap=700.0):
    rng = random.Random(seed)
    a, b = Fighter(a_name, kits, lv, gold, 0), Fighter(b_name, kits, lv, gold, 1)
    a.x, b.x = 0.0, start_gap
    t = 0.0
    trades = 0
    while t < seconds and a.hp > 0 and b.hp > 0:
        for x in (a, b):
            x.refill(t)
            x.tick_charges(t)
        for me, op in ((a, b), (b, a)):
            if me.hp <= 0 or op.hp <= 0 or t < me.busy_until or t < me.cc_until:
                continue
            ready = [s for s in me.spells.values() if s["ready"] <= t and me.mana >= s["cost"] and not s["self_cast"]
                     and (not s["needs_cc"] or op.cc_until > t)]
            op_ready = [s for s in op.spells.values() if s["ready"] <= t]
            my_burst = sum(me.spell_damage(s, op) for s in ready)
            op_burst = sum(op.spell_damage(s, me) for s in op_ready)
            low = me.hp / me.max_hp < HUMAN["retreat_hp"] and me.hp < op.hp
            want = ready and not low and (my_burst >= op_burst * 0.8 or not op_ready
                                          or my_burst >= op.hp)
            d = dist(me, op)
            step = me.ms * DT * (-1 if me.x > op.x else 1)
            if want:
                best = max(ready, key=lambda s: (me.spell_damage(s, op) / max(s["cd"], 1), s["range"]))
                dashes = [sp for sp in ready if sp["dash"]]
                if d > best["range"] and dashes:
                    dsp = max(dashes, key=lambda sp: sp["range"])
                    if d - dsp["range"] <= max(best["range"], me.range) + 50 or d <= dsp["range"]:
                        gap = min(dsp["range"], max(d - me.range, 0.0))
                        me.x += gap * (-1 if me.x > op.x else 1)
                        best = dsp
                        d = dist(me, op)
                if d <= best["range"]:
                    me.busy_until = t + best["cast"]
                    me.mana -= best["cost"]
                    me.use(best, t)
                    trades += 1
                    me.empowered = True
                    if not op.blocks(best, t) and resolve_hit(best, me, op, rng, t):
                        dmg = me.spell_damage(best, op) + me.proc(op, t)
                        me.dealt += op.take(dmg, best["magic"], t)
                        if best["cc"]:
                            op.cc_until = max(op.cc_until, t + best["cast"] + best["cc"])
                else:
                    me.x += step
            else:
                # 물러나며 상대 최대 사거리 + 여유 밖에 서기. 평타 사거리 안에 상대가 오면 평타(턴 교환)
                keep = max([s["range"] for s in op_ready] + [op.range]) + HUMAN["spacing"]
                home = -HUMAN["retreat_room"] if me.side == 0 else start_gap + HUMAN["retreat_room"]
                if d < keep and abs(me.x - home) > 1 and (me.x - step - home) * (me.x - home) > 0:
                    me.x -= step
                elif d > keep + 50:
                    me.x += step * 0.5
            if d <= me.range + 35 and t >= me.aa_ready and t >= me.busy_until and HUMAN["aa_trade"] > rng.random():
                dmg = me.ad * 100 / (100 + op.armor)
                me.dealt += op.take(dmg, False, t) + op.take(me.aa_bonus(op, t, rng), True, t)
                me.aa_ready = t + 1 / me.as_
                me.busy_until = t + 0.25 / me.as_
        t += DT
    return {"a_hp": max(a.hp, 0) / a.max_hp, "b_hp": max(b.hp, 0) / b.max_hp, "a_kill": b.hp <= 0, "b_kill": a.hp <= 0,
            "a_dealt": a.dealt, "b_dealt": b.dealt, "casts": trades, "t": t}


def matchup(a, b, kits, n=200, **kw):
    s = {"edge": 0.0, "a_kill": 0, "b_kill": 0, "casts": 0}
    for i in range(n):
        # 양쪽 자리를 번갈아 써서 '먼저 움직이는 쪽' 치우침을 없앰
        if i % 2 == 0:
            r = duel(a, b, kits, seed=i, **kw)
            ea, ka, kb = r["a_hp"] - r["b_hp"], r["a_kill"], r["b_kill"]
        else:
            r = duel(b, a, kits, seed=i, **kw)
            ea, ka, kb = r["b_hp"] - r["a_hp"], r["b_kill"], r["a_kill"]
        s["edge"] += ea / n
        s["a_kill"] += ka / n
        s["b_kill"] += kb / n
        s["casts"] += r["casts"] / n
    return s


MID10 = ["Locke", "Syndra", "Sylas", "Galio", "Akali", "Yone", "Viktor", "TwistedFate", "Ekko", "Yasuo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--gold", type=float, default=1000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    kits = load()
    if a.pair:
        print(json.dumps(matchup(*a.pair, kits, n=a.n, lv=a.level, gold=a.gold), ensure_ascii=False))
        return
    res = {}
    for x in MID10:
        for y in MID10:
            if x < y:
                res[f"{x}|{y}"] = matchup(x, y, kits, n=a.n, lv=a.level, gold=a.gold)
    if a.out:
        json.dump({"level": a.level, "gold": a.gold, "n": a.n, "human": HUMAN, "pairs": res}, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
