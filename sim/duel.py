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
              "W": {"dmg": ["MinDamage"], "range": 400},
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
    "Locke": {"Q": {"dmg": ["MissileDamage", "NailDamage"], "missile": "LockeQMissile"},
              "W": {"dmg": []},
              "E": {"dmg": ["DashDamage", "OnHitDamage"], "hit": "targeted"},
              "R": {"dmg": ["Damage"], "missile": "LockeRNailMissile"}},
}
MAGE_AD = {"Yone", "Yasuo", "Talon", "Zed", "Qiyana", "Naafiri"}


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
        st = k["stats"]
        self.name, self.side, self.lv = name, side, lv
        prof = PROFILE["ad" if name in MAGE_AD else "ap"]
        self.ap = gold * prof.get("ap", 0)
        self.bonus_ad = gold * prof.get("ad", 0)
        self.ad = grow(st["ad"], st["ad_lv"], lv) + self.bonus_ad
        self.max_hp = grow(st["hp"], st["hp_lv"], lv) + gold * prof["hp"]
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
            if rk == 0 or not (names or is_dash):
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
                "ready": 0.0,
            }
        self.x = 0.0
        self.cc_until = 0.0
        self.busy_until = 0.0
        self.aa_ready = 0.0
        self.dealt = 0.0

    def stat(self, r):
        v = {"ap": self.ap, "ad": self.ad if r["formula"] == "total" else self.bonus_ad, "cur_hp": self.hp,
             "max_hp": self.max_hp, "armor": self.armor, "mr": self.mr, "ms": 0.0}.get(r["stat"], 0.0)
        return v

    def spell_damage(self, sp, target):
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

    def cdr(self, cd):
        return cd * 100 / (100 + self.haste)


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
        for me, op in ((a, b), (b, a)):
            if me.hp <= 0 or op.hp <= 0 or t < me.busy_until or t < me.cc_until:
                continue
            ready = [s for s in me.spells.values() if s["ready"] <= t and me.mana >= s["cost"]
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
                    best["ready"] = t + me.cdr(best["cd"])
                    trades += 1
                    if resolve_hit(best, me, op, rng, t):
                        dmg = me.spell_damage(best, op)
                        op.hp -= dmg
                        me.dealt += dmg
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
                op.hp -= dmg
                me.dealt += dmg
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
