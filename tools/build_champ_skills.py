"""CommunityDragon 게임 파일(<챔피언>.bin.json)로 챔피언 스킬·능력치 표 만들기.

    python tools/build_champ_skills.py data/skills/cdragon --ddragon data/ddragon/16.23.1/en_US/championFull.json \
        --meraki data/skills/meraki_champions.json --out rules/champ_skills_16.23.json

칸(스킬마다):
  targeting   게임 파일 mTargetingTypeData. 없으면 Target(대상 지정). kind = targeted / skillshot / area / self
  cast_time   선딜(초). mCastTime, 없으면 spellCastTime
  total_time  시전 전체 시간(spellTotalTime, 후딜 포함 상한. 이동 취소 가능 여부는 별도)
  range       castRangeDisplayOverride > castRange(25000 같은 자리값은 무시) > 투사체 사거리
  missiles    같은 스킬 폴더의 투사체: 속도(가속 포함), 폭, 사거리
  cooldown/cost  1~5(궁 1~3)레벨 값
  damage      mSpellCalculations → 기본값(레벨별) + 계수(능력치 종류·전체/추가)
  tags        mSpellTags(Trait_* 원본)
능력치는 CharacterRecords/Root(레벨당 AD 포함). Data Dragon 쿨과 다르면 report에 남긴다.
미검증: mStat 번호→능력치 이름 대응(STAT), 대상 지정 기본값, 투사체 묶기 규칙.
"""
import argparse
import glob
import json
import os
import re

# CDTB 계산식 mStat 번호(일부만 확인: 0=AP는 아리 Q 50% AP로 확인, 2=AD). 나머지는 번호 그대로 둔다.
STAT = {None: "ap", 0: "ap", 1: "armor", 2: "ad", 5: "mr", 6: "ms", 11: "max_hp", 12: "cur_hp"}
FORMULA = {None: "total", 0: "total", 1: "base", 2: "bonus"}
TARGET_KIND = {"Target": "targeted", "TargetOrLocation": "targeted_or_ground",
               "Direction": "skillshot", "Location": "skillshot", "LocationClamped": "skillshot",
               "DragDirection": "skillshot", "Cone": "skillshot",
               "Area": "area", "AreaClamped": "area", "SelfAoe": "self_area", "Self": "self"}
PLACEHOLDER_RANGE = 20000


def ranks(v, n):
    if not isinstance(v, list):
        return None
    # 길이 7(0번 자리값 + 1~5레벨 + 여유) 배열은 1번부터, mana처럼 6칸 배열은 0번부터가 1레벨
    v = v[1:n + 1] if len(v) >= 7 else v[:n]
    return [round(x, 3) for x in v]


def first(v):
    return v[1] if isinstance(v, list) and len(v) > 1 else (v[0] if isinstance(v, list) and v else v)


def missile_info(s):
    spec = s.get("mMissileSpec") or {}
    mv = spec.get("movementComponent") or {}
    speed = mv.get("mSpeed") or mv.get("mInitialSpeed") or s.get("missileSpeed")
    out = {"speed": speed, "width": spec.get("mMissileWidth") or s.get("mLineWidth"),
           "range": first(s.get("castRange")), "type": mv.get("__type")}
    if out["range"] and out["range"] >= PLACEHOLDER_RANGE:
        out["range"] = None
    if mv.get("mAcceleration"):
        out.update(accel=mv["mAcceleration"], min_speed=mv.get("mMinSpeed"), max_speed=mv.get("mMaxSpeed"))
    if mv.get("mTracksTarget"):
        out["homing"] = True
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if v is not None}


def calc_parts(calc, dv, n):
    """계산식 한 개 → {base:[..], ratios:[{stat,formula,coef or coef_by_rank}], other:[..]}"""
    out = {"base": None, "ratios": [], "other": []}
    for p in calc.get("mFormulaParts", []) if isinstance(calc, dict) else []:
        t = p.get("__type")
        if t == "NamedDataValueCalculationPart":
            vals = ranks(dv.get(p.get("mDataValue")), n)
            out["base"] = vals if out["base"] is None else out["base"]
            if out["base"] is not vals:
                out["other"].append({"datavalue": p.get("mDataValue"), "values": vals})
        elif t == "StatByCoefficientCalculationPart":
            out["ratios"].append({"stat": STAT.get(p.get("mStat"), p.get("mStat")),
                                  "formula": FORMULA.get(p.get("mStatFormula"), p.get("mStatFormula")),
                                  "coef": round(p.get("mCoefficient", 0), 4)})
        elif t == "StatByNamedDataValueCalculationPart":
            out["ratios"].append({"stat": STAT.get(p.get("mStat"), p.get("mStat")),
                                  "formula": FORMULA.get(p.get("mStatFormula"), p.get("mStatFormula")),
                                  "coef_by_rank": ranks(dv.get(p.get("mDataValue")), n)})
        elif t == "NumberCalculationPart":
            out["other"].append({"number": p.get("mNumber")})
        else:
            out["other"].append({"part": t})
    if not out["other"]:
        del out["other"]
    return out


def champion(path, maxrank=None):
    b = json.load(open(path, encoding="utf-8"))
    root = next(v for k, v in b.items() if k.endswith("/CharacterRecords/Root"))
    if "spells" not in root:
        return None

    def mv(key, sub="baseValue"):
        v = root.get(key)
        return round(v.get(sub, 0.0), 4) if isinstance(v, dict) else v

    res = root.get("primaryAbilityResource") or {}
    stats = {"hp": mv("baseHPModifiable"), "hp_lv": mv("hpPerLevelModifiable"),
             "ad": mv("baseDamageModifiable"), "ad_lv": mv("damagePerLevelModifiable"),
             "armor": mv("baseArmorModifiable"), "armor_lv": mv("armorPerLevelModifiable"),
             "mr": mv("baseMR"), "mr_lv": mv("mrPerLevel"),
             "ms": mv("baseMoveSpeedModifiable"), "range": mv("attackRangeModifiable"),
             "as": mv("attackSpeedModifiable"), "as_ratio": mv("attackSpeedRatioModifiable"),
             "as_lv": mv("attackSpeedPerLevelModifiable"),
             "mana": round((res.get("{726ee5cd}") or {}).get("baseValue", 0.0), 2),
             "mana_lv": round((res.get("{6216bf7b}") or {}).get("baseValue", 0.0), 2),
             "resource_type": res.get("arType"),
             "windup_offset": (root.get("basicAttack") or {}).get("mAttackDelayCastOffsetPercent"),
             "collision_radius": root.get("pathfindingCollisionRadius"),
             "identity": root.get("purchaseIdentities")}
    lower = {k.lower(): v for k, v in b.items()}
    spells = {}
    for i, sp in enumerate(root["spells"][:4]):
        slot = "QWER"[i]
        n = (maxrank or {}).get(slot) or (3 if slot == "R" else 5)
        obj = b.get(sp) or lower.get(sp.lower()) or {}
        s = obj.get("mSpell") or {}
        dv = {d.get("name"): d.get("values") for d in s.get("DataValues", []) or []}
        t = (s.get("mTargetingTypeData") or {}).get("__type") or "Target"
        rng = first(s.get("castRangeDisplayOverride")) or first(s.get("castRange"))
        folder = sp.rsplit("/", 1)[0] + "/"
        missiles = {}
        for k, v in b.items():
            if k.lower().startswith(folder.lower()) and k != sp and isinstance(v, dict) and v.get("__type") == "SpellObject":
                ms = v.get("mSpell") or {}
                if ms.get("mMissileSpec") or "Missile" in k:
                    missiles[k.rsplit("/", 1)[1]] = missile_info(ms)
        if not missiles and TARGET_KIND.get(t) == "skillshot" and (s.get("missileSpeed") or 0) > 100:
            missiles["_main"] = {"speed": s["missileSpeed"], "width": s.get("mLineWidth"), "from": "main_spell"}
        if (rng is None or rng >= PLACEHOLDER_RANGE) and missiles:
            rng = max((m.get("range") or 0) for m in missiles.values()) or rng
        dmg = {}
        for name, calc in (s.get("mSpellCalculations") or {}).items():
            dmg[name] = calc_parts(calc, dv, n)
        spells[slot] = {
            "name": obj.get("mScriptName"), "targeting": t, "kind": TARGET_KIND.get(t, t),
            "cast_time": s.get("mCastTime", s.get("spellCastTime")),
            "total_time": s.get("spellTotalTime"),
            "range": round(rng, 1) if isinstance(rng, (int, float)) and rng < PLACEHOLDER_RANGE else None,
            "radius": first(s.get("castRadius")),
            # Cooldown(새 칸, Data Dragon과 같음) 우선. cooldownTime은 옛 칸이라 패치 전 값이 남아 있기도 함(신드라 W)
            "cooldown": ranks((s.get("Cooldown") or {}).get("values") or s.get("cooldownTime"), n), "cost": ranks((s.get("manaValues") or {}).get("values") or s.get("mana"), n),
            "missiles": missiles, "damage": dmg, "tags": s.get("mSpellTags") or [],
            "flags": {k: True for k in ("mCantCancelWhileWindingUp", "mCantCancelWhileChanneling",
                                        "mChannelIsInterruptedByDisables", "canMoveAndCastWhileChanneling")
                      if s.get(k)},
        }
    return {"name": root.get("mCharacterName"), "stats": stats, "spells": spells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cdragon_dir")
    ap.add_argument("--ddragon")
    ap.add_argument("--meraki")
    ap.add_argument("--version", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    champs, bad = {}, []
    ranks_by = {}
    if a.ddragon:
        for v in json.load(open(a.ddragon, encoding="utf-8"))["data"].values():
            ranks_by[v["id"].lower()] = {"QWER"[i]: sp["maxrank"] for i, sp in enumerate(v["spells"])}
    for p in sorted(glob.glob(os.path.join(a.cdragon_dir, "*.bin.json"))):
        try:
            c = champion(p, ranks_by.get(os.path.basename(p).split(".")[0]))
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(p), repr(e)))
            continue
        if c:
            champs[c["name"]] = c
    rep = [f"# 챔피언 스킬 표 검수 {a.version}", "", f"- 챔피언 {len(champs)}명, 읽기 실패 {len(bad)}"]
    rep += [f"  - {f}: {e}" for f, e in bad]
    if a.ddragon:
        dd = json.load(open(a.ddragon, encoding="utf-8"))["data"]
        dd = {v["id"].lower(): v for v in dd.values()}
        cd_bad, ms_bad, cost_bad, n = [], [], [], 0
        for name, c in champs.items():
            d = dd.get(name.lower())
            if not d:
                rep.append(f"- Data Dragon에 없음: {name}")
                continue
            if d["stats"]["movespeed"] != c["stats"]["ms"]:
                ms_bad.append(f"{name} {d['stats']['movespeed']}→{c['stats']['ms']}")
            for i, slot in enumerate("QWER"):
                n += 1
                dc = d["spells"][i]["cooldown"]
                gc = c["spells"].get(slot, {}).get("cooldown")
                if not gc or [round(x, 2) for x in gc[:len(dc)]] != [round(x, 2) for x in dc]:
                    cd_bad.append(f"{name} {slot}: Data Dragon {dc} / 게임 파일 {gc}")
                dcost = d["spells"][i]["cost"]
                gcost = c["spells"].get(slot, {}).get("cost") or [0] * len(dcost)
                if any(dcost) and [round(x) for x in gcost[:len(dcost)]] != [round(x) for x in dcost]:
                    cost_bad.append(f"{name} {slot}: Data Dragon {dcost} / 게임 파일 {gcost}")
        rep += ["", f"## Data Dragon 쿨 대조: {n - len(cd_bad)}/{n} 일치", ""] + [f"- {x}" for x in cd_bad]
        rep += ["", f"## Data Dragon 소모값 대조: 다름 {len(cost_bad)}", ""] + [f"- {x}" for x in cost_bad]
        rep += ["", f"## 이동속도 차이 {len(ms_bad)}", ""] + [f"- {x}" for x in ms_bad]
    if a.meraki:
        mk = {v["key"].lower(): v for v in json.load(open(a.meraki, encoding="utf-8")).values()}
        agree = tot = 0
        diff = []
        mk_kind = {"Direction": "skillshot", "Location": "skillshot", "Unit": "targeted", "Auto": "self",
                   "Passive": "self", "Vector": "skillshot", "Varied": None}
        for name, c in champs.items():
            m = mk.get(name.lower())
            if not m:
                continue
            for slot, s in c["spells"].items():
                mt = (m["abilities"].get(slot) or [{}])[0].get("targeting")
                want = mk_kind.get(mt)
                if want is None:
                    continue
                tot += 1
                got = s["kind"].split("_")[0]
                got = {"area": "skillshot", "self": "self"}.get(got, got)
                if got == want or (want == "self" and s["kind"] == "self_area"):
                    agree += 1
                else:
                    diff.append(f"{name} {slot}: 게임 파일 {s['targeting']} / Meraki {mt}")
        rep += ["", f"## Meraki(25.15) 타게팅 대조: {agree}/{tot} 일치", ""] + [f"- {x}" for x in diff]
    out = {"_doc": __doc__.strip().splitlines()[0], "_source": f"CommunityDragon {a.version}",
           "_unverified": ["STAT 번호 대응(0=AP·2=AD만 확인)", "대상 지정 기본값", "투사체 묶기", "cast_time이 공격 속도에 비례하는 스킬"],
           "champions": champs}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.report:
        open(a.report, "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print(a.out, len(champs), "명")
    print("\n".join(rep[:4]))


if __name__ == "__main__":
    main()
