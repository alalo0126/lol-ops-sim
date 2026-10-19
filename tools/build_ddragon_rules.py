"""2단계: Data Dragon 챔피언·아이템·소환사 주문 JSON을 받아 rules/champions.json, rules/items.json을 만든다.

클라우드에서는 Riot 서버 접속이 막혀 있어 kk PC에서 실행한다.
    python tools\build_ddragon_rules.py            # 기본 16.20.1 (패치 26.20)
    python tools\build_ddragon_rules.py --version 16.20.1
원본은 data/ddragon/<버전>/ 에 저장(git 제외), 정리본은 rules/ 에 저장한다.
이미 받은 원본이 있으면 다시 받지 않는다(--refresh로 강제).
"""
import argparse
import collections
import glob
import json
import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDN = "https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/{name}.json"
FILES = ["champion", "item", "summoner"]
LANGS = ["en_US", "ko_KR"]
SR_MAP = "11"

# Data Dragon에는 근거리/원거리 표시가 없다. 16.20 기준 근거리는 모두 사거리 225 이하,
# 원거리 최소는 Rakan 300이라 300 미만을 근거리로 본다. 예외: Lillia는 사거리 325지만 근거리 분류.
# 폼을 바꾸는 챔피언은 기본 사거리만으로 판단할 수 없어 따로 표시한다.
RANGED_MIN_RANGE = 300
MELEE_OVERRIDE = {"Lillia"}
FORM_SWAP = {"Elise", "Gnar", "Jayce", "Kayle", "Nidalee"}
# item.json의 maps["11"]은 믿을 수 없다(아레나·다른 모드 변형 아이템도 true).
# 16.20 실제 경기 190판에서 6자리 id(32xxxx, 66xxxx 등)와 정글 펫 1105~1107은 한 번도 구매되지 않았다.
SR_MAX_ITEM_ID = 100000
NOT_SR_IDS = {"1105", "1106", "1107"}

# rules/positions.json에 적은 값. Data Dragon과 어긋나면 경고한다.
EXPECTED_SMITE = {"cooldown_s": 15, "range": 500}


def load(ver, lang, name, refresh):
    path = os.path.join(ROOT, "data", "ddragon", ver, lang, name + ".json")
    if refresh or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(CDN.format(ver=ver, lang=lang, name=name), timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag(ver):
    return {"src": "ddragon", "level": "verified", "ddragon_version": ver}


def build_champions(en, ko, ver):
    out = {}
    for cid, c in en["data"].items():
        rng = c["stats"]["attackrange"]
        if cid in FORM_SWAP:
            range_type = "form_swap"
        elif cid in MELEE_OVERRIDE:
            range_type = "melee"
        else:
            range_type = "melee" if rng < RANGED_MIN_RANGE else "ranged"
        out[cid] = {
            "key": int(c["key"]),
            "name_en": c["name"],
            "name_ko": ko["data"][cid]["name"],
            "tags": c["tags"],
            "partype": c["partype"],
            "stats": c["stats"],
            "range_type": range_type,
        }
    return {
        "_doc": "Data Dragon champion.json 정리본. range_type은 기본 사거리 300 미만을 근거리로 본 파생값(Data Dragon에 근거리 표시 없음, Lillia만 예외).",
        "_source": tag(ver),
        "_range_type_source": {"level": "unverified", "note": "사거리 기준 파생값. form_swap 챔피언은 수동 확인 필요"},
        "champions": out,
    }


def stage1_purchases():
    """1단계 원본(data/raw)의 ITEM_PURCHASED 횟수. 원본이 없으면 빈 Counter."""
    counts = collections.Counter()
    for path in glob.glob(os.path.join(ROOT, "data", "raw", "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                tl = json.load(f).get("timeline")
        except (OSError, ValueError):
            continue
        if not tl:
            continue
        for fr in tl["info"]["frames"]:
            for e in fr["events"]:
                if e["type"] == "ITEM_PURCHASED":
                    counts[str(e["itemId"])] += 1
    return counts


def build_items(en, ko, ver, bought):
    out = {}
    for iid, it in en["data"].items():
        if not it.get("maps", {}).get(SR_MAP):
            continue
        if int(iid) >= SR_MAX_ITEM_ID or iid in NOT_SR_IDS:
            continue
        gold = it.get("gold", {})
        out[iid] = {
            "name_en": it["name"],
            "name_ko": ko["data"].get(iid, {}).get("name"),
            "gold": gold,
            "tags": it.get("tags", []),
            "stats": it.get("stats", {}),
            "from": it.get("from", []),
            "into": it.get("into", []),
            "depth": it.get("depth"),
            "in_store": it.get("inStore", True),
            "required_champion": it.get("requiredChampion"),
            "required_ally": it.get("requiredAlly"),
            "consumed": it.get("consumed", False),
            "bought_in_stage1": bought[iid],
        }
    return {
        "_doc": "Data Dragon item.json 중 협곡 아이템 정리본. 다른 모드 변형(6자리 id, 정글 펫 1105~1107)은 뺐다. bought_in_stage1은 1단계 경기에서 구매된 횟수. 구매 제한은 rules/items_restrictions.json이 기준.",
        "_source": tag(ver),
        "items": out,
    }


def checks(champs, items, summ_en, restrictions):
    msgs = []
    smite = summ_en["data"].get("SummonerSmite")
    if smite is None:
        msgs.append("경고: summoner.json에 SummonerSmite가 없음")
    else:
        cd = smite["cooldown"][0]
        rng = smite["range"][0]
        if cd != EXPECTED_SMITE["cooldown_s"]:
            msgs.append(f"경고: 강타 쿨타임 Data Dragon {cd} vs rules {EXPECTED_SMITE['cooldown_s']}")
        if rng != EXPECTED_SMITE["range"]:
            msgs.append(f"경고: 강타 사거리 Data Dragon {rng} vs rules {EXPECTED_SMITE['range']}")
        msgs.append(f"강타 Data Dragon: 쿨 {cd}, 사거리 {rng}, 모드 {smite.get('modes')}")
    names = {v["name_en"]: k for k, v in items["items"].items()}
    for n in ["Runaan's Hurricane", "Gustwalker Hatchling", "Mosstomper Seedling", "Scorchclaw Pup", "World Atlas"]:
        if n not in names:
            msgs.append(f"경고: 협곡 아이템에 {n} 없음")
    listed = list(restrictions["range_purchase_restrictions"]["ranged_only"]["value"])
    for n in restrictions["range_dependent_effects"]["value"]:
        listed += [x.strip() for x in n.split("/")]
    for n in listed:
        if n not in names:
            msgs.append(f"경고: items_restrictions.json의 {n}이 items.json에 없음")
    dup = collections.Counter(v["name_en"] for v in items["items"].values())
    msgs.append("이름이 겹치는 아이템: " + ", ".join(sorted(n for n, c in dup.items() if c > 1)))
    if any(v["bought_in_stage1"] for v in items["items"].values()):
        never = sorted(v["name_en"] for v in items["items"].values()
                       if v["in_store"] and v["gold"].get("purchasable") and not v["bought_in_stage1"])
        msgs.append(f"상점 아이템 중 1단계 경기에서 한 번도 안 산 것 {len(never)}개: " + ", ".join(never))
    counts = {}
    for c in champs["champions"].values():
        counts[c["range_type"]] = counts.get(c["range_type"], 0) + 1
    msgs.append(f"챔피언 {len(champs['champions'])}명, 사거리 분류 {counts}")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="16.20.1")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ver = args.version

    raw = {(lang, name): load(ver, lang, name, args.refresh) for lang in LANGS for name in FILES}
    champs = build_champions(raw[("en_US", "champion")], raw[("ko_KR", "champion")], ver)
    items = build_items(raw[("en_US", "item")], raw[("ko_KR", "item")], ver, stage1_purchases())

    rules_dir = os.path.join(ROOT, "rules")
    for name, obj in [("champions", champs), ("items", items)]:
        with open(os.path.join(rules_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    with open(os.path.join(rules_dir, "items_restrictions.json"), encoding="utf-8") as f:
        restrictions = json.load(f)
    msgs = checks(champs, items, raw[("en_US", "summoner")], restrictions)
    report = os.path.join(ROOT, "data", "ddragon", ver, "check.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(msgs) + "\n")
    print("\n".join(msgs))
    print(f"저장: rules/champions.json, rules/items.json, {os.path.relpath(report, ROOT)}")
    return 1 if any(m.startswith("경고") for m in msgs) else 0


if __name__ == "__main__":
    sys.exit(main())
