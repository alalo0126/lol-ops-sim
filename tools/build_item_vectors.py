"""아이템 능력치·효과 벡터 표(공정 비교 설계 1장). 패치마다 다시 만든다.

    python tools/build_item_vectors.py data/ddragon/16.20.1/en_US/item.json            → rules/item_vectors_16.20.1.json
    python tools/build_item_vectors.py 새.json --prev rules/item_vectors_16.20.1.json   → + reports/item_diff_<새 버전>.md

능력치 12개(이속·관통은 고정값과 %를 나눔)는 설명 <stats> 블록에서 읽는다(Data Dragon stats에는 스킬 가속·관통이 없음).
효과 태그 8개는 설명 문구 키워드로 붙인다(미검증: 새로 생기거나 바뀐 아이템은 사람이 확인).
골드 환산은 기본 아이템 단가(롱소드 등)로 능력치를 골드로 바꾼 값. 협곡(맵 11)·상점 판매·완성 단계만 담는다.
"""
import argparse
import json
import os
import re
import sys

STATS = {   # 설명 문구 → 벡터 키
    "Attack Damage": "ad", "Ability Power": "ap", "Health": "hp", "Armor": "armor", "Magic Resist": "mr",
    "Attack Speed": "as", "Critical Strike Chance": "crit", "Ability Haste": "haste", "Move Speed": "ms",
    "Armor Penetration": "pen", "Magic Penetration": "pen", "Lethality": "pen",
}
KEYS = ["ad", "ap", "hp", "armor", "mr", "as", "crit", "haste", "ms", "ms_pct", "pen", "pen_pct"]
PCT_SPLIT = {"ms", "pen"}   # 같은 이름이라도 %면 다른 축(이속 %, 관통 %)
EFFECTS = {   # 효과 태그 → 설명 문구 정규식(소문자). 'health' 같은 단어에 걸리지 않게 단어 경계를 둔다
    "antiheal": [r"\bwounds\b"], "shield": [r"\bshield"], "sustain": [r"\bomnivamp\b", r"\blife ?steal\b", r"\bheals?\b", r"\bhealing\b", r"\brestores?\b"],
    "area": [r"nearby enem", r"\barea\b", r"around you"], "slow": [r"\bslow"], "stasis": [r"\bstasis\b", r"\brevive", r"restores? +50% base health"],
    "structure": [r"\bturrets?\b", r"\bstructures?\b", r"\btowers?\b"], "mana": [r"\bmana\b"],
}
# 골드 단가: 1·2단계 재료 아이템(신발·소모품 제외)의 가격을 능력치로 회귀해 능력치 하나당 골드를 낸다(패치마다 다시 계산)


def strip(s):
    return re.sub(r"<[^>]+>", " ", s)


def parse_stats(desc):
    m = re.search(r"<stats>(.*?)</stats>", desc, re.S)
    out = {k: 0.0 for k in KEYS}
    if not m:
        return out
    for line in re.split(r"<br\s*/?>", m.group(1)):
        t = strip(line).strip()
        n = re.search(r"(-?\d+(?:\.\d+)?)\s*(%?)", t)
        if not n:
            continue
        name = re.sub(r"[-\d.%]+", "", t).strip()
        key = STATS.get(name)
        if key:
            if key in PCT_SPLIT and n.group(2) == "%":
                key += "_pct"
            out[key] += float(n.group(1))
    return out


def effect_tags(desc):
    m = re.search(r"</stats>(.*)", desc, re.S)
    text = strip(m.group(1) if m else desc).lower()
    return sorted(k for k, pats in EFFECTS.items() if any(re.search(p, text) for p in pats))


def unit_prices(items, rows):
    """rows: [(가격, 능력치 dict)] 재료 아이템. 음이 아닌 최소제곱(좌표 하강)으로 능력치 단가."""
    keys = [k for k in KEYS if sum(1 for _, st in rows if st.get(k)) >= 2]
    w = {k: 20.0 for k in keys}
    for _ in range(4000):
        for k in keys:
            num = den = 0.0
            for g, st in rows:
                x = st.get(k, 0.0)
                if not x:
                    continue
                r = g - sum(w[j] * st.get(j, 0.0) for j in keys if j != k)
                num += x * r
                den += x * x
            w[k] = max(0.0, num / den) if den else 0.0
    return w


def build(path):
    d = json.load(open(path, encoding="utf-8"))
    ver = d.get("version", "unknown")
    items = d["data"]
    out, rows = {}, []
    for iid, it in items.items():
        if len(iid) > 4 or not it.get("maps", {}).get("11") or not it.get("gold", {}).get("purchasable") \
                or it.get("requiredChampion") or it.get("inStore") is False:
            continue
        st = parse_stats(it.get("description", ""))
        eff = effect_tags(it.get("description", ""))
        out[iid] = {"name": it["name"], "gold": it["gold"]["total"], "depth": it.get("depth", 1),
                    "stats": {k: v for k, v in st.items() if v}, "effects": eff}
        if it.get("depth", 1) <= 2 and it.get("into") and not eff and any(st.values()):
            rows.append((it["gold"]["total"], st))
    up = unit_prices(items, rows)
    for v in out.values():
        v["stat_gold"] = round(sum(v["stats"].get(k, 0.0) * up.get(k, 0.0) for k in KEYS), 1)
    return {"_doc": "tools/build_item_vectors.py. stats=설명 <stats> 값, effects=키워드 태그(미검증), stat_gold=기본 아이템 단가 환산",
            "_source": {"src": "ddragon", "version": ver, "unit_prices": {k: round(v, 2) for k, v in up.items()}, "unit_price_rows": len(rows),
                        "level": "stats verified(ddragon), effects unverified"}, "items": out}, ver


def diff(new, old):
    a, b = old["items"], new["items"]
    lines = [f"# 아이템 변경 {old['_source']['version']} → {new['_source']['version']}", "",
             "| 종류 | 아이템 | 내용 |", "|---|---|---|"]
    for i in sorted(set(b) - set(a)):
        lines.append(f"| 추가 | {b[i]['name']} | {b[i]['gold']}G {b[i]['stats']} {b[i]['effects']} |")
    for i in sorted(set(a) - set(b)):
        lines.append(f"| 삭제 | {a[i]['name']} | |")
    for i in sorted(set(a) & set(b)):
        ch = [f"{k}: {a[i][k]} → {b[i][k]}" for k in ("gold", "stats", "effects") if a[i][k] != b[i][k]]
        if ch:
            lines.append(f"| 변경 | {b[i]['name']} | {'; '.join(map(str, ch))} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item_json")
    ap.add_argument("--prev")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    table, ver = build(a.item_json)
    p = os.path.join(a.out_dir, "rules", f"item_vectors_{ver}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(table, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(p, len(table["items"]), "개")
    if a.prev:
        rp = os.path.join(a.out_dir, "reports", f"item_diff_{ver}.md")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        open(rp, "w", encoding="utf-8").write(diff(table, json.load(open(a.prev, encoding="utf-8"))))
        print(rp)


if __name__ == "__main__":
    sys.exit(main())
