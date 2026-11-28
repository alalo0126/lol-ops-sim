"""챔피언 기본 능력치·스킬 표(공정 비교 설계 1-2장). 패치마다 다시 만든다.

    python tools/build_champ_kits.py data/ddragon/<버전>/en_US/championFull.json [--prev rules/champ_kits_<이전>.json]
    → rules/champ_kits_<버전>.json (+ reports/champ_diff_<버전>.md)

Data Dragon에서 바로 나오는 값(검증됨): 기본 능력치와 레벨당 성장, 이동 속도, 사거리, 스킬별 레벨당 쿨다운·비용·사거리.
설명 문구 키워드로 붙이는 기능 태그(미검증): 돌진, 하드 CC, 둔화, 보호막, 회복, 은신, 시야 제공, 광역.
타게팅/논타겟 구분·시전 시간·투사체 속도·폭·계수는 Data Dragon에 없어 CommunityDragon이 필요하다(아직 미연결).
"""
import argparse
import json
import os
import re

TAGS = {
    "dash": [r"\bdash", r"\bleap", r"\bblink", r"\bjumps?\b", r"\blunges?\b"],
    "hard_cc": [r"\bstun", r"\bknock(s|ing|ed)? (up|back|aside)", r"\bairborne\b", r"\broot", r"\bsuppress", r"\bcharm", r"\btaunt", r"\bfear", r"\bsleep"],
    "slow": [r"\bslow"],
    "shield": [r"\bshield"],
    "heal": [r"\bheals?\b", r"\bhealing\b", r"\brestores?\b"],
    "stealth": [r"\binvisib", r"\bcamouflage"],
    "vision": [r"\breveal", r"grants? (true )?sight", r"\bvision\b"],
    "area": [r"\bnearby enem", r"\barea\b", r"\bin a (line|cone|circle)"],
}
BASE_KEYS = ["hp", "hpperlevel", "mp", "mpperlevel", "movespeed", "armor", "armorperlevel", "spellblock", "spellblockperlevel",
             "attackrange", "attackdamage", "attackdamageperlevel", "attackspeed", "attackspeedperlevel", "hpregen", "hpregenperlevel"]


def text(s):
    return re.sub(r"<[^>]+>", " ", s or "").lower()


def tags(s):
    t = text(s)
    return sorted(k for k, pats in TAGS.items() if any(re.search(p, t) for p in pats))


def build(path):
    d = json.load(open(path, encoding="utf-8"))
    ver = d.get("version", "unknown")
    out = {}
    for key, c in d["data"].items():
        spells = []
        for slot, s in zip("QWER", c.get("spells", [])):
            spells.append({"slot": slot, "name": s.get("name"), "cooldown": s.get("cooldown"), "cost": s.get("cost"),
                           "range": s.get("range"), "maxrank": s.get("maxrank"),
                           "tags": tags(s.get("description", "") + " " + s.get("tooltip", ""))})
        p = c.get("passive", {})
        out[key] = {"name": c.get("name"), "classes": c.get("tags", []), "partype": c.get("partype"),
                    "stats": {k: c["stats"].get(k) for k in BASE_KEYS if k in c.get("stats", {})},
                    "melee": c.get("stats", {}).get("attackrange", 0) < 300,
                    "passive": {"name": p.get("name"), "tags": tags(p.get("description", ""))}, "spells": spells}
    return {"_doc": "tools/build_champ_kits.py. stats·spells의 cooldown/cost/range는 Data Dragon(verified), tags는 키워드(unverified)",
            "_source": {"src": "ddragon championFull", "version": ver,
                        "missing": "타게팅/논타겟, 시전 시간, 투사체 속도·폭, 계수 → CommunityDragon 필요"},
            "champions": out}, ver


def diff(new, old):
    a, b = old["champions"], new["champions"]
    lines = [f"# 챔피언 변경 {old['_source']['version']} → {new['_source']['version']}", "", "| 종류 | 챔피언 | 내용 |", "|---|---|---|"]
    for k in sorted(set(b) - set(a)):
        lines.append(f"| 추가 | {b[k]['name']} | |")
    for k in sorted(set(a) & set(b)):
        ch = [f"{s}: {a[k]['stats'].get(s)} → {b[k]['stats'].get(s)}" for s in BASE_KEYS if a[k]["stats"].get(s) != b[k]["stats"].get(s)]
        for sa, sb in zip(a[k]["spells"], b[k]["spells"]):
            for f in ("cooldown", "cost", "range", "tags"):
                if sa.get(f) != sb.get(f):
                    ch.append(f"{sb['slot']} {f}: {sa.get(f)} → {sb.get(f)}")
        if a[k]["passive"] != b[k]["passive"]:
            ch.append("패시브 변경")
        if ch:
            lines.append(f"| 변경 | {b[k]['name']} | {'; '.join(map(str, ch))} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("champion_full_json")
    ap.add_argument("--prev")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    table, ver = build(a.champion_full_json)
    p = os.path.join(a.out_dir, "rules", f"champ_kits_{ver}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(table, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(p, len(table["champions"]), "명")
    if a.prev:
        rp = os.path.join(a.out_dir, "reports", f"champ_diff_{ver}.md")
        open(rp, "w", encoding="utf-8").write(diff(table, json.load(open(a.prev, encoding="utf-8"))))
        print(rp)


if __name__ == "__main__":
    main()
