"""룬·아이템 선택 추론: 라이너가 누구를 상대로 무엇을 골랐고, 그 선택이 초반 라인·갱 결과를 바꿨나.

    python -m tools.choice_fit [--raw data/raw] [--prefix 16.20.] [--out sim/choice_fit.json]

라이너(탑·미드·원딜·서폿)마다
  - 자기: 챔피언 유형(Data Dragon 첫 태그), 근거리/원거리, 핵심 룬(룬 트리 첫 칸), 보조 트리, 첫 완성 아이템 종류,
    14분 전 첫 방어 재료가 마법 저항인가 방어력인가
  - 라인 상대(같은 포지션 적): 유형, 근거리/원거리, 피해 종류(마법사·첫 완성템 AP면 AP)
  - 결과(14분 전): 사망, 적 정글러가 관여한 사망(갱 사망), 라인 상대를 혼자 죽인 수(솔로킬), 라인 상대에게 혼자 죽은 수
를 뽑아
  1) 선택 패턴: 포지션×유형별 핵심 룬 비율, 상대 근거리/원거리에 따른 핵심 룬 변화, 상대 AP일 때 마법 저항 먼저 사는 비율,
  2) 효과: 갱 사망·솔로킬을 핵심 룬·유형·근거리·상대 근거리로 로지스틱 회귀(포지션 고정효과. 10분 레벨·골드 차는
     사망의 결과이기도 해서 통제하지 않음)
를 낸다. 룬 이름은 Data Dragon 16.20.1 runesReforged(ko_KR).
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from tools.item_builds import ROOT, item_kinds, logistic
from tools.item_kinds_fit import kind_of

LANERS = ["TOP", "MIDDLE", "BOTTOM", "UTILITY"]
KEYSTONE = {8112: "감전", 8128: "어둠의 수확", 9923: "칼날비", 8351: "빙결 강화", 8360: "봉인 풀린 주문서", 8369: "선제공격",
            8005: "집중 공격", 8008: "치명적 속도", 8021: "기민한 발놀림", 8010: "정복자", 8437: "착취의 손아귀", 8439: "여진",
            8465: "수호자", 8214: "콩콩이 소환", 8229: "신비로운 유성", 8230: "폭풍전사의 포효", 8992: "죽음불꽃 손길"}
TREE = {8100: "지배", 8300: "영감", 8000: "정밀", 8400: "결의", 8200: "마법"}
GANK_BEFORE = 14 * 60


def participants(d, champs, items, legendary):
    info, fr = d["detail"]["info"], d["timeline"]["info"]["frames"]
    parts = info["participants"]
    ev = [e for f in fr for e in f["events"]]
    buys = collections.defaultdict(list)
    for e in ev:
        if e["type"] == "ITEM_PURCHASED":
            buys[e["participantId"]].append((e["timestamp"] / 1000, e["itemId"]))
    kills = [e for e in ev if e["type"] == "CHAMPION_KILL" and e["timestamp"] / 1000 < GANK_BEFORE]
    by = {(p["teamId"], p["teamPosition"]): p for p in parts}
    out = []
    for p in parts:
        role = p.get("teamPosition")
        if role not in LANERS or (300 - p["teamId"], role) not in by:
            continue
        o = by[(300 - p["teamId"], role)]
        ej = by.get((300 - p["teamId"], "JUNGLE"))
        pid, oid = p["participantId"], o["participantId"]

        def first_leg(q):
            return next((kind_of(items[str(i)]) for t, i in buys[q["participantId"]] if i in legendary), None)

        def info_of(q):
            c = champs.get(q["championName"], {})
            return (c.get("tags") or ["?"])[0], c.get("range_type", "?")

        tag, rng = info_of(p)
        otag, orng = info_of(o)
        ofl = first_leg(o)
        def_first = None
        for t, i in buys[pid]:
            tg = set(items.get(str(i), {}).get("tags", []))
            if t < GANK_BEFORE and tg & {"SpellBlock", "MagicResist"}:
                def_first = "mr"
                break
            if t < GANK_BEFORE and "Armor" in tg:
                def_first = "armor"
                break
        st = p["perks"]["styles"]
        ks = st[0]["selections"][0]["perk"]
        deaths = [k for k in kills if k["victimId"] == pid]
        gank = [k for k in deaths if ej and (k.get("killerId") == ej["participantId"] or ej["participantId"] in k.get("assistingParticipantIds", []))]
        solo_k = sum(1 for k in kills if k.get("killerId") == pid and k["victimId"] == oid and not k.get("assistingParticipantIds"))
        solo_d = sum(1 for k in deaths if k.get("killerId") == oid and not k.get("assistingParticipantIds"))
        f10 = fr[min(10, len(fr) - 1)]["participantFrames"]
        lv = f10[str(pid)]["level"] - f10[str(oid)]["level"]
        gd = (f10[str(pid)]["totalGold"] - f10[str(oid)]["totalGold"]) / 1000
        out.append({"role": role, "champ": p["championName"], "tag": tag, "range": rng, "keystone": KEYSTONE.get(ks, str(ks)),
                    "tree2": TREE.get(st[1]["style"], str(st[1]["style"])), "first_item": first_leg(p), "def_first": def_first,
                    "opp_tag": otag, "opp_range": orng, "opp_ap": otag == "Mage" or ofl == "ap",
                    "deaths14": len(deaths), "gank_deaths14": len(gank), "solo_kills14": solo_k, "solo_deaths14": solo_d,
                    "lv_diff10": lv, "gold_diff10": gd, "win": p["win"]})
    return out


def share(rows, key):
    c = collections.Counter(r[key] for r in rows)
    n = sum(c.values()) or 1
    return {k: round(v / n, 3) for k, v in c.most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "sim", "choice_fit.json"))
    a = ap.parse_args()
    champs = json.load(open(os.path.join(ROOT, "rules", "champions.json"), encoding="utf-8"))["champions"]
    items = json.load(open(os.path.join(ROOT, "rules", "items.json"), encoding="utf-8"))["items"]
    legendary, _, _ = item_kinds()
    rows, games = [], 0
    for path in sorted(glob.glob(os.path.join(a.raw, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(a.prefix):
            continue
        games += 1
        rows += participants(d, champs, items, legendary)
    res = {"_doc": "tools/choice_fit.py: 라이너 룬·아이템 선택 패턴과 초반(14분 전) 라인·갱 결과", "games": games, "laners": len(rows)}
    # 1) 패턴
    pat = {}
    for role in LANERS:
        R = [r for r in rows if r["role"] == role]
        pat[role] = {"tag": share(R, "tag"), "keystone": share(R, "keystone"), "by_tag": {}}
        for tag, n in collections.Counter(r["tag"] for r in R).most_common():
            if n < 30:
                continue
            S = [r for r in R if r["tag"] == tag]
            vs = {}
            for orng in ("melee", "ranged"):
                T = [r for r in S if r["opp_range"] == orng]
                if len(T) >= 15:
                    vs[orng] = {"n": len(T), "keystone": dict(list(share(T, "keystone").items())[:4])}
            pat[role]["by_tag"][tag] = {"n": n, "keystone": dict(list(share(S, "keystone").items())[:5]), "vs_opp_range": vs}
    res["patterns"] = pat
    D = [r for r in rows if r["def_first"]]
    res["def_first_mr"] = {"opp_ap": round(float(np.mean([r["def_first"] == "mr" for r in D if r["opp_ap"]])), 3) if D else None,
                           "opp_ad": round(float(np.mean([r["def_first"] == "mr" for r in D if not r["opp_ap"]])), 3) if D else None,
                           "n": len(D)}
    # 2) 효과: 갱 사망(14분 전 1회 이상), 솔로킬(1회 이상)
    ks_top = [k for k, n in collections.Counter(r["keystone"] for r in rows).most_common() if n >= 40]
    tags = [t for t, n in collections.Counter(r["tag"] for r in rows).most_common() if n >= 40][1:]
    feats = ["melee"] + [f"role_{x}" for x in LANERS[1:]] + [f"tag_{t}" for t in tags] + [f"ks_{k}" for k in ks_top[1:]] + ["opp_melee"]

    def x(r):
        return [1.0, float(r["range"] == "melee")] + [float(r["role"] == x) for x in LANERS[1:]] + [float(r["tag"] == t) for t in tags] \
            + [float(r["keystone"] == k) for k in ks_top[1:]] + [float(r["opp_range"] == "melee")]
    X = np.array([x(r) for r in rows])
    eff = {"base_keystone": ks_top[0] if ks_top else None, "base_tag": None}
    for target, f in (("gank_death", lambda r: r["gank_deaths14"] > 0), ("solo_kill", lambda r: r["solo_kills14"] > 0),
                      ("solo_death", lambda r: r["solo_deaths14"] > 0)):
        y = np.array([float(f(r)) for r in rows])
        w, se = logistic(X, y)
        eff[target] = {"mean": round(float(y.mean()), 3),
                       **{n: {"coef": round(float(c), 3), "ci95": [round(float(c - 1.96 * s), 3), round(float(c + 1.96 * s), 3)]}
                          for n, c, s in zip(["intercept"] + feats, w, se)}}
    eff["base_tag"] = [t for t, n in collections.Counter(r["tag"] for r in rows).most_common() if n >= 40][0]
    res["effects"] = eff
    res["by_role"] = {role: {k: round(float(np.mean([r[k] > 0 for r in rows if r["role"] == role])), 3)
                             for k in ("gank_deaths14", "solo_kills14", "deaths14")} for role in LANERS}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(a.out.replace(".json", "_rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "patterns"}, ensure_ascii=False, indent=1))
    for role in LANERS:
        print(role, json.dumps(res["patterns"][role]["by_tag"], ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
