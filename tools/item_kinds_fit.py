"""아이템 종류 검수(1단계 1,000판): 완성 아이템을 종류(AD·AP·브루저·탱커·서폿)로 나눠
  1) 포지션별 종류 비율·첫 완성 시각,
  2) 킬 회귀: 같은 레벨·아이템 골드에서 종류별 완성 아이템 수 차가 '누가 죽였나'를 바꾸는가,
  3) 정글 캠프 속도: 분당 정글 CS를 레벨·아이템 골드·종류로 회귀(정글러가 캠프를 돈 분만),
  4) 정글러 상점 방문(30초 안 구매를 한 번으로): 20분까지 횟수, 방문 시각, 방문당 구매 골드.

    python -m tools.item_kinds_fit        :: data/raw → sim/item_kinds.json
"""
import collections
import glob
import json
import os
import random

import numpy as np

from tools.item_builds import ROOT, ROLES, item_kinds, logistic, pct

KINDS = ["ad", "ap", "bruiser", "tank", "support"]


def kind_of(v):
    t = set(v.get("tags", []))
    if "GoldPer" in t or ("Aura" in t and not t & {"Damage", "SpellDamage"}):
        return "support"
    if "SpellDamage" in t:
        return "ap"
    dmg = bool(t & {"Damage", "CriticalStrike", "AttackSpeed", "ArmorPenetration", "OnHit"})
    tanky = bool(t & {"Health", "Armor", "SpellBlock", "MagicResist"})
    if dmg and tanky:
        return "bruiser"
    if dmg:
        return "ad"
    if tanky:
        return "tank"
    return "support"


def ols(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ w
    s2 = float(r @ r) / max(1, len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    return w, se


def main(prefix="16.20."):
    legendary, boots, names = item_kinds()
    items = json.load(open(os.path.join(ROOT, "rules", "items.json"), encoding="utf-8"))["items"]
    kind = {i: kind_of(items[str(i)]) for i in legendary}
    rng = random.Random(0)
    share = collections.defaultdict(collections.Counter)      # role → 첫 완성 종류
    t_first = collections.defaultdict(lambda: collections.defaultdict(list))
    KX, Ky = [], []
    JX, Jy = [], []
    shops = collections.defaultdict(list)
    n = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        n += 1
        info, fr = d["detail"]["info"], d["timeline"]["info"]["frames"]
        role = {p["participantId"]: p["teamPosition"] for p in info["participants"]}
        buys = collections.defaultdict(list)
        kills = []
        for f in fr:
            for e in f["events"]:
                ty = e["type"]
                if ty == "ITEM_PURCHASED":
                    buys[e["participantId"]].append((e["timestamp"], e["itemId"]))
                elif ty == "ITEM_UNDO":
                    b = buys[e["participantId"]]
                    for i in range(len(b) - 1, -1, -1):
                        if b[i][1] == e.get("beforeId"):
                            del b[i]
                            break
                elif ty == "CHAMPION_KILL" and e.get("killerId") in role and e.get("victimId") in role:
                    kills.append((e["timestamp"], e["killerId"], e["victimId"]))
        legs = {pid: [(t, i) for t, i in b if i in legendary] for pid, b in buys.items()}

        def kinds_at(pid, t):
            c = collections.Counter(kind[i] for bt, i in legs.get(pid, []) if bt <= t)
            return [c[k] for k in KINDS]

        def state(pid, t):
            m = min(int(t // 60000), len(fr) - 1)
            pf = fr[m]["participantFrames"][str(pid)]
            return pf["level"], (pf["totalGold"] - pf["currentGold"]) / 1000

        for pid, r in role.items():
            if r not in ROLES:
                continue
            lg = legs.get(pid, [])
            if lg:
                share[r][kind[lg[0][1]]] += 1
                t_first[r][kind[lg[0][1]]].append(round(lg[0][0] / 1000))
            if r == "JUNGLE":
                # 상점 방문: 1:30 이후 구매를 30초 간격으로 묶음(시작 아이템 제외)
                visits = []
                for t, i in buys.get(pid, []):
                    if t < 90000 or t > 1200000:
                        continue
                    g = items.get(str(i), {}).get("gold", {}).get("total", 0)
                    if visits and t - visits[-1][1] <= 30000:
                        visits[-1][1] = t
                        visits[-1][2] += g
                    else:
                        visits.append([t, t, g])
                shops["count20"].append(len(visits))
                shops["first_s"] += [round(v[0] / 1000) for v in visits[:1]]
                shops["gold_per_visit"] += [v[2] for v in visits]
                for k in range(5, 20):
                    if k + 1 >= len(fr):
                        break
                    a, b = fr[k]["participantFrames"][str(pid)], fr[k + 1]["participantFrames"][str(pid)]
                    dcs = b["jungleMinionsKilled"] - a["jungleMinionsKilled"]
                    if dcs < 4:   # 그 분에 캠프를 거의 안 돈 경우 제외(갱·귀환·사망)
                        continue
                    lv, ig = a["level"], (a["totalGold"] - a["currentGold"]) / 1000
                    JX.append([1.0, lv, ig] + kinds_at(pid, k * 60000)[:4] + [float(k)])
                    Jy.append(dcs)
        for t, k, v in kills:
            if t > 30 * 60000:
                continue
            a, b = (k, v) if rng.random() < 0.5 else (v, k)
            la, ga = state(a, t)
            lb, gb = state(b, t)
            ka, kb = kinds_at(a, t), kinds_at(b, t)
            KX.append([la - lb, ga - gb] + [x - y for x, y in zip(ka, kb)])
            Ky.append(1.0 if a == k else 0.0)
    KX, Ky = np.array(KX), np.array(Ky)
    w, se = logistic(KX, Ky)
    feats = ["level_diff", "item_gold_diff_k"] + [f"{k}_diff" for k in KINDS]
    kill = {f: {"coef": round(float(a), 3), "ci95": [round(float(a - 1.96 * s), 3), round(float(a + 1.96 * s), 3)]} for f, a, s in zip(feats, w, se)}
    kill["n"] = len(Ky)
    JX, Jy = np.array(JX), np.array(Jy, float)
    jw, jse = ols(JX, Jy)
    jfe = ["intercept", "level", "item_gold_k"] + [f"{k}_count" for k in KINDS[:4]] + ["minute"]
    jungle = {f: {"coef": round(float(a), 3), "se": round(float(s), 3)} for f, a, s in zip(jfe, jw, jse)}
    jungle["n"] = len(Jy)
    jungle["mean_cs_per_min"] = round(float(Jy.mean()), 2)
    out = {
        "_doc": "tools/item_kinds_fit.py: 완성 아이템 종류(태그로 분류)별 비율·킬 회귀·정글 캠프 속도·정글러 상점 방문(1단계 1,000판)",
        "games": n,
        "kinds": {i: kind[i] for i in sorted(kind)},
        "first_kind_share": {r: {k: round(c / max(1, sum(share[r].values())), 3) for k, c in share[r].most_common()} for r in ROLES},
        "first_kind_time_s": {r: {k: pct(v)["p50"] for k, v in t_first[r].items() if v} for r in ROLES},
        "kill_regression": kill,
        "jungle_cs_per_min": jungle,
        "jungle_shops": {k: pct(v) for k, v in shops.items()},
    }
    with open(os.path.join(ROOT, "sim", "item_kinds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "kinds"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
