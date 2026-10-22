"""아이템 빌드: 1단계 1,000판 구매 기록으로 포지션·챔피언별 실제 빌드 순서 표를 만들고,
'같은 골드면 완성 아이템 수가 교전 결과를 바꾸는가'를 킬로 검수한다.

    python -m tools.item_builds        :: data/raw → sim/item_builds.json

완성 아이템 = 상위 아이템이 없고(into 없음) 2,000골드 이상인 아이템, 신발 = 장화(1001)에서 올라가는 2단계 신발.
되돌리기(ITEM_UNDO)는 그 구매를 지운다. 판매는 빌드 순서에는 반영하지 않는다(완성품 판매는 20분 전에 드묾).

검수 회귀(킬 한 건 = 죽인 사람 vs 죽은 사람 한 쌍, 무작위 순서): P(앞 사람이 죽임) = σ(a·레벨 차 + b·아이템 골드 차(천) + c·완성 아이템 수 차).
아이템 골드 = 직전 분 프레임의 총 골드 − 보유 골드. c가 0과 다르면 골드 총액만 쓰는 지금 교전식에 완성 아이템 효과를 더한다.
"""
import collections
import glob
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def item_kinds():
    items = json.load(open(os.path.join(ROOT, "rules", "items.json"), encoding="utf-8"))["items"]
    legendary = {int(k) for k, v in items.items() if not v.get("into") and v["gold"]["total"] >= 2000 and not v.get("consumed")}
    boots = {int(k) for k, v in items.items() if "1001" in (v.get("from") or [])}
    names = {int(k): v.get("name_ko") or v.get("name_en") for k, v in items.items()}
    return legendary, boots, names


def logistic(X, y, iters=40):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y))
    p = 1 / (1 + np.exp(-X @ w))
    se = np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)))
    return w, se


def pct(v):
    v = sorted(v)
    return {f"p{q}": v[min(len(v) - 1, int(q / 100 * len(v)))] for q in (10, 25, 50, 75, 90)} | {"n": len(v)} if v else {}


def main(prefix="16.20."):
    legendary, boots, names = item_kinds()
    rng = random.Random(0)
    builds = collections.defaultdict(collections.Counter)        # (role, champ) → 첫 완성 3개 순서
    first_items = collections.defaultdict(collections.Counter)   # role → 첫 완성 아이템
    champ_first = collections.defaultdict(collections.Counter)   # (role, champ) → 첫 완성 아이템
    boots_by = collections.defaultdict(collections.Counter)      # role → 2단계 신발
    t_leg = collections.defaultdict(lambda: collections.defaultdict(list))   # role → k번째 완성 → 시각(초)
    t_boots = collections.defaultdict(list)
    X, y = [], []
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
        champ = {p["participantId"]: p["championName"] for p in info["participants"]}
        buys = collections.defaultdict(list)   # pid → [(t, item)]
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
        for pid, r in role.items():
            if r not in ROLES:
                continue
            seq = [i for _, i in legs.get(pid, [])]
            for k, (t, _) in enumerate(legs.get(pid, [])[:3]):
                t_leg[r][k + 1].append(round(t / 1000))
            if seq:
                first_items[r][seq[0]] += 1
                champ_first[(r, champ[pid])][seq[0]] += 1
            if len(seq) >= 2:
                builds[(r, champ[pid])][tuple(seq[:3])] += 1
            bt = next(((t, i) for t, i in buys.get(pid, []) if i in boots), None)
            if bt:
                boots_by[r][bt[1]] += 1
                t_boots[r].append(round(bt[0] / 1000))

        def state(pid, t):
            m = min(int(t // 60000), len(fr) - 1)
            pf = fr[m]["participantFrames"][str(pid)]
            return pf["level"], (pf["totalGold"] - pf["currentGold"]) / 1000, sum(1 for bt, _ in legs.get(pid, []) if bt <= t)
        for t, k, v in kills:
            if t > 30 * 60000:
                continue
            a, b = (k, v) if rng.random() < 0.5 else (v, k)
            la, ga, ca = state(a, t)
            lb, gb, cb = state(b, t)
            X.append([la - lb, ga - gb, ca - cb])
            y.append(1.0 if a == k else 0.0)
    X, y = np.array(X), np.array(y)
    w, se = logistic(X, y)
    reg = {"features": ["level_diff", "item_gold_diff_k", "legendary_count_diff"], "coef": [round(float(v), 3) for v in w],
           "ci95": [[round(float(a - 1.96 * s), 3), round(float(a + 1.96 * s), 3)] for a, s in zip(w, se)], "n": len(y)}
    w2, se2 = logistic(X[:, :2], y)
    reg["without_legendary"] = {"coef": [round(float(v), 3) for v in w2]}
    name = lambda i: f"{i} {names.get(i, '')}"
    out = {
        "_doc": "tools/item_builds.py: 1단계 1,000판 실제 빌드. builds = 포지션·챔피언별 첫 완성 아이템 3개 순서(상위 5개)",
        "games": n,
        "kill_regression": reg,
        "legendary_time_s": {r: {k: pct(v) for k, v in t_leg[r].items()} for r in ROLES},
        "boots_time_s": {r: pct(v) for r, v in t_boots.items()},
        "first_legendary": {r: [[name(i), c] for i, c in first_items[r].most_common(10)] for r in ROLES},
        "boots": {r: [[name(i), c] for i, c in boots_by[r].most_common(5)] for r in ROLES},
        "champion_first": {f"{r}|{ch}": [[i, c] for i, c in cnt.most_common(4)] for (r, ch), cnt in champ_first.items() if sum(cnt.values()) >= 5},
        "builds": {f"{r}|{ch}": [[list(b), c] for b, c in cnt.most_common(5)] for (r, ch), cnt in builds.items() if sum(cnt.values()) >= 5},
    }
    with open(os.path.join(ROOT, "sim", "item_builds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("games", "kill_regression", "legendary_time_s", "boots_time_s", "first_legendary", "boots")}, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
