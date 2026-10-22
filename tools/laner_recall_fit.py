"""라이너 귀환(상점 방문) 모델: 1단계 원본에서 '다음 1분 안에 스스로 귀환해 상점에 감' 확률을 분 프레임 상태로 로지스틱 회귀한다.

    python -m tools.laner_recall_fit        :: data/raw → sim/laner_recall.json

귀환 시각은 기록에 없어서 구매 묶음(20초 안 연속 구매)의 첫 구매를 귀환 도착으로 본다. 죽은 뒤 90초 안에 시작한
구매 묶음은 사망 상점으로 보고 뺀다(그 분 표본도 뺀다). 특징: 보유 골드(천), 체력 비율, 분/10, 상수. 포지션별로 따로 맞춘다.
시뮬은 초당 위험률 1 − (1 − p)^(1/60)로 쓴다(sim/bots.py LanerBot).
"""
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANERS = ["TOP", "MIDDLE", "BOTTOM", "UTILITY"]
FEATS = ["const", "gold_k", "hp", "minute_10"]
SHOP_GAP_MS = 20000
DEATH_SHOP_MS = 90000


def logistic(X, y, l2=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    p = 1 / (1 + np.exp(-X @ w))
    se = np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)))
    return w, se


def main(prefix="16.20."):
    rows = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        info, tl = d["detail"]["info"], d["timeline"]["info"]
        if info.get("gameDuration", 0) < 20 * 60:
            continue
        role = {p["participantId"]: p["teamPosition"] for p in info["participants"]}
        ev = [e for f in tl["frames"] for e in f["events"]]
        deaths = collections.defaultdict(list)
        shops = collections.defaultdict(list)
        for e in ev:
            if e["type"] == "CHAMPION_KILL":
                deaths[e.get("victimId")].append(e["timestamp"])
            elif e["type"] == "ITEM_PURCHASED" and e["timestamp"] > 60000:
                s = shops[e["participantId"]]
                if not s or e["timestamp"] - s[-1][1] > SHOP_GAP_MS:
                    s.append([e["timestamp"], e["timestamp"]])
                else:
                    s[-1][1] = e["timestamp"]
        for pid, r in role.items():
            if r not in LANERS:
                continue
            dts = deaths.get(pid, [])
            vol = [s[0] for s in shops.get(pid, []) if not any(0 <= s[0] - t <= DEATH_SHOP_MS for t in dts)]
            for m in range(2, 20):
                t0, t1 = m * 60000, (m + 1) * 60000
                if any(t0 - DEATH_SHOP_MS <= t <= t1 for t in dts):
                    continue
                pf = tl["frames"][m]["participantFrames"][str(pid)]
                cs = pf["championStats"]
                # 이 분 프레임 직전 20초 안에 이미 상점에 있었으면 빼기(구매 직후 골드가 비어 있는 표본)
                if any(t0 - SHOP_GAP_MS <= s <= t0 for s in vol):
                    continue
                y = 1.0 if any(t0 < s <= t1 for s in vol) else 0.0
                rows[r].append([1.0, pf["currentGold"] / 1000, cs["health"] / max(1, cs["healthMax"]), m / 10, y])
    out = {"_doc": "tools/laner_recall_fit.py: P(다음 1분 안에 스스로 귀환) = σ(w·[1, 보유 골드(천), 체력 비율, 분/10]). 1단계 1,000판(20분 이상)",
           "features": FEATS, "roles": {}}
    for r in LANERS:
        a = np.array(rows[r])
        X, y = a[:, :4], a[:, 4]
        w, se = logistic(X, y)
        bins = {}
        for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 9)):
            m = (X[:, 1] >= lo) & (X[:, 1] < hi)
            bins[f"{lo}-{hi}k"] = {"n": int(m.sum()), "rate": round(float(y[m].mean()), 3) if m.any() else None}
        out["roles"][r] = {"w": [round(float(v), 3) for v in w], "se": [round(float(v), 3) for v in se], "n": len(y),
                           "base_rate": round(float(y.mean()), 3), "rate_by_gold": bins}
        print(r, out["roles"][r])
    with open(os.path.join(ROOT, "sim", "laner_recall.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
