"""유충·전령이 포탑 공성에 주는 효과(1,000판 타임라인).

팀-분 단위(8~20분) 표본: 그 분에 깬 방패·포탑 수를
  유충 스택(분 시작 시 우리 보유, 양 팀 합 — 합이 대부분 3이라 상대 스택 대신 합을 넣음), 전령 창(전령 처치 뒤 window_s 이내),
  골드·경험치 차, 용 수 차, 분 고정효과
로 푸아송 회귀(IRLS). 라인별(탑·미드·바텀)로도 따로 맞춘다.

  python -m tools.upper_value_fit [--raw data/raw] [--out sim/upper_value.json]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

LANES = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}
MIN0, MIN1 = 8, 20


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["timeline"]["info"]["frames"]


def team_of(pid):
    return 100 if pid <= 5 else 200


def rows_from(frames, window_s):
    """Riot 타임라인 → 팀-분 행."""
    ev = sorted((e for f in frames for e in f["events"]), key=lambda e: e["timestamp"])
    grubs, heralds, dragons, struct = [], [], [], []
    for e in ev:
        t = e["timestamp"] / 1000
        if e["type"] == "ELITE_MONSTER_KILL":
            T = e.get("killerTeamId") or team_of(e["killerId"])
            if e["monsterType"] == "HORDE":
                grubs.append((t, T))
            elif e["monsterType"] == "RIFTHERALD":
                heralds.append((t, T))
            elif e["monsterType"] == "DRAGON":
                dragons.append((t, T))
        elif e["type"] == "TURRET_PLATE_DESTROYED":
            struct.append((t, 300 - e["teamId"], LANES.get(e.get("laneType"))))
        elif e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
            struct.append((t, 300 - e["teamId"], LANES.get(e.get("laneType"))))
    snaps = []
    for f in frames:
        pf = f["participantFrames"]
        snaps.append({T: (sum(pf[str(p)]["totalGold"] for p in range(1, 11) if team_of(p) == T),
                          sum(pf[str(p)]["xp"] for p in range(1, 11) if team_of(p) == T)) for T in (100, 200)})
    return panel(grubs, heralds, dragons, struct, snaps, window_s)


def rows_from_sim(g, window_s):
    """시뮬 한 판(Game, 20분까지 진행, 분마다 record_frame) → 같은 팀-분 행. 시뮬 유충은 3마리를 한 번에 잡는다."""
    grubs, heralds, dragons, struct = [], [], [], []
    for e in g.events:
        if e["type"] == "objective":
            if e["name"] == "grubs":
                grubs += [(e["t"], e["team"])] * 3
            elif e["name"] == "herald":
                heralds.append((e["t"], e["team"]))
            else:
                dragons.append((e["t"], e["team"]))
        elif e["type"] in ("plate", "tower"):
            struct.append((e["t"], e["team"], e["lane"]))
    fr = {f["t"] // 60: f for f in g.frames}
    snaps = []
    for m in range(max(fr) + 1):
        p = fr[m]["p"] if m in fr else {}
        snaps.append({T: (sum(v["gold"] for v in p.values() if v["team"] == T),
                          sum(v["xp"] for v in p.values() if v["team"] == T)) for T in (100, 200)})
    return panel(grubs, heralds, dragons, struct, snaps, window_s)


def panel(grubs, heralds, dragons, struct, snaps, window_s):
    out = []
    for m in range(MIN0, MIN1):
        if m >= len(snaps):
            break
        t0, t1 = m * 60, (m + 1) * 60
        for T in (100, 200):
            O = 300 - T
            g_us = sum(1 for t, k in grubs if k == T and t < t0)
            g_op = sum(1 for t, k in grubs if k == O and t < t0)
            h_us = any(k == T and t0 - window_s < t < t1 for t, k in heralds)
            h_op = any(k == O and t0 - window_s < t < t1 for t, k in heralds)
            d_diff = sum(1 for t, k in dragons if k == T and t < t0) - sum(1 for t, k in dragons if k == O and t < t0)
            ys = {ln: sum(1 for t, a, l in struct if a == T and l == ln and t0 <= t < t1) for ln in ("top", "mid", "bot")}
            (gu, xu), (go, xo) = snaps[m][T], snaps[m][O]
            out.append({"m": m, "g_us": g_us, "g_all": g_us + g_op, "h_us": int(h_us), "h_op": int(h_op), "d_diff": d_diff,
                        "gold_diff": (gu - go) / 1000, "xp_diff": (xu - xo) / 1000, **ys})
    return out


def poisson(X, y, iters=50):
    b = np.zeros(X.shape[1])
    b[0] = math.log(max(y.mean(), 1e-3))
    for _ in range(iters):
        mu = np.exp(X @ b)
        H = X.T @ (X * mu[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(X.T @ (X * np.exp(X @ b)[:, None]))
    # 과산포 보정(준-푸아송)
    mu = np.exp(X @ b)
    phi = float(((y - mu) ** 2 / mu).sum() / (len(y) - X.shape[1]))
    return b, np.sqrt(np.diag(cov) * max(phi, 1.0))


FEATS = ["g_us", "g_all", "h_us", "h_op", "d_diff", "gold_diff", "xp_diff"]


def design(rows):
    mins = sorted({r["m"] for r in rows})
    X = []
    for r in rows:
        X.append([1.0] + [float(r["m"] == m) for m in mins[1:]] + [float(r[f]) for f in FEATS])
    return np.array(X), 1 + len(mins) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="sim/upper_value.json")
    ap.add_argument("--window-s", type=float, default=180)
    a = ap.parse_args()
    rows = []
    files = sorted(glob.glob(os.path.join(a.raw, "*.json")))
    for f in files:
        try:
            rows += rows_from(load(f), a.window_s)
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
    res = fit(rows, a.window_s)
    res["matches"] = len(files)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def fit(rows, window_s):
    X, off = design(rows)
    res = {"rows": len(rows), "window_s": window_s, "minutes": [MIN0, MIN1], "fit": {}}
    for target in ("top", "mid", "bot", "all"):
        y = np.array([r["top"] + r["mid"] + r["bot"] if target == "all" else r[target] for r in rows], float)
        b, se = poisson(X, y)
        res["fit"][target] = {"mean_per_min": round(float(y.mean()), 4),
                              **{f: {"coef": round(float(b[off + i]), 4), "se": round(float(se[off + i]), 4),
                                     "rate_ratio": round(math.exp(b[off + i]), 3)} for i, f in enumerate(FEATS)}}
    # 서술: 유충 스택별, 전령 창 평균
    desc = {}
    for k in range(4):
        sub = [r for r in rows if r["g_us"] == k and r["m"] >= 14]
        if sub:
            desc[f"g_us={k}"] = {ln: round(sum(r[ln] for r in sub) / len(sub), 3) for ln in ("top", "mid", "bot")} | {"n": len(sub)}
    for h in (0, 1):
        sub = [r for r in rows if r["h_us"] == h and r["m"] >= 15]
        desc[f"h_us={h} (15분+)"] = {ln: round(sum(r[ln] for r in sub) / len(sub), 3) for ln in ("top", "mid", "bot")} | {"n": len(sub)}
    res["describe"] = desc
    return res


if __name__ == "__main__":
    main()
