"""5단계 보상용 승률 모델: 1단계 경기로 '분 m 시점 상태 → 블루(100) 팀 승리 확률' 로지스틱 회귀를 분마다 맞춘다.

    python tools\\fit_winprob.py                 :: data\\raw → data\\winprob_features.json + sim\\winprob.json
    python tools/fit_winprob.py --features-only  :: 특징만 뽑기
    python tools/fit_winprob.py --from-features data/winprob_features.json   :: 뽑아 둔 특징으로 다시 맞추기

입력(블루 − 레드 차이): 팀 골드(천 단위), 팀 경험치(천 단위), 드래곤, 공허 유충, 전령, (참고용) 킬, 포탑.
보상 모델(sim)은 시뮬레이터가 만들 수 있는 값만 쓴다: 골드·경험치·드래곤·유충·전령. 포탑·킬은 시뮬레이터에
포탑이 없고 킬은 골드에 이미 들어 있어 뺀다. 참고 모델(full)과의 정확도 차이를 함께 저장한다.
분 m 모델은 m분 넘게 이어진 경기만 쓴다(그 전에 끝난 경기는 그 시각의 상태가 없음).
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINUTES = list(range(1, 21))
SIM_FEATURES = ["gold", "xp", "dragons", "grubs", "heralds"]
FULL_FEATURES = SIM_FEATURES + ["kills", "towers"]
SCALE = {"gold": 1000.0, "xp": 1000.0}


def extract(prefix="16.20."):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        det, tl = d.get("detail"), d.get("timeline")
        if not det or not tl or not det["info"]["gameVersion"].startswith(prefix):
            continue
        info = det["info"]
        team_of = {str(p["participantId"]): p["teamId"] for p in info["participants"]}
        win100 = next(t["win"] for t in info["teams"] if t["teamId"] == 100)
        duration = info.get("gameDuration", 0)
        frames = tl["info"]["frames"]
        events = [e for fr in frames for e in fr["events"]]
        for m in MINUTES:
            if m >= len(frames) or duration <= m * 60:
                break
            fr = frames[m]
            diff = {k: 0.0 for k in FULL_FEATURES}
            for pid, pf in fr["participantFrames"].items():
                s = 1 if team_of[pid] == 100 else -1
                diff["gold"] += s * pf["totalGold"]
                diff["xp"] += s * pf["xp"]
            for e in events:
                if e.get("timestamp", 0) > m * 60000:
                    continue
                if e["type"] == "CHAMPION_KILL" and str(e.get("killerId")) in team_of:
                    diff["kills"] += 1 if team_of[str(e["killerId"])] == 100 else -1
                elif e["type"] == "ELITE_MONSTER_KILL":
                    s = 1 if e.get("killerTeamId") == 100 else -1
                    key = {"DRAGON": "dragons", "HORDE": "grubs", "RIFTHERALD": "heralds"}.get(e.get("monsterType"))
                    if key:
                        diff[key] += s
                elif e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                    diff["towers"] += 1 if e["teamId"] == 200 else -1   # teamId = 잃은 팀
            rows.append({"match": os.path.basename(path)[:-5], "m": m, "win": int(win100), **{k: round(v, 1) for k, v in diff.items()}})
    return rows


def design(rows, feats):
    X = np.array([[r[k] / SCALE.get(k, 1.0) for k in feats] for r in rows], dtype=float)
    return np.hstack([np.ones((len(rows), 1)), X]), np.array([r["win"] for r in rows], dtype=float)


def fit_logistic(X, y, l2=1.0, iters=50):
    """뉴턴법(IRLS). 절편은 규제하지 않는다."""
    w = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * l2
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def cv_score(rows, feats, folds=5):
    """경기 단위 5겹 교차검증 정확도·로그손실."""
    matches = sorted({r["match"] for r in rows})
    fold_of = {mid: i % folds for i, mid in enumerate(matches)}
    acc = ll = n = 0
    for k in range(folds):
        tr = [r for r in rows if fold_of[r["match"]] != k]
        te = [r for r in rows if fold_of[r["match"]] == k]
        if not tr or not te:
            continue
        w = fit_logistic(*design(tr, feats))
        X, y = design(te, feats)
        p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-6, 1 - 1e-6)
        acc += ((p > 0.5) == (y > 0.5)).sum()
        ll += -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        n += len(y)
    return round(acc / n, 4), round(ll / n, 4)


def fit_all(rows):
    out = {"features": SIM_FEATURES, "scale": SCALE, "matches": len({r["match"] for r in rows}), "minutes": {}}
    for m in MINUTES:
        rm = [r for r in rows if r["m"] == m]
        if len(rm) < 50:
            continue
        w = fit_logistic(*design(rm, SIM_FEATURES))
        acc, ll = cv_score(rm, SIM_FEATURES)
        facc, fll = cv_score(rm, FULL_FEATURES)
        out["minutes"][m] = {"n": len(rm), "blue_win_rate": round(float(np.mean([r["win"] for r in rm])), 4),
                             "intercept": round(float(w[0]), 5),
                             "coef": {k: round(float(c), 5) for k, c in zip(SIM_FEATURES, w[1:])},
                             "cv_accuracy": acc, "cv_logloss": ll, "full_cv_accuracy": facc, "full_cv_logloss": fll}
        print(f"{m:2d}분 n={len(rm)} 정확도 {acc:.3f} (포탑·킬 포함 {facc:.3f}) 계수", out["minutes"][m]["coef"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--from-features")
    a = ap.parse_args()
    feat_path = os.path.join(ROOT, "data", "winprob_features.json")
    if a.from_features:
        with open(a.from_features, encoding="utf-8") as f:
            rows = json.load(f)
    else:
        rows = extract()
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"특징 {len(rows)}행 → {feat_path}")
    if a.features_only:
        return
    out = fit_all(rows)
    out["_doc"] = ("분 m 시점 블루 팀 승리 확률 = logistic(intercept + Σ coef·(블루−레드 차이 ÷ scale)). "
                   "tools/fit_winprob.py로 1단계 경기에서 맞춤. 시뮬레이터는 m 사이를 선형 보간한다.")
    dst = os.path.join(ROOT, "sim", "winprob.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
