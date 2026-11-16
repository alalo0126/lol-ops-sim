"""논문 주장 재현 점검(reports/paper_check.md). 1단계 1,000판(data/raw, 16.20)으로:

1) 포지션별 골드·경험치 차(Gaina·Nordmoen 2018 방식)가 팀 합계보다 승패를 더 잘 맞히나
2) 위치 기반 영역 지표(Silva·Lopes 2026 '공간 장악', Rioult 외 2014 '뭉침')가 승률 모델에 더하는 게 있나
3) 3분 정글러 위치(탑 쪽/바텀 쪽)가 첫 유충·첫 용·초반 킬 위치를 바꾸나(동선 → 오브젝트 주도권)

    python -m tools.paper_check            :: 셋 다
    python -m tools.paper_check --part 3   :: 3번만
"""
import argparse
import glob
import json
import os

import numpy as np

from tools.fit_winprob import fit_logistic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
MINUTES = [5, 10, 15, 20]
MAP = 14800.0


def load_games(prefix="16.20."):
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        det, tl = d.get("detail"), d.get("timeline")
        if det and tl and det["info"]["gameVersion"].startswith(prefix):
            yield os.path.basename(path)[:-5], det["info"], tl["info"]["frames"]


def state_rows():
    rows = []
    for mid, info, frames in load_games():
        ps = info["participants"]
        team = {str(p["participantId"]): p["teamId"] for p in ps}
        role = {str(p["participantId"]): p.get("teamPosition", "") for p in ps}
        if sorted(role.values()) != sorted(ROLES * 2):
            continue
        win = int(next(t["win"] for t in info["teams"] if t["teamId"] == 100))
        events = [e for fr in frames for e in fr["events"]]
        for m in MINUTES:
            if m >= len(frames) or info.get("gameDuration", 0) <= m * 60:
                break
            r = {"match": mid, "m": m, "win": win}
            keys = ["gold", "xp", "dragons", "grubs", "heralds", "spread", "terr"] + [f"{a}_{b}" for a in ROLES for b in "gx"]
            r.update({k: 0.0 for k in keys})
            pos = {100: [], 200: []}
            for pid, pf in frames[m]["participantFrames"].items():
                s = 1 if team[pid] == 100 else -1
                r["gold"] += s * pf["totalGold"]
                r["xp"] += s * pf["xp"]
                r[role[pid] + "_g"] += s * pf["totalGold"]
                r[role[pid] + "_x"] += s * pf["xp"]
                x, y = pf["position"]["x"], pf["position"]["y"]
                pos[team[pid]].append((x, y))
                t = (x + y) / 2 / MAP                       # 0 = 블루 우물, 1 = 레드 우물
                r["terr"] += s * (t if team[pid] == 100 else 1 - t)   # 적 진영으로 얼마나 나가 있나
            for tm in (100, 200):
                a = np.array(pos[tm], float)
                r["spread"] += (1 if tm == 100 else -1) * np.linalg.norm(a - a.mean(0), axis=1).mean()
            for e in events:
                if e.get("timestamp", 0) <= m * 60000 and e["type"] == "ELITE_MONSTER_KILL":
                    k = {"DRAGON": "dragons", "HORDE": "grubs", "RIFTHERALD": "heralds"}.get(e.get("monsterType"))
                    if k:
                        r[k] += 1 if e.get("killerTeamId") == 100 else -1
            rows.append(r)
    return rows


def scale(k):
    return 1000.0 if k in ("gold", "xp", "spread") or k[-2:] in ("_g", "_x") else 1.0


def design(rs, feats):
    X = np.array([[r[k] / scale(k) for k in feats] for r in rs], float)
    return np.hstack([np.ones((len(rs), 1)), X]), np.array([r["win"] for r in rs], float)


def cv_rowwise(rs, feats, reps=5, folds=10):
    """경기 단위 10겹 × 5번 반복. 행마다 평균 로그손실(짝 비교용)과 정확도."""
    ms = sorted({r["match"] for r in rs})
    lls, accs = [], []
    for rep in range(reps):
        perm = np.random.default_rng(rep).permutation(len(ms))
        fold = {ms[i]: j % folds for j, i in enumerate(perm)}
        ll, ac = np.zeros(len(rs)), np.zeros(len(rs))
        for k in range(folds):
            tr = [r for r in rs if fold[r["match"]] != k]
            te = [i for i, r in enumerate(rs) if fold[r["match"]] == k]
            w = fit_logistic(*design(tr, feats))
            X, y = design([rs[i] for i in te], feats)
            p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-6, 1 - 1e-6)
            ll[te] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            ac[te] = (p > 0.5) == (y > 0.5)
        lls.append(ll)
        accs.append(ac.mean())
    return float(np.mean(accs)), np.mean(lls, 0)


BASE = ["gold", "xp", "dragons", "grubs", "heralds"]
SETS = {
    "기준(sim/winprob 특징)": BASE,
    "포지션별 골드+경험치": [f"{r}_{b}" for b in "gx" for r in ROLES] + ["dragons", "grubs", "heralds"],
    "기준+영역": BASE + ["terr"],
    "기준+뭉침": BASE + ["spread"],
    "기준+영역+뭉침": BASE + ["terr", "spread"],
}


def part12():
    rows = state_rows()
    print(f"상태 {len(rows)}행, 경기 {len({r['match'] for r in rows})}판")
    for m in MINUTES:
        rs = [r for r in rows if r["m"] == m]
        print(f"\n== {m}분 (n={len(rs)})")
        _, base_ll = cv_rowwise(rs, BASE)
        for name, feats in SETS.items():
            acc, ll = cv_rowwise(rs, feats)
            d = ll - base_ll
            print(f"  {name:22s} 정확도 {acc:.3f} 로그손실 {ll.mean():.4f} (기준 대비 {d.mean():+.4f} ± {1.96 * d.std() / np.sqrt(len(d)):.4f})")
        w = fit_logistic(*design(rs, SETS["포지션별 골드+경험치"]))
        print("  포지션별 계수(천 단위):", {k: round(float(c), 3) for k, c in zip(SETS["포지션별 골드+경험치"], w[1:])})
        terr = np.mean([(r["terr"] > 0) == (r["win"] == 1) for r in rs if r["terr"] != 0])
        gold = np.mean([(r["gold"] > 0) == (r["win"] == 1) for r in rs if r["gold"] != 0])
        print(f"  이긴 팀이 영역 우위였던 비율 {terr:.3f} / 골드 우위였던 비율 {gold:.3f}")


def part3():
    out = []
    for _, info, frames in load_games():
        ps = info["participants"]
        jg = {p["teamId"]: str(p["participantId"]) for p in ps if p.get("teamPosition") == "JUNGLE"}
        if len(jg) != 2 or len(frames) < 7:
            continue
        team = {str(p["participantId"]): p["teamId"] for p in ps}
        events = [e for fr in frames for e in fr["events"]]
        first = {}
        for e in events:
            if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in ("HORDE", "DRAGON"):
                first.setdefault(e["monsterType"], e.get("killerTeamId"))
        for tm, pid in jg.items():
            p = frames[3]["participantFrames"][pid]["position"]
            kt = kb = 0
            for e in events:
                if e["type"] != "CHAMPION_KILL" or e["timestamp"] >= 360000 or team.get(str(e.get("killerId"))) != tm:
                    continue
                if pid in [str(e.get("killerId"))] + [str(a) for a in e.get("assistingParticipantIds", [])]:
                    q = e["position"]
                    kt, kb = (kt + 1, kb) if q["y"] - q["x"] > 0 else (kt, kb + 1)
            out.append({"top": p["y"] - p["x"] > 0, "grub": first.get("HORDE") == tm, "drag": first.get("DRAGON") == tm,
                        "kt": kt, "kb": kb, "win": next(t["win"] for t in info["teams"] if t["teamId"] == tm)})
    print(f"\n정글러 {len(out)}명 (3분 위치: 맵 대각선 위 = 탑 쪽)")
    for side in (True, False):
        s = [o for o in out if o["top"] == side]
        se = lambda v: 1.96 * np.std(v) / np.sqrt(len(v))
        g, dr, w = ([o[k] for o in s] for k in ("grub", "drag", "win"))
        print(f"  {'탑 쪽' if side else '바텀 쪽'} {len(s)}명: 첫 유충 {np.mean(g):.3f}±{se(g):.3f} 첫 용 {np.mean(dr):.3f}±{se(dr):.3f} "
              f"6분 전 관여 킬 탑 쪽 {np.mean([o['kt'] for o in s]):.2f} / 바텀 쪽 {np.mean([o['kb'] for o in s]):.2f} 승률 {np.mean(w):.3f}±{se(w):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    a = ap.parse_args()
    if a.part in (None, 1, 2):
        part12()
    if a.part in (None, 3):
        part3()
