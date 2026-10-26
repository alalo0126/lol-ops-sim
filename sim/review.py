"""경기 리뷰 묶음(sim_review/1): 내 경기의 분마다 '실제 선택 / AI 추천 / 20분 승률 차 / 근거'를 LLM 코치에 넘긴다.

    python tools\\my_compact.py D:\\AI\\lol\\match_cache APDO        :: → data/my_compact.json.gz
    python -m sim.review                                            :: → data/reviews/<경기>.json (전부)
    python -m sim.review --match KR_8395595585 --reps 16             :: 한 판만

방법(6단계 반사실 실험과 같음):
- 실제 정글러가 minute → minute+1분에 한 일을 운영 범주 6개 중 하나로 붙인다(stage6.label).
- 같은 실제 상태에서 학습 정글러의 첫 1분을 각 범주로 묶어 20분까지 reps번 돌린다(stage6.rollout_value).
  범주끼리 같은 시드를 써서(짝 비교) 범주 간 차이의 잡음을 줄인다.
- 추천 = 짝수 번째 반복에서 20분 예측 승률 평균이 가장 높은 범주. 승률 차 = 홀수 번째 반복의 추천 − 실제
  (짝 차이 평균), ci = 95% 구간. 고르는 반복과 재는 반복을 나눠 차이가 부풀려지지 않게 한다.

근거(evidence):
- 1,000판 천상계가 같은 시기·상황에서 추천·실제 범주를 고른 비율(reports/stage6_compare.json의 real_share).
- 반복 중 추천이 실제 선택보다 나았던 비율(시뮬).
- 그 1분의 실제 기록(정글 CS·골드 증가, 킬·사망).

verified: 조언 검증(같은 선택을 한 실제 경기가 더 이겼나)은 다른 작업이 맡는다. 결과가 나오기 전까지 전부 false로 둔다.
한계: 내 경기는 패치 16.13~16.19, 시뮬레이터 규칙은 26.20 기준. 실제 프레임은 1분 간격이라 캠프 상태·와드는 추정.
"""
import argparse
import json
import math
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np

from .data import ROOT
from .game import Game
from .ppo import Policy
from .stage6 import CATS, CAT_KO, RealMatch, label, load_compact, rollout_value, situation
from .winprob import WinProb

MINUTES = range(3, 18)
POLICY = os.environ.get("REVIEW_POLICY") or os.path.join(ROOT, "models", "stage12_a", "policy_latest.npz")
COMPARE = os.path.join(ROOT, "reports", "stage6_compare.json")
SIT_DIMS = (("dragon", "드래곤"), ("gold", "팀 골드"), ("need", "정글러 상태"), ("top_obj", "윗쪽 오브젝트"))
_S = {}


def _init():
    _S["wp"] = WinProb()
    _S["pol"] = Policy.from_file(POLICY)
    _S["nodes"] = Game(seed=0).map.nodes
    with open(COMPARE, encoding="utf-8") as f:
        _S["entries"] = json.load(f)["entries"]


def _fn(seed):
    pol = _S["pol"]
    pol.rng = np.random.default_rng(seed)
    return lambda o, m, mw, ms: pol.act(o, m, mw, ms)


def _rollouts(job):
    m, team, minute, cat, seeds = job
    rm = RealMatch(m)
    return [rollout_value(rm, team, minute, _fn(s), _S["wp"], seed=s, cat=cat)[0] for s in seeds]


def mean_ci(xs):
    n = len(xs)
    mu = float(np.mean(xs))
    if n < 2:
        return mu, mu, mu
    half = 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(n)
    return mu, mu - half, mu + half


def pro_share(sit, entries):
    """같은 시기에서 상황 값이 맞는 1,000판 표 행들. [(설명, 행)]"""
    out = []
    for key, dim in SIT_DIMS:
        for e in entries:
            if e["phase"] == sit["phase"] and e["dim"] == dim and e["value"] == sit[key]:
                out.append((f"{sit['phase']}·{sit[key]}", e))
                break
    if not out:
        for e in entries:
            if e["phase"] == sit["phase"] and e["dim"] == "시기 전체":
                out.append((sit["phase"], e))
    return out


def _pct(x):
    return f"{100 * x:.0f}%"


def decision(rm, team, minute, q, nid, entries, nodes):
    """한 분의 결정. q = {범주: [시드별 20분 승률]} (같은 시드 순서)."""
    real = label(rm.window(team, minute, nodes), team, nodes)
    means = {c: float(np.mean(v)) for c, v in q.items()}
    # 추천은 짝수 번째 반복으로 고르고 승률 차는 홀수 번째로 잰다. 같은 반복으로 고르고 재면 '가장 높게 나온 범주'를
    # 고른 탓에 차이가 부풀려진다(승자의 저주).
    pick = {c: float(np.mean(v[0::2])) for c, v in q.items()}
    rec = max(CATS, key=lambda c: pick[c])
    paired = [a - b for a, b in zip(q[rec][1::2], q[real][1::2])]
    diff, lo, hi = mean_ci(paired) if rec != real else (0.0, 0.0, 0.0)
    j = rm.jungler(team)
    a, b = rm.pf(minute, j), rm.pf(minute + 1, j)
    w = rm.window(team, minute, nodes)
    sit = situation(rm, team, minute)
    did = f"정글 CS +{b['jcs'] - a['jcs']}, 골드 +{b['total_gold'] - a['total_gold']}"
    if w["died"]:
        did += ", 사망"
    elif w["kills_for"]:
        did += f", 킬 관여 {w['kills_for']}"
    d = {
        "id": f"D{nid[0]}", "t": f"{minute}:00",
        "state": f"{sit['phase']}, {sit['dragon']}, {sit['gold']}, 레벨 {a['level']}, 체력 {_pct(a['hp'])}",
        "actual": {"action": real, "label": CAT_KO[real]},
        "recommended": {"action": rec, "label": CAT_KO[rec]},
        "winprob": {"actual": round(float(np.mean(q[real][1::2])), 4), "recommended": round(float(np.mean(q[rec][1::2])), 4),
                    "diff": round(diff, 4), "ci": [round(lo, 4), round(hi, 4)]},
        "all": {c: round(means[c], 4) for c in CATS},
        "evidence": [],
        "verified": False,
    }
    nid[0] += 1

    def add(text, value, source):
        d["evidence"].append({"id": f"S{nid[1]}", "text": text, "value": value, "source": source})
        nid[1] += 1

    add("그 1분 실제 기록", did, "Riot 타임라인")
    if rec != real:
        better = sum(1 for x in paired if x > 0) / len(paired)
        add(f"시뮬 반복 중 '{CAT_KO[rec]}'이(가) '{CAT_KO[real]}'보다 나았던 비율", _pct(better), f"시뮬 {len(paired)}회")
        for desc, e in pro_share(sit, entries)[:2]:
            sh = e["real_share"]
            add(f"천상계 {desc} 상황에서 '{CAT_KO[rec]}' 선택 비율 (실제 선택 '{CAT_KO[real]}'은 {_pct(sh[real])})",
                _pct(sh[rec]), f"1,000판 중 {e['n']}상황")
    return d


def review_match(m, reps, pool):
    rm = RealMatch(m)
    team = rm.team_of[m["me"]]
    nodes, entries = _S["nodes"], _S["entries"]
    mins = [mi for mi in MINUTES if len(rm.frames) > mi + 1 and m["duration"] > (mi + 1) * 60]
    seeds = {mi: [zlib.crc32(f"{m['match']}/{mi}/{r}".encode()) for r in range(reps)] for mi in mins}
    jobs = [(m, team, mi, c, seeds[mi]) for mi in mins for c in CATS]
    vals = pool.map(_rollouts, jobs, chunksize=1)
    q = {}
    for (m_, t_, mi, c, _), v in zip(jobs, vals):
        q.setdefault(mi, {})[c] = v
    nid = [1, 1]
    decisions = [decision(rm, team, mi, q[mi], nid, entries, nodes) for mi in mins]
    me = m["me"]
    return {
        "format": "sim_review/1",
        "match_id": m["match"], "participant_id": me, "champion": rm.champ_of[me], "role": rm.role_of[me],
        "patch": m["version"][:5], "win": bool(m["win100"] if team == 100 else 1 - m["win100"]),
        "policy": os.path.relpath(POLICY, ROOT).replace("\\", "/"), "reps": reps,
        "note": "verified는 조언 검증 결과가 나오기 전까지 모두 false. 승률은 시뮬레이터 안의 20분 예측 승률.",
        "decisions": decisions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "my_compact.json.gz"))
    ap.add_argument("--match", action="append")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    matches = [m for m in load_compact(args.src) if not args.match or m["match"] in args.match]
    _init()
    dst_dir = os.path.join(ROOT, "data", "reviews")
    os.makedirs(dst_dir, exist_ok=True)
    with Pool(args.workers, initializer=_init) as pool:
        for m in matches:
            if not RealMatch(m).valid:
                print(f"{m['match']}: 포지션이 다 안 채워져 건너뜀")
                continue
            t = time.time()
            r = review_match(m, args.reps, pool)
            dst = os.path.join(dst_dir, f"{m['match']}.json")
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            n_diff = sum(1 for d in r["decisions"] if d["actual"]["action"] != d["recommended"]["action"])
            print(f"{m['match']}: {len(r['decisions'])}분, 추천과 다름 {n_diff} → {dst} ({time.time() - t:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
