"""5단계 판정: 학습 정글러 vs 규칙 봇 정글러, 같은 시드·같은 팀에서 짝지어 비교한다.

    python -m sim.evaluate --policy models/stage5/policy_latest.npz --games 2000   :: → reports/stage5_eval.md

완료 기준(설계 문서): 규칙 봇 정글러보다 확실히 높은 승률 + 행동이 시뮬레이터 버그 악용이 아님.
- 승률: 20분 예측 승률(Φ, 1단계 1,000판 로지스틱 회귀)의 평균. 같은 시드에서 '그 팀 정글러만 학습 정글러로 바꾼'
  경기와 '규칙 봇 그대로'인 경기를 짝지어 차이의 95% 신뢰구간을 낸다. 학습 경기 시드(100000~)와 겹치지 않는 0~ 시드를 쓴다.
- 버그 악용 점검: 행동 분포, 10분 정글 CS·골드·킬·데스, 갱·카정·라인 미니언·오브젝트 횟수를 규칙 봇과 1단계 실제 분포
  옆에 놓는다. 실제 p10~p90 밖으로 크게 벗어난 지표가 악용 후보다.
"""
import argparse
import collections
import json
import os
import statistics
from multiprocessing import Pool

import numpy as np

from .agent import ACTION_NAMES
from .data import ROOT, Rules, load_json
from .ppo import Policy
from .rollout import play

_policy = None


def apply_overrides(rules, overrides):
    """"gank.coef.bias=-0.4" 같은 문자열로 sim/params.json 값을 바꾼다(민감도 점검용)."""
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        *path, last = key.split(".")
        d = rules.params
        for k in path:
            d = d[k]
        d[last] = json.loads(val)
    return rules


def _init(path, greedy, overrides=None):
    global _policy, _greedy, _rules
    _policy = Policy.from_file(path) if path else None
    _greedy = greedy
    _rules = apply_overrides(Rules(), overrides) if overrides else None


def _run(job):
    seed, team = job
    base = play(seed, team, None, rules=_rules)
    fn = (lambda o, m, mw, ms: _policy.act(o, m, mw, ms, greedy=_greedy)) if _policy else None
    _policy and setattr(_policy, "rng", np.random.default_rng(seed))
    agent = play(seed, team, fn, rules=_rules)
    agent.pop("traj", None)
    return base, agent


def pct(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"p10": q(0.1), "p25": q(0.25), "median": statistics.median(v), "p75": q(0.75), "p90": q(0.9), "mean": statistics.fmean(v)}


def evaluate(policy_path, games, workers, greedy=False, seed0=0, overrides=None):
    jobs = [(s, 100 if s % 2 == 0 else 200) for s in range(seed0, seed0 + games)]
    with Pool(workers, initializer=_init, initargs=(policy_path, greedy, overrides)) as pool:
        pairs = pool.map(_run, jobs, chunksize=8)
    return pairs


def summarize(pairs):
    base = np.array([b["phi"] for b, a in pairs])
    agent = np.array([a["phi"] for b, a in pairs])
    d = agent - base
    se = d.std(ddof=1) / np.sqrt(len(d))
    out = {"games": len(pairs), "phi_rule": float(base.mean()), "phi_agent": float(agent.mean()),
           "diff": float(d.mean()), "diff_ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
           "agent_better_share": float((d > 0).mean())}
    for team in (100, 200):
        sel = np.array([b["team"] == team for b, a in pairs])
        out[f"phi_rule_{team}"], out[f"phi_agent_{team}"] = float(base[sel].mean()), float(agent[sel].mean())
    keys = list(pairs[0][0]["stats"])
    out["stats"] = {k: {"rule": pct([b["stats"][k] for b, a in pairs if b["stats"][k] is not None]),
                        "agent": pct([a["stats"][k] for b, a in pairs if a["stats"][k] is not None])} for k in keys}
    counts = np.sum([a["counts"] for b, a in pairs], axis=0)
    out["action_share"] = {ACTION_NAMES[i]: round(float(c / counts.sum()), 4) for i, c in enumerate(counts)}
    out["wards_per_game"] = float(np.mean([a.get("wards", 0) for b, a in pairs]))
    return out


REAL_KEYS = {"cs10": "jungle_cs", "gold10": "gold", "level10": "level", "xp10": "xp"}
REAL20 = {"cs20": "cs20", "gold20": "gold20", "kills": "kills20", "deaths": "deaths20", "assists": "assists20",
          "dragons": "dragons20", "grubs": "grubs20", "heralds": "heralds20", "enemy_cs_share": "enemy_jungle_cs_share"}


def report(out, policy_path, dst):
    real = load_json("sim", "targets_stage1.json")
    r10 = real["both_junglers_by_minute"]["10"]
    kda = real["both_junglers_kda_by_10min"]
    real20 = load_json("sim", "targets_stage5.json")
    L = []
    L.append("# 5단계 판정: 학습 정글러 vs 규칙 봇 정글러\n")
    L.append(f"정책: `{os.path.relpath(policy_path, ROOT)}` · 경기 {out['games']}판(같은 시드·같은 팀에서 짝지어 비교, 블루/레드 절반씩)\n")
    L.append("## 20분 예측 승률\n")
    L.append("| | 규칙 봇 정글러 | 학습 정글러 | 차이 (95% CI) |\n|---|---|---|---|")
    L.append(f"| 전체 | {out['phi_rule']:.3f} | {out['phi_agent']:.3f} | {out['diff']:+.3f} ({out['diff_ci95'][0]:+.3f} ~ {out['diff_ci95'][1]:+.3f}) |")
    for t, name in ((100, "블루"), (200, "레드")):
        L.append(f"| {name} | {out[f'phi_rule_{t}']:.3f} | {out[f'phi_agent_{t}']:.3f} | {out[f'phi_agent_{t}'] - out[f'phi_rule_{t}']:+.3f} |")
    L.append(f"\n같은 경기에서 학습 정글러 쪽 승률이 더 높았던 비율: {out['agent_better_share']:.1%}\n")
    L.append("## 행동 지표 (중앙값, 괄호는 평균)\n")
    L.append("| 지표 | 실제 1,000판 p10~p90 (중앙) | 규칙 봇 | 학습 정글러 | 판정 |\n|---|---|---|---|---|")
    names = {"cs10": "10분 정글 CS", "gold10": "10분 골드", "level10": "10분 레벨", "xp10": "10분 경험치", "k10": "10분 킬",
             "d10": "10분 데스", "a10": "10분 어시스트", "cs20": "20분 정글 CS", "gold20": "20분 골드", "kills": "20분 킬",
             "deaths": "20분 데스", "assists": "20분 어시스트", "ganks": "갱 시도", "gank_success": "갱 성공",
             "invade_camps": "카정 캠프", "lanefarm": "라인 미니언 먹기", "lane_cs20": "20분 라인 CS", "dragons": "드래곤",
             "grubs": "유충(3마리 묶음)", "heralds": "전령", "enemy_dragons": "적 드래곤", "first_clear_24_min": "정글 CS 24 첫 도달(분)",
             "team_gold_diff20": "20분 팀 골드 차", "enemy_cs_share": "적 정글 캠프 비중"}
    for k, s in out["stats"].items():
        realcol, verdict = "", ""
        if k in REAL20:
            r = dict(real20[REAL20[k]])
            if k == "grubs":      # 실제는 마리 수, 시뮬레이터는 3마리 묶음 처치 횟수
                r = {q: v / 3 for q, v in r.items()}
        elif k in REAL_KEYS:
            r = r10[REAL_KEYS[k]]
        elif k in ("k10", "d10", "a10"):
            r = kda[{"k10": "kills", "d10": "deaths", "a10": "assists"}[k]]
        elif k == "first_clear_24_min":
            r = real["both_first_minute_jungle_cs_ge_24"]
        else:
            r = None
        if r:
            num = lambda x: f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"
            realcol = f"{num(r['p10'])}~{num(r['p90'])} ({num(r['median'])})"
            m = s["agent"]["median"]
            verdict = "OK" if r["p25"] <= m <= r["p75"] else ("주의" if r["p10"] <= m <= r["p90"] else "밖")
        fmt = lambda x: f"{x['median']:g} ({x['mean']:.2f})"
        L.append(f"| {names.get(k, k)} | {realcol} | {fmt(s['rule'])} | {fmt(s['agent'])} | {verdict} |")
    L.append(f"\n학습 정글러 와드: 경기당 {out['wards_per_game']:.1f}개\n")
    L.append("## 학습 정글러 행동 분포 (결정 기준)\n")
    L.append("| 행동 | 비율 |\n|---|---|")
    for k, v in sorted(out["action_share"].items(), key=lambda x: -x[1]):
        if v >= 0.002:
            L.append(f"| {k} | {v:.1%} |")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "models", "stage5", "policy_latest.npz"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--greedy", action="store_true", help="가장 확률 높은 행동만(기본은 학습 때처럼 확률적으로)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "stage5_eval.md"))
    ap.add_argument("--set", dest="overrides", action="append", help="params.json 값 바꾸기, 예: --set gank.coef.bias=-0.4")
    ap.add_argument("--summary-only", action="store_true", help="보고서 없이 승률 차이만 출력")
    a = ap.parse_args()
    pairs = evaluate(a.policy, a.games, a.workers, a.greedy, overrides=a.overrides)
    if a.summary_only:
        out = summarize(pairs)
        st = out["stats"]
        print(f"{' '.join(a.overrides or ['기본'])}: 규칙 봇 {out['phi_rule']:.3f} → 학습 {out['phi_agent']:.3f} "
              f"(차이 {out['diff']:+.3f}, 95% CI {out['diff_ci95'][0]:+.3f}~{out['diff_ci95'][1]:+.3f}) "
              f"20분 킬 {st['kills']['agent']['mean']:.1f}/{st['kills']['rule']['mean']:.1f} 드래곤 {st['dragons']['agent']['mean']:.2f}/{st['dragons']['rule']['mean']:.2f}")
        return
    out = summarize(pairs)
    report(out, a.policy, a.out)
    with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"규칙 봇 {out['phi_rule']:.3f} → 학습 {out['phi_agent']:.3f} (차이 {out['diff']:+.3f}, 95% CI {out['diff_ci95'][0]:+.3f}~{out['diff_ci95'][1]:+.3f}) → {a.out}")


if __name__ == "__main__":
    main()
