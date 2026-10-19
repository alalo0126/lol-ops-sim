"""4단계 완료 기준 점검: 규칙 봇끼리 N판을 돌려 1단계 실제 분포(sim/targets_stage1.json)와 비교한다.

    python -m sim.run_baseline --games 300

판정: 시뮬레이터 중앙값이 실제 p25~p75 안이면 OK, p10~p90 안이면 주의, 밖이면 불일치.
'보정' 표시가 붙은 지표는 그 값에 맞춰 모델 값을 정했으므로 검증 근거로 쓰지 않는다.
"""
import argparse
import collections
import json
import os
import statistics

from .data import ROOT
from .game import Game

MINUTES = [2, 3, 4, 5, 6, 8, 10, 15]
CALIBRATED = {("jungle_cs", 3), ("gold", 2)}   # dps_level1, first_bonus_treat_s를 맞춘 지표


def pct(values):
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "p10": q(0.1), "p25": q(0.25), "median": statistics.median(v), "p75": q(0.75), "p90": q(0.9)}


def verdict(sim_median, real):
    if real["p25"] <= sim_median <= real["p75"]:
        return "OK"
    if real["p10"] <= sim_median <= real["p90"]:
        return "주의"
    return "불일치"


def simulate(games, seed0=0, cognition="challenger"):
    by_min = collections.defaultdict(lambda: collections.defaultdict(list))
    first24, kda, first_recall, first_clear = [], collections.defaultdict(list), [], []
    obj_first = collections.defaultdict(list)
    ganks = collections.Counter()
    for s in range(seed0, seed0 + games):
        g = Game(seed=s, cognition=cognition).run()
        frames = {f["t"] // 60: f for f in g.frames}
        for c in g.champs:
            if c.role != "JUNGLE":
                continue
            for m in MINUTES:
                pf = frames[m]["p"][c.pid]
                for k in ("jungle_cs", "lane_cs", "gold", "level", "xp"):
                    by_min[m][k].append(pf[k])
            hit = next((m for m in sorted(frames) if frames[m]["p"][c.pid]["jungle_cs"] >= 24), None)
            if hit is not None:
                first24.append(hit)
            for k in ("kills", "deaths", "assists"):
                kda[k].append(frames[10]["p"][c.pid][k])
            if c.first_recall_s:
                first_recall.append(c.first_recall_s / 60)
            if c.first_clear_done_s:
                first_clear.append(c.first_clear_done_s / 60)
        for name in ("dragon", "grubs", "herald"):
            t = next((e["t"] for e in g.events if e["type"] == "objective" and e["name"] == name), None)
            if t is not None:
                obj_first[name].append(t / 60)
        for e in g.events:
            if e["type"] == "gank":
                ganks["ganks"] += 1
                ganks["success"] += e["success"]
    return {
        "by_min": {m: {k: pct(v) for k, v in d.items()} for m, d in by_min.items()},
        "first24": pct(first24) if first24 else None,
        "first24_hist": dict(sorted(collections.Counter(first24).items())),
        "kda10": {k: pct(v) for k, v in kda.items()},
        "first_recall_min": pct(first_recall) if first_recall else None,
        "first_clear_min": pct(first_clear) if first_clear else None,
        "objective_first_min": {k: pct(v) for k, v in obj_first.items()},
        "objective_taken_share": {k: round(len(v) / games, 2) for k, v in obj_first.items()},
        "ganks_per_game": round(ganks["ganks"] / games, 2),
        "gank_success_rate": round(ganks["success"] / max(1, ganks["ganks"]), 2),
        "games": games, "cognition": cognition,
    }


def compare(sim, real):
    rows = []
    rj = real["jungler_by_minute"]
    for m in MINUTES:
        for k in ("jungle_cs", "gold", "level", "xp"):
            r, s = rj[str(m)][k], sim["by_min"][m][k]
            rows.append({"metric": f"{m}분 {k}", "real": r, "sim": s, "verdict": verdict(s["median"], r),
                         "calibrated": (k, m) in CALIBRATED})
    for k in ("kills", "deaths", "assists"):
        r, s = real["jungler_kda_by_10min"][k], sim["kda10"][k]
        rows.append({"metric": f"10분 {k}", "real": r, "sim": s, "verdict": verdict(s["median"], r), "calibrated": False})
    r, s = real["first_minute_jungle_cs_ge_24"], sim["first24"]
    rows.append({"metric": "정글 CS 24 첫 도달(분 프레임)", "real": r, "sim": s, "verdict": verdict(s["median"], r), "calibrated": True})
    return rows


def write_report(rows, sim, real, path):
    r2 = lambda x: f"{round(x, 2):g}"
    fmt = lambda d: f'{r2(d["median"])} ({r2(d["p10"])}~{r2(d["p90"])})'
    lines = [
        "# 4단계 기준선: 규칙 봇 시뮬레이터 vs 1단계 실제 경기",
        "",
        f"시뮬레이터 {sim['games']}판(정글러 {sim['games'] * 2}명, 인지 수준 {sim['cognition']}) · 실제 {real['matches']}판(16.20 KR 챌린저·GM 정글러).",
        "값은 중앙값 (p10~p90). 판정: 시뮬 중앙값이 실제 p25~p75 안이면 OK, p10~p90 안이면 주의, 밖이면 불일치.",
        "'보정'은 그 지표에 맞춰 모델 값을 정했다는 뜻이라 검증 근거가 아니다.",
        "기준값은 1단계 1,000판(tools/stage4_targets.py)에서 뽑는다.",
        "",
        "| 지표 | 실제 | 시뮬 | 판정 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        tag = " (보정)" if r["calibrated"] else ""
        lines.append(f'| {r["metric"]}{tag} | {fmt(r["real"])} | {fmt(r["sim"])} | {r["verdict"]} |')
    lines += [
        "",
        f'정글 CS 24 첫 도달 분 분포: 실제 {real["first_minute_jungle_cs_ge_24_hist"]} / 시뮬 {sim["first24_hist"]}',
        "",
        "## 시뮬레이터에서만 보는 값",
        "",
        f'- 첫 풀캠 완료(초 단위): {fmt(sim["first_clear_min"])}분',
        f'- 첫 귀환: {fmt(sim["first_recall_min"])}분 (실제 대리값: 1:30 이후 첫 상점 구매 중앙 2.9분, p10 1.9, p90 4.7 · 3단계 문서)',
        f'- 갱: 판당 {sim["ganks_per_game"]}회, 성공률 {sim["gank_success_rate"]}',
    ]
    for k, v in sim["objective_first_min"].items():
        lines.append(f'- 첫 {k}: {fmt(v)}분, 20분 안에 잡힌 판 비율 {sim["objective_taken_share"][k]}')
    lines += ["", "차이 원인 분석은 [stage4_notes.md](stage4_notes.md)."]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cognition", default="challenger")
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "stage4_baseline.md"))
    args = ap.parse_args()
    with open(os.path.join(ROOT, "sim", "targets_stage1.json"), encoding="utf-8") as f:
        real = json.load(f)
    sim = simulate(args.games, args.seed, args.cognition)
    rows = compare(sim, real)
    for r in rows:
        tag = "(보정)" if r["calibrated"] else ""
        print(f'{r["metric"]:<26}{tag:<6} 실제 {r["real"]["median"]:>7g}  시뮬 {r["sim"]["median"]:>7g}  {r["verdict"]}')
    write_report(rows, sim, real, args.report)
    bad = [r["metric"] for r in rows if r["verdict"] == "불일치" and not r["calibrated"]]
    print(f"\n불일치 {len(bad)}개: {bad}" if bad else "\n불일치 없음")
    print(f"보고서: {args.report}")


if __name__ == "__main__":
    main()
