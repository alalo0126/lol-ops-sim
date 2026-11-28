"""라인 10분 모델(sim.lane_duel) vs 미드 리플레이 68판(26.21, 1초 위치 + 타임라인 CS).

    python -m sim.mid_lane_check --real data/mid_lane_real.json --n 20
실제 지표(3~10분, 두 미드가 모두 미드 라인 띠에 있을 때): 두 미드 거리 중앙값·25%, 5/8/10분 CS 차 크기, 10분 CS.
쌍 비교: 실제 경기의 같은 챔피언 쌍을 시뮬로 n판 → 10분 CS 차 부호·크기 상관.
시뮬 거리는 1차원(라인 방향)이라 실제 2차원 거리보다 작게 나오는 쪽(옆 간격 없음).
"""
import argparse
import json
import math
import statistics as st

from sim import duel, lane_duel


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    real = json.load(open(a.real))
    kits = duel.load()
    sim_rows, skipped = [], []
    for r in real:
        if r["blue"] not in kits or r["red"] not in kits:
            skipped.append(r["match"])
            continue
        ds, d5, d8, d10, c10 = [], [], [], [], []
        try:
            lane_duel.lane_game(r["blue"], r["red"], kits, seconds=5.0)
        except Exception as e:  # noqa: BLE001  킷을 못 읽는 챔피언은 건너뜀(보고서에 남김)
            skipped.append(f"{r['match']} {r['blue']}|{r['red']}: {e!r}")
            continue
        for i in range(a.n):
            g = lane_duel.lane_game(r["blue"], r["red"], kits, seed=i)
            ds += g["dists"]
            d5.append(g["cs_at"][300][0] - g["cs_at"][300][1])
            d8.append(g["cs_at"][480][0] - g["cs_at"][480][1])
            d10.append(g["cs_at"][600][0] - g["cs_at"][600][1])
            c10 += g["cs_at"][600]
        sim_rows.append({"match": r["match"], "pair": f"{r['blue']}|{r['red']}", "real_d10": r["cs10"][0] - r["cs10"][1],
                         "sim_d10": st.mean(d10), "sim_abs": [st.mean(map(abs, d5)), st.mean(map(abs, d8)), st.mean(map(abs, d10))],
                         "sim_dist_med": st.median(ds) if ds else None, "sim_cs10": st.mean(c10),
                         "real_dist_med": r["dist_med"], "real_abs": [abs(r[k][0] - r[k][1]) for k in ("cs5", "cs8", "cs10")],
                         "real_cs10": st.mean(r["cs10"])})
    rr = [x for x in sim_rows if x["real_dist_med"]]
    out = {
        "games": len(sim_rows), "skipped": skipped,
        "dist_median": {"real": round(st.median(x["real_dist_med"] for x in rr)),
                        "sim": round(st.median(x["sim_dist_med"] for x in rr if x["sim_dist_med"]))},
        "abs_cs_diff_5_8_10": {"real": [round(st.mean(x["real_abs"][i] for x in sim_rows), 1) for i in range(3)],
                               "sim_per_game": [round(st.mean(x["sim_abs"][i] for x in sim_rows), 1) for i in range(3)]},
        "cs10": {"real": round(st.mean(x["real_cs10"] for x in sim_rows), 1), "sim": round(st.mean(x["sim_cs10"] for x in sim_rows), 1)},
        "pair_corr_d10": round(pearson([x["real_d10"] for x in sim_rows], [x["sim_d10"] for x in sim_rows]), 3),
        "pair_same_sign": f"{sum((x['real_d10'] > 0) == (x['sim_d10'] > 0) for x in sim_rows if x['real_d10'] and x['sim_d10'])}"
                          f"/{sum(1 for x in sim_rows if x['real_d10'] and x['sim_d10'])}",
        "sd_pair_mean_sim_d10": round(st.pstdev([x["sim_d10"] for x in sim_rows]), 1),
        "sd_real_d10": round(st.pstdev([x["real_d10"] for x in sim_rows]), 1),
        "rows": sim_rows,
    }
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
