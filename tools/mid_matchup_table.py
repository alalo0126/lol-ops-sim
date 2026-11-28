"""미드 상성 표: 1,000판에 나온 미드 쌍(같은 쌍 2판 이상, 킷 있음)을 sim.lane_duel로 돌려 10분 골드·CS·킬 차를 표로.

    python -m tools.mid_matchup_table --real data/duel_real_mid.json --n 12 --out sim/mid_matchups.json
메인 시뮬레이터(sim/lanes.py)가 미드 0~10분 동안 이 골드 차를 상성 몫으로 나눠 준다(params lanes.mid_matchup).
"""
import argparse
import collections
import json
from multiprocessing import Pool

from sim import duel, lane_duel

KITS = None


def run(pair):
    global KITS
    if KITS is None:
        KITS = duel.load()
    a, b, n = pair
    try:
        r = lane_duel.matchup(a, b, KITS, n=n)
    except Exception as e:  # noqa: BLE001
        return a, b, None, repr(e)
    return a, b, {k: round(v, 2) for k, v in r.items()}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-games", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = json.load(open(a.real))["rows"]
    kits = duel.load()
    cnt = collections.Counter(tuple(sorted(r[:2])) for r in rows if r[0] != r[1])
    pairs = [(x, y, a.n) for (x, y), c in cnt.items() if c >= a.min_games and x in kits and y in kits]
    with Pool(a.workers) as p:
        res = p.map(run, pairs)
    table, fail = {}, []
    for x, y, r, err in res:
        if r is None:
            fail.append(f"{x}|{y}: {err}")
        else:
            table[f"{x}|{y}"] = {**r, "games": cnt[(x, y)]}
    # 한 판 기록 비율(이 표가 덮는 실제 미드 경기 비율)
    covered = sum(c for (x, y), c in cnt.items() if f"{x}|{y}" in table)
    json.dump({"_doc": __doc__.strip().splitlines()[0], "n_per_pair": a.n, "pairs": len(table), "failed": fail,
               "covered_games": covered, "total_games": sum(cnt.values()),
               "note": "키는 사전순 'A|B', 값은 A 기준(gold_diff = A − B). 상성 폭이 실제보다 큼(reports/duel_notes.md 5차) → 쓰는 쪽에서 줄여서 씀",
               "table": table}, open(a.out, "w"), indent=0)
    print(len(table), "쌍, 실패", len(fail), "덮는 판", covered, "/", sum(cnt.values()))


if __name__ == "__main__":
    main()
