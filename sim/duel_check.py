"""1:1 교전 층(sim.duel) 결과를 1,000판(16.20) 미드 맞대결과 비교.

    python -m sim.duel_check --real data/duel_real_mid.json --n 100
실제 지표: 10분 골드 차, 10분 전 서로 솔로킬. 챔피언별로 (실제 상대 빈도 가중) 평균 우위를 놓고 순위 상관을 본다.
쌍 표본이 적어(쌍당 대부분 <10판) 쌍 단위 비교는 참고만.
"""
import argparse
import collections
import json
import math

from sim import duel


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--gold", type=float, default=1000)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = json.load(open(a.real))["rows"]
    champs = duel.MID10
    pairs = collections.defaultdict(lambda: [0, 0.0, 0, 0])
    for x, y, gd, sx, sy, kx, ky, v in rows:
        if x in champs and y in champs and x != y:
            for p, q, g, s1, s2 in ((x, y, gd, sx, sy), (y, x, -gd, sy, sx)):
                e = pairs[(p, q)]
                e[0] += 1; e[1] += g; e[2] += s1; e[3] += s2
    kits = duel.load()
    sim = {}
    for x in champs:
        for y in champs:
            if x < y:
                r = duel.matchup(x, y, kits, n=a.n, lv=a.level, gold=a.gold)
                sim[(x, y)] = r["edge"]
                sim[(y, x)] = -r["edge"]
    out = {"champions": {}, "pairs": []}
    xs, ys_g, ys_s = [], [], []
    for c in champs:
        opp = [(q, pairs[(c, q)]) for q in champs if q != c and pairs[(c, q)][0]]
        n = sum(e[0] for _, e in opp)
        if not n:
            continue
        se = sum(sim[(c, q)] * e[0] for q, e in opp) / n
        rg = sum(e[1] for _, e in opp) / n
        rs = sum(e[2] - e[3] for _, e in opp) / n
        out["champions"][c] = {"games": n, "sim_edge": round(se, 3), "real_gold10": round(rg), "real_solo_diff": round(rs, 3)}
        xs.append(se); ys_g.append(rg); ys_s.append(rs)
    agree = tot = 0
    for (p, q), e in pairs.items():
        if p < q and e[0] >= 5:
            tot += 1
            same = (sim[(p, q)] > 0) == (e[1] > 0)
            agree += same
            out["pairs"].append({"pair": f"{p}|{q}", "games": e[0], "sim_edge": round(sim[(p, q)], 3),
                                 "real_gold10": round(e[1] / e[0]), "same_sign": same})
    out["spearman_gold10"] = round(spearman(xs, ys_g), 3)
    out["spearman_solo"] = round(spearman(xs, ys_s), 3)
    out["pairs_same_sign"] = f"{agree}/{tot}"
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for c, v in sorted(out["champions"].items(), key=lambda kv: -kv[1]["sim_edge"]):
        print(c, v)
    print("순위 상관: 10분 골드", out["spearman_gold10"], "솔로킬 차", out["spearman_solo"], "쌍 부호 일치", out["pairs_same_sign"])


if __name__ == "__main__":
    main()
