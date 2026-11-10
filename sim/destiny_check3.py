"""데스티니 카드 3차 검수(라이너·공통): 1,000판 타임라인(data/stage6_compact.json.gz, 0~21분).

    python -m sim.destiny_check3

- D29·D131(미드), D95(바텀): 라인 초반(1:30~6분) 첫 킬 뒤 60초 안 구매(귀환)했나 → 60초 뒤~4분 사망률, 그 뒤 8분 골드 차.
- D57: 원딜 시작 아이템 도란검(1055) vs 그 밖 → 10분 CS·골드 차(상대 원딜 대비).
- D110: 12~19분 30초 안 3킬 이상 우리 쪽 교전 승리 뒤 90초 안 용을 먹었나 → 승리(교전 직후 분 골드 차 보정).
결과는 reports/destiny_check3.md.
"""
import os

import numpy as np

from .data import ROOT
from .destiny_check2 import G, effect, load


def early_kill_recall(games, role):
    rec = {1: [0, 0, []], 0: [0, 0, []]}
    for G_ in games:
        if len(G_.fr) <= 9:
            continue
        for team in (100, 200):
            me = G_.role.get((team, role))
            foe = G_.role.get((300 - team, role))
            if not me or not foe:
                continue
            ks = [k for k in G_.ev["kill"] if (k[1] == me or me in k[3]) and G_.team.get(k[2]) != team and 90000 <= k[0] <= 360000]
            if not ks:
                continue
            t = ks[0][0] / 1000
            if any(k[2] == me and t - 1 <= k[0] / 1000 <= t + 60 for k in G_.ev["kill"]):
                continue          # 맞교환·60초 안 사망(죽어서 구매)은 뺌
            bought = int(any(b[1] == me and t < b[0] / 1000 <= t + 60 for b in G_.ev["buy"]))
            died = any(k[2] == me and t + 60 < k[0] / 1000 <= t + 240 for k in G_.ev["kill"])
            m = 9
            gd = (G_.fr[m][str(me)][6] - G_.fr[m][str(foe)][6]) - (G_.fr[int(t // 60)][str(me)][6] - G_.fr[int(t // 60)][str(foe)][6])
            rec[bought][0] += 1
            rec[bought][1] += died
            rec[bought][2].append(gd)
    return rec


def d57(games):
    """26.20 원딜 시작 아이템 분포(수확의 낫 1083은 거의 없음)."""
    out = {True: [], False: []}
    for G_ in games:
        if len(G_.fr) <= 10:
            continue
        for team in (100, 200):
            me, foe = G_.role.get((team, "BOTTOM")), G_.role.get((300 - team, "BOTTOM"))
            if not me or not foe:
                continue
            first = [b[2] for b in G_.ev["buy"] if b[1] == me and b[0] < 60000]
            if not first:
                continue
            dor = 1055 in first
            f, o = G_.fr[10][str(me)], G_.fr[10][str(foe)]
            out[dor].append((f[9] - o[9], f[6] - o[6]))
    return out


def d110(games):
    rows = []
    for G_ in games:
        ks = G_.ev["kill"]
        for team in (100, 200):
            last = -999
            for k in ks:
                t = k[0] / 1000
                if not (720 <= t <= 1140) or t - last < 90:
                    continue
                win_k = [x for x in ks if t <= x[0] / 1000 <= t + 30 and G_.team.get(x[2]) == 300 - team]
                lose_k = [x for x in ks if t <= x[0] / 1000 <= t + 30 and G_.team.get(x[2]) == team]
                if len(win_k) >= 3 and len(win_k) > len(lose_k):
                    end = max(x[0] for x in win_k) / 1000
                    alive_drag = any(e[1] == "DRAGON" and 0 <= e[0] / 1000 - end <= 90 for e in G_.ev["elite"])
                    if not alive_drag:
                        last = t
                        continue
                    took = int(any(e[1] == "DRAGON" and e[3] == team and 0 <= e[0] / 1000 - end <= 90 for e in G_.ev["elite"]))
                    rows.append((took, G_.gdiff(team, min(20, int(end // 60) + 1)), G_.win(team)))
                    last = t
    return effect(rows)


def main():
    games = [G(g) for g in load()]
    L = ["# 데스티니 카드 3차 검수 (라이너·공통, 1,000판 16.20 타임라인 0~21분)", ""]
    for cid, role in [("D29·D131", "MIDDLE"), ("D95", "BOTTOM")]:
        r = early_kill_recall(games, role)
        f = lambda b: (f"{r[b][1] / max(1, r[b][0]):.0%} ({r[b][1]}/{r[b][0]}), 9분까지 맞라이너 대비 골드 변화 중앙 "
                       f"{np.median(r[b][2]) if r[b][2] else float('nan'):+.0f}")
        L += [f"## {cid} {role} 라인 초반 첫 킬 뒤 귀환", "",
              f"- 1:30~6분 첫 킬(맞교환 제외) 뒤 60초 안 구매함: 60초 뒤~4분 사망 {f(1)}",
              f"- 구매 안 함: 60초 뒤~4분 사망 {f(0)}", ""]
    o = d57(games)
    s = lambda v, i: f"{np.mean([x[i] for x in v]):+.1f}" if v else "-"
    L += ["## D57 원딜 시작 아이템", "",
          f"- 도란검 시작 {len(o[True])}명: 10분 맞원딜 대비 CS {s(o[True], 0)}, 골드 {s(o[True], 1)}",
          f"- 그 밖(수확의 낫·도란방패 등) {len(o[False])}명: CS {s(o[False], 0)}, 골드 {s(o[False], 1)}",
          "- 챔피언이 다르므로 아이템 효과와 챔피언 효과가 섞임.", ""]
    e = d110(games)
    L += ["## D110 교전 승리 뒤 바로 용", "",
          ("- 12~19분 3킬 이상 교전 승리 뒤 90초 안 용 처치가 있었던 경우, 그 용을 우리가 먹음: " +
           ("-" if e is None else f"{e['n1']}팀 승률 {e['w1']:.0%} / 상대가 먹음 {e['n0']}팀 {e['w0']:.0%}, "
                                  f"골드 차 보정 계수 {e['coef']:+.2f} (95% {e['lo']:+.2f}~{e['hi']:+.2f})")), ""]
    out = os.path.join(ROOT, "reports", "destiny_check3.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
