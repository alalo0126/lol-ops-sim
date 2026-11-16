"""데스티니 카드 2차 검수: 1,000판 타임라인(data/stage6_compact.json.gz, 16.20, 0~21분)과 리플레이 70판.

    python -m sim.destiny_check2

카드마다 '데스티니 조언대로 한 경우 vs 아닌 경우'를 같은 시점 팀 골드 차로 묶어 비교한다(로지스틱 회귀: 승리 ~ 조언 행동 + 골드 차,
부트스트랩 95% 구간). 결과는 reports/destiny_check2.md. 1분 프레임 자료라 '그 판단을 했는지'는 대리 지표로만 본다.
"""
import collections
import gzip
import json
import math
import os

import numpy as np

from .data import ROOT

MAP = 14870


def load():
    with gzip.open(os.path.join(ROOT, "data", "stage6_compact.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def logit(X, y, iters=50, ridge=1e-3):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        w += np.linalg.solve(H, g)
    return w


def effect(rows, boot=400, seed=0):
    """rows: [(행동 0/1, 골드 차(천), 승 0/1)] → 행동 계수(로그오즈)와 95% 구간, 행동 1/0 승률."""
    if len(rows) < 20 or len({r[0] for r in rows}) < 2:
        return None
    a = np.array(rows, float)
    w = logit(a[:, :2], a[:, 2])[1]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        b = a[rng.integers(0, len(a), len(a))]
        if len(set(b[:, 0])) < 2:
            continue
        bs.append(logit(b[:, :2], b[:, 2])[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    w1 = a[a[:, 0] == 1, 2].mean()
    w0 = a[a[:, 0] == 0, 2].mean()
    return {"n1": int((a[:, 0] == 1).sum()), "n0": int((a[:, 0] == 0).sum()), "w1": w1, "w0": w0, "coef": w, "lo": lo, "hi": hi}


def team_gold(fr, team_pids):
    return sum(fr[str(p)][6] for p in team_pids)


class G:
    def __init__(self, g):
        self.g = g
        self.team = {p[0]: p[1] for p in g["parts"]}
        self.role = {(p[1], p[2]): p[0] for p in g["parts"]}
        self.pids = {t: [p[0] for p in g["parts"] if p[1] == t] for t in (100, 200)}
        self.fr = g["frames"]
        self.ev = g["events"]

    def win(self, team):
        return int(self.g["win100"] == (team == 100))

    def gdiff(self, team, minute):
        m = min(minute, len(self.fr) - 1)
        return (team_gold(self.fr[m], self.pids[team]) - team_gold(self.fr[m], self.pids[300 - team])) / 1000


def d32(games):
    """상대 서폿 구매(귀환 대리) 뒤 60초 안 그 팀 원딜 사망률 vs 같은 시각대 아무 60초."""
    hit = base = n = nb = 0
    for G_ in games:
        for team in (100, 200):
            sup, adc = G_.role.get((team, "UTILITY")), G_.role.get((team, "BOTTOM"))
            if not sup or not adc:
                continue
            deaths = [k[0] / 1000 for k in G_.ev["kill"] if k[2] == adc]
            adc_buys = [b[0] / 1000 for b in G_.ev["buy"] if b[1] == adc]
            buys = sorted({round(b[0] / 1000) for b in G_.ev["buy"] if b[1] == sup and 180 <= b[0] / 1000 <= 840
                           and not any(abs(a - b[0] / 1000) <= 40 for a in adc_buys)})
            sess, last = [], -99
            for t in buys:
                if t - last > 30:
                    sess.append(t)
                last = t
            for t in sess:
                n += 1
                hit += any(t < d <= t + 60 for d in deaths)
            for t in range(180, 840, 60):
                nb += 1
                base += any(t < d <= t + 60 for d in deaths)
    return hit, n, base, nb


def trade(games, kinds, lanes_side=None):
    """상대가 에픽(kinds)을 가져간 ±60초 안 내준 팀이 포탑·킬을 얻었나 → 승리(그 분 골드 차 보정)."""
    rows = []
    for G_ in games:
        for e in G_.ev["elite"]:
            if e[1] not in kinds:
                continue
            t, taker = e[0] / 1000, e[3]
            if taker not in (100, 200) or t > 20 * 60:
                continue
            giver = 300 - taker
            ex, ey = e[6] or 0, e[7] or 0
            far_lane = "TOP_LANE" if e[1] == "DRAGON" else "BOT_LANE"
            got = any(abs(tw[0] / 1000 - t) <= 60 and tw[1] == taker and tw[2] in (far_lane, "MID_LANE") for tw in G_.ev["tower"]) or \
                any(abs(k[0] / 1000 - t) <= 60 and G_.team.get(k[2]) == taker and G_.team.get(k[1]) == giver
                    and k[4] is not None and math.hypot(k[4] - ex, k[5] - ey) > 5000 for k in G_.ev["kill"])
            rows.append((int(got), G_.gdiff(giver, int(t // 60)), G_.win(giver)))
    return effect(rows)


def d46(games):
    """14분 팀 골드 +5,000 이상 팀: 정글러 14→17분 라인 CS 증가가 많은(≥10) 경우 vs 적은 경우 → 승리·경기 시간."""
    rows, dur = [], collections.defaultdict(list)
    for G_ in games:
        if len(G_.fr) <= 17:
            continue
        for team in (100, 200):
            if G_.gdiff(team, 14) < 5:
                continue
            j = str(G_.role[(team, "JUNGLE")])
            farm = int(G_.fr[17][j][9] - G_.fr[14][j][9] < 10)
            rows.append((farm, G_.gdiff(team, 14), G_.win(team)))
            dur[farm].append(G_.g["duration"] / 60)
    return effect(rows), {k: float(np.median(v)) for k, v in dur.items()}


def d83(games):
    """6분 정글 CS +20 이상 앞선 정글러: 6~9분에 구매(귀환) 했나 → 9~15분 사망 수·승리."""
    rows, deaths = [], collections.defaultdict(list)
    for G_ in games:
        if len(G_.fr) <= 15:
            continue
        for team in (100, 200):
            j, oj = G_.role[(team, "JUNGLE")], G_.role[(300 - team, "JUNGLE")]
            if G_.fr[6][str(j)][8] - G_.fr[6][str(oj)][8] < 20:
                continue
            bought = int(any(b[1] == j and 360 <= b[0] / 1000 <= 540 for b in G_.ev["buy"]))
            rows.append((bought, G_.gdiff(team, 6), G_.win(team)))
            deaths[bought].append(sum(1 for k in G_.ev["kill"] if k[2] == j and 540 <= k[0] / 1000 <= 900))
    return effect(rows), {k: float(np.mean(v)) for k, v in deaths.items()}


def top_side(x, y):
    return y - x > 2500


def d100(games):
    """5분 전 우리 탑 2데스 이상: 정글러 6~10분 프레임 중 탑 쪽(맵 위 삼각형) 비율 ≥ 절반 vs 미만 → 승리."""
    rows = []
    for G_ in games:
        if len(G_.fr) <= 10:
            continue
        for team in (100, 200):
            top = G_.role[(team, "TOP")]
            if sum(1 for k in G_.ev["kill"] if k[2] == top and k[0] < 300000) < 2:
                continue
            j = str(G_.role[(team, "JUNGLE")])
            share = np.mean([top_side(*G_.fr[m][j][:2]) for m in range(6, 11)])
            rows.append((int(share < 0.5), G_.gdiff(team, 5), G_.win(team)))
    return effect(rows)


def d53(games):
    """정글러가 직전 프레임 체력 40% 미만일 때 킬 관여(합류)한 교전에서 15초 안 정글러 사망률 vs 40% 이상."""
    c = {True: [0, 0], False: [0, 0]}
    for G_ in games:
        for team in (100, 200):
            j = G_.role[(team, "JUNGLE")]
            for k in G_.ev["kill"]:
                if not (k[1] == j or j in k[3]) or k[0] < 180000:
                    continue
                m = int(k[0] // 60000)
                hp = G_.fr[m][str(j)][2] / max(1, G_.fr[m][str(j)][3])
                low = hp < 0.4
                c[low][0] += 1
                c[low][1] += any(kk[2] == j and k[0] <= kk[0] <= k[0] + 15000 for kk in G_.ev["kill"])
    return c


def d107(games):
    """첫 유충(HORDE) 직전 분 우리 탑+미드 골드가 뒤질 때 vs 앞설 때: 그 유충을 우리가 가져간 비율, 우리 정글러 ±30초 사망률."""
    c = {True: [0, 0, 0], False: [0, 0, 0]}
    for G_ in games:
        hs = [e for e in G_.ev["elite"] if e[1] == "HORDE"]
        if not hs:
            continue
        t = hs[0][0] / 1000
        m = int(t // 60)
        for team in (100, 200):
            lanes = lambda tm: sum(G_.fr[m][str(G_.role[(tm, r)])][6] for r in ("TOP", "MIDDLE"))
            behind = lanes(team) < lanes(300 - team)
            j = G_.role[(team, "JUNGLE")]
            c[behind][0] += 1
            c[behind][1] += hs[0][3] == team
            c[behind][2] += any(k[2] == j and abs(k[0] / 1000 - t) <= 30 for k in G_.ev["kill"])
    return c


def main():
    games = [G(g) for g in load()]
    f = lambda e: "-" if e is None else (f"조언대로 {e['n1']}팀 승률 {e['w1']:.0%} / 아님 {e['n0']}팀 {e['w0']:.0%}, "
                                         f"골드 차 보정 계수 {e['coef']:+.2f} (95% {e['lo']:+.2f}~{e['hi']:+.2f})")
    L = ["# 데스티니 카드 2차 검수 (1,000판 16.20 타임라인)", "",
         "계수 = 같은 시점 팀 골드 차를 넣은 로지스틱 회귀의 '조언대로 함' 로그오즈. 95% 구간이 0을 넘지 않으면 차이 있음.", ""]
    hit, n, base, nb = d32(games)
    L += ["## D32 상대 서폿 귀환 뒤 바텀 다이브", "",
          f"- 상대 서폿만 구매(3~14분, 원딜은 ±40초 안 구매 없음 = 원딜 혼자 라인) 뒤 60초 안 그 팀 원딜 사망 {hit / n:.1%} ({hit}/{n}) vs 아무 60초 {base / nb:.1%} ({base}/{nb}).", ""]
    for cid, name, kinds in [("D24", "용·전령을 내주고 반대편 교환", ("DRAGON", "RIFTHERALD")),
                             ("D49", "전령을 내주고 교환", ("RIFTHERALD",)),
                             ("D75", "유충을 내주고 교환", ("HORDE",))]:
        L += [f"## {cid} {name}", "", f"- 상대가 가져간 ±60초 안 내준 팀이 반대편(용→탑·미드, 전령·유충→바텀·미드) 포탑 또는 에픽 5,000 밖 킬 획득: {f(trade(games, kinds))}", ""]
    e, dur = d46(games)
    L += ["## D46 크게 앞설 때 정글이 사이드 웨이브 먹지 않기", "",
          f"- 14분 골드 +5,000 팀, 정글러 14→17분 라인 CS +10 미만(=웨이브 안 먹음)을 '조언대로'로: {f(e)}",
          f"- 경기 시간 중앙(분): 안 먹음 {dur.get(1, float('nan')):.1f} / 먹음 {dur.get(0, float('nan')):.1f}", ""]
    e, de = d83(games)
    L += ["## D83 앞선 정글러는 귀환해 템 맞추기", "",
          f"- 6분 정글 CS +20 이상 정글러, 6~9분 구매함: {f(e)}",
          f"- 9~15분 정글러 사망 평균: 구매함 {de.get(1, float('nan')):.2f} / 안 함 {de.get(0, float('nan')):.2f}", ""]
    L += ["## D100 지는 탑 봐주기 멈추고 아래로", "",
          f"- 5분 전 탑 2데스 이상 팀, 정글러 6~10분 탑 쪽 비율 절반 미만(=조언대로): {f(d100(games))}", ""]
    c = d53(games)
    L += ["## D53 피 없을 때 질 교전 합류하지 않기", "",
          f"- 3분 이후 정글러 킬 관여 교전에서 15초 안 정글러 사망: 직전 프레임 체력 40% 미만 {c[True][1] / max(1, c[True][0]):.0%} ({c[True][1]}/{c[True][0]})"
          f" vs 40% 이상 {c[False][1] / max(1, c[False][0]):.0%} ({c[False][1]}/{c[False][0]}). 1분 전 체력이라 교전 순간 체력은 아님.", ""]
    c = d107(games)
    L += ["## D107 상체 주도권 없으면 유충 억지로 치지 않기", "",
          f"- 첫 유충 직전 분 탑+미드 골드 뒤짐: 유충 가져감 {c[True][1] / max(1, c[True][0]):.0%}, 정글러 ±30초 사망 {c[True][2] / max(1, c[True][0]):.0%} (n={c[True][0]})",
          f"- 앞섬: 가져감 {c[False][1] / max(1, c[False][0]):.0%}, 정글러 사망 {c[False][2] / max(1, c[False][0]):.0%} (n={c[False][0]})", ""]
    inv, ret, other = replay_checks()
    pc = lambda c: f"{c[1] / max(1, c[0]):.0%} ({c[1]}/{c[0]})"
    L += ["## 리플레이 70판", "",
          f"- D136 1레벨 카정: 3분 전 적 정글 진입(3초 이상) 뒤 30초 안 사망, 75초 전 진입(1레벨) {pc(inv[True])} vs 이후 {pc(inv[False])}.",
          f"- D116 리턴 갱: 적 정글의 킬 없는 바텀 방문이 끝나고 90초 안 우리 정글 바텀 방문 킬률 {pc(ret)} vs 그 밖 바텀 방문 {pc(other)}.", ""]
    out = os.path.join(ROOT, "reports", "destiny_check2.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))



# ---------------------------------------------------------------------- 리플레이 70판
def replay_checks():
    """D136: 3분 전 적 정글 진입(1레벨 = 75초 전 vs 이후) 뒤 30초 안 사망. D116: 적 정글 바텀 방문(킬 없음) 뒤 90초 안 우리 정글 바텀 방문의 킬률."""
    import csv
    import glob
    from .game import Game
    from .lane_visits import AFTER, lane_of, outcome, visits
    from .stage6 import zone
    nodes = Game(seed=0).map.nodes
    inv = {True: [0, 0], False: [0, 0]}
    ret, other = [0, 0], [0, 0]
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replay_raw", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        mid = os.path.basename(path)[:-5]
        parts = d["detail"]["info"]["participants"]
        kills = [{"t": e["timestamp"] / 1000, "killer": e.get("killerId"), "victim": e["victimId"],
                  "assists": e.get("assistingParticipantIds", [])}
                 for fr in d["timeline"]["info"]["frames"] for e in fr["events"] if e["type"] == "CHAMPION_KILL"]
        jv = {}
        for p in parts:
            if p.get("teamPosition") != "JUNGLE":
                continue
            fp = os.path.join(ROOT, "data", "replay_tracks", f"{mid}_{p['championName']}.csv")
            if not os.path.exists(fp):
                continue
            zones, xy = {}, {}
            with open(fp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("valid", "1") == "1":
                        t, x, y = round(float(r["time"])), float(r["x"]), float(r["y"])
                        zones[t], xy[t] = zone(x, y, p["teamId"], nodes), (x, y)
            pid, team = p["participantId"], p["teamId"]
            tp = {q["participantId"] for q in parts if q["teamId"] == team}
            # D136
            run = 0
            for t in range(30, 180):
                run = run + 1 if zones.get(t) == "enemy_jungle" else 0
                if run == 3:
                    lv1 = t - 2 < 75
                    inv[lv1][0] += 1
                    inv[lv1][1] += any(k["victim"] == pid and t - 2 <= k["t"] <= t + 30 for k in kills)
                    break
            vs = []
            for span in visits(zones):
                if lane_of(*xy[span[0]]) == "bot":
                    vs.append((span, outcome(span, kills, pid, tp)))
            jv[team] = vs
        if len(jv) < 2:
            continue
        for team in (100, 200):
            foe_fail = [s for s, o in jv[300 - team] if not o["kill"]]
            for s, o in jv[team]:
                r = any(0 <= s[0] - fs[1] <= 90 for fs in foe_fail)
                c = ret if r else other
                c[0] += 1
                c[1] += o["kill"]
    return inv, ret, other


if __name__ == "__main__":
    main()
