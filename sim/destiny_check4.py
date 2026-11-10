"""데스티니 카드 4차 검수(21분 이후): 1,000판 타임라인 경기 끝까지(data/late_compact.json.gz, tools/late_compact.py로 PC에서 생성).

    python -m sim.destiny_check4

방법은 destiny_check2와 같다(승리 또는 결과 ~ 조언 행동 + 팀 골드 차, 부트스트랩 95%). 1분 프레임·이벤트만 있어
미니언 웨이브·넥서스 체력·시야·궁극기는 모른다. 결과는 reports/destiny_check4.md.
"""
import gzip
import json
import math
import os

from .data import ROOT
from .destiny_check2 import G, effect

BARON = (5007, 10471)
DRAGON = (9866, 4414)
TP = 12


def load():
    with gzip.open(os.path.join(ROOT, "data", "late_compact.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def pos(G_, t, pid):
    f = G_.fr[min(int(round(t / 60)), len(G_.fr) - 1)][str(pid)]
    return f[0], f[1]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def side(x, y):
    return "top" if y - x > 3500 else "bot" if x - y > 3500 else "mid"


def lane_strip(x, y):
    """라인 위(정글·강가 빼고): top / mid / bot / None."""
    rx, ry = 14870 - x, 14870 - y
    if (x < 1900 and y > 4500) or (y > 12900 and x < 10000) or (x < 3300 and y > 11500):
        return "top"
    if (rx < 1900 and ry > 4500) or (ry > 12900 and rx < 10000) or (rx < 3300 and ry > 11500):
        return "bot"
    if abs(x - y) < 1000 and 4000 < x + y < 25740:
        return "mid"
    return None


def barons(G_):
    return [(e[0] / 1000, e[3]) for e in G_.ev["elite"] if e[1] == "BARON_NASHOR"]


def baron_alive(G_, t):
    return t >= 1200 and not any(t - 360 < bt <= t for bt, _ in barons(G_))


def built(G_, team, t0, t1, lane=None, inhib=None):
    """team이 [t0, t1]초에 부순 적 건물(포탑·억제기)."""
    return [b for b in G_.ev["tower"] if b[1] == 300 - team and t0 <= b[0] / 1000 <= t1
            and (lane is None or b[2] == lane) and (inhib is None or (b[3] == "INHIBITOR_BUILDING") == inhib)]


def fight_wins(G_, t_min, need=3):
    """(이긴 팀, 교전 끝 초): 30초 안 적 사망 need 이상이고 우리 사망보다 많은 교전, 90초 간격."""
    out, last = [], {100: -999, 200: -999}
    ks = G_.ev["kill"]
    for k in ks:
        t = k[0] / 1000
        if t < t_min:
            continue
        for team in (100, 200):
            if t - last[team] < 90:
                continue
            w = [x for x in ks if t <= x[0] / 1000 <= t + 30 and G_.team.get(x[2]) == 300 - team]
            l_ = [x for x in ks if t <= x[0] / 1000 <= t + 30 and G_.team.get(x[2]) == team]
            if len(w) >= need and len(w) > len(l_):
                out.append((team, max(x[0] for x in w) / 1000, len(w) - len(l_)))
                last[team] = t
    return out


def gd(G_, team, t):
    return G_.gdiff(team, int(t // 60) + 1)


def d11(games):
    rows = []
    for G_ in games:
        ks = G_.ev["kill"]
        for k in ks:
            t = k[0] / 1000
            if not baron_alive(G_, t) or any(0 < t - x[0] / 1000 <= 30 for x in ks):
                continue                       # 앞 30초 킬 없는 '끊기'만
            victim, K = k[2], 300 - G_.team.get(k[2], 0)
            jg = int(victim == G_.role.get((300 - K, "JUNGLE")))
            took = int(any(bt - t <= 90 and bt > t and bk == K for bt, bk in barons(G_)))
            rows.append((jg, gd(G_, K, t), took))
    return effect(rows)


def d19(games):
    rows, tw = [], [0, 0, 0, 0]
    for G_ in games:
        for bt, K in barons(G_):
            if bt < 1500 or K not in (100, 200):
                continue
            top = G_.role.get((K, "TOP"))
            if not top:
                continue
            far = int(dist(pos(G_, bt, top), BARON) > 6000)
            rows.append((far, gd(G_, K, bt), G_.win(K)))
            tw[far * 2] += 1
            tw[far * 2 + 1] += bool([b for b in built(G_, K, bt - 60, bt + 60) if b[2] in ("TOP_LANE", "BOT_LANE")])
    return effect(rows), tw


def d35(games):
    rows = []
    for G_ in games:
        bs = barons(G_)
        if not bs or bs[0][1] not in (100, 200):
            continue
        bt, K = bs[0]
        rows.append((int(len(built(G_, K, bt, bt + 180)) >= 2), gd(G_, K, bt), G_.win(K)))
    return effect(rows)


def d37_66(games):
    r37, r66 = [], []
    for G_ in games:
        for K, end, _ in fight_wins(G_, 1500):
            if not baron_alive(G_, end):
                continue
            bt = next((b for b, k in barons(G_) if k == K and 0 <= b - end <= 180), None)
            inh = built(G_, K, end, end + 60, inhib=True)
            if bt is not None and bt - end <= 60:
                r37.append((1, gd(G_, K, end), G_.win(K)))
                r66.append((1, gd(G_, K, end), G_.win(K)))
            elif bt is not None:
                r66.append((0, gd(G_, K, end), G_.win(K)))
            elif inh:
                r37.append((0, gd(G_, K, end), G_.win(K)))
    return effect(r37), effect(r66)


def d67(games):
    rows = []
    for G_ in games:
        for bt, K in barons(G_):
            if K not in (100, 200):
                continue
            lost = sum(1 for k in G_.ev["kill"] if 0 < k[0] / 1000 - bt <= 60 and G_.team.get(k[2]) == K)
            rows.append((int(lost < 2), gd(G_, K, bt), G_.win(K)))
    return effect(rows)


def d68(games):
    rows = []
    for G_ in games:
        for bt, K in barons(G_):
            if bt < 1500 or K not in (100, 200):
                continue
            b = built(G_, K, bt, bt + 180)
            if b:
                rows.append((int(any(x[2] == "MID_LANE" for x in b)), gd(G_, K, bt), G_.win(K)))
    return effect(rows)


def epic_soon(G_, team, t, s=60):
    return any(0 < e[0] / 1000 - t <= s and e[3] == team for e in G_.ev["elite"])


def d71(games):
    rows = []
    for G_ in games:
        for k in G_.ev["kill"]:
            t, v = k[0] / 1000, k[2]
            if not (900 <= t <= 1500) or k[4] is None or side(k[4], k[5]) == "mid":
                continue
            T = G_.team.get(v)
            if T not in (100, 200):
                continue
            alone = all(dist(pos(G_, t, p), (k[4], k[5])) > 1500 for p in G_.pids[T] if p != v)
            rows.append((int(not alone), gd(G_, T, t), int(not epic_soon(G_, 300 - T, t))))
    return effect(rows)


def d79(games):
    rows = []
    for G_ in games:
        sp = {p[0]: p[6] for p in G_.g["parts"]}
        for e in G_.ev["elite"]:
            t = e[0] / 1000
            if e[1] != "DRAGON" or not (900 <= t <= 1500):
                continue
            for T in (100, 200):
                me, foe = G_.role.get((T, "TOP")), G_.role.get((300 - T, "TOP"))
                if not me or not foe or TP in (sp[me] or []) or TP not in (sp[foe] or []):
                    continue
                near = int(dist(pos(G_, t - 30, me), DRAGON) < 5000)
                rows.append((near, gd(G_, T, t - 60), int(e[3] == T)))
    return effect(rows)


def d120(games, tags):
    rows = []
    for G_ in games:
        champ = {p[0]: p[3] for p in G_.g["parts"]}
        for e in G_.ev["elite"]:
            t = e[0] / 1000
            if e[1] not in ("DRAGON", "BARON_NASHOR") or t < 1200:
                continue
            pit = BARON if e[1] == "BARON_NASHOR" else DRAGON
            for T in (100, 200):
                far = max(G_.pids[T], key=lambda p: dist(pos(G_, t - 30, p), pit))
                if dist(pos(G_, t - 30, far), pit) < 6000:
                    continue
                tank = "Tank" in tags.get(champ[far], [])[:1]
                rows.append((int(not tank), gd(G_, T, t - 60), int(e[3] == T)))
    return effect(rows)


def d132(games):
    rows, deaths = [], {0: [], 1: []}
    for G_ in games:
        if len(G_.fr) <= 25:
            continue
        for T in (100, 200):
            ad = G_.role.get((T, "BOTTOM"))
            if not ad:
                continue
            ls = [lane_strip(*G_.fr[m][str(ad)][:2]) for m in range(15, 26)]
            mid, sd = ls.count("mid"), ls.count("top") + ls.count("bot")
            if mid == sd:
                continue
            a = int(mid > sd)
            rows.append((a, G_.gdiff(T, 15), G_.win(T)))
            deaths[a].append(sum(1 for k in G_.ev["kill"] if k[2] == ad and 900 <= k[0] / 1000 <= 1500))
    return effect(rows), {a: (len(v), sum(v) / max(1, len(v))) for a, v in deaths.items()}


def d133(games):
    rows = []
    for G_ in games:
        for K, end, net in fight_wins(G_, 1200):
            if net < 3:
                continue
            rows.append((int(bool(built(G_, K, end, end + 60))), gd(G_, K, end), G_.win(K)))
    return effect(rows)


def d48(games):
    """20~25분 억제기 파괴 뒤 3분 안 그 팀 첫 에픽: 바론(1) vs 용(0)."""
    rows = []
    for G_ in games:
        seen = set()
        for b in G_.ev["tower"]:
            t, K = b[0] / 1000, 300 - b[1]
            if b[3] != "INHIBITOR_BUILDING" or not (1200 <= t <= 1500) or K in seen:
                continue
            seen.add(K)
            e = next((e for e in G_.ev["elite"] if e[3] == K and 0 <= e[0] / 1000 - t <= 180
                      and e[1] in ("DRAGON", "BARON_NASHOR")), None)
            if e:
                rows.append((int(e[1] == "BARON_NASHOR"), gd(G_, K, t), G_.win(K)))
    return effect(rows)


def d94(games):
    return sum(1 for G_ in games for e in G_.ev["elite"] if e[2] == "ELDER_DRAGON")


def fmt(e, a="조언대로", b="아님", what="승률"):
    if e is None:
        return "표본 부족"
    return (f"{a} {e['n1']}건 {what} {e['w1']:.0%} / {b} {e['n0']}건 {e['w0']:.0%}, "
            f"골드 차 보정 계수 {e['coef']:+.2f} (95% {e['lo']:+.2f}~{e['hi']:+.2f})")


def main():
    with open(os.path.join(ROOT, "rules", "champions.json"), encoding="utf-8") as f:
        tags = {k: v.get("tags", []) for k, v in json.load(f)["champions"].items()}
    games = [G(g) for g in load()]
    L = [f"# 데스티니 카드 4차 검수 (21분 이후, 1,000판 16.20 타임라인 경기 끝까지, {len(games)}판)", "",
         "계수는 '조언대로 한 쪽'의 로그오즈 차(같은 시점 팀 골드 차 보정). 95% 구간이 0을 넘지 않으면 차이 없음.", ""]
    L += ["## D11 바론 생성 뒤 끊기: 적 정글을 끊으면 바론", "",
          "- 20분 이후 바론이 살아 있을 때 앞 30초 킬 없는 단독 킬, 90초 안 바론 획득: " +
          fmt(d11(games), "적 정글 끊음", "다른 포지션", "바론 획득"), ""]
    e19, tw = d19(games)
    L += ["## D19 후반 바론 때 탑은 사이드에", "",
          "- 25분 이후 바론 처치 팀, 탑이 바론에서 6,000 밖: " + fmt(e19, "탑 사이드", "탑 합류"),
          f"- 바론 ±60초 사이드 포탑 파괴: 탑 사이드 {tw[3]}/{tw[2]}, 탑 합류 {tw[1]}/{tw[0]}", ""]
    L += ["## D35 첫 바론 뒤 포탑", "", "- 첫 바론 뒤 3분 안 건물 2개 이상: " + fmt(d35(games), "2개 이상", "0~1개"), ""]
    e37, e66 = d37_66(games)
    L += ["## D37·D66 25분 이후 교전 승리 뒤 바론 먼저", "",
          "- D37 60초 안 바론 vs 억제기만(바론 3분 안 없음): " + fmt(e37, "바론", "억제기만"),
          "- D66 60초 안 바론 vs 60~180초 뒤 바론: " + fmt(e66, "바로 바론", "늦은 바론"), ""]
    L += ["## D48 20~25분 억제기 뒤 바론 쪽 vs 용 쪽", "",
          "- 억제기 파괴 뒤 3분 안 그 팀 첫 에픽: " + fmt(d48(games), "바론", "용"), ""]
    L += ["## D67 바론 먹고 바로 빠지기", "", "- 바론 뒤 60초 안 바론 팀 사망 2명 미만: " + fmt(d67(games), "2명 미만", "2명 이상"), ""]
    L += ["## D68 바론 버프로 미드", "", "- 25분 이후 바론 뒤 3분 건물 파괴 중 미드 포함: " + fmt(d68(games), "미드 포함", "사이드만"), ""]
    L += ["## D71 사이드 혼자 죽지 않기", "",
          "- 15~25분 사이드 사망, 아군 1,500 안에 있었음(결과=60초 안 상대 에픽 안 내줌): " +
          fmt(d71(games), "아군 근처", "혼자", "에픽 안 내줌"), ""]
    L += ["## D79 텔포 없는 탑은 용에 미리 붙기", "",
          "- 15~25분 용, 우리 탑 텔포 없음·상대 탑 텔포, 용 30초 전 우리 탑 용 5,000 안: " + fmt(d79(games), "용 근처", "멀리", "용 획득"), ""]
    L += ["## D120 오브젝트 때 사이드는 딜러가", "",
          "- 20분 이후 용·바론 때 오브젝트에서 6,000 밖 아군(가장 먼 1명)이 탱커 아님: " + fmt(d120(games, tags), "딜러", "탱커", "오브젝트 획득"), ""]
    e132, dd = d132(games)
    L += ["## D132 원딜 15~25분 미드", "",
          "- 15~25분 원딜 1분 프레임이 사이드 라인보다 미드 라인에 더 많음: " + fmt(e132, "미드", "사이드"),
          f"- 15~25분 원딜 사망 평균: 미드 {dd[1][1]:.2f}({dd[1][0]}명), 사이드 {dd[0][1]:.2f}({dd[0][0]}명)", ""]
    L += ["## D133 20분 이후 교전 순이득 3 이상 뒤 바로 귀환 말고 건물", "",
          "- 60초 안 건물 파괴: " + fmt(d133(games), "건물", "없음"), ""]
    L += ["## 읽는 법(주의)", "",
          "- D35·D67·D68·D133은 결과(건물 파괴·사망 수)가 이미 앞선 정도를 반영해, 골드 차를 보정해도 '조언대로 해서 이겼다'보다 "
          "'잘 풀린 팀이 그렇게 했다'가 섞임. 조언과 방향이 같다는 정도로만 읽는다.",
          "- D37: 억제기만 간 팀(36건)은 끝내기 직전 상황이 많아 승률이 높음. 표본이 작고 구간이 넓다.",
          "- D48: 바론 12건, 둘 다 승률 90% 이상이라 구별 불가(표본 부족).",
          "- D132: 15~25분 천상계 원딜 97%가 미드 라인에 더 오래 있음(조언과 같은 행동이 기본값), 사이드 원딜 33명뿐.", ""]
    L += ["## 표본 부족·자료 없음", "", f"- D94 장로 처치 {d94(games)}번뿐(표본 부족).",
          "- D18·D92·D109·D111: 미니언 웨이브·넥서스 체력 기록 없음 → 확인 불가.", ""]
    out = os.path.join(ROOT, "reports", "destiny_check4.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
