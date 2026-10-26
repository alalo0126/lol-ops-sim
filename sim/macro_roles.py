"""운영 단계(14~30분) 포지션별 행동을 라인전 결과로 나눠 천상계 1,000판에서 잰다.

    python -m sim.macro_roles      :: data/macro_compact.json.gz(tools/macro_compact.py) → reports/macro_roles.md, sim/macro_roles.json

라인전 결과(14분, 같은 포지션 상대와 총 골드 차): 이김(+1,000 이상) / 비김 / 짐(−1,000 이하).
상성(탑): 내 챔피언·상대 챔피언 첫 태그(Data Dragon) 묶음 — 사이드형(Fighter·Assassin 근접) / 탱커형(Tank) / 원거리형.
분 프레임마다 행동:
  - 구역: 기지 / 사이드 라인(탑·바텀) / 미드 라인 / 우리 정글 / 적 정글 / 강가·둥지
  - 라인 깊이(라인에 있을 때): 우리 본진까지 거리 ÷ (우리 + 적 본진) → 우리 2차 쪽(<0.38) / 우리 절반 / 적 절반 / 적 2차 쪽(>0.62)
  - 뭉침: 2,500 안 아군 2명 이상
  - 정글러: 원딜·서폿 3,000 안, 사이드 라이너(탑) 3,000 안이면서 둘 다 사이드 라인
한타: 15초 안에 이어진 킬 3개 이상(첫 킬에서 3,500 안). 참여 = 그 킬들의 킬·데스·어시스트.
'잘한 행동' = 같은 포지션·같은 라인전 결과에서 그 행동을 한 분의 3분 뒤 예측 승률 변화 − 안 한 분(지금 예측 승률·분 통제).
  예측 승률은 이 1,000판으로 분마다 맞춘 로지스틱(골드·경험치·포탑·드래곤·바론 차). 선택 편향이 섞인 연관이지 인과가 아니다.
"""
import collections
import gzip
import json
import math
import os

import numpy as np

from .data import ROOT, load_json
from .game import Game
from .lane_visits import lane_of
from .stage6 import zone

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_KO = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "BOTTOM": "원딜", "UTILITY": "서폿"}
M0, M1, AHEAD = 14, 30, 3
LEAD = 1000
STATE_KO = {"won": "라인전 이김", "even": "비김", "lost": "라인전 짐"}


def load():
    with gzip.open(os.path.join(ROOT, "data", "macro_compact.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def champ_kind(name, champs):
    c = champs.get(name, {})
    tags = c.get("tags") or ["?"]
    if c.get("range_type") == "ranged" and tags[0] not in ("Tank",):
        return "ranged"
    if tags[0] == "Tank":
        return "tank"
    return "side"       # Fighter·Assassin 등 근접


# ---------------------------------------------------------------------- 예측 승률(경기 끝까지)
def team_feats(m, minute):
    """블루 − 레드: 골드(천), 경험치(천), 포탑, 드래곤, 바론."""
    fr = m["frames"][minute]
    team = {p[0]: p[1] for p in m["parts"]}
    g = x = 0.0
    for pid, v in fr.items():
        s = 1 if team[int(pid)] == 100 else -1
        g += s * v[6]
        x += s * v[5]
    t = minute * 60000
    tw = sum((1 if e[1] == 200 else -1) for e in m["events"]["tower"] if e[0] <= t and e[3] == "TOWER_BUILDING")
    dr = sum((1 if e[3] == 100 else -1) for e in m["events"]["elite"] if e[0] <= t and e[1] == "DRAGON")
    ba = sum((1 if e[3] == 100 else -1) for e in m["events"]["elite"] if e[0] <= t and e[1] == "BARON_NASHOR")
    return [g / 1000, x / 1000, tw, dr, ba]


def logit_fit(X, y, ridge=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w += np.linalg.solve(H, X.T @ (y - p))
    return w


def fit_wp(M):
    """분(14~33)마다 로지스틱. 표본이 적은 뒤쪽 분은 앞 분 모델을 씀."""
    W = {}
    for minute in range(10, 34):
        X, y = [], []
        for m in M:
            if len(m["frames"]) > minute:
                X.append(team_feats(m, minute))
                y.append(m["win100"])
        if len(X) >= 150:
            W[minute] = logit_fit(np.array(X), np.array(y, float))
        elif minute - 1 in W:
            W[minute] = W[minute - 1]
    return W


def phi(W, m, minute, team):
    w = W[min(minute, max(W))]
    x = np.array([1.0] + team_feats(m, minute))
    p = 1 / (1 + math.exp(-float(x @ w)))
    return p if team == 100 else 1 - p


# ---------------------------------------------------------------------- 행동
def fights(m):
    """[(시작 ms, 중심 x, y, {참여 pid}, 킬 수)]."""
    ks = sorted(k for k in m["events"]["kill"] if k[4] is not None)
    out, cur = [], []
    for k in ks + [None]:
        if k is not None and cur and k[0] - cur[-1][0] <= 15000 and math.dist((k[4], k[5]), (cur[0][4], cur[0][5])) <= 3500:
            cur.append(k)
            continue
        if len(cur) >= 3:
            ppl = set()
            for c in cur:
                ppl |= {c[1], c[2], *c[3]}
            ppl.discard(0)
            out.append((cur[0][0], float(np.mean([c[4] for c in cur])), float(np.mean([c[5] for c in cur])), ppl, len(cur)))
        cur = [k] if k is not None else []
    return out


def depth(x, y, team, nodes):
    own = nodes["B_fountain" if team == 100 else "R_fountain"]
    foe = nodes["R_fountain" if team == 100 else "B_fountain"]
    a, b = math.dist((x, y), (own["x"], own["y"])), math.dist((x, y), (foe["x"], foe["y"]))
    p = a / (a + b)
    return "own_deep" if p < 0.38 else "own_half" if p < 0.5 else "enemy_half" if p < 0.62 else "enemy_deep"


def place(x, y, team, nodes):
    z = zone(x, y, team, nodes)
    if z == "lane":
        return "side" if lane_of(x, y) in ("top", "bot") else "mid"
    return {"pit": "river"}.get(z, z)


def rows_for(m, W, nodes, champs):
    parts = {p[0]: p for p in m["parts"]}
    by_role = {(p[1], p[2]): p[0] for p in m["parts"]}
    if len(by_role) != 10 or len(m["frames"]) <= M0 + AHEAD:
        return [], []
    fr14 = m["frames"][M0]
    state, kind = {}, {}
    for pid, p in parts.items():
        opp = by_role.get((300 - p[1], p[2]))
        if opp is None:
            return [], []
        d = fr14[str(pid)][6] - fr14[str(opp)][6]
        state[pid] = "won" if d >= LEAD else "lost" if d <= -LEAD else "even"
        kind[pid] = (champ_kind(p[3], champs), champ_kind(parts[opp][3], champs))
    win = {100: m["win100"], 200: 1 - m["win100"]}
    F = fights(m)
    deaths = {(int(k[0] // 60000), k[2]) for k in m["events"]["kill"]}
    out = []
    last = min(M1, len(m["frames"]) - 1 - AHEAD)
    for minute in range(M0, last + 1):
        fr = m["frames"][minute]
        ph = {T: phi(W, m, minute, T) for T in (100, 200)}
        ph3 = {T: phi(W, m, minute + AHEAD, T) for T in (100, 200)}
        # 다음 1분 안에 시작하는 한타
        upcoming = [f for f in F if minute * 60000 <= f[0] < (minute + 1) * 60000]
        for pid, p in parts.items():
            T, role = p[1], p[2]
            x, y = fr[str(pid)][0], fr[str(pid)][1]
            pl = place(x, y, T, nodes)
            mates = [q for q in parts if parts[q][1] == T and q != pid]
            near = sum(1 for q in mates if math.dist((x, y), fr[str(q)][:2]) <= 2500)
            r = {"match": m["match"], "pid": pid, "team": T, "role": role, "minute": minute, "died1": (minute, pid) in deaths, "state": state[pid], "kind": kind[pid][0],
                 "opp_kind": kind[pid][1], "place": pl, "depth": depth(x, y, T, nodes) if pl in ("side", "mid") else None,
                 "lane": lane_of(x, y) if pl in ("side", "mid") else None, "grouped": near >= 2,
                 "hp": fr[str(pid)][2] / max(1, fr[str(pid)][3]),
                 "cs1": m["frames"][minute + 1][str(pid)][9] - fr[str(pid)][9],
                 "jcs1": m["frames"][minute + 1][str(pid)][8] - fr[str(pid)][8],
                 "phi": ph[T], "dphi3": ph3[T] - ph[T], "win": win[T]}
            if role == "JUNGLE":
                adc, sup, top = by_role[(T, "BOTTOM")], by_role[(T, "UTILITY")], by_role[(T, "TOP")]
                r["near_bot"] = min(math.dist((x, y), fr[str(adc)][:2]), math.dist((x, y), fr[str(sup)][:2])) <= 3000
                tx, ty = fr[str(top)][:2]
                r["with_side_top"] = pl == "side" and place(tx, ty, T, nodes) == "side" and math.dist((x, y), (tx, ty)) <= 3000
                r["top_state"] = state[top]
                r["quad"] = ("top" if y - x > 0 else "bot") if pl in ("own_jungle", "enemy_jungle") else None
            out.append(r)
    # 한타 참여
    frows = []
    for t0, fx, fy, ppl, nk in F:
        minute = int(t0 // 60000)
        if not (M0 <= minute <= M1) or minute >= len(m["frames"]):
            continue
        fr = m["frames"][minute]
        obj = any(abs(e[0] - t0) <= 60000 and e[1] in ("DRAGON", "BARON_NASHOR", "RIFTHERALD", "ATAKHAN") for e in m["events"]["elite"])
        sides = {T: [q for q in ppl if parts.get(q, [0, 0])[1] == T] for T in (100, 200)}
        for pid, p in parts.items():
            x, y = fr[str(pid)][0], fr[str(pid)][1]
            frows.append({"match": m["match"], "team": p[1], "role": p[2], "minute": minute, "state": state[pid], "kind": kind[pid][0],
                          "joined": pid in ppl, "dist": math.dist((x, y), (fx, fy)), "place": place(x, y, p[1], nodes),
                          "hp": fr[str(pid)][2] / max(1, fr[str(pid)][3]), "objective": obj, "size": nk,
                          "won_fight": len(sides[p[1]]) > 0 and sum(1 for k in m["events"]["kill"] if t0 <= k[0] <= t0 + 20000 and parts.get(k[2], [0, 0])[1] != p[1]) >
                          sum(1 for k in m["events"]["kill"] if t0 <= k[0] <= t0 + 20000 and parts.get(k[2], [0, 0])[1] == p[1])})
    return out, frows


# ---------------------------------------------------------------------- 통계
def ols_diff(rows, flag):
    """dphi3 ~ 1 + logit(phi) + minute + flag → (계수, 95% 구간), 경기 단위 부트스트랩 300회."""
    def est(rs):
        X = np.array([[1.0, math.log(max(1e-4, r["phi"]) / max(1e-4, 1 - r["phi"])), r["minute"], float(flag(r))] for r in rs])
        Y = np.array([r["dphi3"] for r in rs])
        if X[:, 3].sum() < 10 or X[:, 3].sum() > len(rs) - 10:
            return None
        return float(np.linalg.lstsq(X, Y, rcond=None)[0][3])
    b = est(rows)
    if b is None:
        return None, None, None
    by = collections.defaultdict(list)
    for r in rows:
        by[r["match"]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(300):
        rs = [r for i in rng.integers(0, len(keys), len(keys)) for r in by[keys[i]]]
        v = est(rs)
        if v is not None:
            bs.append(v)
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def share(rows, f):
    return float(np.mean([f(r) for r in rows])) if rows else float("nan")


def pct(v):
    return "-" if v is None or v != v else f"{100 * v:.0f}%"


def pp(t):
    b, lo, hi = t
    if b is None:
        return "-"
    star = " ✓" if lo > 0 else " ✗" if hi < 0 else ""
    return f"{100 * b:+.1f} ({100 * lo:+.1f}~{100 * hi:+.1f}){star}"


BEHAVIORS = {
    "TOP": [("사이드 라인에 있음", lambda r: r["place"] == "side"),
            ("적 절반 이상 깊이 사이드", lambda r: r["place"] == "side" and r["depth"] in ("enemy_half", "enemy_deep")),
            ("우리 절반에서 사이드(받아먹기)", lambda r: r["place"] == "side" and r["depth"] in ("own_deep", "own_half")),
            ("미드·본대와 뭉침", lambda r: r["grouped"]),
            ("적 정글", lambda r: r["place"] == "enemy_jungle")],
    "MIDDLE": [("사이드 라인에 있음", lambda r: r["place"] == "side"),
               ("미드 라인", lambda r: r["place"] == "mid"),
               ("뭉침", lambda r: r["grouped"]),
               ("적 정글", lambda r: r["place"] == "enemy_jungle")],
    "BOTTOM": [("사이드 라인에 있음", lambda r: r["place"] == "side"),
               ("미드 라인", lambda r: r["place"] == "mid"),
               ("뭉침", lambda r: r["grouped"]),
               ("적 절반 이상 깊이 라인", lambda r: r["place"] in ("side", "mid") and r["depth"] in ("enemy_half", "enemy_deep"))],
    "UTILITY": [("원딜·본대와 뭉침", lambda r: r["grouped"]),
                ("적 정글(시야)", lambda r: r["place"] == "enemy_jungle"),
                ("강가", lambda r: r["place"] == "river"),
                ("사이드 라인", lambda r: r["place"] == "side")],
    "JUNGLE": [("원딜·서폿 3,000 안", lambda r: r["near_bot"]),
               ("사이드 탑과 함께", lambda r: r["with_side_top"]),
               ("우리 정글", lambda r: r["place"] == "own_jungle"),
               ("적 정글", lambda r: r["place"] == "enemy_jungle"),
               ("뭉침", lambda r: r["grouped"])],
}


def join_model(frows, role):
    """한타 참여 ~ 거리(천) + 사이드 라인 + 체력 + 라인전 결과 + 오브젝트 한타 (로지스틱 계수, 표준화 안 함)."""
    rs = [r for r in frows if r["role"] == role]
    X = np.array([[r["dist"] / 1000, float(r["place"] == "side"), r["hp"], float(r["state"] == "won"), float(r["state"] == "lost"),
                   float(r["objective"])] for r in rs])
    y = np.array([float(r["joined"]) for r in rs])
    w = logit_fit(X, y)
    return len(rs), dict(zip(["거리 천당", "사이드 라인", "체력 비율", "라인전 이김", "라인전 짐", "오브젝트 한타"], w[1:]))


def obj_timer(m, minute):
    """minute 시작 기준 다음 드래곤·바론 생성까지 초(이미 살아 있으면 0). 처치 기록으로 근사."""
    t = minute * 60
    out = {}
    for name, first, resp in (("DRAGON", 300, 300), ("BARON_NASHOR", 1200, 360)):
        spawn = first
        for e in sorted(x for x in m["events"]["elite"] if x[1] == name):
            if e[0] / 1000 > t:
                break
            spawn = e[0] / 1000 + resp
        out[name] = max(0.0, spawn - t)
    return out


def next_obj(r):
    return min(r["drag_s"], r["baron_s"] if r["minute"] >= 19 else 1e9)


def guide_checks(R):
    """가이드 주장을 같은 분 표본으로 대조."""
    top = [r for r in R if r["role"] == "TOP"]
    mid = [r for r in R if r["role"] == "MIDDLE"]
    jg = [r for r in R if r["role"] == "JUNGLE"]
    side = lambda r: r["place"] == "side"
    deep = lambda r: r["place"] == "side" and r["depth"] in ("enemy_half", "enemy_deep")
    L = ["## 가이드 주장 대조", "", "| 주장(출처) | 데이터 | 판정 |", "|---|---|---|"]
    for st in ("won", "lost"):
        rs = [r for r in top if r["state"] == st]
        a = [r for r in rs if 14 <= r["minute"] < 20]
        b = [r for r in rs if 20 <= r["minute"] < 26]
        c = [r for r in rs if 26 <= r["minute"] <= 30]
        L.append(f"| 탑 {STATE_KO[st]}: 15~20분 스플릿, 20~25분 바론 나오면 뭉치고, 25분 넘어 다시 스플릿(eloascend) | "
                 f"사이드 비율 14~19분 {pct(share(a, side))}·20~25분 {pct(share(b, side))}·26~30분 {pct(share(c, side))}, "
                 f"뭉침 {pct(share(a, lambda r: r['grouped']))}·{pct(share(b, lambda r: r['grouped']))}·{pct(share(c, lambda r: r['grouped']))} | 아래 해석 |")
    for st in ("won", "lost"):
        rs = [r for r in top if r["state"] == st]
        L.append(f"| 탑 {STATE_KO[st]}: 드래곤·바론 곧 나올 땐 스플릿 말고 합류(eloascend) | 다음 오브젝트 1분 안 사이드 "
                 f"{pct(share([r for r in rs if next_obj(r) < 60], side))} vs 2분 넘게 {pct(share([r for r in rs if next_obj(r) >= 120], side))} | 아래 해석 |")
    rs = [r for r in top]
    L.append(f"| 첫 포탑이 깨진 뒤부터 스플릿(dignitas) | 탑 깊은 사이드(적 절반 이상) 첫 포탑 전 {pct(share([r for r in rs if not r['tower_down']], deep))} "
             f"vs 후 {pct(share([r for r in rs if r['tower_down']], deep))} (14분 이후 분만) | 아래 해석 |")
    L.append(f"| 밀리면 포탑 근처에서 받아먹기(eloascend) | 탑 라인에 있을 때 우리 쪽 절반 비율: 이김 "
             f"{pct(share([r for r in top if r['state'] == 'won' and r['depth']], lambda r: r['depth'] in ('own_deep', 'own_half')))} vs 짐 "
             f"{pct(share([r for r in top if r['state'] == 'lost' and r['depth']], lambda r: r['depth'] in ('own_deep', 'own_half')))} | 맞음 |")
    L.append(f"| 미드: 다음 오브젝트가 곧 나오면 사이드 말고 먼저 자리(boostroom) | 미드 사이드: 오브젝트 살아 있음 {pct(share([r for r in mid if next_obj(r) == 0], side))}, "
             f"1분 안 {pct(share([r for r in mid if 0 < next_obj(r) < 60], side))}, 2분 넘게 {pct(share([r for r in mid if next_obj(r) >= 120], side))} | 아래 해석 |")
    L.append(f"| 미드: 밀리면 미드에서 웨이브 정리·안전하게(boostroom) | 미드 라인 비율 이김 {pct(share([r for r in mid if r['state'] == 'won'], lambda r: r['place'] == 'mid'))} "
             f"vs 짐 {pct(share([r for r in mid if r['state'] == 'lost'], lambda r: r['place'] == 'mid'))}, 사이드 {pct(share([r for r in mid if r['state'] == 'won'], side))} vs "
             f"{pct(share([r for r in mid if r['state'] == 'lost'], side))} | 아래 해석 |")
    L.append(f"| 정글: 드래곤 나올 때 풀캠 돌지 말고 오브젝트로(buildzcrank) | 정글 우리 정글 비율: 드래곤 1분 안 {pct(share([r for r in jg if 0 < r['drag_s'] < 60], lambda r: r['place'] == 'own_jungle'))} "
             f"vs 2분 넘게 {pct(share([r for r in jg if r['drag_s'] >= 120], lambda r: r['place'] == 'own_jungle'))}; 원딜·서폿 근처 {pct(share([r for r in jg if 0 < r['drag_s'] < 60], lambda r: r['near_bot']))} vs "
             f"{pct(share([r for r in jg if r['drag_s'] >= 120], lambda r: r['near_bot']))} | 아래 해석 |")
    return L + [""]


def main():
    M = load()
    nodes = Game(seed=0).map.nodes
    champs = load_json("rules", "champions.json")["champions"]
    W = fit_wp(M)
    R, FR = [], []
    timers = {}
    for m in M:
        a, b = rows_for(m, W, nodes, champs)
        R += a
        FR += b
        for minute in range(M0, min(M1, len(m["frames"]) - 1) + 1):
            timers[(m["match"], minute)] = obj_timer(m, minute)
    first_tower = {m["match"]: min([e[0] for e in m["events"]["tower"] if e[3] == "TOWER_BUILDING"], default=1e12) / 60000 for m in M}
    for r in R:
        t = timers[(r["match"], r["minute"])]
        r["drag_s"], r["baron_s"] = t["DRAGON"], t["BARON_NASHOR"]
        r["tower_down"] = r["minute"] >= first_tower[r["match"]]
    L = ["# 운영 단계 포지션별 행동 × 라인전 결과 (천상계 1,000판, 14~30분 분 프레임)", "",
         f"- 정글러·라이너 분 {len(R)}개, 한타(15초 안 킬 3개 이상) 참여 판단 {len(FR)}개.",
         f"- 라인전 결과 = 14분 같은 포지션 상대와 총 골드 차: 이김 +{LEAD} 이상 / 짐 −{LEAD} 이하.",
         "- 효과 = 그 행동을 한 분의 3분 뒤 예측 승률 변화 − 안 한 분(지금 예측 승률·분 통제, 경기 단위 부트스트랩 95%). ✓ = 0보다 큼, ✗ = 0보다 작음.",
         "  선택 편향(이기고 있어서 그 행동을 할 수 있었던 것)이 섞인 연관이다.", ""]
    table = {}
    for role in ROLES:
        L += [f"## {ROLE_KO[role]}", ""]
        rr = [r for r in R if r["role"] == role]
        L += ["### 어디에 있나(분 비율)", "", "| 라인전 | 분 | 사이드 라인 | 미드 라인 | 우리 정글 | 적 정글 | 강가 | 기지 | 뭉침 | 라인 CS/분 | 정글 CS/분 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for st in ("won", "even", "lost"):
            rs = [r for r in rr if r["state"] == st]
            L.append(f"| {STATE_KO[st]} | {len(rs)} | {pct(share(rs, lambda r: r['place'] == 'side'))} | {pct(share(rs, lambda r: r['place'] == 'mid'))} | "
                     f"{pct(share(rs, lambda r: r['place'] == 'own_jungle'))} | {pct(share(rs, lambda r: r['place'] == 'enemy_jungle'))} | "
                     f"{pct(share(rs, lambda r: r['place'] == 'river'))} | {pct(share(rs, lambda r: r['place'] == 'base'))} | {pct(share(rs, lambda r: r['grouped']))} | "
                     f"{np.mean([r['cs1'] for r in rs]):.1f} | {np.mean([r['jcs1'] for r in rs]):.1f} |")
        if role != "JUNGLE":
            L += ["", "### 라인에 있을 때 깊이(어디까지 먹나)", "", "| 라인전 | 우리 2차 쪽 | 우리 절반 | 적 절반 | 적 2차 쪽 |", "|---|---|---|---|---|"]
            for st in ("won", "even", "lost"):
                rs = [r for r in rr if r["state"] == st and r["depth"]]
                L.append(f"| {STATE_KO[st]} | " + " | ".join(pct(share(rs, lambda r, d=d: r["depth"] == d)) for d in ("own_deep", "own_half", "enemy_half", "enemy_deep")) + " |")
        if role == "TOP":
            L += ["", "### 상성(내 챔피언 유형 vs 상대 유형)별 사이드 비율·깊이", "", "| 나 vs 상대 | 라인전 | 분 | 사이드 | 적 절반 이상 | 뭉침 |", "|---|---|---|---|---|---|"]
            for k in ("side", "tank", "ranged"):
                for ok in ("side", "tank", "ranged"):
                    for st in ("won", "lost"):
                        rs = [r for r in rr if r["kind"] == k and r["opp_kind"] == ok and r["state"] == st]
                        if len(rs) < 150:
                            continue
                        L.append(f"| {k} vs {ok} | {STATE_KO[st]} | {len(rs)} | {pct(share(rs, lambda r: r['place'] == 'side'))} | "
                                 f"{pct(share(rs, lambda r: r['place'] == 'side' and r['depth'] in ('enemy_half', 'enemy_deep')))} | {pct(share(rs, lambda r: r['grouped']))} |")
        if role == "MIDDLE":
            L += ["", "### 언제 사이드로 가나: 다음 드래곤·바론 생성까지 남은 시간별 미드 사이드 비율", "",
                  "| 다음 오브젝트까지 | 분 | 사이드 라인 | 미드 라인 | 뭉침 |", "|---|---|---|---|---|"]
            for lo, hi, lab in ((0, 1, "살아 있음"), (1, 60, "1분 안"), (60, 120, "1~2분"), (120, 1e9, "2분 넘게")):
                rs = [r for r in rr if lo <= min(r["drag_s"], r["baron_s"] if r["minute"] >= 19 else 1e9) < hi]
                L.append(f"| {lab} | {len(rs)} | {pct(share(rs, lambda r: r['place'] == 'side'))} | {pct(share(rs, lambda r: r['place'] == 'mid'))} | {pct(share(rs, lambda r: r['grouped']))} |")
        if role == "JUNGLE":
            L += ["", "### 정글러 위치", "", "| 라인전(정글 골드 차) | 원딜·서폿 근처 | 사이드 탑과 함께 | 우리 정글 위쪽/아래쪽 | 적 정글 위쪽/아래쪽 |", "|---|---|---|---|---|"]
            for st in ("won", "even", "lost"):
                rs = [r for r in rr if r["state"] == st]
                L.append(f"| {STATE_KO[st]} | {pct(share(rs, lambda r: r['near_bot']))} | {pct(share(rs, lambda r: r['with_side_top']))} | "
                         f"{pct(share(rs, lambda r: r['place'] == 'own_jungle' and r['quad'] == 'top'))}/{pct(share(rs, lambda r: r['place'] == 'own_jungle' and r['quad'] == 'bot'))} | "
                         f"{pct(share(rs, lambda r: r['place'] == 'enemy_jungle' and r['quad'] == 'top'))}/{pct(share(rs, lambda r: r['place'] == 'enemy_jungle' and r['quad'] == 'bot'))} |")
            L += ["", "사이드 탑과 함께 있는 분 비율 — 탑 라인전 결과 × 다음 오브젝트:", "", "| 탑 라인전 | 바론 2분 안 | 드래곤 2분 안 | 둘 다 2분 넘게 |", "|---|---|---|---|"]
            for st in ("won", "even", "lost"):
                rs = [r for r in rr if r["top_state"] == st]
                a = [r for r in rs if r["minute"] >= 19 and r["baron_s"] < 120]
                b = [r for r in rs if r["drag_s"] < 120 and not (r["minute"] >= 19 and r["baron_s"] < 120)]
                c = [r for r in rs if r["drag_s"] >= 120 and not (r["minute"] >= 19 and r["baron_s"] < 120)]
                L.append(f"| {STATE_KO[st]} | {pct(share(a, lambda r: r['with_side_top']))} ({len(a)}) | {pct(share(b, lambda r: r['with_side_top']))} ({len(b)}) | "
                         f"{pct(share(c, lambda r: r['with_side_top']))} ({len(c)}) |")
        L += ["", "### 한타 참여", ""]
        fr_ = [r for r in FR if r["role"] == role]
        L += ["| 라인전 | 한타 | 참여 | 사이드 라인에서 참여 | 오브젝트 한타 참여 | 참여 안 했을 때 거리 중앙 |", "|---|---|---|---|---|---|"]
        for st in ("won", "even", "lost"):
            rs = [r for r in fr_ if r["state"] == st]
            side = [r for r in rs if r["place"] == "side"]
            obj = [r for r in rs if r["objective"]]
            nj = [r["dist"] for r in rs if not r["joined"]]
            L.append(f"| {STATE_KO[st]} | {len(rs)} | {pct(share(rs, lambda r: r['joined']))} | {pct(share(side, lambda r: r['joined']))} ({len(side)}) | "
                     f"{pct(share(obj, lambda r: r['joined']))} | {np.median(nj) if nj else float('nan'):.0f} |")
        n, co = join_model(FR, role)
        L += ["", f"참여 로지스틱({n}개, 한타 시작 분 프레임 기준): " + ", ".join(f"{k} {v:+.2f}" for k, v in co.items()) + ".", ""]
        L += ["### 잘한 행동(3분 뒤 예측 승률 변화, %p)", "", "| 행동 | " + " | ".join(STATE_KO[s] for s in ("won", "even", "lost")) + " |", "|---|---|---|---|"]
        table[role] = {}
        for name, f in BEHAVIORS[role]:
            cells = []
            table[role][name] = {}
            for st in ("won", "even", "lost"):
                rs = [r for r in rr if r["state"] == st]
                t = ols_diff(rs, f)
                cells.append(f"{pp(t)} · {pct(share(rs, f))}")
                table[role][name][st] = {"share": share(rs, f), "dphi3": t[0], "ci": [t[1], t[2]], "n": len(rs)}
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", "칸 = 효과(95% 구간) · 그 행동 분 비율.", ""]
    L += guide_checks(R)
    out = {"format": "macro_roles/1", "minutes": [M0, M1], "lead_gold": LEAD, "ahead_min": AHEAD,
           "note": "role × 라인전 결과(won/even/lost) × 행동: 천상계 분 비율(share)과 3분 뒤 예측 승률 변화 차이(dphi3, ci). 보상 표 재료. 선택 편향 포함.",
           "table": table}
    with open(os.path.join(ROOT, "sim", "macro_roles.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "reports", "macro_roles.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
