"""레딧 운영 조언을 천상계 1,000판(14~30분 분 프레임)으로 대조.

    python -m sim.reddit_check      :: → reports/reddit_check.md

표본은 sim/macro_roles.py와 같다(rows_for, 라인전 결과 = 14분 같은 포지션 골드 차 ±1,000).
글쓴이 티어는 글에 적힌 것만 옮긴다. 레딧 원문은 클라우드에서 직접 못 열어 검색 결과의 레딧 미러 페이지로 읽었다.
"""
import os

import numpy as np

from .data import ROOT, load_json
from .game import Game
from .macro_roles import M0, M1, fit_wp, load, obj_timer, pct, rows_for, share

SRC = {
    "kled": ("r/KledMains 'Tips on playing from behind/even'(2025-03, 글쓴이 브론즈, 댓글 티어 없음)",
             "https://lr.psf.lt/r/KledMains/comments/1j2up5q/tips_on_playing_from_behindeven"),
    "wiki": ("r/summonerschool 위키 탑 101(/u/GarciLP, 티어 없음)", "https://r.datuan.dev/r/summonerschool/wiki/101/top"),
    "teamcomp": ("r/summonerschool 'A New Guide to Understanding Team-Comps' 댓글(2016-11, 티어 없음)", "https://r.datuan.dev/5bmaug"),
    "cs": ("r/summonerschool 'Maintaining CS'(2024-10, 댓글 uuam·Warwicks_Paws_owo, 티어 없음)",
           "https://reddit.birdcat.cafe/r/summonerschool/comments/1gdas6d/maintaining_cs"),
    "csing": ("r/summonerschool 'Help with CSing' 댓글 KiaraKawaii(2025-01, 티어 없음)",
              "https://pol1.lr.ggtyler.dev/r/summonerschool/comments/1ib1sur/help_with_csing/m9k971a/?context=3"),
    "adc": ("r/summonerschool 'What should an ADC do'(2024-07, 티어 없음)", "https://lr.us.psf.lt/r/summonerschool/comments/1duknnt/what_should_an_adc_do"),
    "midmacro": ("r/summonerschool 'Mid macro maybe?' 댓글 killerchand(2024-07, 에메랄드 2)",
                 "https://lr.us.psf.lt/r/summonerschool/comments/1dtj36w/mid_macro_maybe/lb9oqok/?context=3"),
    "overstay": ("r/summonerschool 'Tend to overstay a lot'(2025-02, 글쓴이·답글 다이아 2)",
                 "https://nyc1.lr.ggtyler.dev/r/summonerschool/comments/1ifl87z/tend_to_overstay_a_lot"),
    "camps": ("r/summonerschool 'Junglers, you have to clear camps now'(2019-11, 티어 없음)",
              "https://lr.ggtyler.dev/r/summonerschool/comments/dzdxzk/junglers_you_have_to_clear_camps_now/f87bnq6/?context=3"),
    "jgmains": ("r/Jungle_Mains 정글 기본 가이드(브라질 에메랄드 1)", "https://lr.in.psf.lt/r/Jungle_Mains"),
    "malz": ("r/MalzaharMains 'Any tips for mid game Malzahar'(2024-09, 티어 없음)",
             "https://lr.ggtyler.dev/r/MalzaharMains/comments/1fcsxxi/any_tips_for_mid_game_malzahar"),
}


def build():
    M = load()
    nodes = Game(seed=0).map.nodes
    champs = load_json("rules", "champions.json")["champions"]
    W = fit_wp(M)
    R = []
    for m in M:
        rows, _ = rows_for(m, W, nodes, champs)
        if not rows:
            continue
        tw = m["events"]["tower"]
        drags = sorted(e for e in m["events"]["elite"] if e[1] == "DRAGON")
        for r in rows:
            t = obj_timer(m, r["minute"])
            r["drag_s"], r["baron_s"] = t["DRAGON"], t["BARON_NASHOR"]
            ms = r["minute"] * 60000
            foe = 300 - r["team"]
            # 적 바텀 외곽 포탑이 깨졌나(건물 teamId = 주인 팀)
            r["foe_bot_outer_down"] = any(e[0] <= ms and e[1] == foe and e[2] == "BOT_LANE" and e[4] == "OUTER_TURRET" for e in tw)
            r["own_bot_outer_down"] = any(e[0] <= ms and e[1] == r["team"] and e[2] == "BOT_LANE" and e[4] == "OUTER_TURRET" for e in tw)
            r["foe_drag_recent"] = any(ms - 60000 <= e[0] < ms and e[3] == foe for e in drags)
            r["drag_no"] = sum(1 for e in drags if e[0] < ms) + 1
        R += rows
    return R


def ci(rows, f, n_boot=400, seed=0):
    """경기 단위 부트스트랩 95% 구간(비율)."""
    if not rows:
        return float("nan"), float("nan")
    by = {}
    for r in rows:
        by.setdefault(r["match"], []).append(float(f(r)))
    keys = list(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        s = [v for i in pick for v in by[keys[i]]]
        vals.append(np.mean(s))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pc(rows, f):
    lo, hi = ci(rows, f)
    return f"{pct(share(rows, f))} ({lo:.0%}~{hi:.0%}, {len(rows)}분)"


def main():
    R = build()
    by = lambda role, **kw: [r for r in R if r["role"] == role and all(r[k] == v for k, v in kw.items())]
    side = lambda r: r["place"] == "side"
    top_lane = lambda r: r["lane"] == "top"
    deep = lambda r: r["depth"] == "enemy_deep"
    own_half = lambda r: r["depth"] in ("own_deep", "own_half")
    died = lambda r: r["died1"]
    drag_next = lambda r: r["drag_s"] < 120 and not (r["minute"] >= 19 and r["baron_s"] < r["drag_s"])
    baron_next = lambda r: r["minute"] >= 19 and r["baron_s"] < 120 and r["baron_s"] <= r["drag_s"]
    far = lambda r: r["drag_s"] >= 120 and (r["minute"] < 19 or r["baron_s"] >= 120)
    L = ["# 레딧 운영 조언 × 천상계 1,000판", "",
         "- 표본: 천상계 1,000판 14~30분 선수-분(sim/macro_roles.py와 같음). 괄호 = 경기 단위 부트스트랩 95% 구간, 표본 분 수.",
         "- 레딧은 클라우드에서 직접 접속·검색이 막혀(reddit.com·pullpush·arctic-shift 403, 미러 검색 robots 금지) 웹 검색에 잡힌 레딧 미러 글을 읽었다.",
         "- '드래곤/바론 2분 안'은 이미 살아 있는 경우를 포함한다. 사망 = 그 분(프레임 다음 1분) 안에 죽음.",
         "- 읽은 글 중 천상계(마스터 이상) 작성자는 없었다. 적힌 티어는 브론즈·실버·골드·에메랄드·다이아뿐이고 대부분 티어가 없다.", ""]
    rows = []

    top_l, top_w = by("TOP", state="lost"), by("TOP", state="won")
    # 1 진 탑 스플릿·2차 포탑
    rows.append(("kled", "진 탑: 스플릿해서 2차 포탑까지 밀어라",
                 f"사이드에 있을 때 적 2차 쪽 깊이: 짐 {pc([r for r in top_l if side(r)], deep)} vs 이김 {pc([r for r in top_w if side(r)], deep)}",
                 "반대(진 탑은 깊게 안 민다)"))
    rows.append(("kled", "진 탑: 오브젝트 반대쪽 라인을 밀어라",
                 f"드래곤 2분 안, 사이드에 있을 때 탑 라인 비율: 짐 {pc([r for r in top_l if side(r) and drag_next(r)], top_lane)} · "
                 f"바론 2분 안 {pc([r for r in top_l if side(r) and baron_next(r)], top_lane)}", None))
    rows.append(("wiki", "밀리면 안전하게, 받아먹을 수 있는 것만",
                 f"사이드에 있을 때 우리 절반: 짐 {pc([r for r in top_l if side(r)], own_half)} vs 이김 {pc([r for r in top_w if side(r)], own_half)}", "맞음"))
    rows.append(("wiki", "앞섰지만 스플릿 챔피언이 아니면 다른 라인 포탑을 압박",
                 f"이긴 탑 미드 라인 비율: 탱커 {pc(by('TOP', state='won', kind='tank'), lambda r: r['place'] == 'mid')} vs 사이드형 {pc(by('TOP', state='won', kind='side'), lambda r: r['place'] == 'mid')}",
                 None))
    t_all = by("TOP")
    rows.append(("teamcomp", "시간이 갈수록 스플릿은 어려워지고 뭉치는 쪽이 강해진다",
                 f"탑 사이드 비율 14~19분 {pc([r for r in t_all if r['minute'] < 20], side)} · 26~30분 {pc([r for r in t_all if r['minute'] >= 26], side)}", "맞음"))
    # uuam: 바론은 꼭 뭉치고 첫 두 드래곤은 덜
    rows.append(("cs", "탑은 최대한 사이드에 남고, 바론엔 뭉치되 첫 두 드래곤은 덜 급하다",
                 f"탑 사이드 비율: 드래곤 2분 안(1~2번째) {pc([r for r in t_all if drag_next(r) and r['drag_no'] <= 2], side)} · "
                 f"(3번째 이후) {pc([r for r in t_all if drag_next(r) and r['drag_no'] >= 3], side)} · 바론 2분 안 {pc([r for r in t_all if baron_next(r)], side)} · "
                 f"둘 다 2분 넘게 {pc([r for r in t_all if far(r)], side)}", None))
    mid = by("MIDDLE")
    rows.append(("cs", "미드는 다음 오브젝트 쪽 사이드, 탑은 반대쪽",
                 f"사이드에 있을 때 탑 라인 비율 — 드래곤 2분 안: 미드 {pc([r for r in mid if side(r) and drag_next(r)], top_lane)}, 탑 {pc([r for r in t_all if side(r) and drag_next(r)], top_lane)}; "
                 f"바론 2분 안: 미드 {pc([r for r in mid if side(r) and baron_next(r)], top_lane)}, 탑 {pc([r for r in t_all if side(r) and baron_next(r)], top_lane)}", None))
    rows.append(("csing", "마법사는 사이드의 우리 쪽 절반(수확 구역)에서 먹는다",
                 f"미드가 사이드에 있을 때 우리 절반: {pc([r for r in mid if side(r)], own_half)} (탑 {pc([r for r in t_all if side(r)], own_half)})", None))
    adc = by("BOTTOM")
    rows.append(("midmacro", "원딜은 바텀 포탑을 깬 뒤 미드로 간다",
                 f"원딜 미드 라인 비율: 적 바텀 외곽 깨진 뒤 {pc([r for r in adc if r['foe_bot_outer_down']], lambda r: r['place'] == 'mid')} vs 전 {pc([r for r in adc if not r['foe_bot_outer_down']], lambda r: r['place'] == 'mid')}",
                 None))
    rows.append(("adc", "원딜은 운영 단계에 미드를 먹고 혼자 사이드에 있지 마라",
                 f"원딜 미드 {pc(adc, lambda r: r['place'] == 'mid')}, 사이드 {pc(adc, side)}", "맞음"))
    # 사이드 오래 있으면 죽는다
    for role, ko in (("TOP", "탑"), ("MIDDLE", "미드")):
        rs = by(role)
        rows.append(("overstay", f"{ko}: 사이드에 오래(깊게) 있으면 죽는다",
                     f"그 분 사망: 사이드 적 2차 쪽 {pc([r for r in rs if side(r) and deep(r)], died)} vs 사이드 우리 절반 {pc([r for r in rs if side(r) and own_half(r)], died)} vs 뭉침 {pc([r for r in rs if r['grouped']], died)}",
                     None))
    jg = by("JUNGLE")
    rows.append(("camps", "정글러는 캠프를 먹어야 한다(갱만 하면 뒤처짐)",
                 f"정글러 우리 정글 비율 {pc(jg, lambda r: r['place'] == 'own_jungle')}, 분당 정글 CS 5 이상 {pc(jg, lambda r: r['jcs1'] >= 5)}", None))
    rows.append(("jgmains", "바텀이 도움이 필요하면 거기 있어라(하체 중심)",
                 f"정글러 원딜·서폿 3,000 안 {pc(jg, lambda r: r.get('near_bot'))}; 우리 바텀 외곽 깨진 뒤 {pc([r for r in jg if r['own_bot_outer_down']], lambda r: r.get('near_bot'))}",
                 None))
    rows.append(("malz", "적이 오브젝트를 먹으면 반대쪽 사이드를 민다",
                 f"적 드래곤 직후 1분, 사이드에 있는 미드·탑의 탑 라인 비율 {pc([r for r in mid + t_all if side(r) and r['foe_drag_recent']], top_lane)} vs 평소 {pc([r for r in mid + t_all if side(r) and not r['foe_drag_recent']], top_lane)}",
                 None))

    V = {"진 탑: 오브젝트 반대쪽 라인을 밀어라": "맞음(드래곤 때 탑 라인, 바론 때 바텀 쪽)",
         "앞섰지만 스플릿 챔피언이 아니면 다른 라인 포탑을 압박": "맞음(약함, 탱커 표본 적음)",
         "탑은 최대한 사이드에 남고, 바론엔 뭉치되 첫 두 드래곤은 덜 급하다": "맞음",
         "미드는 다음 오브젝트 쪽 사이드, 탑은 반대쪽": "대체로 맞음(드래곤 때 뚜렷, 바론 때 미드는 약함)",
         "마법사는 사이드의 우리 쪽 절반(수확 구역)에서 먹는다": "부분(우리 절반이 많지만 탑과 차이 없음)",
         "원딜은 바텀 포탑을 깬 뒤 미드로 간다": "차이 없음(14분이면 대부분 이미 미드)",
         "탑: 사이드에 오래(깊게) 있으면 죽는다": "확인 안 됨(1분 프레임 위치로는 차이 없음)",
         "미드: 사이드에 오래(깊게) 있으면 죽는다": "확인 안 됨(1분 프레임 위치로는 차이 없음)",
         "정글러는 캠프를 먹어야 한다(갱만 하면 뒤처짐)": "판단 불가(설명 통계만)",
         "바텀이 도움이 필요하면 거기 있어라(하체 중심)": "맞음",
         "적이 오브젝트를 먹으면 반대쪽 사이드를 민다": "차이 없음"}
    rows = [(k, c, d, v or V.get(c)) for k, c, d, v in rows]
    L += ["| 레딧 주장 | 출처 | 1,000판 데이터 | 판정 |", "|---|---|---|---|"]
    for key, claim, data, verdict in rows:
        L.append(f"| {claim} | [{SRC[key][0]}]({SRC[key][1]}) | {data} | {verdict or '아래 해석'} |")
    out = os.path.join(ROOT, "reports", "reddit_check.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
