"""경기 리뷰 묶음(sim_review/1), 상황 칸 단위 근거판. LLM 코치(D:\\AI\\lol\\coach_sim.py) 입력.

    python tools\\my_compact.py D:\\AI\\lol\\match_cache APDO   :: → data/my_compact.json.gz
    python -m sim.review_cells                                 :: → data/reviews/<경기>.json

한 판·한 분마다 반사실을 돌리는 방식(sim/review.py)은 잡음이 커서 쓰지 않는다(reports/review_noise.md).
대신 내 경기의 각 분(3~17분)을 상황 칸(stage6.situation의 시기·드래곤·팀 골드·정글러 상태)에 넣고,
조언 검증 표(sim/advice_cells.json, sim/advice_cells.py)에서 그 칸의 결과를 근거로 붙인다.

결정을 만드는 규칙:
- 그 칸에서 verified(시뮬 구간과 실제 1,000판 3분 뒤 구간이 둘 다 0 밖, 부호 같음)이고 시뮬 차이가 +인 범주가 '검증된 좋은 선택'.
  그중 sim_vs_mix가 가장 큰 범주를 추천으로 둔다.
- 검증된 좋은 선택이 없는 칸에서는 천상계 데이터 사실만 붙인다(fact_decision, basis=pro_share, verified=false):
  내 선택이 그 칸 천상계에서 15% 미만으로 드물고 천상계 최다 선택이 30% 이상일 때만. 코치는 '참고'로만 쓴다.
  그 밖의 분은 넣지 않고 skipped에 남긴다.
- winprob 값은 절대 승률이 아니라 '천상계 선택 비율로 섞은 평균 대비 20분 예측 승률 차이'(sim_vs_mix)다.
  diff = 추천 − 실제, ci = 추천 범주의 sim_vs_mix 구간(실제 범주와의 차이 구간이 아님).
- objective는 실제 라벨이 처치 결과로 붙어 효과가 부풀려지므로 지금은 추천에서 뺀다(HOLD).
- 실제 범주가 contradicted(시뮬과 실제가 반대 방향)면 그 분은 넣지 않는다.
"""
import argparse
import json
import os

from .data import ROOT
from .game import Game
from .stage6 import CAT_KO, CATS, RealMatch, label, load_compact, situation

MINUTES = range(3, 18)
CELLS = os.path.join(ROOT, "sim", "advice_cells.json")
# 실제 쪽 라벨이 결과를 보고 붙는 범주(stage6.label: 정글러가 에픽 처치에 관여해야 objective)는 실제 효과가
# 처치 효과를 그대로 담아 부풀려진다. 표에 real_label(결과 없이 붙인 라벨)이 없으면 추천 근거로 쓰지 않는다.
HOLD = {"objective"}
# 천상계 사실만 붙이는 분의 조건: 칸 표본, 내 선택이 드문 정도, 천상계 최다 선택의 비율
FACT_MIN_N, FACT_RARE, FACT_MODAL = 60, 0.15, 0.30


def _pct(x):
    return f"{100 * x:.0f}%"


def _pp(x):
    return f"{100 * x:+.1f}%p"


def load_cells(path=CELLS):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    if t.get("format") != "advice_cells/1":
        raise ValueError(f"{path}: format이 advice_cells/1이 아님")
    return t


def cell_key(sit, table):
    return table.get("key_sep", "|").join(sit[k] for k in table["cell_keys"])


def good_cats(cell, hold=HOLD):
    """검증된 좋은 선택: verified이고 시뮬 차이가 +인 범주, 큰 순."""
    cats = cell["cats"]
    ok = [c for c in CATS if c in cats and c not in hold and cats[c].get("verified") and cats[c]["sim_vs_mix"] > 0]
    return sorted(ok, key=lambda c: -cats[c]["sim_vs_mix"])


def decision(rm, team, minute, cell, key, nid, nodes, outcome_free=False, hold=()):
    w = rm.window(team, minute, nodes)
    if outcome_free:
        # 조언 검증 표와 같은 기준: 처치·교전 결과 없이 1분 끝 위치와 귀환만으로 범주를 붙인다
        real = label(dict(w, elite=0, fights=[]), team, nodes)
    else:
        real = label(w, team, nodes)
    cats = cell["cats"]
    good = good_cats(cell, hold)
    if cats.get(real, {}).get("contradicted"):
        return None
    if not good:
        return fact_decision(rm, team, minute, cell, key, nid, w, real)
    rec = real if real in good else good[0]
    cr, ca = cats[rec], cats.get(real, {})
    a_val = ca.get("sim_vs_mix", 0.0)
    j = rm.jungler(team)
    a, b = rm.pf(minute, j), rm.pf(minute + 1, j)
    sit = situation(rm, team, minute)
    did = f"정글 CS +{b['jcs'] - a['jcs']}, 골드 +{b['total_gold'] - a['total_gold']}"
    did += ", 사망" if w["died"] else (f", 킬 관여 {w['kills_for']}" if w["kills_for"] else "")
    d = {
        "id": f"D{nid[0]}", "t": f"{minute}:00",
        "state": f"{key.replace('|', ', ')}, 레벨 {a['level']}, 체력 {_pct(a['hp'])}",
        "actual": {"action": real, "label": CAT_KO[real]},
        "recommended": {"action": rec, "label": CAT_KO[rec]},
        "winprob": {"actual": round(a_val, 4), "recommended": round(cr["sim_vs_mix"], 4),
                    "diff": round(cr["sim_vs_mix"] - a_val, 4), "ci": [round(x, 4) for x in cr["sim_vs_mix_ci"]]},
        "cell": key,
        "evidence": [],
        "verified": True,
    }
    nid[0] += 1

    def add(text, value, source, kind):
        d["evidence"].append({"id": f"S{nid[1]}", "kind": kind, "text": text, "value": value, "source": source})
        nid[1] += 1

    add("그 1분 실제 기록", did, "Riot 타임라인", "record")
    n = cell.get("n_real", cr.get("real_n"))
    add(f"이 상황 천상계가 '{CAT_KO[rec]}'을(를) 고른 비율", _pct(cr["real_share"]), f"1,000판 {n}상황", "share")
    lo, hi = cr["real_dwp3_ci"]
    add(f"'{CAT_KO[rec]}'을(를) 고른 천상계의 3분 뒤 예측 승률 차이(95% 구간 {_pp(lo)}~{_pp(hi)})",
        _pp(cr["real_dwp3_diff"]), f"1,000판, 고른 {cr.get('real_n', '?')}상황", "real_effect")
    add(f"시뮬에서 '{CAT_KO[rec]}'의 20분 승률, 천상계 평균 선택 대비", _pp(cr["sim_vs_mix"]),
        f"시뮬 {cell.get('n_sim_states', '?')}상태 × {cell.get('reps', '?')}회", "sim_effect")
    if rec != real and ca:
        add(f"이 상황 천상계가 실제 선택 '{CAT_KO[real]}'을(를) 고른 비율", _pct(ca["real_share"]), f"1,000판 {n}상황", "share")
    return d


def _record(rm, team, minute, w):
    j = rm.jungler(team)
    a, b = rm.pf(minute, j), rm.pf(minute + 1, j)
    did = f"정글 CS +{b['jcs'] - a['jcs']}, 골드 +{b['total_gold'] - a['total_gold']}"
    did += ", 사망" if w["died"] else (f", 킬 관여 {w['kills_for']}" if w["kills_for"] else "")
    return a, did


def fact_decision(rm, team, minute, cell, key, nid, w, real):
    """시뮬로 검증된 추천이 없는 칸: 천상계 데이터 사실만 붙인다(verified=false, 코치는 '참고'로만 씀).
    내 선택이 그 칸 천상계에서 드물고(FACT_RARE 미만) 천상계가 뚜렷하게 많이 고른 범주(FACT_MODAL 이상)가 있을 때만."""
    cats = cell["cats"]
    if cell.get("n_real", 0) < FACT_MIN_N:
        return None
    modal = max(CATS, key=lambda c: cats.get(c, {}).get("real_share", 0.0))
    cm, ca = cats[modal], cats.get(real, {})
    if modal == real or ca.get("real_share", 0.0) >= FACT_RARE or cm["real_share"] < FACT_MODAL:
        return None
    if cm.get("real_dwp3_diff") is None:
        return None
    a, did = _record(rm, team, minute, w)
    a_eff = ca.get("real_dwp3_diff") or 0.0
    d = {
        "id": f"D{nid[0]}", "t": f"{minute}:00",
        "state": f"{key.replace('|', ', ')}, 레벨 {a['level']}, 체력 {_pct(a['hp'])}",
        "actual": {"action": real, "label": CAT_KO[real]},
        "recommended": {"action": modal, "label": CAT_KO[modal]},
        "basis": "pro_share",
        "winprob": {"actual": round(a_eff, 4), "recommended": round(cm["real_dwp3_diff"], 4),
                    "diff": round(cm["real_dwp3_diff"] - a_eff, 4), "ci": [round(x, 4) for x in cm["real_dwp3_ci"]]},
        "cell": key, "evidence": [], "verified": False,
    }
    nid[0] += 1

    def add(text, value, source, kind):
        d["evidence"].append({"id": f"S{nid[1]}", "kind": kind, "text": text, "value": value, "source": source})
        nid[1] += 1

    n = cell["n_real"]
    add("그 1분 실제 기록", did, "Riot 타임라인", "record")
    add(f"이 상황 천상계가 '{CAT_KO[modal]}'을(를) 고른 비율", _pct(cm["real_share"]), f"1,000판 {n}상황", "share")
    add(f"이 상황 천상계가 '{CAT_KO[real]}'을(를) 고른 비율", _pct(ca.get("real_share", 0.0)), f"1,000판 {n}상황", "share")
    lo, hi = cm["real_dwp3_ci"]
    add(f"'{CAT_KO[modal]}'을(를) 고른 천상계의 3분 뒤 예측 승률 차이(95% 구간 {_pp(lo)}~{_pp(hi)}, 고른 상황이 원래 달랐을 수 있어 인과 아님)",
        _pp(cm["real_dwp3_diff"]), f"1,000판, 고른 {cm.get('real_n', '?')}상황", "real_effect")
    return d


def review_match(m, table, nodes):
    rm = RealMatch(m)
    team = rm.team_of[m["me"]]
    mins = [mi for mi in MINUTES if len(rm.frames) > mi + 1 and m["duration"] > (mi + 1) * 60]
    nid, decisions, skipped = [1, 1], [], []
    # 표에 real_label이 있으면 실제 범주가 결과 없이 붙은 표다. 그때만 objective 보류를 푼다.
    outcome_free = "real_label" in table
    hold = () if outcome_free else HOLD
    for mi in mins:
        key = cell_key(situation(rm, team, mi), table)
        cell = table["cells"].get(key)
        d = decision(rm, team, mi, cell, key, nid, nodes, outcome_free, hold) if cell else None
        if d is None:
            skipped.append(f"{mi}:00")
        else:
            decisions.append(d)
    me = m["me"]
    return {
        "format": "sim_review/1",
        "match_id": m["match"], "participant_id": me, "champion": rm.champ_of[me], "role": rm.role_of[me],
        "patch": m["version"][:5], "win": bool(m["win100"] if team == 100 else 1 - m["win100"]),
        "winprob_kind": "vs_mix",
        "source": {"cells": "sim/advice_cells.json", "policy": table.get("policy")},
        "skipped": skipped,
        "note": "winprob은 천상계 평균 선택 대비 20분 예측 승률 차이. 검증된 근거가 없는 분은 skipped.",
        "decisions": decisions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "my_compact.json.gz"))
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--match", action="append")
    args = ap.parse_args()
    table = load_cells(args.cells)
    nodes = Game(seed=0).map.nodes
    dst_dir = os.path.join(ROOT, "data", "reviews")
    os.makedirs(dst_dir, exist_ok=True)
    total = kept = 0
    for m in load_compact(args.src):
        if args.match and m["match"] not in args.match:
            continue
        if not RealMatch(m).valid:
            print(f"{m['match']}: 포지션이 다 안 채워져 건너뜀")
            continue
        r = review_match(m, table, nodes)
        with open(os.path.join(dst_dir, f"{m['match']}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        n_bad = sum(1 for d in r["decisions"] if d["actual"]["action"] != d["recommended"]["action"])
        total += len(r["decisions"]) + len(r["skipped"])
        kept += len(r["decisions"])
        print(f"{m['match']}: 근거 있는 분 {len(r['decisions'])} (추천과 다름 {n_bad}), 근거 없음 {len(r['skipped'])}")
    print(f"전체 {total}분 중 근거 있는 분 {kept}")


if __name__ == "__main__":
    main()
