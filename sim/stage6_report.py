"""6단계 보고서: 상황별 차이 목록과 원인 분류.

입력: data/stage6_situations.json, data/stage6_cf.json(기본 가정), data/stage6_cf_<태그>.json(가정 바꾼 반사실).
출력: reports/stage6_compare.md, reports/stage6_compare.json

차이 하나 = 상황 칸(시기 × 다른 조건 하나)에서 학습 정글러가 실제보다 가장 많이 고른 범주 X와 가장 덜 고른 범주 Y.
- 시뮬 반사실: 같은 실제 상태에서 첫 1분을 X로 묶은 것과 Y로 묶은 것의 20분 예측 승률 차(짝지은 차이).
- 실제 경기 대조: 그 칸의 실제 정글러 중 X를 한 경우와 Y를 한 경우의 3분 뒤 예측 승률 변화 차.
  선택 편향(갱이 될 때 갱을 감)이 섞인 관찰 값이라 방향만 본다.
원인 분류
- 더 나은 전략 후보: 시뮬에서 X가 낫고, 실제 경기에서도 X가 Y보다 나쁘지 않음
- 조건 차이: 시뮬에서 X가 낫지만, 상대 행동 가정(규칙 봇)을 실제 쪽으로 바꾸면 우위가 절반 이하로 줄거나 뒤집힘
- 시뮬레이터 오류: 시뮬에서 X가 낫지만 실제 경기에서는 X가 나쁘고, 상대 가정을 바꿔도 우위가 남음
  → 해당 범주의 결과 모델(갱 성공률 등)이 실제와 다른지 함께 적는다
- 그 밖: 시뮬 안에서도 X와 Y가 차이 없음(판정 보류) / 시뮬에서도 Y가 나음(학습 부족)
"""
import collections
import json
import math
import os

import numpy as np

from .data import ROOT
from .stage6 import CATS, CAT_KO

DIMS = [("dragon", "드래곤"), ("top_obj", "윗쪽 오브젝트"), ("gold", "팀 골드"), ("need", "정글러 상태")]
PHASES = ["3-5분", "6-9분", "10-13분", "14-17분"]
MIN_N = 150
MIN_GAP = 0.10
# 가정 바꾼 반사실. kind: condition = 상대(규칙 봇) 행동 가정, mechanic = 시뮬레이터 결과 모델
SENS = {"contest": {"ko": "적 정글이 오브젝트를 자주 다툼", "kind": "condition",
                    "set": "jungler.objective_go_prob 0.25→0.8"},
        "gank": {"ko": "갱 성공 확률 낮춤", "kind": "mechanic", "set": "gank.coef.bias 0.4→−0.8"},
        "lanehp": {"ko": "라이너 체력을 실제 분포에 맞춤", "kind": "mechanic",
                   "set": "laner.trade_damage [0.03,0.18]→[0.01,0.08], laner.hp_regen_per_s 0.0015→0.003"},
        "camps": {"ko": "캠프 타이머 1분 이르게 추정", "kind": "condition", "set": "--camp-shift 60"},
        "items": {"ko": "아이템 골드가 갱·교전 승률에 반영", "kind": "mechanic",
                  "set": "gank.coef.item_gold_diff, skirmish.coef.item_gold_diff 0→0.35(천 골드당, 레벨 1 차이 정도로 가정)"}}


def load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def share(labels):
    c = collections.Counter(labels)
    n = max(1, sum(c.values()))
    return {k: c[k] / n for k in CATS}


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se)


def diff_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return (float("nan"),) * 3
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


def key(r):
    return (r["mi"], r["team"], r["minute"])


def cf_index(cf, rows):
    """반사실 결과를 상황 행과 잇고, 표본 가중치(칸 크기 ÷ 칸 표본 수)를 붙인다."""
    if cf is None:
        return {}
    by = {key(r): r for r in rows}
    cell_n = collections.Counter((r["phase"], r["dragon"], r["real"]) for r in rows)
    out = {}
    res = [x for x in cf["results"] if key(x) in by]
    cell_s = collections.Counter((by[key(x)]["phase"], by[key(x)]["dragon"], by[key(x)]["real"]) for x in res)
    for x in res:
        r = by[key(x)]
        c = (r["phase"], r["dragon"], r["real"])
        x["w"] = cell_n[c] / cell_s[c]
        out[key(x)] = x
    return out


def _wmean_ci(d, w):
    d, w = np.asarray(d, float), np.asarray(w, float) / np.sum(w)
    m = float((w * d).sum())
    neff = 1.0 / float((w ** 2).sum())
    sd = math.sqrt(float((w * (d - m) ** 2).sum()) * len(d) / max(1, len(d) - 1))
    se = sd / math.sqrt(neff)
    return {"n": len(d), "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se}


NAN = {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}


def cf_contrast(cfi, rows, X, Y):
    """칸 안 표본에서 Q(X) − Q(Y) (반복 평균끼리 짝지음), 가중 평균과 95% CI."""
    d, w = [], []
    for r in rows:
        x = cfi.get(key(r))
        if x is None:
            continue
        d.append(np.mean(x["q"][X]) - np.mean(x["q"][Y]))
        w.append(x["w"])
    return _wmean_ci(d, w) if len(d) >= 5 else dict(NAN, n=len(d))


def sens_shift(base, alt, rows, X, Y):
    """가정 바꾼 반사실과 기본 반사실을 같은 상황끼리 짝지어: 바꾼 쪽 Q(X)−Q(Y), 기본 쪽 같은 값, 그 차이."""
    a, b, w = [], [], []
    for r in rows:
        k = key(r)
        if k in base and k in alt:
            a.append(np.mean(alt[k]["q"][X]) - np.mean(alt[k]["q"][Y]))
            b.append(np.mean(base[k]["q"][X]) - np.mean(base[k]["q"][Y]))
            w.append(base[k]["w"])
    if len(a) < 5:
        return None
    return {"alt": _wmean_ci(a, w), "base": _wmean_ci(b, w), "shift": _wmean_ci(np.array(a) - np.array(b), w)}


def classify(sim, real, sens):
    if not (sim["n"] >= 5) or math.isnan(sim["mean"]):
        return "판정 보류(반사실 표본 부족)"
    if sim["lo"] <= 0 <= sim["hi"]:
        return "판정 보류(시뮬 안에서도 우열 없음)"
    if sim["hi"] < 0:
        return "학습 부족(시뮬에서도 실제 선택이 나음)"
    # 같은 상황끼리 비교해 우위가 절반 이하로 줄고 그 감소가 유의하면 그 가정 탓으로 본다
    hit = {t for t, v in sens.items() if v and v["base"]["mean"] > 0 and v["alt"]["mean"] <= v["base"]["mean"] / 2
           and v["shift"]["hi"] < 0}
    real_worse = not math.isnan(real[2]) and real[2] < 0
    if any(SENS[t]["kind"] == "mechanic" for t in hit):
        return "시뮬레이터 오류(" + ", ".join(SENS[t]["ko"] for t in hit if SENS[t]["kind"] == "mechanic") + ")"
    if hit:
        return "조건 차이(" + ", ".join(SENS[t]["ko"] for t in hit) + ")"
    if real_worse:
        return "시뮬레이터 오류 후보(실제 경기에서는 반대, 원인 모델 미확인)"
    return "더 나은 전략 후보"


def mechanism_checks(rows):
    """범주별 결과 모델 대조: 같은 범주로 붙은 1분에서 실제와 시뮬(학습 정글러)의 결과."""
    out = {}
    real = collections.defaultdict(list)
    ai = collections.defaultdict(list)
    for r in rows:
        real[r["real"]].append(r["real_gain"])
        for lab, g in zip(r["ai"], r["ai_gain"]):
            ai[lab].append(g)
    for c in CATS:
        out[c] = {"real_n": len(real[c]), "ai_n": len(ai[c])}
        for k in ("gold", "jcs", "xp"):
            out[c][f"real_{k}"] = float(np.mean([g[k] for g in real[c]])) if real[c] else None
            out[c][f"ai_{k}"] = float(np.mean([g[k] for g in ai[c]])) if ai[c] else None
    return out


def report(extra=()):
    rows = load("stage6_situations.json")
    cf = load("stage6_cf.json")
    cfi = cf_index(cf, rows)
    by_key = {key(r): r for r in rows}
    sens_cf = {t: cf_index(load(f"stage6_cf_{t}.json"), rows) for t in SENS}
    # 가정 바꾼 반사실은 같은 표본·같은 난수로 가정만 그대로 둔 반복(repeat)과 비교한다(기본 반사실과 비교하면
    # 기본 쪽 우연 오차가 모든 가정에 똑같이 실려 '다 줄어든 것'처럼 보임)
    ref = cf_index(load("stage6_cf_repeat.json"), rows) or cfi
    res_ref = "repeat" if load("stage6_cf_repeat.json") else "base"
    res = {"situations": len(rows), "matches": len({r["mi"] for r in rows}), "cf_samples": len(cfi)}
    # 전체·시기별 분포
    dist = {}
    for ph in ["전체"] + PHASES:
        sel = [r for r in rows if ph == "전체" or r["phase"] == ph]
        dist[ph] = {"n": len(sel), "real": share([r["real"] for r in sel]),
                    "ai": share([a for r in sel for a in r["ai"]]), "rule": share([r["rule"] for r in sel])}
    res["dist"] = dist
    res["mechanism"] = mechanism_checks(rows)
    res["outcomes"] = load("stage6_outcomes.json")
    # 1분 수입 대조(상태 옮기기 점검): 규칙 봇·학습 정글러 vs 실제
    gains = {}
    for ph in PHASES:
        sel = [r for r in rows if r["phase"] == ph]
        gains[ph] = {who: {k: float(np.mean([(g[k] if who != "ai" else np.mean([x[k] for x in g])) for g in
                                             [r[f"{who}_gain"] for r in sel]])) for k in ("gold", "jcs", "xp")}
                     for who in ("real", "rule", "ai")}
    res["gains"] = gains
    # 차이 목록
    entries = []
    for ph in PHASES:
        for dim, dim_ko in [("_all", "시기 전체")] + DIMS:
            vals = sorted({r.get(dim, "-") for r in rows})
            for v in vals:
                sel = [r for r in rows if r["phase"] == ph and r.get(dim, "-") == v]
                if len(sel) < MIN_N:
                    continue
                sr, sa = share([r["real"] for r in sel]), share([a for r in sel for a in r["ai"]])
                gap = {c: sa[c] - sr[c] for c in CATS}
                X = max(gap, key=gap.get)
                Y = min(gap, key=gap.get)
                if gap[X] < MIN_GAP and -gap[Y] < MIN_GAP:
                    continue
                sim = cf_contrast(cfi, sel, X, Y) if cfi else {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
                sens = {t: (sens_shift(ref, ci, sel, X, Y) if ci and cfi else None) for t, ci in sens_cf.items()}
                rx = [r for r in sel if r["real"] == X and r["phi3"] is not None]
                ry = [r for r in sel if r["real"] == Y and r["phi3"] is not None]
                real_d3 = diff_ci([r["phi3"] - r["phi"] for r in rx], [r["phi3"] - r["phi"] for r in ry])
                real_win = diff_ci([r["win"] - r["phi"] for r in rx], [r["win"] - r["phi"] for r in ry])
                e = {"phase": ph, "dim": dim_ko, "value": v, "n": len(sel), "X": X, "Y": Y,
                     "real_share": sr, "ai_share": sa, "gap_X": gap[X], "gap_Y": gap[Y],
                     "sim": sim, "sens": sens, "real_dphi3": real_d3, "real_win": real_win,
                     "n_real_X": len(rx), "n_real_Y": len(ry)}
                e["cause"] = classify(sim, real_win, sens)
                entries.append(e)
    res["entries"] = entries
    # 차이 지점 목록(핵심): 같은 실제 상태에서 학습 정글러의 1분 범주가 실제와 다른 경우를 (실제 → 학습) 쌍으로 묶는다
    pairs = collections.defaultdict(list)
    for r in rows:
        x = cfi.get(key(r))
        if x is None:
            continue
        for a in r["ai"]:
            if a != r["real"]:
                pairs[(r["real"], a)].append(r)
    total_w = sum(cfi[key(r)]["w"] for v in pairs.values() for r in v)
    exc = {ph: {c: [rr["win"] - rr["phi"] for rr in rows if rr["phase"] == ph and rr["real"] == c] for c in CATS} for ph in PHASES}
    plist = []
    for (Y, X), sel in pairs.items():
        if len(sel) < 50:
            continue
        sim = cf_contrast(cfi, sel, X, Y)
        sens = {t: (sens_shift(ref, ci, sel, X, Y) if ci else None) for t, ci in sens_cf.items()}
        # 실제 대조: 같은 시기 실제 정글러 중 X를 한 1분 − Y를 한 1분의 (최종 승리 − 예측 승률), 쌍의 시기 분포로 가중
        phw = collections.Counter(r["phase"] for r in sel)
        d, var = 0.0, 0.0
        for ph, n in phw.items():
            a, b = np.array(exc[ph][X]), np.array(exc[ph][Y])
            wgt = n / len(sel)
            d += wgt * (a.mean() - b.mean())
            var += wgt ** 2 * (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        real_win = (float(d), float(d - 1.96 * math.sqrt(var)), float(d + 1.96 * math.sqrt(var)))
        e = {"real": Y, "ai": X, "n": len(sel), "share": sum(cfi[key(r)]["w"] for r in sel) / total_w,
             "phases": dict(phw), "sim": sim, "sens": sens, "real_win": real_win}
        e["cause"] = classify(sim, real_win, sens)
        plist.append(e)
    plist.sort(key=lambda e: -e["share"])
    res["pairs"] = plist
    res["sens_reference"] = res_ref
    d, w = [], []
    for k, x in cfi.items():
        r = next(rr for rr in [by_key[k]])
        d.append(np.mean(x["q"][r["real"]]) - np.mean(x["q"]["free"]))
        w.append(x["w"])
    res["follow_real"] = _wmean_ci(d, w) if d else NAN
    res["follow_real_phase"] = {}
    for ph in PHASES:
        dd = [(np.mean(x["q"][by_key[k]["real"]]) - np.mean(x["q"]["free"]), x["w"]) for k, x in cfi.items() if by_key[k]["phase"] == ph]
        res["follow_real_phase"][ph] = _wmean_ci([a for a, _ in dd], [b for _, b in dd]) if dd else NAN
    # 범주별 1분 묶음 가치(전체 표본): Q(c) − Q(자유)
    qv = {}
    for c in CATS:
        if cfi:
            qv[c] = cf_contrast(cfi, rows, c, "free")
    res["q_vs_free"] = qv
    # 시기별 범주 표: 실제 대비 학습 정글러의 선택 비율 차이와 그 범주로 첫 1분을 묶었을 때의 가치
    cat_phase = {}
    for ph in PHASES:
        sel = [r for r in rows if r["phase"] == ph]
        sr, sa = share([r["real"] for r in sel]), share([a for r in sel for a in r["ai"]])
        cat_phase[ph] = {c: {"real": sr[c], "ai": sa[c], "q": cf_contrast(cfi, sel, c, "free") if cfi else NAN,
                             "sens": {t: (sens_shift(ref, ci, sel, c, "free") if ci and cfi else None) for t, ci in sens_cf.items()}}
                         for c in CATS}
    res["cat_phase"] = cat_phase
    if cfi:
        comp = collections.defaultdict(list)
        for x in cfi.values():
            for c in CATS:
                comp[c] += [bool(l[1]) for l in x["lab"][c]]
        res["compliance"] = {c: float(np.mean(v)) for c, v in comp.items()}
    with open(os.path.join(ROOT, "reports", "stage6_compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_md(res)
    print(f"차이 {len(entries)}개 → reports/stage6_compare.md")


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def ci_txt(m, lo, hi, scale=100, unit="%p"):
    if m is None or math.isnan(m):
        return "-"
    return f"{scale * m:+.1f}{unit} ({scale * lo:+.1f}~{scale * hi:+.1f})"


def write_md(res):
    L = []
    L.append("# 6단계: 천상계 정글러와 학습 정글러 비교 (자동 생성 표)\n")
    L.append(f"실제 {res['matches']}판 × 양 팀 정글러 × 3~17분 = 상황 {res['situations']:,}개. 반사실 표본 {res['cf_samples']}개.\n")
    L.append("범주는 다음 1분의 위치·이벤트로 붙인다(실제와 시뮬 같은 함수): " + ", ".join(f"{CAT_KO[c]}" for c in CATS) + ".\n")
    L.append("## 범주 분포 (실제 / 학습 정글러 / 규칙 봇)\n")
    L.append("| 시기 | 상황 수 | " + " | ".join(CAT_KO[c] for c in CATS) + " |")
    L.append("|---|---|" + "---|" * len(CATS))
    for ph, d in res["dist"].items():
        L.append(f"| {ph} | {d['n']:,} | " + " | ".join(f"{pct(d['real'][c])} / {pct(d['ai'][c])} / {pct(d['rule'][c])}" for c in CATS) + " |")
    L.append("\n## 같은 상태에서 1분 수입 (상태 옮기기 점검)\n")
    L.append("| 시기 | 골드 실제 / 규칙 봇 / 학습 | 정글 CS 실제 / 규칙 봇 / 학습 | 경험치 실제 / 규칙 봇 / 학습 |")
    L.append("|---|---|---|---|")
    for ph, g in res["gains"].items():
        L.append(f"| {ph} | " + " | ".join(f"{g['real'][k]:.0f} / {g['rule'][k]:.0f} / {g['ai'][k]:.0f}" for k in ("gold", "jcs", "xp")) + " |")
    L.append("\n## 범주별 결과 모델 대조 (그 범주로 붙은 1분의 평균 수입, 실제 / 학습 정글러)\n")
    L.append("| 범주 | 실제 수 | 학습 수 | 골드 | 정글 CS | 경험치 |")
    L.append("|---|---|---|---|---|---|")
    for c, m in res["mechanism"].items():
        f = lambda k: f"{m['real_' + k]:.0f} / {m['ai_' + k]:.0f}" if m["real_" + k] is not None and m["ai_" + k] is not None else "-"
        L.append(f"| {CAT_KO[c]} | {m['real_n']:,} | {m['ai_n']:,} | {f('gold')} | {f('jcs')} | {f('xp')} |")
    if res.get("outcomes"):
        L.append("\n## 범주별 1분 결과 (킬 관여 / 정글러 사망 / 에픽 몬스터 처치 비율, 실제 / 학습 정글러 / 규칙 봇)\n")
        L.append("| 범주 | 표본 수 실제/학습/규칙 | 킬 관여 | 사망 | 에픽 몬스터 처치 |")
        L.append("|---|---|---|---|---|")
        o = res["outcomes"]
        for c in CATS:
            get = lambda who, k: o[who].get(c, {}).get(k)
            L.append(f"| {CAT_KO[c]} | " + "/".join(str(o[w].get(c, {}).get("n", 0)) for w in ("real", "ai", "rule")) + " | "
                     + " | ".join(" / ".join(pct(get(w, k)) for w in ("real", "ai", "rule")) for k in ("kill", "death", "elite")) + " |")
    if res.get("q_vs_free"):
        L.append("\n## 첫 1분을 범주로 묶었을 때 20분 예측 승률 (자유 행동 대비, 전체 표본)\n")
        L.append("| 범주 | 묶음 대비 자유 | 묶은 범주 행동을 시작한 비율 |")
        L.append("|---|---|---|")
        for c, q in res["q_vs_free"].items():
            L.append(f"| {CAT_KO[c]} | {ci_txt(q['mean'], q['lo'], q['hi'])} | {pct(res.get('compliance', {}).get(c))} |")
    if res.get("cat_phase"):
        L.append("\n## 시기별 범주: 선택 비율(실제→학습)과 첫 1분을 그 범주로 묶은 가치(자유 행동 대비, %p)\n")
        L.append("| 시기 | " + " | ".join(CAT_KO[c] for c in CATS) + " |")
        L.append("|---|" + "---|" * len(CATS))
        for ph, d in res["cat_phase"].items():
            L.append(f"| {ph} | " + " | ".join(f"{pct(d[c]['real'])}→{pct(d[c]['ai'])}, {ci_txt(d[c]['q']['mean'], d[c]['q']['lo'], d[c]['q']['hi'])}" for c in CATS) + " |")
    if res.get("pairs"):
        fr = res["follow_real"]
        L.append("\n## 실제 선택을 따라 했을 때\n")
        L.append(f"같은 실제 상태에서 첫 1분만 천상계 정글러가 실제로 한 범주로 묶고 나머지는 학습 정글러가 둔 경우 − 학습 정글러가 처음부터 고른 경우: "
                 f"{ci_txt(fr['mean'], fr['lo'], fr['hi'])} (표본 {fr['n']}개). 시기별: "
                 + ", ".join(f"{ph} {ci_txt(v['mean'], v['lo'], v['hi'])}" for ph, v in res["follow_real_phase"].items()) + ".\n")
        L.append("\n## 차이 지점 목록 (실제 → 학습 정글러)\n")
        L.append("같은 실제 상태에서 학습 정글러의 다음 1분 범주가 실제와 다른 경우를 쌍으로 묶었다. 비중 = 다른 경우 전체에서 이 쌍의 몫(표본 가중). "
                 "시뮬 반사실 = 학습 정글러 범주로 묶은 1분 − 실제 범주로 묶은 1분의 20분 예측 승률(같은 상태, 같은 난수). "
                 "실제 대조 = 같은 시기 실제 정글러가 학습 쪽 범주를 한 1분 − 실제 쪽 범주를 한 1분의 (최종 승리 − 그때 예측 승률), 관찰 값이라 방향 참고. "
                 "가정 바꿈 = 그 가정으로 같은 상황을 다시 돌렸을 때 시뮬 반사실이 기본보다 변한 양.\n")
        L.append("| 실제 → 학습 | 비중 | 표본 | 시뮬 반사실 | 실제 대조 | " + " | ".join(v["ko"] for v in SENS.values()) + " | 분류 |")
        L.append("|---|---|---|---|---|" + "---|" * len(SENS) + "---|")
        for e in res["pairs"]:
            sm = e["sim"]
            sens = [ci_txt(v["shift"]["mean"], v["shift"]["lo"], v["shift"]["hi"]) if v else "-" for v in e["sens"].values()]
            L.append(f"| {CAT_KO[e['real']]} → {CAT_KO[e['ai']]} | {pct(e['share'])} | {e['n']} | {ci_txt(sm['mean'], sm['lo'], sm['hi'])} | "
                     f"{ci_txt(*e['real_win'])} | " + " | ".join(sens) + f" | {e['cause']} |")
    L.append("\n## 부록: 상황 칸별 선택 비율 차이\n")
    L.append("X = 학습 정글러가 실제보다 많이 고른 범주, Y = 덜 고른 범주. 시뮬 반사실 = Q(X) − Q(Y), 20분 예측 승률 %p. "
             "실제 대조 = 그 칸 실제 정글러가 X를 했을 때 − Y를 했을 때의 3분 뒤 예측 승률 변화 / 최종 승리 − 지금 예측 승률. "
             "범주가 1분 동안의 킬·사망 위치로도 정해져서(라인 개입에 킬이 섞임) 실제 대조는 방향 참고용이다. "
             "가정 바꿈 열 = 같은 상황에서 그 가정으로 다시 돌린 X−Y가 기본보다 얼마나 변했는지.\n")
    L.append("| 시기 | 조건 | 상황 수 | X (실제→학습) | Y (실제→학습) | 시뮬 반사실 X−Y | 실제 대조 X−Y | "
             + " | ".join(f"가정 바꿈: {v['ko']}" for v in SENS.values()) + " | 분류 |")
    L.append("|---|---|---|---|---|---|---|" + "---|" * len(SENS) + "---|")
    for e in res["entries"]:
        X, Y = e["X"], e["Y"]
        s = e["sim"]
        sens = [ci_txt(v["shift"]["mean"], v["shift"]["lo"], v["shift"]["hi"]) if v else "-" for v in e["sens"].values()]
        L.append(f"| {e['phase']} | {e['dim']}: {e['value']} | {e['n']:,} | {CAT_KO[X]} {pct(e['real_share'][X])}→{pct(e['ai_share'][X])} | "
                 f"{CAT_KO[Y]} {pct(e['real_share'][Y])}→{pct(e['ai_share'][Y])} | {ci_txt(s['mean'], s['lo'], s['hi'])} | "
                 f"{ci_txt(*e['real_dphi3'])} / {ci_txt(*e['real_win'])} | " + " | ".join(sens) + f" | {e['cause']} |")
    with open(os.path.join(ROOT, "reports", "stage6_compare.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
