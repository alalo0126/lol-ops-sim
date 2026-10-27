"""밖에서 뽑아 온 .rofl 디코딩 결과(위치·와드)를 우리가 이미 검증한 데이터와 맞대서 믿을 만한지 판정한다 (kk PC).

  python -m tools.rofl_decoded_check                 # data/rofl_decoded/ 의 positions*.csv, wards*.csv 전부
  python -m tools.rofl_decoded_check --match KR_8427110811

입력 형식 (UTF-8 CSV, 머리줄 필수):
  positions*.csv: match_id, t_s, riot_id, x, y
  wards*.csv:     match_id, t_place_s, t_end_s, x, y, ward_type, team, owner_riot_id
riot_id는 '이름#태그' 또는 '이름'. 좌표는 Riot 맵 좌표(0~15000, 블루 우물이 왼쪽 아래).

기준:
  위치 ↔ 리플레이 API 1초 동선(data/replay_tracks/<경기>_<챔피언>.csv, mark_valid로 유효한 행만), ±1초
  위치 ↔ 타임라인 분 위치(data/replay_raw/<경기>.json, 10명 전원), ±3초
  와드 ↔ 타임라인 WARD_PLACED(위치 없음, 시각·설치자·종류만), 같은 설치자 ±3초
리플레이 동선 자체가 타임라인과 중앙값 ~130 차이이므로, 위치는 동선 대비 중앙값 300 이내·1,000 이내 90% 이상이면 통과로 본다.
"""
import argparse
import bisect
import csv
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DECODED = os.path.join(ROOT, "data", "rofl_decoded")
RAW = os.path.join(ROOT, "data", "replay_raw")
TRACKS = os.path.join(ROOT, "data", "replay_tracks")

PASS_MEDIAN = 300
PASS_NEAR = 1000
PASS_NEAR_SHARE = 0.9
WARD_DT = 3


def ward_kind(s):
    """와드 종류 표기를 묶음으로: control / blue / stealth. 모르면 None (대소문자·표기 차이 무시)."""
    s = (s or "").lower()
    if "control" in s or "pink" in s:
        return "control"
    if "blue" in s or "farsight" in s:
        return "blue"
    if any(k in s for k in ("yellow", "sight", "stealth", "trinket", "totem")):
        return "stealth"
    return None


def read_csv(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(DECODED, pattern))):
        with open(path, encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    return rows


def name_of(riot_id):
    return riot_id.split("#")[0].strip()


def load_raw(mid):
    path = os.path.join(RAW, f"{mid}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_track(mid, champ):
    """리플레이 API 동선 중 유효한 (시각, x, y). 파일 없으면 []."""
    path = os.path.join(TRACKS, f"{mid}_{champ}.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    if rows and "valid" not in rows[0]:
        from tools.replay_track import mark_valid
        cols = ["time", "x", "y", "cam_x", "cam_y", "cam_z", "jump"]
        rows = [dict(zip(cols + ["valid"], m)) for m in mark_valid([tuple(r.get(c, 0) for c in cols) for r in rows])]
    return [(r["time"], r["x"], r["y"]) for r in rows if r["valid"]]


def nearest(series, t, tol):
    """시각순 [(t, x, y)]에서 t±tol 안의 가장 가까운 점."""
    i = bisect.bisect_left(series, (t,))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(series) and abs(series[j][0] - t) <= tol:
            if best is None or abs(series[j][0] - t) < abs(best[0] - t):
                best = series[j]
    return best


def summary(dists):
    if not dists:
        return None
    d = sorted(dists)
    return {"n": len(d), "median": round(d[len(d) // 2]),
            "near_share": round(sum(x <= PASS_NEAR for x in d) / len(d), 3)}


def passed(s):
    return s is not None and s["median"] <= PASS_MEDIAN and s["near_share"] >= PASS_NEAR_SHARE


def check_positions(mid, rows, raw):
    """선수별 {'track': 요약, 'timeline': 요약}. rows는 이 경기의 positions 행."""
    parts = {p["riotIdGameName"]: p for p in raw["detail"]["info"]["participants"]}
    frames = raw["timeline"]["info"]["frames"]
    by_player = {}
    for r in rows:
        by_player.setdefault(name_of(r["riot_id"]), []).append((float(r["t_s"]), float(r["x"]), float(r["y"])))
    out = {}
    for name, series in by_player.items():
        series.sort()
        p = parts.get(name)
        if p is None:
            out[name] = {"error": "경기 참가자에 없는 이름"}
            continue
        td = []
        for t, x, y in load_track(mid, p["championName"]):
            q = nearest(series, t, 1.0)
            if q:
                td.append(math.hypot(q[1] - x, q[2] - y))
        tl = []
        for fr in frames[1:]:
            pos = fr["participantFrames"][str(p["participantId"])]["position"]
            q = nearest(series, fr["timestamp"] / 1000, 3.0)
            if q:
                tl.append(math.hypot(q[1] - pos["x"], q[2] - pos["y"]))
        out[name] = {"champion": p["championName"], "track": summary(td), "timeline": summary(tl)}
    return out


def check_wards(mid, rows, raw):
    """타임라인 WARD_PLACED와 설치자·시각으로 짝짓기. 위치는 타임라인에 없어 맞춰 볼 수 없다."""
    pid = {p["riotIdGameName"]: p["participantId"] for p in raw["detail"]["info"]["participants"]}
    events = [e for fr in raw["timeline"]["info"]["frames"] for e in fr["events"]
              if e["type"] == "WARD_PLACED" and e.get("creatorId")]
    used = set()
    matched = kind_ok = kind_n = 0
    for r in rows:
        who = pid.get(name_of(r["owner_riot_id"]))
        t = float(r["t_place_s"])
        cand = [i for i, e in enumerate(events)
                if i not in used and e["creatorId"] == who and abs(e["timestamp"] / 1000 - t) <= WARD_DT]
        if not cand:
            continue
        i = min(cand, key=lambda i: abs(events[i]["timestamp"] / 1000 - t))
        used.add(i)
        matched += 1
        want = ward_kind(events[i].get("wardType"))
        if want:
            kind_n += 1
            kind_ok += want == ward_kind(r.get("ward_type"))
    in_map = sum(0 <= float(r["x"]) <= 15000 and 0 <= float(r["y"]) <= 15000 for r in rows)
    return {"decoded": len(rows), "timeline": len(events), "matched": matched,
            "kind_agree": round(kind_ok / kind_n, 3) if kind_n else None, "in_map": in_map}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    args = ap.parse_args()
    pos, wards = read_csv("positions*.csv"), read_csv("wards*.csv")
    if not pos and not wards:
        sys.exit(f"{DECODED}에 positions*.csv / wards*.csv 없음")
    mids = sorted({r["match_id"] for r in pos + wards})
    if args.match:
        mids = [m for m in mids if m == args.match]
    report = {}
    for mid in mids:
        raw = load_raw(mid)
        if raw is None:
            print(f"{mid}: data/replay_raw에 타임라인 없음, 건너뜀")
            continue
        res = {}
        prow = [r for r in pos if r["match_id"] == mid]
        if prow:
            res["positions"] = check_positions(mid, prow, raw)
            for name, s in res["positions"].items():
                if "error" in s:
                    print(f"{mid} {name}: {s['error']}")
                    continue
                verdict = "통과" if passed(s["track"]) else ("동선 없음" if s["track"] is None else "불합격")
                print(f"{mid} {s['champion']:>12} 동선 대비 {s['track']}  타임라인 대비 {s['timeline']}  → {verdict}")
        wrow = [r for r in wards if r["match_id"] == mid]
        if wrow:
            res["wards"] = check_wards(mid, wrow, raw)
            w = res["wards"]
            print(f"{mid} 와드: 디코딩 {w['decoded']} / 타임라인 {w['timeline']}, 짝지음 {w['matched']}, "
                  f"종류 일치 {w['kind_agree']}, 맵 안 좌표 {w['in_map']}")
        report[mid] = res
    os.makedirs(DECODED, exist_ok=True)
    with open(os.path.join(DECODED, "check_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"저장: {os.path.join(DECODED, 'check_report.json')}")


if __name__ == "__main__":
    main()
