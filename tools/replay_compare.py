"""리플레이에서 뽑은 정글러 좌표(data/replay_tracks)를 Riot 타임라인 1분 위치와 맞대 본다 (kk PC, RIOT_API_KEY 필요).

  python -m tools.replay_compare KR_8427110811

타임라인은 data/replay_raw/<경기>.json이 있으면 쓰고, 없으면 API로 받아 같은 형식으로 저장한다.
(1단계 1,000판이 있는 data/raw와 섞지 않는다. 예전에 data/raw에 저장된 파일은 replay_raw로 옮긴다.)
분마다 타임라인 위치와 같은 시각 리플레이 좌표의 거리를 출력한다. 거리가 작으면 좌표 변환이 맞다는 뜻이다.
"""
import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "collect"))
RAW = os.path.join(ROOT, "data", "replay_raw")
OLD_RAW = os.path.join(ROOT, "data", "raw")
TRACKS = os.path.join(ROOT, "data", "replay_tracks")


def load_match(mid, api=None):
    path = os.path.join(RAW, f"{mid}.json")
    old = os.path.join(OLD_RAW, f"{mid}.json")
    if not os.path.exists(path) and os.path.exists(old):
        with open(old, encoding="utf-8") as f:
            version = json.load(f)["detail"]["info"]["gameVersion"]
        if not version.startswith("16.20"):
            os.makedirs(RAW, exist_ok=True)
            os.replace(old, path)
            print(f"{old} → {path} 로 옮김 (1단계 데이터와 분리)")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if api is None:
        from riot_api import RiotAPI
        api = RiotAPI(platform="kr")
    data = {"detail": api.match(mid), "timeline": api.timeline(mid)}
    os.makedirs(RAW, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def load_track(path):
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def nearest(rows, t):
    """같은 시각 ±3초 안의 유효한 행. 없으면 None."""
    ok = [r for r in rows if abs(r["time"] - t) <= 3]
    return min(ok, key=lambda r: abs(r["time"] - t)) if ok else None


def compare(match_id, api=None):
    """정글러별 (챔피언, 오차 중앙값, 500 이내 분 수, 비교한 분 수)를 돌려주고 표를 출력한다."""
    data = load_match(match_id, api)
    parts = data["detail"]["info"]["participants"]
    frames = data["timeline"]["info"]["frames"]
    result = []
    for p in parts:
        if p.get("teamPosition") != "JUNGLE":
            continue
        champ = p["championName"]
        path = os.path.join(TRACKS, f"{match_id}_{champ}.csv")
        if not os.path.exists(path):
            print(f"{champ}: {path} 없음")
            continue
        rows = load_track(path)
        if "valid" not in rows[0]:
            from tools.replay_track import mark_valid
            cols = ["time", "x", "y", "cam_x", "cam_y", "cam_z", "jump"]
            marked = mark_valid([tuple(r.get(c, 0) for c in cols) for r in rows])
            rows = [dict(zip(cols + ["valid"], m)) for m in marked]
        bad = sum(1 for r in rows if not r["valid"])
        rows = [r for r in rows if r["valid"]]
        print(f"\n{champ} (참가자 {p['participantId']}, 무효로 뺀 행 {bad})")
        print("  분  타임라인(x,y)    리플레이(x,y)    거리")
        dists = []
        for fr in frames[1:]:
            t = fr["timestamp"] / 1000
            if t > rows[-1]["time"] + 1:
                break
            pos = fr["participantFrames"][str(p["participantId"])]["position"]
            r = nearest(rows, t)
            if r is None:
                print(f"  {t / 60:4.0f}  ({pos['x']:5d},{pos['y']:5d})  유효한 기록 없음")
                continue
            d = math.hypot(pos["x"] - r["x"], pos["y"] - r["y"])
            dists.append(d)
            print(f"  {t / 60:4.0f}  ({pos['x']:5d},{pos['y']:5d})  ({r['x']:5.0f},{r['y']:5.0f})  {d:5.0f}")
        if dists:
            dists.sort()
            med = dists[len(dists) // 2]
            near = sum(d <= 500 for d in dists)
            print(f"  거리 중앙값 {med:.0f}, 500 이내 {near}/{len(dists)}분")
            result.append((champ, round(med), near, len(dists)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    args = ap.parse_args()
    compare(args.match)


if __name__ == "__main__":
    main()
