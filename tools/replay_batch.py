"""천상계 정글 경기를 찾아 리플레이로 양 팀 정글러 동선을 기록하고 타임라인과 맞대 본다 (kk PC).

  python -m tools.replay_batch --count 20

필요: 롤 클라이언트 로그인, RIOT_API_KEY, game.cfg EnableReplayApi=1 (python -m tools.replay_track --enable-api).
1. 찾기: 챌린저·그랜드마스터 상위부터 최근 --hours 시간 솔랭 중 지금 클라이언트와 같은 빌드, 20분 이상,
   양 팀 정글러가 있는 경기. 선수 한 명당 --per-player 판까지 (소수 선수 편중 방지).
   경기·타임라인은 data/replay_raw/<경기>.json (1단계 data/raw와 분리).
2. 기록: 경기마다 리플레이를 받아 틀고 양 팀 정글러를 차례로 추적 → data/replay_tracks/<경기>_<챔피언>.csv,
   끝나면 리플레이 게임 창을 닫는다.
3. 비교: 타임라인 1분 위치와의 오차를 data/replay_tracks/summary.csv에 쌓는다.
이미 CSV가 있는 경기는 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 한다. 실행 중엔 키보드·마우스를 쓰지 말 것.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "collect"))

from riot_api import RiotAPI  # noqa: E402
from tools.replay_compare import compare  # noqa: E402
from tools.replay_track import (DEFAULT_GAME_DIR, LCU, OUT, REPLAY_RAW, Replay, download_and_watch,  # noqa: E402
                                find_game_dir, record)

SUMMARY = os.path.join(OUT, "summary.csv")
MIN_DURATION = 20 * 60


def junglers(detail):
    return [p["championName"] for p in detail["info"]["participants"] if p.get("teamPosition") == "JUNGLE"]


def find_matches(api, build, count, hours, per_player, tiers):
    """조건에 맞는 경기 번호를 count개까지 모아 data/replay_raw에 저장한다. 이미 저장된 것도 포함해 돌려준다."""
    os.makedirs(REPLAY_RAW, exist_ok=True)
    found = sorted(n[:-5] for n in os.listdir(REPLAY_RAW) if n.endswith(".json"))
    seen = set(found)
    since = time.time() - hours * 3600
    for tier in tiers:
        entries = sorted(api.apex_entries(tier), key=lambda e: -e.get("leaguePoints", 0))
        for e in entries:
            if len(found) >= count:
                return found
            puuid, taken = e.get("puuid"), 0
            if not puuid:
                continue
            for mid in api.match_ids(puuid, count=20, start_time=since):
                if taken >= per_player or len(found) >= count:
                    break
                if mid in seen:
                    continue
                seen.add(mid)
                detail = api.match(mid)
                if not detail:
                    continue
                info = detail["info"]
                seed = next((p for p in info["participants"] if p["puuid"] == puuid), None)
                if (not info["gameVersion"].startswith(build) or info["gameDuration"] < MIN_DURATION
                        or any(p.get("gameEndedInEarlySurrender") for p in info["participants"])
                        or len(junglers(detail)) != 2 or not seed or seed.get("teamPosition") != "JUNGLE"):
                    continue
                timeline = api.timeline(mid)
                if not timeline:
                    continue
                with open(os.path.join(REPLAY_RAW, f"{mid}.json"), "w", encoding="utf-8") as f:
                    json.dump({"detail": detail, "timeline": timeline,
                               "seed": {"puuid": puuid, "tier": tier, "participantId": seed["participantId"]}},
                              f, ensure_ascii=False)
                found.append(mid)
                taken += 1
                print(f"  찾음 {mid} {' vs '.join(junglers(detail))} ({len(found)}/{count})")
    return found


def close_replay():
    subprocess.run(["taskkill", "/IM", "League of Legends.exe", "/F"], capture_output=True)
    time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--hours", type=float, default=24, help="최근 몇 시간 경기에서 찾을지")
    ap.add_argument("--per-player", type=int, default=2)
    ap.add_argument("--tiers", default="challenger,grandmaster")
    ap.add_argument("--until", type=float, default=1200)
    ap.add_argument("--speed", type=float, default=4)
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--find-only", action="store_true", help="경기만 찾고 기록은 하지 않는다")
    args = ap.parse_args()

    game_dir = find_game_dir(args.game_dir)
    lcu = LCU(game_dir)
    version = (lcu.get("/lol-replays/v1/configuration") or {}).get("gameVersion", "")
    build = ".".join(version.split(".")[:3]) + "."  # 리플레이는 빌드까지 같아야 열린다 (예: 16.21.829.)
    print(f"클라이언트 빌드 {version} → gameVersion {build}* 경기만")

    api = RiotAPI(platform="kr")
    matches = find_matches(api, build, args.count, args.hours, args.per_player, args.tiers.split(","))
    print(f"대상 {len(matches)}판")
    if args.find_only:
        return

    done = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as f:
            done = {r["match"] for r in csv.DictReader(f)}

    for i, mid in enumerate(matches, 1):
        with open(os.path.join(REPLAY_RAW, f"{mid}.json"), encoding="utf-8") as f:
            data = json.load(f)
        champs = junglers(data["detail"])
        todo = [c for c in champs if not os.path.exists(os.path.join(OUT, f"{mid}_{c}.csv"))]
        print(f"\n[{i}/{len(matches)}] {mid} {' vs '.join(champs)}")
        if todo:
            if not data["detail"]["info"]["gameVersion"].startswith(build):
                print("  빌드가 달라져 재생 불가, 건너뜀")
                continue
            try:
                download_and_watch(lcu, mid.split("_")[-1])
                rp = Replay()
                rp.wait_ready()
                for c in todo:
                    record(rp, mid, c, args.until, args.speed)
            except SystemExit as e:
                print(f"  실패: {e}")
            except Exception as e:  # 한 판 실패로 전체를 멈추지 않는다
                print(f"  오류: {e!r}")
            finally:
                close_replay()
        if mid in done:
            continue
        try:
            rows = compare(mid, api)
        except Exception as e:
            print(f"  비교 오류: {e!r}")
            continue
        new = not os.path.exists(SUMMARY)
        with open(SUMMARY, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["match", "champion", "median_dist", "within_500", "minutes"])
            for r in rows:
                w.writerow((mid,) + r)


if __name__ == "__main__":
    main()
