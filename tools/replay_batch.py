"""천상계 정글 경기를 찾아 리플레이로 양 팀 정글러 동선을 기록하고 타임라인과 맞대 본다 (kk PC).

  python -m tools.replay_batch --count 20
  python -m tools.replay_batch --count 20 --role MIDDLE     # 양 팀 미드 (data/replay_raw에 있는 경기부터)
  python -m tools.replay_batch --count 20 --role MIDDLE,UTILITY,TOP,BOTTOM   # 한 경기를 열어 8명 모두
  python -m tools.replay_batch --count 400 --hours 168 --download-only    # 기록 없이 경기 JSON·.rofl만 받아 둠
  python -m tools.replay_batch --count 400 --until 0 --role JUNGLE,TOP,MIDDLE  # 경기 끝까지(20분까지 해 둔 건 이어서)

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

from riot_api import KeyExpired, RiotAPI  # noqa: E402
from tools.replay_compare import compare  # noqa: E402
from tools.replay_track import (DEFAULT_GAME_DIR, LCU, LCU_BODY, OUT, REPLAY_RAW, Replay, download_and_watch,  # noqa: E402
                                find_game_dir, last_time, record, replay_state)

SUMMARY = os.path.join(OUT, "summary.csv")
MIN_DURATION = 20 * 60
MIN_VALID = 0.7  # 유효 행이 이보다 적으면 카메라 고정이 실패한 것으로 보고 다시 기록한다


def junglers(detail):
    return by_role(detail, "JUNGLE")


def by_role(detail, role):
    return [p["championName"] for p in detail["info"]["participants"] if p.get("teamPosition") == role]


def find_matches(api, build, count, hours, per_player, tiers):
    """조건에 맞는 경기 번호를 count개까지 모아 data/replay_raw에 저장한다. 이미 저장된 것도 포함해 돌려준다."""
    os.makedirs(REPLAY_RAW, exist_ok=True)
    seen = {n[:-5] for n in os.listdir(REPLAY_RAW) if n.endswith(".json")}
    found = []
    for mid in sorted(seen):   # 이미 받아 둔 경기 중 지금 빌드와 같은 것부터 쓴다
        with open(os.path.join(REPLAY_RAW, f"{mid}.json"), encoding="utf-8") as f:
            if json.load(f)["detail"]["info"]["gameVersion"].startswith(build):
                found.append(mid)
    since = time.time() - hours * 3600
    if api is None:
        if len(found) < count:
            print(f"  RIOT_API_KEY가 없어 받아 둔 {len(found)}판만 씁니다")
        return found
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


class Tee:
    """print를 화면과 로그 파일에 같이 쓴다(클라우드에서 PC 폴더의 로그로 진행을 본다)."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s.replace("\r", "\n") if s.endswith("\r") else s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def download_only(lcu, matches, timeout=180):
    """리플레이(.rofl)만 받아 둔다. 다음 패치 뒤엔 클라이언트가 옛 빌드 경기를 못 받으므로 지금 받아 두는 것."""
    ok = 0
    for i, mid in enumerate(matches, 1):
        gid = mid.split("_")[-1]
        state, _ = replay_state(lcu, gid)
        if state != "watch":
            lcu.post(f"/lol-replays/v1/rofls/{gid}/download/graceful", LCU_BODY)
            t0 = time.time()
            while time.time() - t0 < timeout:
                time.sleep(2)
                state, _ = replay_state(lcu, gid)
                if state in ("watch", "incompatible", "lost", "missingOrExpired", "error", "unsupported"):
                    break
        ok += state == "watch"
        print(f"  [{i}/{len(matches)}] {mid} {state}")
    print(f"리플레이 받아 둠 {ok}/{len(matches)}판")


def close_replay():
    subprocess.run(["taskkill", "/IM", "League of Legends.exe", "/F"], capture_output=True)
    time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--hours", type=float, default=24, help="최근 몇 시간 경기에서 찾을지")
    ap.add_argument("--per-player", type=int, default=2)
    ap.add_argument("--tiers", default="challenger,grandmaster")
    ap.add_argument("--until", type=float, default=1200, help="몇 초까지 기록 (0 = 경기 끝까지, 이미 짧게 기록한 건 이어서)")
    ap.add_argument("--speed", type=float, default=4)
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--find-only", action="store_true", help="경기만 찾고 기록은 하지 않는다")
    ap.add_argument("--download-only", action="store_true", help="경기를 찾고 리플레이(.rofl)만 받는다")
    ap.add_argument("--log", default=os.path.join(OUT, "collect_log.txt"), help="진행 로그 파일 (빈 문자열이면 안 씀)")
    ap.add_argument("--role", default="JUNGLE",
                    help="기록할 포지션(양 팀), 쉼표로 여러 개(예: MIDDLE,UTILITY,TOP,BOTTOM). 한 경기를 한 번 열어 그 포지션을 모두 기록. "
                         "이미 받아 둔 data/replay_raw 경기부터 쓴다 (기본 JUNGLE)")
    args = ap.parse_args()
    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        sys.stdout = Tee(args.log)
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}")
    roles = [r.strip().upper() for r in args.role.split(",") if r.strip()]
    bad = [r for r in roles if r not in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")]
    if bad:
        ap.error(f"모르는 포지션: {bad}")

    game_dir = find_game_dir(args.game_dir)
    lcu = LCU(game_dir)
    version = (lcu.get("/lol-replays/v1/configuration") or {}).get("gameVersion", "")
    build = ".".join(version.split(".")[:3]) + "."  # 리플레이는 빌드까지 같아야 열린다 (예: 16.21.829.)
    print(f"클라이언트 빌드 {version} → gameVersion {build}* 경기만")

    # 받아 둔 경기로 충분하면 API 키 없이도 돈다(키가 없으면 새 경기는 찾지 않음)
    api = RiotAPI(platform="kr") if os.environ.get("RIOT_API_KEY", "").startswith("RGAPI-") else None
    try:
        matches = find_matches(api, build, args.count, args.hours, args.per_player, args.tiers.split(","))
    except KeyExpired as e:   # 키가 만료돼도 받아 둔 경기로 계속 기록한다
        print(f"  {e}\n  -> 새 경기 찾기는 멈추고 받아 둔 경기만 씁니다")
        matches = find_matches(None, build, args.count, args.hours, args.per_player, args.tiers.split(","))
    matches = matches[:args.count]   # 이미 받아 둔 경기가 더 많아도 --count판까지만
    print(f"대상 {len(matches)}판 ({', '.join(roles)})")
    if args.find_only:
        return
    if args.download_only:
        download_only(lcu, matches)
        return

    done = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as f:
            done = {r["match"] for r in csv.DictReader(f)}

    for i, mid in enumerate(matches, 1):
        with open(os.path.join(REPLAY_RAW, f"{mid}.json"), encoding="utf-8") as f:
            data = json.load(f)
        champs = [c for r in roles for c in by_role(data["detail"], r)]
        end = data["detail"]["info"]["gameDuration"] if args.until <= 0 else args.until
        todo = []   # (챔피언, 시작 초): 없으면 0부터, 짧게 기록해 둔 건 마지막 시각 5초 전부터 이어서
        for c in champs:
            lt = last_time(mid, c)
            if lt is None:
                todo.append((c, 0.0))
            elif lt < end - 30:
                todo.append((c, max(0.0, lt - 5)))
        print(f"\n[{i}/{len(matches)}] {mid} {' vs '.join(champs)} ({end / 60:.0f}분까지, 할 일 {len(todo)}명)")
        if todo:
            if not data["detail"]["info"]["gameVersion"].startswith(build):
                print("  빌드가 달라져 재생 불가, 건너뜀")
                continue
            try:
                close_replay()   # 열려 있던 리플레이 창이 있으면 API가 그 경기에 붙으므로 먼저 닫는다
                download_and_watch(lcu, mid.split("_")[-1])
                rp = Replay()
                rp.wait_ready()
                for c, start in todo:
                    _, ok = record(rp, mid, c, args.until, args.speed, start=start)
                    if ok < MIN_VALID:
                        print(f"  유효 기록 {ok:.0%}뿐이라 4배속으로 한 번 더 기록")
                        record(rp, mid, c, args.until, min(args.speed, 4), start=start)
            except SystemExit as e:
                print(f"  실패: {e}")
            except Exception as e:  # 한 판 실패로 전체를 멈추지 않는다
                print(f"  오류: {e!r}")
            finally:
                close_replay()
        if mid in done or roles != ["JUNGLE"]:   # 타임라인 1분 위치 비교는 정글 기록 때만 summary.csv에 쌓음
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
