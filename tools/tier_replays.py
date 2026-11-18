"""티어 기준선 경기(data/tiers/<TIER>/raw)에서 골라 리플레이로 1초 동선을 기록한다 (kk PC).

  python -m tools.tier_replays --tier GOLD --count 20                       # 시드 팀 정글·서폿·미드
  python -m tools.tier_replays --tier GOLD --count 20 --roles JUNGLE,UTILITY,MIDDLE --both-teams
  python -m tools.tier_replays --root data/tiers_26.23 --extra-from data/tiers --tier GOLD --count 20   # 새 패치 경기로 채우기

천상계 기록(data/replay_raw, data/replay_tracks)과 섞이지 않게 data/tiers/<TIER>/replay_raw, replay_tracks에 쓴다.
고르는 기준: 지금 클라이언트와 같은 빌드, 20분 이상, 시드 1명당 1판, 피딩(10데스 이상·KDA 0.3 이하)·
잠수(서폿 빼고 팀 피해 5% 미만) 선수가 없는 판, 최근 경기부터.
받기가 연속 --give-up판 실패하면 하루 받기 한도로 보고 멈춘다. 다시 실행하면 기록한 선수는 건너뛰고 이어서 한다.
필요: 롤 클라이언트 로그인, game.cfg EnableReplayApi=1. 실행 중엔 키보드·마우스를 쓰지 말 것.
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tools.replay_track as rt  # noqa: E402
from tools.replay_batch import MIN_VALID, Tee, close_replay  # noqa: E402

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def suspicious(info):
    dmg = {}
    for p in info["participants"]:
        dmg[p["teamId"]] = dmg.get(p["teamId"], 0) + p.get("totalDamageDealtToChampions", 0)
    for p in info["participants"]:
        kda = (p["kills"] + p["assists"]) / max(p["deaths"], 1)
        if p["deaths"] >= 10 and kda <= 0.3:
            return True
        share = p.get("totalDamageDealtToChampions", 0) / max(dmg[p["teamId"]], 1)
        if share < 0.05 and p.get("teamPosition") != "UTILITY":
            return True
    return False


def pick(tier_dir, build, count):
    raw_out = os.path.join(tier_dir, "replay_raw")
    os.makedirs(raw_out, exist_ok=True)
    picked, seeds = [], set()
    for path in sorted(glob.glob(os.path.join(tier_dir, "raw", "*.json.gz")), reverse=True):
        mid = os.path.basename(path)[:-8]
        out = os.path.join(raw_out, f"{mid}.json")
        if os.path.exists(out):   # 앞서 고른 판은 그대로 쓴다(지금 빌드와 같을 때만)
            with open(out, encoding="utf-8") as f:
                d = json.load(f)
            if not d["detail"]["info"]["gameVersion"].startswith(build):
                continue
            picked.append(mid)
            seeds.add(d["seed"]["puuid"])
            continue
    for path in sorted(glob.glob(os.path.join(tier_dir, "raw", "*.json.gz")), reverse=True):
        if len(picked) >= count:
            break
        mid = os.path.basename(path)[:-8]
        if mid in picked:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        info = d["detail"]["info"]
        if d["seed"]["puuid"] in seeds or not info["gameVersion"].startswith(build):
            continue
        if info["gameDuration"] < 20 * 60 or suspicious(info):
            continue
        if any(p.get("gameEndedInEarlySurrender") for p in info["participants"]):
            continue
        with open(os.path.join(raw_out, f"{mid}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        picked.append(mid)
        seeds.add(d["seed"]["puuid"])
    return picked[:count]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="GOLD")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--roles", default="JUNGLE,UTILITY,MIDDLE")
    ap.add_argument("--both-teams", action="store_true", help="양 팀 모두(기본은 시드 팀만)")
    ap.add_argument("--until", type=float, default=1200, help="몇 초까지 기록 (0 = 경기 끝까지)")
    ap.add_argument("--speed", type=float, default=4)
    ap.add_argument("--give-up", type=int, default=3, help="연속 받기 실패 몇 판이면 멈출지")
    ap.add_argument("--game-dir", default=rt.DEFAULT_GAME_DIR)
    ap.add_argument("--pick-only", action="store_true")
    ap.add_argument("--root", default="data/tiers", help="경기 폴더 루트(예: data/tiers_26.23)")
    ap.add_argument("--extra-from", default="", help="이 루트의 같은 티어 기록(replay_tracks)도 '이미 한 판'으로 쳐서 "
                                                  "--count를 채운다(예: data/tiers, 26.22 골드 18판 + 26.23 2판)")
    args = ap.parse_args()

    tier_dir = os.path.join(ROOT, args.root, args.tier.upper())
    if args.extra_from:
        prev = os.path.join(ROOT, args.extra_from, args.tier.upper(), "replay_tracks")
        done_prev = {n.split("_players.csv")[0] for n in os.listdir(prev) if n.endswith("_players.csv")} if os.path.isdir(prev) else set()
        args.count = max(0, args.count - len(done_prev))
        print(f"{args.extra_from}에 이미 {len(done_prev)}판 → 이번에 {args.count}판")
    rt.REPLAY_RAW = os.path.join(tier_dir, "replay_raw")
    rt.OUT = os.path.join(tier_dir, "replay_tracks")
    os.makedirs(rt.OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(tier_dir, "replay_log.txt"))
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}")
    roles = [r.strip().upper() for r in args.roles.split(",") if r.strip()]
    if any(r not in ROLES for r in roles):
        ap.error(f"모르는 포지션: {roles}")

    game_dir = rt.find_game_dir(args.game_dir)
    lcu = rt.LCU(game_dir)
    version = (lcu.get("/lol-replays/v1/configuration") or {}).get("gameVersion", "")
    build = ".".join(version.split(".")[:3]) + "."
    print(f"클라이언트 빌드 {version} → gameVersion {build}* 경기만")
    matches = pick(tier_dir, build, args.count)
    print(f"대상 {len(matches)}판 ({args.tier}, {', '.join(roles)}, {'양 팀' if args.both_teams else '시드 팀'})")
    if args.pick_only:
        return

    fails = 0
    for i, mid in enumerate(matches, 1):
        with open(os.path.join(rt.REPLAY_RAW, f"{mid}.json"), encoding="utf-8") as f:
            d = json.load(f)
        info = d["detail"]["info"]
        seed_team = next(p["teamId"] for p in info["participants"] if p["puuid"] == d["seed"]["puuid"])
        champs = [p["championName"] for r in roles for p in info["participants"]
                  if p.get("teamPosition") == r and (args.both_teams or p["teamId"] == seed_team)]
        end = info["gameDuration"] if args.until <= 0 else min(args.until, info["gameDuration"])
        todo = []
        for c in champs:
            lt = rt.last_time(mid, c)
            if lt is None:
                todo.append((c, 0.0))
            elif lt < end - 30:
                todo.append((c, max(0.0, lt - 5)))
        print(f"\n[{i}/{len(matches)}] {time.strftime('%H:%M:%S')} {mid} {', '.join(champs)} (할 일 {len(todo)}명)")
        if not todo:
            continue
        try:
            close_replay()
            rt.download_and_watch(lcu, mid.split("_")[-1])
            fails = 0
            rp = rt.Replay()
            rp.wait_ready()
            for c, start in todo:
                _, ok = rt.record(rp, mid, c, args.until, args.speed, start=start)
                if ok < MIN_VALID:
                    print(f"  유효 기록 {ok:.0%}뿐이라 한 번 더 기록")
                    rt.record(rp, mid, c, args.until, min(args.speed, 4), start=start)
        except SystemExit as e:
            fails += 1
            print(f"  실패: {e}")
        except Exception as e:  # 한 판 실패로 전체를 멈추지 않는다
            fails += 1
            print(f"  오류: {e!r}")
        finally:
            close_replay()
        if fails >= args.give_up:
            print(f"  {fails}판 연속 실패: 받기 한도(하루 제한)로 보고 멈춤. 내일 같은 명령으로 이어서 하세요.")
            break
    print(f"끝 {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
