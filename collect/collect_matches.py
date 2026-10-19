"""1단계: 챌린저·그랜드마스터 정글러 솔로랭크 경기의 매치와 타임라인을 모은다.

중간에 끊기거나 키가 만료돼도 같은 명령으로 다시 실행하면 이어서 받는다.

  python collect/collect_matches.py --patch 26.20 --target 1000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from riot_api import RiotAPI, KeyExpired  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
STATE = os.path.join(DATA, "collect_state.json")

# 한 선수의 최근 경기를 이만큼 봐도 정글 경기가 없으면 정글러가 아니라고 보고 넘어간다.
NON_JUNGLER_PROBE = 8


def api_version_prefix(patch):
    """표기 패치(26.20)를 API의 gameVersion 접두어(16.20.)로 바꾼다.

    2026-10-19 확인: 25.S1 이후 연도식 패치 이름을 쓰지만 match-v5 gameVersion은
    시즌 번호(2026년 = 16)를 쓴다. 기존 캐시의 26.19 경기가 "16.19.823.722"로 기록돼 있었다.
    """
    year, minor = patch.split(".")
    return f"{int(year) - 10}.{int(minor)}."


def version_key(game_version):
    major, minor = game_version.split(".")[:2]
    return int(major), int(minor)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"players_done": [], "rejected": {}}


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def saved_count():
    return sum(1 for n in os.listdir(RAW) if n.endswith(".json"))


def collect_player(api, puuid, tier, prefix, state, target):
    """선수 한 명의 해당 패치 경기를 훑는다. 패치보다 오래된 경기가 나오면 멈춘다."""
    target_key = version_key(prefix + "0")
    start, probed, jungle_seen = 0, 0, 0
    while True:
        ids = api.match_ids(puuid, start=start, count=100)
        if not ids:
            return
        for mid in ids:
            if saved_count() >= target:
                return
            path = os.path.join(RAW, f"{mid}.json")
            if os.path.exists(path):
                jungle_seen += 1
                continue
            if mid in state["rejected"]:
                if state["rejected"][mid] == "older_patch":
                    return
                continue
            detail = api.match(mid)
            if not detail:
                state["rejected"][mid] = "not_found"
                continue
            info = detail["info"]
            vkey = version_key(info["gameVersion"])
            if vkey < target_key:
                state["rejected"][mid] = "older_patch"
                return
            if vkey > target_key:
                state["rejected"][mid] = "newer_patch"
                continue
            seed = next((p for p in info["participants"] if p["puuid"] == puuid), None)
            probed += 1
            if any(p.get("gameEndedInEarlySurrender") for p in info["participants"]):
                state["rejected"][mid] = "remake"
                continue
            if not seed or seed.get("teamPosition") != "JUNGLE":
                state["rejected"][mid] = "seed_not_jungle"
                if probed >= NON_JUNGLER_PROBE and jungle_seen == 0:
                    print(f"    최근 {probed}판 정글 없음, 다음 선수로")
                    return
                continue
            timeline = api.timeline(mid)
            if not timeline:
                state["rejected"][mid] = "no_timeline"
                continue
            jungle_seen += 1
            save_json(path, {"detail": detail, "timeline": timeline,
                             "seed": {"puuid": puuid, "tier": tier,
                                      "participantId": seed["participantId"]}})
            n = saved_count()
            print(f"    저장 {mid} ({seed['championName']}) 누적 {n}/{target} · API 호출 {api.calls}")
        start += len(ids)
        if len(ids) < 100:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="26.20", help="표기 패치 (예: 26.20)")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--platform", default="kr")
    ap.add_argument("--tiers", default="challenger,grandmaster")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    prefix = api_version_prefix(args.patch)
    api = RiotAPI(platform=args.platform)
    state = load_state()
    print(f"패치 {args.patch} (API gameVersion {prefix}*) · 목표 {args.target}판 · 현재 {saved_count()}판")

    t0 = time.time()
    try:
        for tier in args.tiers.split(","):
            entries = sorted(api.apex_entries(tier), key=lambda e: -e.get("leaguePoints", 0))
            print(f"[{tier}] {len(entries)}명")
            for i, e in enumerate(entries):
                if saved_count() >= args.target:
                    break
                puuid = e.get("puuid")
                if not puuid or puuid in state["players_done"]:
                    continue
                print(f"  ({i + 1}/{len(entries)}) {e.get('leaguePoints')}LP")
                collect_player(api, puuid, tier, prefix, state, args.target)
                state["players_done"].append(puuid)
                save_json(STATE, state)
    except KeyExpired as e:
        save_json(STATE, state)
        sys.exit(str(e))
    except KeyboardInterrupt:
        save_json(STATE, state)
        sys.exit("\n중단. 다시 실행하면 이어서 받습니다.")
    save_json(STATE, state)
    print(f"완료: {saved_count()}판 · API 호출 {api.calls}회 · {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
