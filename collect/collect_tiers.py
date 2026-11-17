"""티어별 기준선: 그 티어에 오래 머문 선수의 솔로랭크 경기(모든 포지션)를 모은다.

Riot은 티어 이력을 주지 않으므로 league-v4 목록의 값으로 대신 거른다.
  - 이번 시즌 판수(wins+losses) >= --min-games
  - veteran=true (그 리그에서 오래 뛴 선수), freshBlood=false (최근 올라온 선수 아님)
선수 편중을 막기 위해 시드 1명당 최대 --per-seed 판만 받는다.
받은 경기는 data/tiers/<TIER>/raw/<경기>.json.gz (경기 정보+타임라인+시드 정보).

  python collect/collect_tiers.py --probe        # 티어별 시드 기준 분포만 보기(경기 안 받음)
  python collect/collect_tiers.py --target 1000  # 아이언→브론즈→…→마스터 순서

끊기거나 키가 만료돼도 같은 명령으로 다시 실행하면 이어서 받는다.
"""
import argparse
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from riot_api import RiotAPI, KeyExpired  # noqa: E402
from collect_matches import version_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "tiers")
DIVISIONS = ["I", "II", "III", "IV"]
ORDER = "IRON,BRONZE,SILVER,GOLD,PLATINUM,EMERALD,DIAMOND,MASTER"
APEX = {"MASTER", "GRANDMASTER", "CHALLENGER"}


def save_json(path, obj, gz=False):
    tmp = path + ".tmp"
    opener = gzip.open if gz else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def raw_dir(tier):
    d = os.path.join(OUT, tier, "raw")
    os.makedirs(d, exist_ok=True)
    return d


def saved(tier):
    return sum(1 for n in os.listdir(raw_dir(tier)) if n.endswith(".json.gz"))


def all_saved_ids():
    ids = set()
    if os.path.isdir(OUT):
        for t in os.listdir(OUT):
            d = os.path.join(OUT, t, "raw")
            if os.path.isdir(d):
                ids.update(n[:-8] for n in os.listdir(d) if n.endswith(".json.gz"))
    return ids


def fetch_seeds(api, tier, max_pages):
    """구간별 목록을 받아 data/tiers/<TIER>/entries.json에 저장(재실행 시 재사용)."""
    path = os.path.join(OUT, tier, "entries.json")
    cached = load_json(path, None)
    if cached is not None:
        return cached
    entries = []
    if tier in APEX:
        rows = api.apex_entries(tier.lower())
        entries = [{"puuid": r.get("puuid"), "division": "I", "lp": r.get("leaguePoints"),
                    "wins": r.get("wins", 0), "losses": r.get("losses", 0),
                    "veteran": bool(r.get("veteran")), "freshBlood": bool(r.get("freshBlood")),
                    "inactive": bool(r.get("inactive"))} for r in rows if r.get("puuid")]
        print(f"  {tier}: {len(entries)}명")
    for div in ([] if tier in APEX else DIVISIONS):
        for page in range(1, max_pages + 1):
            rows = api.tier_entries(tier, div, page)
            if not rows:
                break
            entries.extend({"puuid": r.get("puuid"), "division": div,
                            "lp": r.get("leaguePoints"), "wins": r.get("wins", 0),
                            "losses": r.get("losses", 0), "veteran": bool(r.get("veteran")),
                            "freshBlood": bool(r.get("freshBlood")),
                            "inactive": bool(r.get("inactive"))} for r in rows if r.get("puuid"))
        print(f"  {tier} {div}: 누적 {len(entries)}명")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, entries)
    return entries


def is_long_stay(e, min_games, need_veteran):
    if e["inactive"] or e["freshBlood"]:
        return False
    if need_veteran and not e["veteran"]:
        return False
    return e["wins"] + e["losses"] >= min_games


def probe(entries, tier):
    import statistics
    n = len(entries)
    games = sorted(e["wins"] + e["losses"] for e in entries)
    q = lambda p: games[min(n - 1, int(p * n))] if n else 0  # noqa: E731
    vet = [e for e in entries if e["veteran"]]
    fresh = sum(e["freshBlood"] for e in entries)
    print(f"[{tier}] {n}명 · 판수 중앙 {q(0.5)} (p25 {q(0.25)}, p75 {q(0.75)}, p90 {q(0.9)})")
    print(f"  veteran {len(vet)}명 ({len(vet) / max(n, 1):.0%}), 그 판수 중앙 "
          f"{statistics.median([e['wins'] + e['losses'] for e in vet]) if vet else '-'}"
          f" · freshBlood {fresh}명 ({fresh / max(n, 1):.0%})")
    for g in (50, 100, 150, 200, 300):
        a = sum(is_long_stay(e, g, False) for e in entries)
        b = sum(is_long_stay(e, g, True) for e in entries)
        print(f"  판수>={g}: {a}명 · +veteran {b}명")


def collect_seed(api, tier, seed, prefix_key, state, per_seed, target, seen):
    rej = state["rejected"]
    got = 0
    ids = api.match_ids(seed["puuid"], start=0, count=20)
    for mid in ids:
        if got >= per_seed or saved(tier) >= target:
            return
        if mid in seen:
            continue
        if rej.get(mid) == "older_patch":
            return
        if mid in rej:
            continue
        detail = api.match(mid)
        if not detail:
            rej[mid] = "not_found"
            continue
        info = detail["info"]
        vkey = version_key(info["gameVersion"])
        if vkey < prefix_key:
            rej[mid] = "older_patch"
            return
        if vkey > prefix_key:
            rej[mid] = "newer_patch"
            continue
        if any(p.get("gameEndedInEarlySurrender") for p in info["participants"]):
            rej[mid] = "remake"
            continue
        me = next((p for p in info["participants"] if p["puuid"] == seed["puuid"]), None)
        if not me:
            rej[mid] = "seed_missing"
            continue
        timeline = api.timeline(mid)
        if not timeline:
            rej[mid] = "no_timeline"
            continue
        save_json(os.path.join(raw_dir(tier), f"{mid}.json.gz"),
                  {"detail": detail, "timeline": timeline,
                   "seed": dict(seed, tier=tier, participantId=me["participantId"],
                                teamPosition=me.get("teamPosition"))}, gz=True)
        seen.add(mid)
        got += 1
        print(f"    저장 {mid} ({me.get('teamPosition')} {me['championName']}) "
              f"{tier} {saved(tier)}/{target} · API {api.calls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default=ORDER,
                    help="쉼표 구분, 순서대로 채움 (기본 아이언→마스터)")
    ap.add_argument("--target", type=int, default=1000, help="티어당 경기 수")
    ap.add_argument("--patch", default="", help="예: 26.22. 비우면 첫 시드의 최신 경기 패치")
    ap.add_argument("--min-games", type=int, default=150)
    ap.add_argument("--no-veteran", action="store_true", help="veteran 조건 끄기")
    ap.add_argument("--per-seed", type=int, default=3)
    ap.add_argument("--max-pages", type=int, default=8, help="구간당 목록 페이지(205명/쪽)")
    ap.add_argument("--platform", default="kr")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    api = RiotAPI(platform=args.platform)
    state_path = os.path.join(OUT, "state.json")
    os.makedirs(OUT, exist_ok=True)
    state = load_json(state_path, {"seeds_done": {}, "rejected": {}, "patch": ""})
    seen = all_saved_ids()
    t0 = time.time()
    try:
        for tier in [t.strip().upper() for t in args.tiers.split(",") if t.strip()]:
            entries = fetch_seeds(api, tier, args.max_pages)
            if args.probe:
                probe(entries, tier)
                continue
            seeds = [e for e in entries if is_long_stay(e, args.min_games, not args.no_veteran)]
            random.Random(0).shuffle(seeds)
            print(f"[{tier}] 시드 후보 {len(seeds)}/{len(entries)}명 · 현재 {saved(tier)}/{args.target}판")
            if args.target > len(seeds) * args.per_seed:
                print(f"  주의: 시드×{args.per_seed}판 < 목표. --min-games를 낮추거나 --max-pages를 늘리세요.")
            patch = args.patch or state["patch"]
            if not patch and seeds:
                ids = api.match_ids(seeds[0]["puuid"], count=1)
                gv = api.match(ids[0])["info"]["gameVersion"] if ids else ""
                patch = f"{int(gv.split('.')[0]) + 10}.{gv.split('.')[1]}"
                print(f"  패치 자동: {patch} (gameVersion {gv})")
            state["patch"] = patch
            year, minor = patch.split(".")
            prefix_key = (int(year) - 10, int(minor))
            done = set(state["seeds_done"].get(tier, []))
            for i, s in enumerate(seeds):
                if saved(tier) >= args.target:
                    break
                if s["puuid"] in done:
                    continue
                print(f"  ({i + 1}/{len(seeds)}) {tier} {s['division']} {s['lp']}LP "
                      f"{s['wins'] + s['losses']}판")
                collect_seed(api, tier, s, prefix_key, state, args.per_seed, args.target, seen)
                state["seeds_done"].setdefault(tier, []).append(s["puuid"])
                done.add(s["puuid"])
                save_json(state_path, state)
            print(f"[{tier}] {saved(tier)}판")
    except KeyExpired as e:
        save_json(state_path, state)
        sys.exit(str(e))
    except KeyboardInterrupt:
        save_json(state_path, state)
        sys.exit("\n중단. 다시 실행하면 이어서 받습니다.")
    save_json(state_path, state)
    print(f"완료 · API 호출 {api.calls}회 · {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
