"""리플레이 파일(.rofl) 끝에 붙은 메타데이터 JSON을 읽고 Riot match-v5와 맞대 본다 (kk PC).

  python -m tools.rofl_meta                       # replays 폴더 전체를 match-v5와 대조
  python -m tools.rofl_meta D:\\AI\\lol\\replays\\KR-8427110811.rofl --show

26.21 형식(2026-10-27 확인): 헤더 'RIOT\\x02\\x00' + 빌드 문자열, 페이로드는 zstd 프레임.
메타데이터는 파일 맨 끝에 있고 마지막 4바이트(little-endian)가 JSON 길이다.
키는 gameLength, lastGameChunkId, lastKeyFrameId, statsJson 넷뿐이고 statsJson은 10명의 경기 끝 누적 통계다.
페이로드(청크·키프레임)는 읽지 않는다.
"""
import argparse
import glob
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYS = os.path.join(os.path.dirname(ROOT), "replays")
RAW = os.path.join(ROOT, "data", "replay_raw")

# statsJson 키 → match-v5 participant 키. 정수 값만, 골드는 반올림 때문에 ±GOLD_TOL 허용.
FIELDS = {
    "CHAMPIONS_KILLED": "kills",
    "NUM_DEATHS": "deaths",
    "ASSISTS": "assists",
    "MINIONS_KILLED": "totalMinionsKilled",
    "NEUTRAL_MINIONS_KILLED": "neutralMinionsKilled",
    "WARD_PLACED": "wardsPlaced",
    "LEVEL": "champLevel",
    "GOLD_EARNED": "goldEarned",
}
GOLD_TOL = 2


def build(path):
    """파일 앞 빌드 문자열 (예: '16.21.829.8620')."""
    with open(path, "rb") as f:
        head = f.read(64)
    if head[:4] != b"RIOT":
        raise ValueError(f"{path}: .rofl 아님")
    s = head[16:48].split(b"\x00")[0].split(b"\x01")[0]
    return s.decode("ascii", errors="replace")


def read_meta(path):
    """메타데이터 dict. statsJson은 풀어서 'stats'(10명 list)로 붙인다."""
    with open(path, "rb") as f:
        f.seek(-4, 2)
        n = struct.unpack("<I", f.read(4))[0]
        f.seek(-4 - n, 2)
        meta = json.loads(f.read(n))
    meta["stats"] = json.loads(meta.get("statsJson", "[]"))
    return meta


def match_id(path):
    return os.path.basename(path)[:-5].replace("-", "_")


def compare(meta, detail):
    """(맞은 사람 수, [(이름, 키, rofl 값, match-v5 값)]). 사람은 Riot ID 이름으로 맞춘다(PUUID는 키마다 다름)."""
    parts = {p["riotIdGameName"]: p for p in detail["info"]["participants"]}
    ok, diffs = 0, []
    for r in meta["stats"]:
        p = parts.get(r.get("RIOT_ID_GAME_NAME"))
        if p is None:
            diffs.append((r.get("RIOT_ID_GAME_NAME"), "이름", None, None))
            continue
        bad = []
        for a, b in FIELDS.items():
            va, vb = int(r[a]), p[b]
            tol = GOLD_TOL if a == "GOLD_EARNED" else 0
            if abs(va - vb) > tol:
                bad.append((r["RIOT_ID_GAME_NAME"], a, va, vb))
        ok += not bad
        diffs += bad
    return ok, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--show", action="store_true", help="첫 파일의 키와 1번 선수 통계 출력")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(os.path.join(REPLAYS, "*.rofl")))
    if not files:
        sys.exit(f"{REPLAYS}에 .rofl 없음")
    if args.show:
        m = read_meta(files[0])
        print(build(files[0]), {k: v for k, v in m.items() if k not in ("statsJson", "stats")})
        print(f"선수 {len(m['stats'])}명, 항목 {len(m['stats'][0])}개")
        for k in sorted(m["stats"][0]):
            print(f"  {k} = {m['stats'][0][k]}")
        return
    games = people = good = 0
    for path in files:
        mid = match_id(path)
        raw = os.path.join(RAW, f"{mid}.json")
        if not os.path.exists(raw):
            continue
        with open(raw, encoding="utf-8") as f:
            detail = json.load(f)["detail"]
        m = read_meta(path)
        ok, diffs = compare(m, detail)
        games += 1
        people += len(m["stats"])
        good += ok
        for d in diffs:
            print(f"{mid} {build(path)} 불일치: {d}")
    print(f"{games}판 {people}명 중 {good}명 일치 (골드 ±{GOLD_TOL} 허용)")


if __name__ == "__main__":
    main()
