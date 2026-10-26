"""코치용 입력: 내 경기 캐시(LLM 코치의 match_cache, {"detail", "timeline"} 형식)에서 내가 정글을 간 솔로랭크를
6단계와 같은 줄인 형식으로 모은다. sim/review.py가 이 파일을 읽는다.

    python tools\\my_compact.py D:\\AI\\lol\\match_cache APDO      :: → data\\my_compact.json.gz

경기마다 stage6_compact.compact() 결과에 "match"(경기 번호), "me"(내 participantId)를 붙인다.
21분 미만 경기는 뺀다(6단계와 같이 20분 예측 승률로 비교하므로).
"""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage6_compact import ROOT, compact  # noqa: E402

SOLO_QUEUE = 420


def main(cache_dir, game_name, role="JUNGLE", dst=None):
    out = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "KR_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        info = d["detail"]["info"]
        me = next((p for p in info["participants"] if p.get("riotIdGameName") == game_name), None)
        if me is None or me.get("teamPosition") != role or info.get("queueId") != SOLO_QUEUE:
            continue
        if info.get("gameDuration", 0) < 21 * 60:
            continue
        m = compact(d)
        m["match"] = os.path.basename(path)[:-5]
        m["me"] = me["participantId"]
        out.append(m)
    dst = dst or os.path.join(ROOT, "data", "my_compact.json.gz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(out)}판 → {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
