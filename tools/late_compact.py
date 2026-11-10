"""데스티니 카드 후반 검수 입력: 1단계 원본(data/raw)을 경기 끝까지 줄인다(stage6_compact는 21분까지).

    python tools\\late_compact.py      :: data\\raw → data\\late_compact.json.gz

stage6_compact와 같은 형식(분 프레임 전부, 이벤트 전부)에 참가자 소환사 주문 2개를 덧붙인다.
"""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage6_compact as s6  # noqa: E402

ROOT = s6.ROOT


def main(prefix="16.20."):
    s6.UNTIL_MS = 10 ** 9
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not d.get("detail") or not d.get("timeline") or not d["detail"]["info"]["gameVersion"].startswith(prefix):
            continue
        frames = d["timeline"]["info"]["frames"]
        m = s6.compact(d)
        full = {"timeline": {"info": {"frames": [dict(fr, events=[]) for fr in frames]}}, "detail": d["detail"]}
        m["frames"] = [s6.compact({"detail": d["detail"], "timeline": {"info": {"frames": [fr]}}})["frames"][0]
                       for fr in full["timeline"]["info"]["frames"]]
        sp = {p["participantId"]: [p.get("summoner1Id"), p.get("summoner2Id")] for p in d["detail"]["info"]["participants"]}
        for p in m["parts"]:
            p.append(sp.get(p[0]))
        m["match"] = os.path.basename(path)[:-5]
        out.append(m)
    dst = os.path.join(ROOT, "data", "late_compact.json.gz")
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(out)}판 → {dst} ({os.path.getsize(dst) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
