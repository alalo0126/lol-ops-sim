"""스킬 세부값 원본 받기(kk PC에서 실행. 클라우드·PC 리눅스 VM은 이 주소들이 막혀 있음).

    python tools/fetch_skill_data.py
    → data/skills/meraki_champions.json   (Meraki Analytics: 전 챔피언 스킬 시전 시간·투사체 속도·폭·타게팅·계수, 레벨당 능력치)
      data/skills/meraki_items.json
      data/skills/cdragon/<챔피언>.bin.json (CommunityDragon 게임 파일 원본, 교차 검수용 일부 챔피언)
      data/skills/cdragon_version.txt
      data/skills/fetch_log.txt

받기만 한다. 표 만들기와 검수는 Claude가 이 파일들을 읽어서 한다.
"""
import json
import os
import sys
import time
import urllib.request

OUT = os.path.join("data", "skills")
MERAKI = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US"
CDRAGON = "https://raw.communitydragon.org/latest"
# 교차 검수용: 미드 1:1을 먼저 만들 챔피언 + 정글 몇 명
SAMPLE = ["ahri", "syndra", "orianna", "azir", "viktor", "sylas", "yone", "yasuo",
          "taliyah", "hwei", "leesin", "vi", "jarvaniv", "nidalee"]


def get(url, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "ops-sim/1.0"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    raise err


def main():
    os.makedirs(os.path.join(OUT, "cdragon"), exist_ok=True)
    log = open(os.path.join(OUT, "fetch_log.txt"), "w", encoding="utf-8")

    def note(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    for name, url in [("meraki_champions.json", f"{MERAKI}/champions.json"),
                      ("meraki_items.json", f"{MERAKI}/items.json")]:
        try:
            b = get(url)
            open(os.path.join(OUT, name), "wb").write(b)
            note(f"OK {name} {len(b)} bytes")
        except Exception as e:  # noqa: BLE001
            note(f"실패 {name}: {e}")
    try:
        meta = get(f"{CDRAGON}/content-metadata.json")
        open(os.path.join(OUT, "cdragon_version.txt"), "wb").write(meta)
        note("OK cdragon version " + meta.decode("utf-8", "replace")[:200])
    except Exception as e:  # noqa: BLE001
        note(f"실패 cdragon version: {e}")
    names = sys.argv[1:] or SAMPLE
    for c in names:
        try:
            b = get(f"{CDRAGON}/game/data/characters/{c}/{c}.bin.json")
            open(os.path.join(OUT, "cdragon", f"{c}.bin.json"), "wb").write(b)
            note(f"OK cdragon {c} {len(b)} bytes")
        except Exception as e:  # noqa: BLE001
            note(f"실패 cdragon {c}: {e}")
    note("끝")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
