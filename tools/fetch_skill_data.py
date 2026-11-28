"""스킬 세부값 원본 받기(kk PC에서 실행. 클라우드·PC 리눅스 VM은 이 주소들이 막혀 있음).

    python tools/fetch_skill_data.py
    → data/skills/meraki_champions.json   (Meraki Analytics: 전 챔피언 스킬 시전 시간·투사체 속도·폭·타게팅·계수, 레벨당 능력치)
      data/skills/meraki_items.json
      data/skills/cdragon/<챔피언>.bin.json (CommunityDragon 게임 파일 원본, 기본은 전 챔피언, 이미 받은 건 건너뜀)
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

    for name, url in [] if os.path.exists(os.path.join(OUT, "meraki_items.json")) else [("meraki_champions.json", f"{MERAKI}/champions.json"),
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
    # 룬·아이템 효과 원본(효과 표 effects_<패치>.json용). 경로가 틀리면 실패로만 남김
    for name, path in [("cdragon_items.bin.json", "game/items.cdtb.bin.json"),
                       ("cdragon_perks.json", "plugins/rcp-be-lol-game-data/global/default/v1/perks.json"),
                       ("cdragon_perkstyles.json", "plugins/rcp-be-lol-game-data/global/default/v1/perkstyles.json"),
                       ("cdragon_items.json", "plugins/rcp-be-lol-game-data/global/default/v1/items.json")]:
        if os.path.exists(os.path.join(OUT, name)):
            continue
        try:
            b = get(f"{CDRAGON}/{path}")
            open(os.path.join(OUT, name), "wb").write(b)
            note(f"OK {name} {len(b)} bytes")
        except Exception as e:  # noqa: BLE001
            note(f"실패 {name}: {e}")
    names = sys.argv[1:]
    if names == ["--all"] or not names:
        # 최신 Data Dragon 챔피언 목록(fetch_patch_data.py가 받아 둔 것) → 전 챔피언
        vers = sorted(os.listdir(os.path.join("data", "ddragon")), key=lambda v: [int(x) for x in v.split(".")])
        full = os.path.join("data", "ddragon", vers[-1], "en_US", "championFull.json")
        names = sorted(c.lower() for c in json.load(open(full, encoding="utf-8"))["data"]) if os.path.exists(full) else SAMPLE
    for c in names:
        p = os.path.join(OUT, "cdragon", f"{c}.bin.json")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
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
