"""새 패치 Data Dragon 파일 받기(kk PC에서 실행, 클라우드는 Data Dragon 접속이 막혀 있음).

    python tools/fetch_patch_data.py            → data/ddragon/<최신 버전>/{en_US,ko_KR}/{item,championFull,runesReforged,summoner}.json
    이어서 표 만들기:
    python tools/build_item_vectors.py data/ddragon/<버전>/en_US/item.json --prev rules/item_vectors_16.20.1.json
    python tools/build_champ_kits.py data/ddragon/<버전>/en_US/championFull.json
"""
import json
import os
import sys
import urllib.request

BASE = "https://ddragon.leagueoflegends.com"
FILES = ["item.json", "championFull.json", "runesReforged.json", "summoner.json"]


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else json.loads(get(f"{BASE}/api/versions.json"))[0]
    for lang in ("en_US", "ko_KR"):
        d = os.path.join("data", "ddragon", ver, lang)
        os.makedirs(d, exist_ok=True)
        for f in FILES:
            p = os.path.join(d, f)
            if not os.path.exists(p):
                open(p, "wb").write(get(f"{BASE}/cdn/{ver}/data/{lang}/{f}"))
            print(p)
    print("버전", ver)


if __name__ == "__main__":
    main()
