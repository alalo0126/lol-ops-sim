"""리플레이 재생 중 체력을 읽을 수 있는 API 칸이 있는지 확인(kk PC).

    1) 롤 클라이언트에서 아무 리플레이나 틀고(게임 화면이 뜬 뒤, 2분 이후 장면이면 좋음)
    2) run_hp_probe.bat 더블클릭 → data/replay_tracks/hp_probe/*.json + hp_probe_result.txt

로컬 게임 창 API(https://127.0.0.1:2999, game.cfg EnableReplayApi=1 이미 켜 둠)만 읽는다. 바깥 서버로는 아무것도 안 보냄.
"""
import json
import os
import time

import requests
import urllib3

urllib3.disable_warnings()
BASE = "https://127.0.0.1:2999"
PATHS = ["/liveclientdata/allgamedata", "/liveclientdata/playerlist", "/liveclientdata/activeplayer",
         "/liveclientdata/gamestats", "/liveclientdata/eventdata", "/replay/game", "/replay/playback",
         "/replay/render", "/replay/sequence", "/replay/recording", "/swagger/v2/swagger.json", "/swagger/v3/openapi.json"]
KEYS = ("health", "currenthealth", "maxhealth", "hp", "resourcevalue", "mana")


def find_keys(o, path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}"
            if k.lower() in KEYS:
                hits.append((p, v if not isinstance(v, (dict, list)) else type(v).__name__))
            hits += find_keys(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o[:10]):
            hits += find_keys(v, f"{path}[{i}]")
    return hits


def main():
    out = os.path.join("data", "replay_tracks", "hp_probe")
    os.makedirs(out, exist_ok=True)
    lines = []
    for rnd in range(2):          # 5초 간격 두 번: 값이 바뀌는지도 봄
        for p in PATHS:
            try:
                r = requests.get(BASE + p, verify=False, timeout=5)
                body = r.json() if "json" in r.headers.get("content-type", "") else r.text[:2000]
            except Exception as e:  # noqa: BLE001
                lines.append(f"{rnd} {p}: 실패 {e!r}")
                continue
            name = p.strip("/").replace("/", "_")
            json.dump(body, open(os.path.join(out, f"{name}_{rnd}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            hits = find_keys(body)
            lines.append(f"{rnd} {p}: HTTP {r.status_code}, 체력류 칸 {len(hits)}개 " + "; ".join(f"{a}={b}" for a, b in hits[:12]))
        time.sleep(5)
    txt = "\n".join(lines)
    open(os.path.join(out, "hp_probe_result.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    print("\n끝. data\\replay_tracks\\hp_probe\\hp_probe_result.txt")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
