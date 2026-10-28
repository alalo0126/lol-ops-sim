"""리플레이 화면에서 미니맵 와드가 보이는지 확인용 캡처 (kk PC 전용, 롤 클라이언트가 켜져 있어야 함).

리플레이 API에는 와드 위치가 없다(playerlist는 wardScore만). 그래서 시야를 양 팀 다 켜고(fogOfWar 끔)
정해진 시각마다 멈춘 화면 전체를 PNG로 저장한다. 클라우드에서 미니맵 부분을 잘라 와드 아이콘을 읽을 수 있는지 본다.

    python -m tools.ward_probe KR_8427110811                 # 리플레이를 받아 열고 캡처
    python -m tools.ward_probe KR_8427110811 --no-launch     # 이미 열린 리플레이에 붙기만

저장: data/ward_probe/<경기>_<초>.png, data/ward_probe/<경기>_render.json (render 설정 값)
Pillow가 필요하다: pip install pillow
"""
import argparse
import json
import os
import sys
import time

from tools.replay_track import LCU, Replay, download_and_watch, find_game_dir, same_game, DEFAULT_GAME_DIR, ROOT

OUT = os.path.join(ROOT, "data", "ward_probe")
TIMES = [90, 150, 240, 360, 480, 600, 720, 900, 1080, 1200]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match")
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    try:
        from PIL import ImageGrab
    except ImportError:
        sys.exit("Pillow가 없습니다. 먼저 pip install pillow 를 실행하세요.")
    if not args.no_launch:
        download_and_watch(LCU(find_game_dir(args.game_dir)), args.match.split("_")[-1])
    rp = Replay()
    pb = rp.wait_ready()
    if not same_game(rp, args.match):
        sys.exit(f"재생 중인 리플레이가 {args.match}가 아닙니다. 열린 리플레이 창을 닫고 다시 실행하세요.")
    os.makedirs(OUT, exist_ok=True)
    rp.post("/replay/render", {"interfaceAll": True, "interfaceMinimap": True, "fogOfWar": False})
    with open(os.path.join(OUT, f"{args.match}_render.json"), "w", encoding="utf-8") as f:
        json.dump(rp.get("/replay/render"), f, ensure_ascii=False, indent=1)
    for t in TIMES:
        if t > pb["length"]:
            break
        rp.post("/replay/playback", {"time": t, "paused": True})
        time.sleep(3)   # 이동 뒤 화면이 그려질 때까지
        path = os.path.join(OUT, f"{args.match}_{t}.png")
        ImageGrab.grab().save(path)
        print("저장", path)
    print(f"끝. {OUT} 폴더의 PNG를 확인하세요.")


if __name__ == "__main__":
    main()
