"""리플레이를 배속 재생하면서 미니맵만 잘라 게임 1초마다 저장한다 (kk PC 전용, 롤 클라이언트가 켜져 있어야 함).

와드 위치는 리플레이 API·타임라인 어디에도 없어서(타임라인 WARD_PLACED는 시각·설치자·종류뿐) 미니맵 아이콘으로 읽는다.
와드는 설치자 챔피언 초상화 밑에 깔리므로 한 장이 아니라 설치 뒤 여러 초를 이어 봐야 한다
→ 1초 간격 미니맵을 모두 남기고, 판독(설치 시각 전후 차이·자리에 남는 아이콘)은 따로 한다.

    python -m tools.minimap_record KR_8427110811                # 리플레이 받아 열고 20분까지 기록
    python -m tools.minimap_record KR_8427110811 --no-launch    # 이미 열린 리플레이에 붙기만

저장: data/minimap/<경기>/<초>.png (미니맵 255x255, 원본 크기), data/minimap/<경기>/index.csv (파일, 재생 API 시각)
미니맵 위치 기본값은 1920x1080 기본 UI 크기 실측(--box로 바꿈). Pillow 필요.
"""
import argparse
import csv
import os
import sys
import time

from tools.replay_track import LCU, Replay, download_and_watch, find_game_dir, same_game, DEFAULT_GAME_DIR, ROOT

OUT = os.path.join(ROOT, "data", "minimap")
BOX = (1654, 812, 1909, 1067)   # 1920x1080 캡처에서 미니맵 테두리 포함 255x255 (kk PC 실측)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match")
    ap.add_argument("--until", type=float, default=1200)
    ap.add_argument("--speed", type=float, default=2, help="재생 배속 (기본 2, 1초 간격을 놓치지 않을 만큼)")
    ap.add_argument("--box", default=",".join(map(str, BOX)), help="미니맵 영역 x0,y0,x1,y1")
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    try:
        from PIL import ImageGrab
    except ImportError:
        sys.exit("Pillow가 없습니다. 먼저 pip install pillow 를 실행하세요.")
    box = tuple(int(v) for v in args.box.split(","))
    if not args.no_launch:
        download_and_watch(LCU(find_game_dir(args.game_dir)), args.match.split("_")[-1])
    rp = Replay()
    pb = rp.wait_ready()
    if not same_game(rp, args.match):
        sys.exit(f"재생 중인 리플레이가 {args.match}가 아닙니다. 열린 리플레이 창을 닫고 다시 실행하세요.")
    out = os.path.join(OUT, args.match)
    os.makedirs(out, exist_ok=True)
    rp.post("/replay/render", {"interfaceAll": True, "interfaceMinimap": True, "fogOfWar": False})
    rp.post("/replay/playback", {"time": 0, "paused": True})
    time.sleep(3)
    rp.post("/replay/playback", {"paused": False, "speed": args.speed})
    until = min(args.until, pb["length"] - 1)
    rows, next_s = [], 0
    while True:
        t = rp.get("/replay/playback").get("time", 0.0)
        if t >= until:
            break
        if t >= next_s:
            img = ImageGrab.grab(bbox=box)
            t2 = rp.get("/replay/playback").get("time", t)   # 캡처 전후 시각 평균을 그 장의 시각으로
            s = int(t)
            name = f"{s:04d}.png"
            img.save(os.path.join(out, name))
            rows.append((name, round((t + t2) / 2, 2)))
            next_s = s + 1
            if s % 60 == 0:
                print(f"  {s // 60}:00", end="\r")
        time.sleep(0.05)
    rp.post("/replay/playback", {"paused": True})
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "time"])
        w.writerows(rows)
    print(f"\n저장 {out} ({len(rows)}장)")


if __name__ == "__main__":
    main()
