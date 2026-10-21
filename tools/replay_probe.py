"""롤 클라이언트가 지난 빌드 경기의 .rofl을 받아 주는지 시험한다 (kk PC, 클라이언트 로그인 상태).

  python -m tools.replay_probe KR_8424846201 KR_8410898414

경기마다 메타데이터 상태 → 다운로드 요청 → 최종 상태와 리플레이 폴더 안 파일 크기를 출력한다.
화면의 다운로드 버튼과 같은 요청이라 게임이나 계정 설정은 바꾸지 않는다.
"""
import argparse
import glob
import json
import os
import subprocess
import time

from tools.replay_track import DEFAULT_GAME_DIR, LCU, LCU_BODY, find_game_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="+")
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--wait", type=float, default=120)
    args = ap.parse_args()

    game_dir = find_game_dir(args.game_dir)
    print("설치 경로:", game_dir)
    exe = os.path.join(game_dir, "Game", "League of Legends.exe")
    if os.path.exists(exe):
        ver = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Item '{exe}').VersionInfo.FileVersion"],
                             capture_output=True, text=True).stdout.strip()
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(os.path.dirname(exe)) for f in fs)
        print(f"게임 실행 파일 버전 {ver}, Game 폴더 {size / 1e9:.1f}GB")
    lcu = LCU(game_dir)
    conf = lcu.get("/lol-replays/v1/configuration")
    print("configuration:", json.dumps(conf, ensure_ascii=False))
    rofl_dir = lcu.get("/lol-replays/v1/rofls/path")
    print("rofl 폴더:", rofl_dir)

    for mid in args.matches:
        gid = mid.split("_")[-1]
        print(f"\n== {mid}")
        before = lcu.get(f"/lol-replays/v1/metadata/{gid}")
        print("  처음 상태:", json.dumps(before, ensure_ascii=False))
        for path in (f"/lol-replays/v1/rofls/{gid}/download/graceful", f"/lol-replays/v1/rofls/{gid}/download"):
            r = lcu.post(path, LCU_BODY)
            print(f"  POST {path.rsplit('/', 1)[-1]}: HTTP {r.status_code} {r.text[:200]}")
            t0 = time.time()
            meta = before
            while time.time() - t0 < args.wait:
                time.sleep(3)
                meta = lcu.get(f"/lol-replays/v1/metadata/{gid}") or {}
                if meta.get("state") not in ("checking", "downloading", "found"):
                    break
            print("  이후 상태:", json.dumps(meta, ensure_ascii=False))
            files = glob.glob(os.path.join(rofl_dir or "", f"*{gid}*.rofl")) if rofl_dir else []
            for f in files:
                print(f"  파일 {f} {os.path.getsize(f) / 1e6:.1f}MB")
            if files:
                break


if __name__ == "__main__":
    main()
