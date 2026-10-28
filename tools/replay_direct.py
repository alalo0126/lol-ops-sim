"""롤 클라이언트를 거치지 않고 게임 exe로 .rofl을 직접 여는지 시험하고, 되면 이 빌드의 Game 폴더를 보관한다 (kk PC).

다음 패치가 나오면 클라이언트는 옛 빌드 리플레이를 열지 않는다. 그 빌드의 Game 폴더와 .rofl이 남아 있고 exe로 직접 열린다면
(ReplayBook 방식) 패치 뒤에도 그 경기들을 계속 기록할 수 있다. 뱅가드 아래에서 되는지는 아직 모른다 → 지금 같은 빌드일 때 시험한다.

  python -m tools.replay_direct --test                 # 받아 둔 .rofl 하나를 exe로 열어 리플레이 API가 뜨는지
  python -m tools.replay_direct --test --backup        # 열리면 Game 폴더를 data/../game_<빌드>로 복사(공간이 넉넉할 때만)
  python -m tools.replay_direct --play KR_8427110811 --game-copy D:\\AI\\lol\\game_16.21.829   # 보관한 폴더로 열기

결과는 data/replay_tracks/direct_launch.txt 에 남긴다.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.replay_track import DEFAULT_GAME_DIR, LCU, OUT, Replay, find_game_dir  # noqa: E402

EXE = "League of Legends.exe"
RESULT = os.path.join(OUT, "direct_launch.txt")


def note(msg):
    print(msg)
    os.makedirs(OUT, exist_ok=True)
    with open(RESULT, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def rofl_dir(lcu):
    p = (lcu.get("/lol-replays/v1/rofls/path") or "") if lcu else ""
    if isinstance(p, str) and os.path.isdir(p):
        return p
    docs = os.path.join(os.path.expanduser("~"), "OneDrive", "문서", "League of Legends", "Replays")
    return docs if os.path.isdir(docs) else os.path.join(os.path.expanduser("~"), "Documents", "League of Legends", "Replays")


def launch(game_dir, rofl):
    """ReplayBook과 같은 인자로 게임 exe에 .rofl을 넘긴다."""
    exe_dir = os.path.join(game_dir, "Game")
    args = [os.path.join(exe_dir, EXE), rofl, f"-GameBaseDir={game_dir}", "-SkipRads", "-SkipBuild",
            "-EnableLNP", "-UseNewX3D=1", "-UseNewX3DFramebuffers=1", "-Locale=ko_KR"]
    return subprocess.Popen(args, cwd=exe_dir)


def try_open(game_dir, rofl, wait=150):
    subprocess.run(["taskkill", "/IM", EXE, "/F"], capture_output=True)
    time.sleep(3)
    proc = launch(game_dir, rofl)
    rp = Replay()
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:
            return False, f"exe가 바로 끝남(코드 {proc.returncode})"
        try:
            pb = rp.get("/replay/playback")
            if pb.get("length", 0) > 0:
                n = len(rp.players())
                return True, f"열림: 길이 {pb['length'] / 60:.1f}분, 선수 {n}명"
        except Exception:  # noqa: BLE001 - 창이 뜨기 전에는 연결 거부
            pass
        time.sleep(3)
    return False, f"{wait}초 안에 리플레이 API 응답 없음"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--play", help="경기 번호 (예: KR_8427110811)")
    ap.add_argument("--game-copy", help="보관한 Game 폴더 위치 (--play용)")
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    args = ap.parse_args()
    game_dir = find_game_dir(args.game_dir)
    try:
        lcu = LCU(game_dir)
    except SystemExit:
        lcu = None
    rdir = rofl_dir(lcu)

    if args.play:
        rofl = os.path.join(rdir, f"{args.play.replace('_', '-')}.rofl")
        ok, msg = try_open(args.game_copy or game_dir, rofl)
        note(f"play {args.play} ({args.game_copy or game_dir}): {msg}")
        return

    if args.test:
        rofls = sorted(glob.glob(os.path.join(rdir, "KR-*.rofl")), key=os.path.getmtime, reverse=True)
        if not rofls:
            note(f"시험할 .rofl이 없음 ({rdir})")
            return
        version = ((lcu.get("/lol-replays/v1/configuration") or {}).get("gameVersion", "?")) if lcu else "?"
        ok, msg = try_open(game_dir, rofls[0])
        subprocess.run(["taskkill", "/IM", EXE, "/F"], capture_output=True)
        note(f"직접 열기 시험 빌드 {version} {os.path.basename(rofls[0])}: {'성공' if ok else '실패'} - {msg}")
        if ok and args.backup:
            src = os.path.join(game_dir, "Game")
            dst = os.path.join(os.path.dirname(ROOT), f"game_{'.'.join(version.split('.')[:3])}", "Game")
            if os.path.exists(dst):
                note(f"이미 보관됨: {dst}")
                return
            size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(src) for f in fs)
            free = shutil.disk_usage(os.path.dirname(ROOT)).free
            if free < size + 30 * 1024 ** 3:
                note(f"공간 부족으로 보관 안 함: 필요 {size / 1024 ** 3:.1f}GB + 여유 30GB, 남은 {free / 1024 ** 3:.1f}GB")
                return
            note(f"Game 폴더 보관 시작 {size / 1024 ** 3:.1f}GB → {dst}")
            shutil.copytree(src, dst)
            note(f"보관 끝: {dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - 실패 이유를 결과 파일에 남긴다(클라우드에서 확인)
        import traceback
        note(f"오류: {e!r}\n{traceback.format_exc()}")
