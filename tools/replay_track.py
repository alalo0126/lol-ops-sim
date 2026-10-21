"""리플레이에서 정글러 위치를 초 단위로 뽑는다 (kk PC 전용, 롤 클라이언트가 켜져 있어야 함).

순서:
  1. 롤 클라이언트(LCU) 로컬 API로 리플레이를 받고 재생을 시작한다.
     클라이언트 전적의 '리플레이 다운로드/보기' 버튼과 같은 동작이다.
  2. 게임 창에 선수 단축키(블루 1~5, 레드 q~t)를 두 번 눌러 카메라를 정글러에 고정하고,
     Riot이 공개한 리플레이 API(https://127.0.0.1:2999/replay/...)로 배속 재생하면서 카메라 좌표를 기록한다.
  3. data/replay_tracks/<경기>_<챔피언>.csv 로 저장한다 (게임 시각 초, 맵 좌표 x·y = 타임라인 position과 같은 좌표계, 원본 카메라 좌표).

리플레이는 현재 클라이언트 빌드와 같은 빌드의 경기만 열린다 (작은 핫픽스도 다르면 안 열림).

처음 한 번은 리플레이 API를 켜야 한다:
  python -m tools.replay_track --enable-api      (game.cfg에 EnableReplayApi=1, 원본은 .bak으로 백업)

사용 예:
  python -m tools.replay_track KR_8424846201                  # data/replay_raw에 경기 JSON이 있으면 양 팀 정글러 자동
  python -m tools.replay_track KR_8424846201 --champion Nidalee --until 900 --speed 4
  python -m tools.replay_track KR_8424846201 --probe          # 좌표가 읽히는지만 확인
"""
import argparse
import base64
import csv
import json
import math
import os
import shutil
import sys
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 리플레이 검증용 경기(26.21~)는 1단계 1,000판(data/raw, 26.20)과 섞이지 않게 따로 둔다
REPLAY_RAW = os.path.join(ROOT, "data", "replay_raw")
OUT = os.path.join(ROOT, "data", "replay_tracks")
DEFAULT_GAME_DIR = r"C:\Riot Games\League of Legends"
REPLAY_API = "https://127.0.0.1:2999"
# LCU 다운로드·재생 요청 본문. 클라이언트 전적 화면 버튼이 보내는 값과 같다.
LCU_BODY = {"componentType": "replay-button_match-history"}


# ---------- 롤 클라이언트 (LCU) ----------

def client_from_process():
    """실행 중인 LeagueClientUx 명령줄에서 설치 경로·포트·비밀번호를 읽는다 (설치 경로를 몰라도 됨)."""
    import re
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='LeagueClientUx.exe'\" | "
             "ForEach-Object { $_.ExecutablePath; $_.CommandLine }"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    port = re.search(r'--app-port=(\d+)', out)
    token = re.search(r'--remoting-auth-token=([\w-]+)', out)
    # LeagueClientUx.exe는 설치 폴더 바로 아래에 있다
    exe = next((l.strip() for l in out.splitlines() if l.strip().lower().endswith("leagueclientux.exe")), None)
    if not (port and token):
        return None
    return port.group(1), token.group(1), os.path.dirname(exe) if exe else None


def find_game_dir(game_dir):
    if os.path.exists(os.path.join(game_dir, "lockfile")):
        return game_dir
    found = client_from_process()
    if found and found[2]:
        return found[2]
    return game_dir


class LCU:
    def __init__(self, game_dir):
        lock = os.path.join(game_dir, "lockfile")
        if os.path.exists(lock):
            with open(lock, encoding="utf-8") as f:
                _, _, port, password, proto = f.read().strip().split(":")
        else:
            found = client_from_process()
            if not found:
                sys.exit(f"롤 클라이언트를 찾지 못했습니다 ({lock} 없음, 실행 중인 클라이언트도 없음). "
                         "클라이언트를 켜고 로그인한 뒤 다시 실행하거나 --game-dir로 경로를 주세요.")
            port, password, install = found
            proto = "https"
            print(f"실행 중인 클라이언트에서 찾음: 설치 경로 {install}")
        self.base = f"{proto}://127.0.0.1:{port}"
        self.s = requests.Session()
        self.s.verify = False
        token = base64.b64encode(f"riot:{password}".encode()).decode()
        self.s.headers["Authorization"] = f"Basic {token}"

    def get(self, path):
        r = self.s.get(self.base + path, timeout=10)
        return r.json() if r.ok and r.content else None

    def post(self, path, body=None):
        return self.s.post(self.base + path, json=body if body is not None else {}, timeout=10)


def replay_state(lcu, game_id):
    meta = lcu.get(f"/lol-replays/v1/metadata/{game_id}") or {}
    return meta.get("state", "unknown"), meta


def download_and_watch(lcu, game_id, timeout=300):
    """리플레이를 받아 재생을 시작한다. 재생 불가 상태면 이유와 함께 종료한다."""
    conf = lcu.get("/lol-replays/v1/configuration") or {}
    print(f"클라이언트 리플레이 버전 {conf.get('gameVersion', '?')}")
    state, _ = replay_state(lcu, game_id)
    print(f"리플레이 상태: {state}")
    if state in ("incompatible", "lost", "missingOrExpired", "unsupported"):
        sys.exit(f"이 경기는 재생할 수 없습니다 ({state}). 대개 클라이언트 빌드가 경기 빌드와 다를 때입니다.")
    if state != "watch":
        lcu.post(f"/lol-replays/v1/rofls/{game_id}/download/graceful", LCU_BODY)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(2)
            state, meta = replay_state(lcu, game_id)
            pct = meta.get("downloadProgress")
            print(f"  다운로드 {state} {pct if pct is not None else ''}", end="\r")
            if state == "watch":
                break
            if state in ("incompatible", "lost", "missingOrExpired", "error"):
                sys.exit(f"\n다운로드 실패: {state}")
        print()
        if state != "watch":
            sys.exit("다운로드가 시간 안에 끝나지 않았습니다.")
    r = lcu.post(f"/lol-replays/v1/rofls/{game_id}/watch", LCU_BODY)
    if not r.ok:
        sys.exit(f"재생 요청 실패: HTTP {r.status_code} {r.text[:200]}")


# ---------- 리플레이 API (게임 창, 포트 2999) ----------

class Replay:
    def __init__(self):
        self.s = requests.Session()
        self.s.verify = False

    def get(self, path):
        r = self.s.get(REPLAY_API + path, timeout=5)
        r.raise_for_status()
        return r.json()

    def post(self, path, body):
        r = self.s.post(REPLAY_API + path, json=body, timeout=5)
        r.raise_for_status()
        return r.json()

    def wait_ready(self, timeout=240):
        """게임 창이 뜨고 리플레이가 로딩될 때까지 기다린다."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                pb = self.get("/replay/playback")
                if pb.get("length", 0) > 0:
                    return pb
            except requests.RequestException:
                pass
            time.sleep(2)
        sys.exit("리플레이 API 응답이 없습니다. --enable-api를 한 번 실행했는지, 게임 창이 떴는지 확인하세요.")

    def players(self):
        try:
            return self.get("/liveclientdata/playerlist")
        except requests.RequestException:
            return []


def enable_replay_api(game_dir):
    cfg = os.path.join(game_dir, "Config", "game.cfg")
    if not os.path.exists(cfg):
        sys.exit(f"{cfg} 가 없습니다. --game-dir를 확인하세요.")
    with open(cfg, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if any(l.strip().replace(" ", "") == "EnableReplayApi=1" for l in lines):
        print("이미 켜져 있습니다.")
        return
    shutil.copyfile(cfg, cfg + ".bak")
    lines = [l for l in lines if not l.strip().startswith("EnableReplayApi")]
    try:
        i = next(i for i, l in enumerate(lines) if l.strip() == "[General]")
        lines.insert(i + 1, "EnableReplayApi=1")
    except StopIteration:
        lines = ["[General]", "EnableReplayApi=1"] + lines
    with open(cfg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"켰습니다: {cfg} (원본 {cfg}.bak). 게임 창을 새로 띄울 때부터 적용됩니다.")


# ---------- 추적 ----------

def junglers_from_raw(match_id):
    path = os.path.join(REPLAY_RAW, f"{match_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        info = json.load(f)["detail"]["info"]
    return [p["championName"] for p in info["participants"] if p.get("teamPosition") == "JUNGLE"]


def find_player(players, champion):
    """playerlist에서 챔피언 영문 id로 선수를 찾는다 (한국어 클라이언트는 championName이 한글)."""
    key = champion.lower()
    for p in players:
        raw = (p.get("rawChampionName") or "").lower()
        if raw.endswith("_" + key) or (p.get("championName") or "").lower() == key:
            return p
    return None


# 리플레이 화면 단축키: 블루팀 1~5, 레드팀 q~t (탑·정글·미드·원딜·서폿 순). 두 번 누르면 계속 고정.
# 리플레이 API의 selectionName 고정은 자동 카메라(교전 따라가기)에 밀려서 키 입력을 쓴다.
SCAN_CODES = {"ORDER": [0x02, 0x03, 0x04, 0x05, 0x06], "CHAOS": [0x10, 0x11, 0x12, 0x13, 0x14]}
GAME_WINDOW = "League of Legends (TM) Client"


def lock_key(players, player):
    team = player.get("team")
    mates = [p for p in players if p.get("team") == team]
    return SCAN_CODES[team][mates.index(player)]


def press_twice(scan):
    """게임 창을 앞으로 가져와 스캔 코드 키를 두 번 누른다 (Windows 전용)."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, GAME_WINDOW)
    if not hwnd:
        sys.exit(f"게임 창({GAME_WINDOW})을 찾지 못했습니다.")
    user32.keybd_event(0x12, 0, 0, 0)  # Alt를 눌렀다 떼야 다른 프로그램 창을 앞으로 가져올 수 있다
    user32.keybd_event(0x12, 0, 2, 0)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    # 마우스가 화면 끝에 있으면 카메라가 그쪽으로 밀려 고정이 풀린다 (kk 확인). 창 가운데로 옮긴다
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    user32.SetCursorPos((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
    time.sleep(0.1)

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 8)]

    def send(flags):
        inp = INPUT(type=1, ki=KEYBDINPUT(0, scan, flags, 0, 0))
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    for _ in range(2):
        send(0x0008)           # KEYEVENTF_SCANCODE
        time.sleep(0.05)
        send(0x0008 | 0x0002)  # + KEYEVENTF_KEYUP
        time.sleep(0.15)


def api_lock(rp, name):
    """리플레이 API로 고정. 같은 값을 다시 보내면 무시되는 듯해서 풀었다가 다시 건다."""
    rp.post("/replay/render", {"cameraAttached": False})
    rp.post("/replay/render", {"selectionName": name, "cameraAttached": True})


START_TIME = 10.0
JUMP_PER_S = 1500  # 점멸·대시를 감안해도 챔피언이 1초에 이만큼 움직일 수는 없다
FOUNTAIN = {"ORDER": (400, 400), "CHAOS": (14300, 14300)}
GROUND_Y = 50.0  # 협곡 지면 높이 대략값. 카메라 높이에서 빼서 수평 거리를 구한다


def to_world(cx, cy, cz, pitch_deg):
    """top 카메라는 대상보다 남쪽(-z)에서 pitch 각도로 내려다본다. 대상 위치 = 카메라 z + 높이/tan(pitch).
    결과는 Riot 타임라인과 같은 좌표계 (x, y=z)."""
    return cx, cz + (cy - GROUND_Y) / math.tan(math.radians(pitch_deg))


def track(rp, champion, until, speed, step, lock="key"):
    players = rp.players()
    player = find_player(players, champion)
    if not player:
        print(f"  선수 목록에서 {champion}를 못 찾았습니다 (목록 {len(players)}명).")
        return []
    rp.post("/replay/render", {"interfaceAll": True, "interfaceMinimap": True, "fogOfWar": False})
    name = player.get("riotIdGameName") or champion
    scan = lock_key(players, player)
    if lock == "manual":
        print(f"  게임 창에서 {'1~5' if player.get('team') == 'ORDER' else 'q~t'} 중 {name}의 키를 두 번 눌러 주세요. 15초 기다립니다.")
        time.sleep(15)
    else:
        # 0초에는 챔피언이 소환 전이라 고정 키가 먹지 않는다. 소환된 뒤(10초)로 옮겨서 누른다 (kk PC 확인)
        rp.post("/replay/playback", {"time": START_TIME, "paused": True})
        time.sleep(2)
        if lock == "key":
            press_twice(scan)
        else:
            api_lock(rp, name)
        print(f"  카메라 고정({lock}): {name} ({player.get('team')})")
    rp.post("/replay/playback", {"paused": False, "speed": speed})
    rows, last = [], -step
    while True:
        pb = rp.get("/replay/playback")
        t = pb.get("time", 0.0)
        if t >= until or t >= pb.get("length", 0) - 1:
            break
        if t - last >= step:
            if lock == "api":
                api_lock(rp, name)
                time.sleep(0.1)
            render = rp.get("/replay/render")
            cam = render.get("cameraPosition") or {}
            pitch = (render.get("cameraRotation") or {}).get("y", 56.0)
            x, z = to_world(cam.get("x", 0), cam.get("y", 0), cam.get("z", 0), pitch)
            jump = 0
            if rows:
                dt = max(t - rows[-1][0], 0.5)
                dist = math.hypot(x - rows[-1][1], z - rows[-1][2])
                home = FOUNTAIN[player.get("team")]
                # 귀환·부활(우물로 순간 이동)이 아닌데 1초에 JUMP_PER_S보다 많이 움직이면 고정이 풀린 것으로 보고 다시 건다
                if dist / dt > JUMP_PER_S and math.hypot(x - home[0], z - home[1]) > 2000:
                    jump = 1
                    if lock == "key":
                        press_twice(scan)
                        time.sleep(0.2)
            rows.append((round(t, 2), round(x), round(z), round(cam.get("x", 0)), round(cam.get("y", 0)), round(cam.get("z", 0)), jump))
            last = t
            if len(rows) % 60 == 0:
                print(f"  {int(t // 60)}:{int(t % 60):02d}  x={rows[-1][1]} y={rows[-1][2]}", end="\r")
        time.sleep(max(0.02, step / speed / 4))
    rp.post("/replay/playback", {"paused": True})
    print()
    return rows


MAX_SPEED = 1000  # 유령·질주를 감안한 챔피언 최대 이동 속도(초당). 이보다 빠르면 카메라가 다른 곳을 비춘 것


def mark_valid(rows):
    """마지막으로 믿을 만한 위치에서 MAX_SPEED로 갈 수 없는 곳이면 무효(0)로 표시한다.
    우물 근처로 순간 이동한 것(귀환·부활)은 허용한다. 각 행 끝에 valid 값을 붙여 돌려준다."""
    out, good = [], None
    for r in rows:
        t, x, y = r[0], r[1], r[2]
        ok = good is None or math.hypot(x - good[1], y - good[2]) <= MAX_SPEED * (t - good[0]) + 300 \
            or any(math.hypot(x - fx, y - fy) < 2000 for fx, fy in FOUNTAIN.values())
        if ok:
            good = r
        out.append(tuple(r) + (int(ok),))
    return out


def probe(rp, champion, lock):
    """좌표가 챔피언을 따라가는지 90초만 보고 출력한다."""
    print("playerlist:", [(p.get("rawChampionName"), p.get("riotIdGameName") or p.get("summonerName"), p.get("team"))
                          for p in rp.players()])
    render = rp.get("/replay/render")
    print("render 카메라 관련:", {k: v for k, v in render.items() if k.startswith(("camera", "selection", "interface"))})
    rows = track(rp, champion, until=90, speed=2, step=3, lock=lock)
    for r in rows:
        print("  t=%.0f 위치 x=%d y=%d  (카메라 %d, %d, %d) 튐=%d" % r)
    if len(rows) >= 2 and len({(r[1], r[3]) for r in rows}) == 1:
        print("좌표가 움직이지 않습니다. 이 방식은 안 되고 미니맵 캡처 방식으로 바꿔야 합니다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match", nargs="?", help="경기 번호 (예: KR_8424846201)")
    ap.add_argument("--champion", action="append", help="따라갈 챔피언 영문 id (여러 번 가능). 없으면 data/raw에서 정글러")
    ap.add_argument("--until", type=float, default=1200, help="몇 초까지 기록 (기본 1200 = 20분)")
    ap.add_argument("--speed", type=float, default=4, help="재생 배속 (기본 4)")
    ap.add_argument("--step", type=float, default=1, help="기록 간격, 게임 초 (기본 1)")
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    ap.add_argument("--enable-api", action="store_true", help="game.cfg에 EnableReplayApi=1을 넣고 끝낸다")
    ap.add_argument("--probe", action="store_true", help="좌표가 읽히는지 1분만 확인")
    ap.add_argument("--lock", choices=["key", "api", "manual"], default="key",
                    help="카메라 고정 방법: key=단축키 자동 입력, api=리플레이 API, manual=직접 키 입력 (기본 key)")
    ap.add_argument("--no-launch", action="store_true", help="이미 재생 중인 리플레이에 붙기만 한다")
    args = ap.parse_args()
    args.game_dir = find_game_dir(args.game_dir)

    if args.enable_api:
        enable_replay_api(args.game_dir)
        return
    if not args.match:
        ap.error("경기 번호가 필요합니다")

    champions = args.champion or junglers_from_raw(args.match)
    if not champions:
        sys.exit(f"data/replay_raw/{args.match}.json이 없어 정글러를 모릅니다. --champion 으로 지정하세요.")
    game_id = args.match.split("_")[-1]

    if not args.no_launch:
        download_and_watch(LCU(args.game_dir), game_id)
    rp = Replay()
    pb = rp.wait_ready()
    print(f"리플레이 로딩 완료: 길이 {pb['length'] / 60:.1f}분")

    if args.probe:
        probe(rp, champions[0], args.lock)
        return

    for champ in champions:
        record(rp, args.match, champ, args.until, args.speed, args.step, args.lock)


def record(rp, match_id, champ, until=1200, speed=4, step=1, lock="key"):
    """한 챔피언을 추적해 data/replay_tracks/<경기>_<챔피언>.csv로 저장하고 경로를 돌려준다."""
    os.makedirs(OUT, exist_ok=True)
    print(f"{champ} 추적 ({speed}배속, {until / 60:.0f}분까지)")
    rows = track(rp, champ, until, speed, step, lock)
    if not rows:
        return None
    rows = mark_valid(rows)
    path = os.path.join(OUT, f"{match_id}_{champ}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "x", "y", "cam_x", "cam_y", "cam_z", "jump", "valid"])
        w.writerows(rows)
    bad = sum(1 for r in rows if not r[-1])
    print(f"  저장 {path} ({len(rows)}행, 무효 {bad}행)")
    return path


if __name__ == "__main__":
    main()
