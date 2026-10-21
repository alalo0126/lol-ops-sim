"""리플레이에서 정글러 위치를 초 단위로 뽑는다 (kk PC 전용, 롤 클라이언트가 켜져 있어야 함).

순서:
  1. 롤 클라이언트(LCU) 로컬 API로 리플레이를 받고 재생을 시작한다.
     클라이언트 전적의 '리플레이 다운로드/보기' 버튼과 같은 동작이다.
  2. Riot이 공개한 리플레이 API(https://127.0.0.1:2999/replay/...)로
     카메라를 정글러에 고정하고, 배속 재생하면서 카메라 좌표를 기록한다.
  3. data/replay_tracks/<경기>_<챔피언>.csv 로 저장한다 (게임 시각 초, x, y, z).

리플레이는 현재 클라이언트 빌드와 같은 빌드의 경기만 열린다 (작은 핫픽스도 다르면 안 열림).

처음 한 번은 리플레이 API를 켜야 한다:
  python -m tools.replay_track --enable-api      (game.cfg에 EnableReplayApi=1, 원본은 .bak으로 백업)

사용 예:
  python -m tools.replay_track KR_8424846201                  # data/raw에 경기 JSON이 있으면 양 팀 정글러 자동
  python -m tools.replay_track KR_8424846201 --champion Nidalee --until 900 --speed 4
  python -m tools.replay_track KR_8424846201 --probe          # 좌표가 읽히는지만 확인
"""
import argparse
import base64
import csv
import json
import os
import shutil
import sys
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
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
    path = os.path.join(RAW, f"{match_id}.json")
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


def attach(rp, player, champion):
    """카메라를 해당 챔피언에 고정한다. selectionName 후보를 차례로 넣어 보고 받아들여진 값을 쓴다."""
    names = [player.get(k) for k in ("riotIdGameName", "summonerName", "riotId", "championName")] if player else []
    names = [n for n in names if n] + [champion]
    for name in names:
        rp.post("/replay/render", {"selectionName": name, "cameraAttached": True})
        time.sleep(0.5)
        got = rp.get("/replay/render")
        if got.get("cameraAttached") and got.get("selectionName"):
            return got["selectionName"]
    return None


def track(rp, champion, until, speed, step):
    players = rp.players()
    player = find_player(players, champion)
    if not player:
        print(f"  주의: 선수 목록에서 {champion}를 못 찾음 (목록 {len(players)}명). 챔피언 이름으로 시도합니다.")
    rp.post("/replay/render", {"interfaceAll": False, "interfaceMinimap": True, "fogOfWar": False})
    rp.post("/replay/playback", {"time": 0.0, "paused": True})
    sel = attach(rp, player, champion)
    if not sel:
        print(f"  {champion}에 카메라를 고정하지 못했습니다. --probe 결과를 Claude에게 보내 주세요.")
        return []
    print(f"  카메라 고정: {sel}")
    rp.post("/replay/playback", {"paused": False, "speed": speed})
    rows, last = [], -step
    while True:
        pb = rp.get("/replay/playback")
        t = pb.get("time", 0.0)
        if t >= until or t >= pb.get("length", 0) - 1:
            break
        if t - last >= step:
            cam = rp.get("/replay/render").get("cameraPosition") or {}
            rows.append((round(t, 2), round(cam.get("x", 0), 1), round(cam.get("y", 0), 1), round(cam.get("z", 0), 1)))
            last = t
            if len(rows) % 60 == 0:
                print(f"  {int(t // 60)}:{int(t % 60):02d}  x={rows[-1][1]} z={rows[-1][3]}", end="\r")
        time.sleep(max(0.02, step / speed / 4))
    rp.post("/replay/playback", {"paused": True})
    print()
    return rows


def probe(rp, champion):
    """좌표가 실제로 움직이는지 30초만 보고 출력한다."""
    print("playerlist:", [(p.get("rawChampionName"), p.get("riotIdGameName") or p.get("summonerName"), p.get("team"))
                          for p in rp.players()])
    print("render 기본값:", json.dumps(rp.get("/replay/render"), ensure_ascii=False)[:600])
    rows = track(rp, champion, until=60, speed=2, step=5)
    for r in rows:
        print("  t=%.0f x=%.0f y=%.0f z=%.0f" % r)
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
        sys.exit(f"data/raw/{args.match}.json이 없어 정글러를 모릅니다. --champion 으로 지정하세요.")
    game_id = args.match.split("_")[-1]

    if not args.no_launch:
        download_and_watch(LCU(args.game_dir), game_id)
    rp = Replay()
    pb = rp.wait_ready()
    print(f"리플레이 로딩 완료: 길이 {pb['length'] / 60:.1f}분")

    if args.probe:
        probe(rp, champions[0])
        return

    os.makedirs(OUT, exist_ok=True)
    for champ in champions:
        print(f"{champ} 추적 ({args.speed}배속, {args.until / 60:.0f}분까지)")
        rows = track(rp, champ, args.until, args.speed, args.step)
        if not rows:
            continue
        path = os.path.join(OUT, f"{args.match}_{champ}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "x", "y", "z"])
            w.writerows(rows)
        print(f"  저장 {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
