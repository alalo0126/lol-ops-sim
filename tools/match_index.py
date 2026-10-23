"""1단계 1,000판 목록: 경기마다 누구(시드 정글러) 경기인지, 날짜·챔피언·승패를 표로 뽑는다.

    python tools\\match_index.py      :: data\\raw → data\\match_index.csv (엑셀에서 바로 열림)

시드 정글러 = 수집 때 고른 챌린저·그랜드마스터 선수(collect_matches.py). 상대 정글러도 같이 적는다.
op.gg 링크로 그 선수 전적을 찾아볼 수 있다. Riot ID는 경기 당시 이름이라 지금은 바뀌었을 수 있다.
"""
import csv
import datetime
import glob
import json
import os
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KST = datetime.timezone(datetime.timedelta(hours=9))
COLUMNS = ["match", "date_kst", "minutes", "seed_riot_id", "seed_tier", "seed_team", "seed_champion", "seed_win",
           "enemy_jungler_riot_id", "enemy_jungler_champion", "opgg_seed"]


def riot_id(p):
    name, tag = p.get("riotIdGameName") or p.get("summonerName") or "?", p.get("riotIdTagline") or ""
    return f"{name}#{tag}" if tag else name


def opgg(p):
    name, tag = p.get("riotIdGameName"), p.get("riotIdTagline")
    if not name or not tag:
        return ""
    return "https://op.gg/lol/summoners/kr/" + urllib.parse.quote(f"{name}-{tag}")


def row(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    info = d["detail"]["info"]
    seed = d.get("seed") or {}
    parts = info["participants"]
    me = next((p for p in parts if p["participantId"] == seed.get("participantId")), None)
    if me is None:
        return None
    foe = next((p for p in parts if p["teamId"] != me["teamId"] and p.get("teamPosition") == "JUNGLE"), {})
    when = datetime.datetime.fromtimestamp(info["gameCreation"] / 1000, KST)
    return {"match": os.path.basename(path)[:-5], "date_kst": when.strftime("%Y-%m-%d %H:%M"),
            "minutes": round(info.get("gameDuration", 0) / 60, 1), "seed_riot_id": riot_id(me), "seed_tier": seed.get("tier", ""),
            "seed_team": "블루" if me["teamId"] == 100 else "레드", "seed_champion": me["championName"],
            "seed_win": "승" if me["win"] else "패", "enemy_jungler_riot_id": riot_id(foe) if foe else "",
            "enemy_jungler_champion": foe.get("championName", ""), "opgg_seed": opgg(me)}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.json"))):
        try:
            r = row(path)
        except (OSError, ValueError, KeyError):
            continue
        if r:
            rows.append(r)
    dst = os.path.join(ROOT, "data", "match_index.csv")
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: 엑셀에서 한글이 안 깨짐
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}판 → {dst} (시드 선수 {len({r['seed_riot_id'] for r in rows})}명)")


if __name__ == "__main__":
    main()
