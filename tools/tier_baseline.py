"""티어별 기준선: collect/collect_tiers.py로 받은 data/tiers/<TIER>/raw/*.json.gz를 요약한다.

타임라인 사건(초 단위)과 1분 프레임으로 만들 수 있는 습관 지표만 쓴다(동선·시야 위치는 없음).
선수-판 한 줄마다 지표를 계산하고, 티어×포지션별 중앙값·사분위를 낸다.
  - seed: 시드(그 티어에 오래 머문 선수) 본인만
  - all : 같은 판 10명 전부(티어 모름, 비슷한 MMR로 매칭된다는 가정)
의심 선수(양학·고의 패배 패턴) 표시: 킬이 그 티어 상위 0.5%, 10데스 이상이면서 KDA 0.3 이하,
팀 내 챔피언 피해 비중 5% 미만(잠수 의심). 표시된 선수를 넣은/뺀 중앙값을 같이 낸다.

  python tools/tier_baseline.py                  # → data/tiers/baseline.json, reports/tier_baseline.md
  python tools/tier_baseline.py --compact        # 판별 지표 줄만 data/tiers/rows.jsonl.gz로(클라우드로 옮기기용)
"""
import argparse
import glob
import gzip
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = os.path.join(ROOT, "data", "tiers")
ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "CHALLENGER"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
EPIC = {"DRAGON", "HORDE", "RIFTHERALD", "BARON_NASHOR", "ATAKHAN"}
CONTROL_WARD = 2055
MIN_GAME_S = 15 * 60

METRICS = [
    ("cs10", "10분 CS"), ("cs14", "14분 CS"), ("gold10", "10분 골드"), ("gold14", "14분 골드"),
    ("gd10", "10분 맞라이너 골드차"), ("gd14", "14분 맞라이너 골드차"), ("xp10", "10분 경험치"),
    ("deaths14", "14분 전 사망"), ("solo_deaths14", "14분 전 1:1 사망"), ("gank_deaths14", "14분 전 정글 낀 사망"),
    ("deaths_late_per10", "14분 뒤 10분당 사망"), ("kills", "킬"), ("deaths", "데스"), ("assists", "어시"),
    ("first_back", "첫 귀환(분, 1:30 뒤 첫 구매)"), ("backs14", "14분 전 상점 방문(사망 포함)"),
    ("recalls14", "14분 전 귀환(사망 뒤 구매 뺌)"),
    ("unspent_back_gold", "귀환 때 들고 간 골드 중앙"),
    ("wards_per_min", "분당 와드 설치"), ("control_wards", "제어 와드 구매"),
    ("epic_share", "팀 에픽 처치 관여율"), ("kp", "킬 관여율"),
    ("dmg_share", "팀 내 챔피언 피해 비중"), ("plates", "방패 골드 획득 수"),
]
# 정수 횟수 지표는 중앙값이 갈리지 않아 리포트에 평균을 쓴다
COUNT = {"deaths14", "solo_deaths14", "gank_deaths14", "kills", "deaths", "assists", "backs14",
         "recalls14", "control_wards", "plates"}
# 맞라이너 골드차는 10명 전체로 보면 대칭(0)이라 seed만 의미 있음
SEED_ONLY = {"gd10", "gd14"}
MATCH_METRICS = [("duration", "경기 시간(분)"), ("first_dragon", "첫 용(분)"),
                 ("first_tower", "첫 포탑(분)"), ("kills_total", "양 팀 킬 합"),
                 ("dragons_winner", "이긴 팀 용 수")]


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def frame_at(frames, minute, pid):
    if minute >= len(frames):
        return None
    return frames[minute]["participantFrames"].get(str(pid))


def rows_for_match(d):
    detail, tl, seed = d["detail"], d["timeline"], d["seed"]
    info = detail["info"]
    parts = info["participants"]
    dur = info["gameDuration"]
    if dur < MIN_GAME_S:
        return [], None
    frames = tl["info"]["frames"]
    events = [e for f in frames for e in f.get("events", [])]
    by_pid = {p["participantId"]: p for p in parts}
    team_of = {p["participantId"]: p["teamId"] for p in parts}
    role_of = {p["participantId"]: p.get("teamPosition") or "" for p in parts}
    jungler = {p["teamId"]: p["participantId"] for p in parts if p.get("teamPosition") == "JUNGLE"}
    opp = {}
    for p in parts:
        for q in parts:
            if q["teamId"] != p["teamId"] and q.get("teamPosition") == p.get("teamPosition") and p.get("teamPosition"):
                opp[p["participantId"]] = q["participantId"]

    team_kills = defaultdict(int)
    team_dmg = defaultdict(int)
    for p in parts:
        team_kills[p["teamId"]] += p["kills"]
        team_dmg[p["teamId"]] += p.get("totalDamageDealtToChampions", 0)
    team_epic = defaultdict(int)
    epic_involved = defaultdict(int)
    kills_ev = []
    purchases = defaultdict(list)
    wards = defaultdict(int)
    plates = defaultdict(int)
    first = {}
    dragons = defaultdict(int)
    for e in events:
        t = e.get("timestamp", 0) / 1000
        typ = e.get("type")
        if typ == "CHAMPION_KILL":
            kills_ev.append(e)
        elif typ == "ITEM_PURCHASED":
            purchases[e.get("participantId")].append((t, e.get("itemId")))
        elif typ == "WARD_PLACED" and e.get("creatorId"):
            wards[e["creatorId"]] += 1
        elif typ == "TURRET_PLATE_DESTROYED" and e.get("killerId"):
            plates[e["killerId"]] += 1
        elif typ == "ELITE_MONSTER_KILL" and e.get("monsterType") in EPIC:
            tid = e.get("killerTeamId") or team_of.get(e.get("killerId"))
            if tid:
                team_epic[tid] += 1
                for pid in [e.get("killerId")] + list(e.get("assistingParticipantIds") or []):
                    if pid in team_of and team_of[pid] == tid:
                        epic_involved[pid] += 1
            if e["monsterType"] == "DRAGON":
                first.setdefault("first_dragon", t / 60)
                if tid:
                    dragons[tid] += 1
        elif typ == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
            first.setdefault("first_tower", t / 60)

    rows = []
    for p in parts:
        pid = p["participantId"]
        r = {"pid": pid, "role": role_of[pid], "win": p["win"], "champ": p["championName"],
             "is_seed": pid == seed["participantId"], "puuid": p["puuid"]}
        for m in (10, 14):
            f = frame_at(frames, m, pid)
            if f:
                r[f"cs{m}"] = f["minionsKilled"] + f["jungleMinionsKilled"]
                r[f"gold{m}"] = f["totalGold"]
                if m == 10:
                    r["xp10"] = f["xp"]
                o = opp.get(pid)
                fo = frame_at(frames, m, o) if o else None
                if fo:
                    r[f"gd{m}"] = f["totalGold"] - fo["totalGold"]
        d14 = solo = gank = late = 0
        for e in kills_ev:
            if e.get("victimId") != pid:
                continue
            t = e["timestamp"] / 1000
            killers = [e.get("killerId")] + list(e.get("assistingParticipantIds") or [])
            if t < 14 * 60:
                d14 += 1
                if not e.get("assistingParticipantIds"):
                    solo += 1
                ej = jungler.get(300 - team_of[pid])
                if r["role"] != "JUNGLE" and ej in killers:
                    gank += 1
            else:
                late += 1
        r.update(deaths14=d14, solo_deaths14=solo, gank_deaths14=gank,
                 deaths_late_per10=late / max((dur - 14 * 60) / 600, 0.1))
        r.update(kills=p["kills"], deaths=p["deaths"], assists=p["assists"])
        tk = team_kills[p["teamId"]]
        r["kp"] = (p["kills"] + p["assists"]) / tk if tk else None
        td = team_dmg[p["teamId"]]
        r["dmg_share"] = p.get("totalDamageDealtToChampions", 0) / td if td else None
        # 귀환: 1:30 이후 구매 묶음(30초 안 구매는 한 번)
        backs = []
        for t, _ in sorted(purchases[pid]):
            if t < 90:
                continue
            if not backs or t - backs[-1] > 30:
                backs.append(t)
        r["first_back"] = backs[0] / 60 if backs else None
        r["backs14"] = sum(1 for t in backs if t < 14 * 60)
        deaths_t = [e["timestamp"] / 1000 for e in kills_ev if e.get("victimId") == pid]
        # 사망 뒤 90초 안의 구매는 부활 후 상점(귀환 아님)
        recalls = [t for t in backs if not any(0 <= t - dt <= 90 for dt in deaths_t)]
        r["recalls14"] = sum(1 for t in recalls if t < 14 * 60)
        gold_at = []
        for t in backs[:3]:
            m = int(t // 60)
            f = frame_at(frames, m, pid)
            if f:
                gold_at.append(f.get("currentGold", 0))
        r["unspent_back_gold"] = statistics.median(gold_at) if gold_at else None
        r["wards_per_min"] = wards[pid] / (dur / 60)
        r["control_wards"] = sum(1 for _, i in purchases[pid] if i == CONTROL_WARD)
        te = team_epic[p["teamId"]]
        r["epic_share"] = epic_involved[pid] / te if te else None
        r["plates"] = plates[pid]
        rows.append(r)
    win_team = next((p["teamId"] for p in parts if p["win"]), None)
    mrow = {"duration": dur / 60, "kills_total": sum(team_kills.values()),
            "dragons_winner": dragons.get(win_team, 0), **first}
    return rows, mrow


def flag_rows(rows_by_tier):
    for tier, rows in rows_by_tier.items():
        ks = sorted(r["kills"] for r in rows)
        k995 = ks[int(0.995 * (len(ks) - 1))] if ks else 99
        for r in rows:
            kda = (r["kills"] + r["assists"]) / max(r["deaths"], 1)
            why = []
            if r["kills"] >= max(k995, 15):
                why.append("kills_p995")
            if r["deaths"] >= 10 and kda <= 0.3:
                why.append("feed")
            if r["dmg_share"] is not None and r["dmg_share"] < 0.05 and r["role"] != "UTILITY":
                why.append("afk")
            r["flag"] = ",".join(why)


def q(vals, p):
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def stat(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return {"n": len(vals), "p25": round(q(vals, 0.25), 3), "median": round(statistics.median(vals), 3),
            "p75": round(q(vals, 0.75), 3), "mean": round(statistics.fmean(vals), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers-dir", default=TIERS)
    ap.add_argument("--compact", action="store_true", help="rows.jsonl.gz만 쓰기")
    ap.add_argument("--rows", default="", help="이미 만든 rows.jsonl.gz에서 요약")
    ap.add_argument("--out", default=os.path.join(TIERS, "baseline.json"))
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "tier_baseline.md"))
    args = ap.parse_args()

    rows_by_tier, match_rows = defaultdict(list), defaultdict(list)
    if args.rows:
        with gzip.open(args.rows, "rt", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o.get("kind") == "match":
                    match_rows[o["tier"]].append(o)
                else:
                    rows_by_tier[o["tier"]].append(o)
    else:
        for tier in ORDER:
            paths = sorted(glob.glob(os.path.join(args.tiers_dir, tier, "raw", "*.json.gz")))
            for i, path in enumerate(paths):
                try:
                    d = load(path)
                    rows, mrow = rows_for_match(d)
                except Exception as ex:  # 깨진 파일은 건너뛰고 알림
                    print(f"  건너뜀 {os.path.basename(path)}: {ex}")
                    continue
                mid = os.path.basename(path)[:-8]
                for r in rows:
                    r.update(tier=tier, match=mid)
                    rows_by_tier[tier].append(r)
                if mrow:
                    mrow.update(tier=tier, match=mid, kind="match")
                    match_rows[tier].append(mrow)
                if (i + 1) % 200 == 0:
                    print(f"  {tier} {i + 1}/{len(paths)}")
    flag_rows(rows_by_tier)

    if args.compact:
        out = os.path.join(args.tiers_dir, "rows.jsonl.gz")
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for tier in rows_by_tier:
                for r in rows_by_tier[tier]:
                    r = dict(r)
                    r.pop("puuid", None)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                for m in match_rows[tier]:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"→ {out}")
        return

    result = {"tiers": {}}
    for tier in [t for t in ORDER if t in rows_by_tier]:
        rows = rows_by_tier[tier]
        tr = {"matches": len(match_rows[tier]),
              "flagged_share": round(sum(bool(r["flag"]) for r in rows) / len(rows), 4),
              "flag_counts": {k: sum(k in r["flag"].split(",") for r in rows)
                              for k in ("kills_p995", "feed", "afk")},
              "match": {k: stat([m.get(k) for m in match_rows[tier]]) for k, _ in MATCH_METRICS},
              "roles": {}}
        for role in ROLES:
            rr = [r for r in rows if r["role"] == role]
            out = {}
            for scope, sel in (("seed", [r for r in rr if r["is_seed"]]), ("all", rr),
                               ("all_clean", [r for r in rr if not r["flag"]])):
                out[scope] = {k: stat([r.get(k) for r in sel]) for k, _ in METRICS}
                out[scope]["_n"] = len(sel)
            tr["roles"][role] = out
        result["tiers"][tier] = tr
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    write_report(result, args.report)
    print(f"→ {args.out}, {args.report}")


def write_report(res, path):
    tiers = list(res["tiers"])
    L = ["# 티어별 기준선 (솔로랭크, 26.22, 15분 이상 판)", "",
         "값은 중앙값(횟수 지표는 평균). 'all'은 같은 판 10명, 'seed'는 그 티어에 오래 머문 선수 본인. "
         "의심 선수(양학·고의 패배·잠수 패턴)를 뺀 값이 all_clean.", "",
         "## 판 단위", "", "| 지표 | " + " | ".join(tiers) + " |", "|---|" + "---|" * len(tiers)]
    for k, name in MATCH_METRICS:
        L.append(f"| {name} | " + " | ".join(
            f"{res['tiers'][t]['match'][k]['median']:.1f}" if res['tiers'][t]['match'].get(k) else "-"
            for t in tiers) + " |")
    L += ["", "## 의심 선수 비율", "", "| 티어 | 판 | 표시 비율 | 킬 상위 | 피딩 | 잠수 |", "|---|---|---|---|---|---|"]
    for t in tiers:
        r = res["tiers"][t]
        c = r["flag_counts"]
        L.append(f"| {t} | {r['matches']} | {r['flagged_share']:.1%} | {c['kills_p995']} | {c['feed']} | {c['afk']} |")
    for role in ROLES:
        L += ["", f"## {role} (all_clean 중앙값, 괄호는 seed)", "",
              "| 지표 | " + " | ".join(tiers) + " |", "|---|" + "---|" * len(tiers)]
        for k, name in METRICS:
            cells = []
            for t in tiers:
                a = res["tiers"][t]["roles"][role]["all_clean"].get(k)
                s = res["tiers"][t]["roles"][role]["seed"].get(k)
                key = "mean" if k in COUNT else "median"
                if k in SEED_ONLY:
                    a, s = s, None
                if not a:
                    cells.append("-")
                    continue
                fmt = (lambda v: f"{v:.2f}") if abs(a[key]) < 10 else (lambda v: f"{v:.0f}")
                cells.append(fmt(a[key]) + (f" ({fmt(s[key])})" if s else ""))
            L.append(f"| {name}{' (seed만)' if k in SEED_ONLY else ''} | " + " | ".join(cells) + " |")
        L.append("| 표본(all_clean / seed) | " + " | ".join(
            f"{res['tiers'][t]['roles'][role]['all_clean']['_n']} / {res['tiers'][t]['roles'][role]['seed']['_n']}"
            for t in tiers) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
