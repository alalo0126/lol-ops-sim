"""라이너 검수: 시뮬 N판의 라이너 분포를 1단계 실제 분포(sim/targets_laner.json, tools/laner_targets.py)와 비교한다.

    python -m sim.laner_eval --games 100        :: → reports/laner_eval.md

판정은 run_baseline과 같다: 시뮬 중앙값이 실제 p25~p75 안이면 OK, p10~p90 안이면 주의, 밖이면 불일치.
실제 위치·체력은 분 프레임 순간값이라 시뮬도 분 프레임 순간값으로 잰다.
"""
import argparse
import collections
import json
import os
import statistics

from .data import ROOT
from .game import Game
from .stage6 import zone
from tools.laner_targets import lane_band, LANERS, LANE_OF

MINUTES = ["3", "5", "10", "14", "20"]
ZONES = ["own_lane", "other_lane", "jungle", "river", "base"]
JUNGLE_Z = {"own_jungle", "enemy_jungle", "pit"}


def frame_zone(xy, team, role, nodes):
    z = zone(xy[0], xy[1], team, nodes)
    if z == "lane":
        return "own_lane" if lane_band(xy[0], xy[1]) == LANE_OF[role] else "other_lane"
    return "jungle" if z in JUNGLE_Z else z


def simulate(games, seed0=0, game_kw=None):
    vals = collections.defaultdict(list)
    where = collections.defaultdict(collections.Counter)
    from .laner_tracks import T0, T1, behaviour, place
    from .stage6_track import _xy
    for s in range(seed0, seed0 + games):
        g = Game(seed=s, **(game_kw or {}))
        g.record_frame()
        laners = [c for c in g.champs if c.role in LANERS]
        tracks = {c.pid: {} for c in laners}
        while g.t < 1200:
            g.step()
            if T0 <= g.t < T1:
                for c in laners:
                    x, y = _xy(g, c)
                    tracks[c.pid][g.t] = "dead" if x is None else place(x, y, c.team, c.role, g.map.nodes)
        kills = [e for e in g.events if e["type"] == "kill"]
        for c in laners:
            b = behaviour(tracks[c.pid], kills, c.pid, {x.pid for x in g.champs if x.team == c.team})
            vals[(c.role, "rp_base_visits")].append(b["base_visits"])
            vals[(c.role, "rp_roams")].append(len(b["roams"]))
            vals[(c.role, "rp_roam_kill")] += [r["kill"] for r in b["roams"]]
            vals[(c.role, "rp_own_lane")].append(b["share"]["own_lane"])
            vals[(c.role, "rp_trips")].append(b["trips"])
        for e in g.events:
            if e["type"] == "plate":
                vals[("plate", e["tier"])].append(e["t"])
            elif e["type"] == "tower":
                vals[("tower", e["tier"])].append(e["t"])
        towers = [e["t"] for e in g.events if e["type"] == "tower"]
        vals[("first_tower",)].append(min(towers) if towers else 1200)
        vals[("outer_by_20",)].append(sum(1 for e in g.events if e["type"] == "tower" and e["tier"] == "outer"))
        vals[("inner_by_20",)].append(sum(1 for e in g.events if e["type"] == "tower" and e["tier"] == "inner"))
        vals[("plates_by_20",)].append(sum(1 for e in g.events if e["type"] == "plate"))
        nodes = g.map.nodes
        frames = {f["t"] // 60: f for f in g.frames}
        for c in g.champs:
            if c.role not in LANERS:
                continue
            r = c.role
            for m in MINUTES:
                pf = frames[int(m)]["p"][c.pid]
                for k in ("lane_cs", "gold", "level", "hp"):
                    vals[(r, m, k)].append(pf[k])
            for m in range(1, 21):
                pf = frames[m]["p"][c.pid]
                where[(r, m)][frame_zone(pf["xy"], c.team, r, nodes)] += 1
            shops = sorted(e["t"] for e in g.events if e["type"] == "shop" and e["pid"] == c.pid and e["t"] > 60)
            vals[(r, "first_shop_s")].append(shops[0] if shops else 1200)
            vals[(r, "shops_by_20")].append(len(shops))
            for m in (10, 20):
                vals[(r, f"deaths_{m}")].append(sum(1 for e in g.events if e["type"] == "kill" and e["victim"] == c.pid and e["t"] <= m * 60))
                vals[(r, f"wards_{m}")].append(sum(1 for e in g.events if e["type"] == "ward" and e["pid"] == c.pid and e["t"] <= m * 60))
    return vals, where


def verdict(sim_median, real):
    if not real:
        return "-"
    if real["p25"] <= sim_median <= real["p75"]:
        return "OK"
    if real["p10"] <= sim_median <= real["p90"]:
        return "주의"
    return "불일치"


def real_zone(tz, r, m):
    z = tz[r][str(m)]
    return {"own_lane": z.get("own_lane", 0), "other_lane": z.get("other_lane", 0),
            "jungle": sum(z.get(k, 0) for k in JUNGLE_Z), "river": z.get("river", 0), "base": z.get("base", 0)}


REPLAY_KEYS = (("rp_base_visits", "base_visits", 0.5, "{:.1f}"), ("rp_roams", "roams", 0.5, "{:.1f}"),
               ("rp_roam_kill", "roam_kill", 0.1, "{:.0%}"), ("rp_own_lane", "own_lane", 0.05, "{:.0%}"),
               ("rp_trips", "trips", 0.75, "{:.1f}"))


def rp_value(vals, r, key):
    v = vals[(r, key)]
    return statistics.mean(v) if v else 0.0


def replay_rows(vals, rp, r):
    """(시뮬 키, 실제 값, 허용 범위, 형식). 실제 = sim/targets_laner_replay.json(sim/laner_tracks.py가 씀)."""
    real = rp["roles"][r]
    out = []
    for k, rk, tol, fmt in REPLAY_KEYS:
        v = real["share"]["own_lane"] if rk == "own_lane" else real[rk]
        if rk == "roam_kill":   # 로밍 표본이 적은 포지션(탑 7번)은 허용 범위를 표본 오차 두 배로 넓힘
            n = max(1.0, real["roams"] * real["n"])
            tol = max(tol, 2 * (v * (1 - v) / n) ** 0.5)
        out.append((k, v, tol, fmt))
    return out


def report(vals, where, games):
    t = json.load(open(os.path.join(ROOT, "sim", "targets_laner.json"), encoding="utf-8"))
    lines = [f"# 라이너 검수 (시뮬 {games}판 vs 실제 {t['games']}판)", "",
             "판정: 시뮬 중앙값이 실제 p25~p75 안 OK, p10~p90 안 주의, 밖 불일치. 위치·체력은 분 프레임 순간값.", ""]
    count = collections.Counter()
    for r in LANERS:
        lines += [f"## {r}", "", "| 지표 | 시뮬 중앙 | 실제 p10 / p25 / 중앙 / p75 / p90 | 판정 |", "|---|---|---|---|"]
        rows = [(f"{m}분 {k}", (r, m, k), t["by_minute"][r][m].get(k)) for m in MINUTES for k in ("lane_cs", "gold", "level", "hp")]
        rows += [("첫 상점(초)", (r, "first_shop_s"), t["first_shop_s"][r]), ("20분까지 상점", (r, "shops_by_20"), t["shops_by_20"][r])]
        rows += [(f"{m}분 사망", (r, f"deaths_{m}"), t["deaths"][f"{r}_{m}"]) for m in (10, 20)]
        rows += [(f"{m}분 와드", (r, f"wards_{m}"), t["wards"][f"{r}_{m}"]) for m in (10, 20)]
        for name, key, real in rows:
            sm = statistics.median(vals[key])
            v = verdict(sm, real)
            count[v] += 1
            rs = " / ".join(str(real[q]) for q in ("p10", "p25", "p50", "p75", "p90")) if real else "-"
            lines.append(f"| {name} | {round(sm, 2)} | {rs} | {v} |")
        lines += ["", "위치 비율(분 프레임): 시뮬 / 실제", "", "| 분 | " + " | ".join(ZONES) + " |", "|---|" + "---|" * len(ZONES)]
        for m in (3, 6, 10, 15, 20):
            tot = sum(where[(r, m)].values()) or 1
            rz = real_zone(t["zone_by_minute"], r, m)
            lines.append(f"| {m} | " + " | ".join(f"{where[(r, m)][z] / tot:.2f} / {rz[z]:.2f}" for z in ZONES) + " |")
        lines.append("")
    rp_path = os.path.join(ROOT, "sim", "targets_laner_replay.json")
    if os.path.exists(rp_path):
        rp = json.load(open(rp_path, encoding="utf-8"))
        lines += ["## 리플레이 1초 동선(3~18분)", "", f"실제는 {rp['_doc']}. 판정: 아래 허용 범위 안 OK, 두 배 안 주의.", "",
                  "| 포지션 | 기지 방문 시뮬 / 실제 | 로밍 시뮬 / 실제 | 로밍당 킬 관여 시뮬 / 실제 | 자기 라인 비율 시뮬 / 실제 | 강가·정글 나들이 시뮬 / 실제 |", "|---|---|---|---|---|---|"]
        for r in LANERS:
            row = []
            for key, real, tol, fmt in replay_rows(vals, rp, r):
                sv = rp_value(vals, r, key)
                v = "OK" if abs(sv - real) <= tol else ("주의" if abs(sv - real) <= 2 * tol else "불일치")
                count[v] += 1
                row.append(f"{fmt.format(sv)} / {fmt.format(real)} {v}")
            lines.append(f"| {r} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## 포탑", "", "실제 시각은 경기 전체(20분 이후 포함) 분포라, 20분에서 끝나는 시뮬과는 20분 안 비율로 같이 본다.", "",
              "| 지표 | 시뮬 | 실제 |", "|---|---|---|"]
    for tier in ("outer", "inner", "inhib"):
        sv = vals[("plate", tier)]
        rv = t["plate_time_by_tier_s"].get(tier, {})
        lines.append(f"| {tier} 방패 시각 중앙(초), 판당 개수 | {statistics.median(sv) if sv else '-'}, {len(sv) / games:.1f} | {rv.get('p50')}, {rv.get('n', 0) / t['games']:.1f}(경기 전체) |")
    real_outer = {k: v for k, v in t["tower_kill_s"].items() if k.startswith("OUTER")}
    lines.append(f"| 첫 포탑 시각 중앙(초) | {statistics.median(vals[('first_tower',)])} | {t['first_tower_s']['p50']} |")
    b20 = t.get("by_20_per_game", {})
    lines.append(f"| 20분까지 외곽 포탑 파괴(양 팀 합, 평균) | {statistics.mean(vals[('outer_by_20',)]):.2f} | {b20.get('outer', {}).get('mean', '-')} (중앙 시각 " + ", ".join(f"{k} {v['p50']}" for k, v in real_outer.items()) + ") |")
    lines.append(f"| 20분까지 2차 포탑 파괴(양 팀 합, 평균) | {statistics.mean(vals[('inner_by_20',)]):.2f} | {b20.get('inner', {}).get('mean', '-')} |")
    lines.append(f"| 20분까지 방패(양 팀 합, 평균) | {statistics.mean(vals[('plates_by_20',)]):.1f} | {b20.get('plates', {}).get('mean', '-')} |")
    lines.append("")
    lines.insert(4, "요약: " + ", ".join(f"{k} {v}" for k, v in count.most_common()) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "laner_eval.md"))
    a = ap.parse_args()
    vals, where = simulate(a.games)
    md = report(vals, where, a.games)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
