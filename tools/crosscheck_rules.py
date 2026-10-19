"""rules/objectives.json·camps.json의 시각을 1단계 실제 경기(data/raw)와 맞춰 본다.

    python tools\crosscheck_rules.py
처치 시각은 생성 시각보다 빠를 수 없고, 사라지는 시각보다 늦을 수 없다는 점만 확인한다.
"""
import collections
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    obj = json.load(open(os.path.join(ROOT, "rules", "objectives.json"), encoding="utf-8"))
    first = collections.defaultdict(list)
    drake_gaps, baron_gaps, elder_ok = [], [], []
    grub_last = herald_last = 0
    grubs_per_game = collections.Counter()
    jg_cs_1min = collections.Counter()
    versions = collections.Counter()
    n = 0
    for path in glob.glob(os.path.join(ROOT, "data", "raw", "*.json")):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        det, tl = d.get("detail"), d.get("timeline")
        if not det or not tl:
            continue
        n += 1
        versions[det["info"]["gameVersion"][:5]] += 1
        pos = {p["participantId"]: p["teamPosition"] for p in det["info"]["participants"]}
        seen, drakes, barons, team_drakes, grubs = set(), [], [], collections.Counter(), 0
        for i, fr in enumerate(tl["info"]["frames"]):
            if i == 1:
                for pid, pf in fr["participantFrames"].items():
                    if pos[int(pid)] == "JUNGLE":
                        jg_cs_1min[pf["jungleMinionsKilled"]] += 1
            for e in fr["events"]:
                if e["type"] != "ELITE_MONSTER_KILL":
                    continue
                t, m = e["timestamp"] / 1000, e["monsterType"]
                if m not in seen:
                    seen.add(m)
                    first[m].append(t)
                if m == "DRAGON":
                    if e.get("monsterSubType") == "ELDER_DRAGON":
                        elder_ok.append(max(team_drakes.values(), default=0) >= 4)
                    else:
                        drakes.append(t)
                        team_drakes[e["killerTeamId"]] += 1
                elif m == "BARON_NASHOR":
                    barons.append(t)
                elif m == "RIFTHERALD":
                    herald_last = max(herald_last, t)
                elif m == "HORDE":
                    grub_last = max(grub_last, t)
                    grubs += 1
        grubs_per_game[grubs] += 1
        drake_gaps += [b - a for a, b in zip(drakes, drakes[1:])]
        baron_gaps += [b - a for a, b in zip(barons, barons[1:])]

    def line(name, ok, detail):
        print(f"[{'OK' if ok else '불일치'}] {name}: {detail}")

    print(f"경기 {n}판, 버전 {dict(versions)}")
    spawn = {"DRAGON": obj["drake"]["first_spawn_s"]["value"],
             "HORDE": obj["voidgrubs"]["spawn_s"]["value"],
             "RIFTHERALD": obj["rift_herald"]["spawn_s"]["value"],
             "BARON_NASHOR": obj["baron"]["spawn_s"]["value"]}
    for m, s in spawn.items():
        v = sorted(first[m])
        if v:
            line(f"{m} 생성 {s}초", v[0] >= s, f"첫 처치 최조 {v[0]:.0f}초, 중앙값 {v[len(v)//2]:.0f}초 ({len(v)}판)")
    if drake_gaps:
        line("드래곤 리스폰", min(drake_gaps) >= obj["drake"]["respawn_s"]["value"], f"최소 간격 {min(drake_gaps):.1f}초")
    if baron_gaps:
        line("바론 리스폰", min(baron_gaps) >= obj["baron"]["respawn_s"]["value"], f"최소 간격 {min(baron_gaps):.1f}초")
    line("유충 소멸", grub_last <= obj["voidgrubs"]["despawn_s"]["in_combat"], f"마지막 처치 {grub_last:.1f}초, 판당 처치 수 {dict(grubs_per_game)}")
    line("전령 소멸", herald_last <= obj["rift_herald"]["despawn_s"]["in_combat"], f"마지막 처치 {herald_last:.1f}초")
    if elder_ok:
        line("장로 조건", all(elder_ok), f"{len(elder_ok)}회")
    line("캠프 첫 생성 0:55 이후", set(jg_cs_1min) == {0}, f"1분 프레임 정글 CS {dict(jg_cs_1min)}")


if __name__ == "__main__":
    main()
