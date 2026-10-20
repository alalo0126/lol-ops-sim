"""맵 노드 그래프(rules/map_nodes.json)를 만든다.

    python tools/build_map_graph.py

블루 진영 노드만 손으로 적고, 레드 진영은 점대칭으로 만든다. 대칭 중심은 1단계 경기의
포탑 처치 위치 쌍(블루 탑 1차 ↔ 레드 봇 1차 등)에서 구했다: x + x' ≈ 14820, y + y' ≈ 14900.

좌표 출처
- stage1: 1단계 경기(16.20, 340판) 이벤트 위치 중앙값. 포탑=BUILDING_KILL, 용·바론 둥지=ELITE_MONSTER_KILL,
  우물=0분 프레임 위치. data/stage4_targets.json에서 다시 확인할 수 있다.
- consensus: 맵 좌표로 흔히 쓰는 값. 캠프 좌표는 1분 프레임 정글러 위치 중앙값과 420~510 떨어져 있어
  (캠프를 때리는 사거리만큼 떨어져 서 있다고 보면) 어긋나지 않는다.
- unverified: 길목·부시 노드는 대략 위치. 이동 시간에만 영향.

엣지 길이 = 직선 거리 × 계수. 계수는 벽을 돌아가는 정도를 나타낸다.
kk 연습 모드 실측(2026-10-20, 마스터 이 355, 신발 없음)으로 맞춘 구간:
  우물→블루 17초 / 우물→레드 16초 / 우물→칼날부리 15초(우물 이속 효과 포함, 첫 엣지 전체에 적용),
  칼날부리→레드 6초, 레드→돌거북 7초, 늑대→칼날부리 11~12초, 돌거북→아래 바위게 13초,
  블루→칼날부리 18초(블루→늑대 구간 계수를 여기서 역산). 나머지 계수는 미검증 추정값.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MX, MY = 14820, 14900

# 블루 진영(팀 100) 노드: id, x, y, 종류, 출처 수준, 부시 여부
BLUE = [
    ("B_fountain", 662, 285, "fountain", "stage1", False),
    ("B_blue", 3821, 8101, "camp", "consensus", False),
    ("B_gromp", 2288, 8448, "camp", "consensus", False),
    ("B_wolves", 3780, 6443, "camp", "consensus", False),
    ("B_raptors", 6943, 5422, "camp", "consensus", False),
    ("B_red", 7765, 4020, "camp", "consensus", False),
    ("B_krugs", 8394, 2641, "camp", "consensus", False),
    ("B_top_entrance", 2600, 9900, "jungle_entrance", "unverified", True),   # 탑 쪽 삼거리 부시
    ("B_bot_entrance", 9900, 2900, "jungle_entrance", "unverified", True),   # 봇 쪽 삼거리 부시
    ("top_B_outer", 981, 10441, "tower", "stage1", False),
    ("mid_B_outer", 5846, 6396, "tower", "stage1", False),
    ("bot_B_outer", 10504, 1029, "tower", "stage1", False),
]
# 중립 노드(대칭 짝을 직접 적음)
NEUTRAL = [
    ("top_center", 2300, 12600, "lane", "unverified", False),
    ("bot_center", 12520, 2300, "lane", "unverified", False),
    ("mid_center", 7410, 7450, "lane", "unverified", False),
    ("river_top_scuttle", 4400, 9640, "scuttle", "consensus", False),
    ("river_bot_scuttle", 10420, 5260, "scuttle", "consensus", False),
    ("river_top_brush", 3300, 11300, "river", "unverified", True),
    ("river_bot_brush", 11520, 3600, "river", "unverified", True),
    ("baron_pit", 5007, 10471, "epic", "stage1", False),
    ("dragon_pit", 9861, 4447, "epic", "stage1", False),
]
# 레드 우물은 0분 프레임 위치(stage1)가 대칭값과 약 300 차이라 실측값을 쓴다.
RED_FOUNTAIN = (14426, 14170)

# 블루 진영 엣지: (a, b, 계수). 1.15 = 보통 정글 길, 1.3 = 미드 라인을 건너거나 벽을 크게 도는 길
BLUE_EDGES = [
    # 우물 출발 엣지와 아래 kk_measured 표시 엣지는 kk 연습 모드 실측(마스터 이 이속 355, 2026-10-20)에서 역산
    ("B_fountain", "B_wolves", 1.25), ("B_fountain", "B_raptors", 1.127), ("B_fountain", "B_red", 1.212),
    ("B_fountain", "B_blue", 1.223),
    ("B_fountain", "top_B_outer", 1.05), ("B_fountain", "mid_B_outer", 1.05), ("B_fountain", "bot_B_outer", 1.05),
    ("B_blue", "B_gromp", 1.15), ("B_blue", "B_wolves", 1.38), ("B_blue", "B_top_entrance", 1.15),
    ("B_blue", "river_top_scuttle", 1.2), ("B_gromp", "B_top_entrance", 1.15), ("B_gromp", "B_wolves", 1.2),
    ("B_wolves", "mid_B_outer", 1.15), ("B_wolves", "B_raptors", 1.3),
    ("B_raptors", "B_red", 1.15), ("B_raptors", "mid_B_outer", 1.15), ("B_raptors", "mid_center", 1.15),
    ("B_raptors", "river_bot_scuttle", 1.25),
    ("B_red", "B_krugs", 1.63), ("B_red", "B_bot_entrance", 1.2), ("B_krugs", "B_bot_entrance", 1.15),
    ("B_bot_entrance", "bot_B_outer", 1.1), ("B_bot_entrance", "river_bot_brush", 1.15),
    ("B_bot_entrance", "dragon_pit", 1.2),
    ("B_top_entrance", "top_B_outer", 1.1), ("B_top_entrance", "river_top_brush", 1.15),
    ("B_top_entrance", "river_top_scuttle", 1.15),
    ("top_B_outer", "top_center", 1.05), ("mid_B_outer", "mid_center", 1.0), ("bot_B_outer", "bot_center", 1.05),
]
NEUTRAL_EDGES = [
    ("river_top_scuttle", "baron_pit", 1.1), ("river_top_scuttle", "mid_center", 1.1),
    ("river_top_scuttle", "river_top_brush", 1.1), ("river_top_brush", "top_center", 1.1),
    ("river_top_brush", "baron_pit", 1.15),
    ("river_bot_scuttle", "dragon_pit", 1.1), ("river_bot_scuttle", "mid_center", 1.1),
    ("river_bot_scuttle", "river_bot_brush", 1.1), ("river_bot_brush", "bot_center", 1.1),
    ("river_bot_brush", "dragon_pit", 1.15),
]


def mirror_id(nid):
    """블루 노드 id → 점대칭 레드 노드 id. 탑↔봇이 뒤바뀐다."""
    swap = {"top": "bot", "bot": "top"}
    if nid.startswith("B_"):
        rest = nid[2:]
        for a, b in swap.items():
            if rest.startswith(a + "_"):
                rest = b + rest[len(a):]
                break
        return "R_" + rest
    lane, _, tail = nid.partition("_")
    if tail.startswith("B_"):
        return f"{swap.get(lane, lane)}_R_{tail[2:]}"
    neutral = {"baron_pit": "dragon_pit", "dragon_pit": "baron_pit"}
    if nid in neutral:
        return neutral[nid]
    for a, b in swap.items():          # river_top_* ↔ river_bot_*, top_center ↔ bot_center
        if f"_{a}_" in f"_{nid}_":
            return f"_{nid}_".replace(f"_{a}_", f"_{b}_", 1).strip("_")
    return nid


def main():
    nodes = {}
    for nid, x, y, kind, level, brush in BLUE:
        nodes[nid] = {"x": x, "y": y, "kind": kind, "side": 100, "brush": brush, "level": level}
        rid = mirror_id(nid)
        rx, ry = (RED_FOUNTAIN if kind == "fountain" else (MX - x, MY - y))
        nodes[rid] = {"x": rx, "y": ry, "kind": kind, "side": 200, "brush": brush,
                      "level": level, "mirror_of": nid}
    for nid, x, y, kind, level, brush in NEUTRAL:
        nodes[nid] = {"x": x, "y": y, "kind": kind, "side": 0, "brush": brush, "level": level}

    # 레드 포탑은 대칭값 대신 실측 위치를 쓴다(stage1).
    red_towers = {"top_R_outer": (4318, 13875), "mid_R_outer": (8955, 8510), "bot_R_outer": (13866, 4505)}
    for nid, (x, y) in red_towers.items():
        nodes[nid].update({"x": x, "y": y})

    edges, seen = [], set()
    for a, b, f in BLUE_EDGES + NEUTRAL_EDGES:
        for x, y in ((a, b), (mirror_id(a), mirror_id(b))):
            key = frozenset((x, y))
            if key not in seen:
                seen.add(key)
                edges.append([x, y, f])
    # 중립 노드도 대칭 짝이 서로 맞는지 확인
    for nid, x, y, *_ in NEUTRAL:
        m = nodes[mirror_id(nid)]
        assert abs(m["x"] - (MX - x)) <= 150 and abs(m["y"] - (MY - y)) <= 150, nid

    missing = {n for e in edges for n in e[:2]} - set(nodes)
    assert not missing, missing

    out = {
        "_doc": "맵 노드 그래프. 좌표는 Riot 타임라인 좌표계(0~약 14900). 엣지 [a, b, 계수]: 이동 거리 = 직선 거리 × 계수. "
                "생성: python tools/build_map_graph.py (손으로 고치지 말고 스크립트를 고칠 것).",
        "level_meaning": {
            "stage1": "1단계 경기 이벤트 위치 중앙값으로 확인(포탑·둥지·우물)",
            "consensus": "흔히 쓰는 좌표. 캠프는 1분 프레임 정글러 위치와 420~510 차이(공격 사거리만큼 떨어져 섬)",
            "unverified": "길목·부시·라인 중앙 대략 위치",
        },
        "edge_factor": {"value": "1.0~1.3", "level": "unverified", "note": "벽을 돌아가는 정도. 실측 이동 시간 자료가 없어 추정값"},
        "fixed_node_attributes": "부시 여부(brush), 종류(kind), 진영(side)만 고정. 적 시야 여부는 매 틱 와드·챔피언 위치로 계산(3단계 결정).",
        "nodes": nodes,
        "edges": edges,
    }
    dst = os.path.join(ROOT, "rules", "map_nodes.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"노드 {len(nodes)}개, 엣지 {len(edges)}개 → {dst}")


if __name__ == "__main__":
    main()
