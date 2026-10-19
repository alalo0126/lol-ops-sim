"""시야와 인지 제약.

3단계 결정: 노드에 '적 와드 시야' 같은 고정 속성은 두지 않고, 매 틱 와드·챔피언·포탑·미니언 위치로
팀별 시야 노드 집합을 계산한다. 진입을 미리 봤는지는 지나온 노드의 노출 시간을 인지 제약에 통과시켜 0~1로 만든다.
"""
import math


def team_vision(team, champs, wards, nodes, now):
    """team이 보는 노드 집합: 살아 있는 아군 챔피언 노드, 유효한 아군 와드, 아군 포탑·우물, 라인 중앙(양 팀 미니언)."""
    seen = {c.node for c in champs if c.team == team and c.alive(now)}
    seen |= {w["node"] for w in wards if w["team"] == team and w["until"] > now}
    for nid, n in nodes.items():
        if n["kind"] == "lane" or (n["side"] == team and n["kind"] in ("tower", "fountain")):
            seen.add(nid)
    return seen


def detect_prob(exposures_s, profile):
    """노출 시간 목록(초) → 한 번이라도 알아챌 확률.

    1초 이상 노출은 확실히 감지, 1초 미만은 short_exposure_detect × 노출 비율로만 감지한다.
    노출이 하나도 없으면 0.
    """
    p_miss = 1.0
    for e in exposures_s:
        if e <= 0:
            continue
        p = 1.0 if e >= 1.0 else profile["short_exposure_detect"] * e
        p_miss *= 1.0 - min(1.0, p)
    return 1.0 - p_miss


def exposure_on_path(path_nodes, enemy_vision, dwell_radius, ms):
    """경로가 지나는 노드 중 적 시야 안인 노드마다 머무는 시간(반지름 ÷ 이속)."""
    dwell = dwell_radius / ms
    return [dwell for n in path_nodes if n in enemy_vision]


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))
