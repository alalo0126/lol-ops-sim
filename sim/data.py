"""rules/*.json과 sim/params.json 읽기. 규칙 항목은 {value, src, level} 형태라 value만 꺼내 쓴다."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


def values(tree):
    """{value: ..} 항목을 값으로 바꾼 사본. _doc 같은 설명 키는 버린다."""
    if isinstance(tree, dict):
        if "value" in tree and ("level" in tree or "src" in tree):
            return tree["value"]
        return {k: values(v) for k, v in tree.items() if not k.startswith("_")}
    return tree


class Rules:
    def __init__(self):
        self.camps = load_json("rules", "camps.json")["camps"]
        self.objectives = values(load_json("rules", "objectives.json"))
        self.positions = values(load_json("rules", "positions.json"))
        self.combat = values(load_json("rules", "combat.json"))
        self.levels = values(load_json("rules", "levels.json"))["xp_to_reach"]
        self.map = load_json("rules", "map_nodes.json")
        self.champions = load_json("rules", "champions.json")["champions"]
        self.params = values(load_json("sim", "params.json"))
        self.passive_start_s = load_json("rules", "combat.json")["passive_gold_per_s"]["start_s"]

    def level_for_xp(self, xp):
        lv = 1
        for i, need in enumerate(self.levels):
            if xp >= need:
                lv = i + 1
        return min(lv, 18)


def lerp_level(rng, level):
    """[레벨 1 값, 최대 레벨 값] 범위를 몬스터 레벨 1~18 선형 보간. 성장 곡선은 미검증(UNVERIFIED.md)."""
    if not isinstance(rng, list):
        return rng
    lo, hi = rng
    return lo + (hi - lo) * (max(1, min(18, level)) - 1) / 17
