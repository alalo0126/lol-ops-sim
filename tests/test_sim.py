import unittest

from sim.data import Rules
from sim.game import Game
from sim.mapgraph import MapGraph
from sim.vision import detect_prob, exposure_on_path, team_vision

RULES = Rules()


class MapGraphTest(unittest.TestCase):
    def setUp(self):
        self.g = MapGraph(RULES.map)

    def test_size_and_connected(self):
        self.assertTrue(30 <= len(self.g.nodes) <= 40)   # 설계 문서: 30~40개 노드
        self.assertTrue(self.g.is_connected())

    def test_sides_are_roughly_symmetric(self):
        for a, b in [("B_fountain", "B_krugs"), ("B_blue", "dragon_pit"), ("B_raptors", "top_B_outer")]:
            ma = a.replace("B_", "R_", 1) if a.startswith("B_") else a
            mb = {"dragon_pit": "baron_pit", "top_B_outer": "bot_R_outer"}.get(b, b.replace("B_", "R_", 1))
            self.assertAlmostEqual(self.g.dist(a, b), self.g.dist(ma, mb), delta=0.05 * self.g.dist(a, b))

    def test_path_ends_at_destination(self):
        p = self.g.path("B_fountain", "R_fountain")
        self.assertEqual(p[-1], "R_fountain")
        self.assertEqual(self.g.path("B_red", "B_red"), [])


class VisionTest(unittest.TestCase):
    """3단계 문서의 4단계 테스트: 적 와드 노드를 지나면 발각 확률 > 0, 없는 노드만 지나면 0."""

    def test_detection_needs_enemy_vision(self):
        nodes = RULES.map["nodes"]
        ward = [{"team": 200, "node": "river_bot_brush", "until": 999}]
        vis = team_vision(200, [], ward, nodes, now=0)
        prof = RULES.params["cognition"]["profiles"]["challenger"]
        through_ward = exposure_on_path(["B_bot_entrance", "river_bot_brush"], vis, 600, 345)
        no_ward = exposure_on_path(["B_red", "B_bot_entrance"], vis, 600, 345)
        self.assertGreater(detect_prob(through_ward, prof), 0)
        self.assertEqual(detect_prob(no_ward, prof), 0)

    def test_short_exposure_is_probabilistic(self):
        prof = {"short_exposure_detect": 0.7}
        self.assertAlmostEqual(detect_prob([0.5], prof), 0.35)
        self.assertEqual(detect_prob([1.5], prof), 1.0)


class RulesTest(unittest.TestCase):
    def test_level_boundaries(self):
        self.assertEqual(RULES.level_for_xp(279), 1)
        self.assertEqual(RULES.level_for_xp(280), 2)
        self.assertEqual(RULES.level_for_xp(1211), 4)

    def test_first_three_camps_match_stage1_two_minute_median(self):
        """랩터 → 레드 → 크루그 레벨 1 보상 = 1단계 2분 중앙값(정글 CS 12, 경험치 676)."""
        g = Game(seed=0)
        j = next(c for c in g.champs if c.role == "JUNGLE")
        gold = xp = 0
        for kind in ("raptors", "red", "krugs"):
            gg, xx = g.camp_rewards(j, g.camps[f"B_{kind}"])
            gold, xp = gold + gg, xp + xx
        self.assertEqual(round(xp), 676)
        self.assertEqual(round(gold), 274)


class GameTest(unittest.TestCase):
    def test_runs_twenty_minutes_deterministically(self):
        a, b = Game(seed=7).run(), Game(seed=7).run()
        self.assertEqual(a.frames[-1], b.frames[-1])
        self.assertEqual(a.frames[-1]["t"], 1200)
        for c in a.champs:
            if c.role == "JUNGLE":
                self.assertEqual(c.jungle_cs % 4, 0)
                self.assertGreater(c.jungle_cs, 40)


if __name__ == "__main__":
    unittest.main()
