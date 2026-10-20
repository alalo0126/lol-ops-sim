import os
import unittest

from sim.data import ROOT
from sim.game import Game
from sim.stage6 import CATS, label, zone
from sim.winprob import WinProb

DATA = os.path.join(ROOT, "data", "stage6_compact.json.gz")


class ZoneTest(unittest.TestCase):
    def test_sim_nodes_fall_in_their_own_zone(self):
        nodes = Game(seed=0).map.nodes
        expect = {"lane": "lane", "tower": "lane", "fountain": "base", "epic": "pit", "river": "river", "scuttle": "river"}
        for nid, n in nodes.items():
            z = zone(n["x"], n["y"], 100, nodes)
            if n["kind"] in expect:
                self.assertEqual(z, expect[n["kind"]], nid)
            else:
                self.assertEqual(z, "own_jungle" if n["side"] == 100 else "enemy_jungle", nid)

    def test_label_priority(self):
        nodes = Game(seed=0).map.nodes
        w = {"end_xy": (nodes["R_red"]["x"], nodes["R_red"]["y"]), "cs_gain": 4, "elite": 0, "fights": [],
             "recalled": False, "obj_live": False, "scuttle_near": False}
        self.assertEqual(label(w, 100, nodes), "invade")
        self.assertEqual(label(w, 200, nodes), "farm")
        self.assertEqual(label({**w, "recalled": True}, 100, nodes), "recall")
        self.assertEqual(label({**w, "elite": 1}, 100, nodes), "objective")
        lane = nodes["mid_center"]
        self.assertEqual(label({**w, "fights": [(lane["x"], lane["y"])]}, 100, nodes), "lane")
        self.assertTrue(all(c in CATS for c in ("farm", "invade", "lane", "objective", "river", "recall")))


@unittest.skipUnless(os.path.exists(DATA), "data/stage6_compact.json.gz 없음(PC에서 tools/stage6_compact.py)")
class ReconstructTest(unittest.TestCase):
    def test_reconstructed_state_has_real_win_prob_and_runs(self):
        from sim.stage6 import RealMatch, load_compact, reconstruct, run_sim_minute
        wp = WinProb()
        rm = next(RealMatch(m) for m in load_compact() if RealMatch(m).valid)
        for minute in (4, 10):
            g, bot = reconstruct(rm, 100, minute, None, wp, seed=1)
            self.assertAlmostEqual(wp.blue(g), wp.blue_from_diffs(rm.diffs(minute), minute * 60), places=6)
            lab, gains = run_sim_minute(rm, 100, minute, None, wp, seed=1)
            self.assertIn(lab, CATS)
            self.assertGreaterEqual(gains["gold"], 0)


if __name__ == "__main__":
    unittest.main()
