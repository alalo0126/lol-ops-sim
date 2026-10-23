import unittest

from sim.ppo import Policy
from sim.replay import POLICY, STEP_S, record_sim
from sim.winprob import WinProb


class ReplayTest(unittest.TestCase):
    def test_sim_record_shapes(self):
        r = record_sim(0, 100, Policy.from_file(POLICY), WinProb())
        n = 1200 // STEP_S + 1
        for key in ("pos", "hp", "task", "camps", "obj", "wp", "stats"):
            self.assertEqual(len(r[key]), n, key)
        self.assertEqual(len(r["pos"][0]), 20)
        self.assertEqual(len(r["cats"]["100"]), 20)
        for row in r["pos"]:
            self.assertTrue(all(0 <= v <= 15000 for v in row))
        self.assertTrue(r["agent_actions"])


if __name__ == "__main__":
    unittest.main()
