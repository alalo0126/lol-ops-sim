import unittest

import numpy as np

from sim.agent import N_MAIN, AgentJungler
from sim.game import Game
from sim.ppo import HEADS, N_OUT, Policy
from sim.rollout import play
from sim.teacher import play_teacher
from sim.winprob import WinProb

WP = WinProb()


def random_policy(seed):
    rng = np.random.default_rng(seed)

    def pol(obs, m, mw, ms):
        return int(rng.choice(np.flatnonzero(m))), int(rng.choice(np.flatnonzero(mw))), int(rng.choice(np.flatnonzero(ms))), {"obs": obs}
    return pol


class WinProbTest(unittest.TestCase):
    def test_teams_sum_to_one_and_even_game_near_half(self):
        g = Game(seed=3).run(until=600)
        self.assertAlmostEqual(WP.for_team(g, 100) + WP.for_team(g, 200), 1.0)
        even = WinProb({"features": ["gold"], "scale": {"gold": 1000.0},
                        "minutes": {"1": {"intercept": 0.0, "coef": {"gold": 0.5}}, "20": {"intercept": 0.0, "coef": {"gold": 0.5}}}})
        g2 = Game(seed=3)
        self.assertAlmostEqual(even.blue(g2), 0.5)

    def test_more_gold_more_win(self):
        g = Game(seed=1).run(until=900)
        before = WP.for_team(g, 100)
        g.give_gold(g.champs[0], 2000)
        self.assertGreater(WP.for_team(g, 100), before)


class AgentTest(unittest.TestCase):
    def test_random_agent_runs_and_rewards_telescope(self):
        for team in (100, 200):
            r = play(7, team, random_policy(team))
            self.assertGreater(len(r["traj"]), 50)
            total = sum(t["r"] for t in r["traj"])
            # 보상 합 = 20분 Φ − 첫 결정 때 Φ(첫 결정 시점엔 양 팀 차이가 없어 0초 상태와 같음)
            self.assertAlmostEqual(total, r["phi"] - WP.for_team(Game(seed=7), team), places=6)
            self.assertTrue(0.0 <= r["phi"] <= 1.0)

    def test_teacher_matches_rule_bot_roughly(self):
        """학습 정글러와 같은 행동 인터페이스로 규칙 봇을 돌리면 10분 CS가 규칙 봇과 비슷해야 한다."""
        a = [play_teacher(s, 100 if s % 2 == 0 else 200)["stats"]["cs10"] for s in range(20)]
        b = [play(s, 100 if s % 2 == 0 else 200)["stats"]["cs10"] for s in range(20)]
        self.assertLess(abs(np.mean(a) - np.mean(b)), 6)

    def test_reroute_keeps_progress(self):
        g = Game(seed=0)
        c = next(x for x in g.champs if x.role == "JUNGLE" and x.team == 100)
        c.node, c.path, c.task = "B_raptors", [], None
        g.move_to(c, "B_red")
        full = c.edge_left
        c.edge_left -= 300          # 300만큼 감
        g.move_to(c, "B_krugs")     # 레드 너머 돌거북으로 목적지 변경: 앞으로 계속
        self.assertEqual(c.path[0], "B_red")
        self.assertAlmostEqual(c.edge_left, full - 300)
        g.move_to(c, "B_raptors")   # 되돌아가기: 간 거리만큼만 돌아감
        self.assertEqual(c.path, ["B_raptors"])
        self.assertAlmostEqual(c.edge_left, 300)


class PPOTest(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(3)
        P = Policy(10, seed=1, hidden=8)
        P.pi[-2] *= 50
        n = 6
        X = rng.standard_normal((n, 10))
        M = rng.random((n, N_OUT)) > 0.3
        for a, b in HEADS:
            M[:, a] = True
        A = np.zeros((n, 3), dtype=int)
        for i in range(n):
            for k, (a, b) in enumerate(HEADS):
                A[i, k] = rng.choice(np.flatnonzero(M[i, a:b]))
        A[:3, 0] = 0
        lp0, adv, ret = rng.normal(-3, 0.3, n), rng.standard_normal(n), rng.standard_normal(n)

        def loss():
            gpi, gvf, info = P.loss_grads(X, M, A, lp0, adv, ret, clip=10.0, ent_coef=0.05)
            return info["pg_loss"] - 0.05 * info["entropy"] + 0.5 * info["v_loss"], gpi, gvf
        _, gpi, gvf = loss()
        for params, grads in ((P.pi, gpi), (P.vf, gvf)):
            for p, g in zip(params, grads):
                idx = tuple(rng.integers(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + 1e-6
                lp = loss()[0]
                p[idx] = old - 1e-6
                lm = loss()[0]
                p[idx] = old
                self.assertAlmostEqual((lp - lm) / 2e-6, g[idx], places=5)

    def test_masked_actions_never_sampled(self):
        P = Policy(5, seed=0)
        m = np.zeros(N_MAIN, dtype=bool)
        m[[3, 9]] = True
        for _ in range(50):
            a, w, s, rec = P.act(np.zeros(5, dtype=np.float32), m, np.array([True, False, False]), np.array([False, True]))
            self.assertIn(a, (3, 9))
            self.assertEqual((w, s), (0, 1))


if __name__ == "__main__":
    unittest.main()
