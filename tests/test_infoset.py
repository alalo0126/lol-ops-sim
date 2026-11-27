"""맵핵 금지(공정 비교 설계 2장): 학습 정글러 관측은 보이지 않는 적의 실제 상태가 바뀌어도 그대로여야 한다."""
import copy
import unittest

import numpy as np

from sim.agent import AgentJungler
from sim.game import Game
from sim.winprob import WinProb


def _game(seed):
    holder = {}

    def pol(obs, m, mw, ms):
        return int(np.flatnonzero(m)[0]), 0, 0, None

    def make(gm, ch):
        holder["a"] = AgentJungler(gm, ch, pol, WinProb())
        return holder["a"]

    g = Game(seed=seed, jungler_override={100: make})
    return g, holder["a"]


class InfoSetTest(unittest.TestCase):
    def test_hidden_enemy_changes_do_not_leak(self):
        checked = 0
        for seed in range(4):
            g, agent = _game(seed)
            for t in (200, 420, 700, 950):
                while g.t < t:
                    g.step()
                vis = g.vision(100)
                hidden = [e for e in g.champs if e.team == 200 and e.alive(g.t) and e.node not in vis]
                if not hidden:
                    continue
                st = (copy.deepcopy(agent.known), copy.deepcopy(agent.known_cs), agent._ev_i)
                base = agent.observe()
                saved = [(e, e.hp, e.flash_at, e.item_gold, e.task, e.busy_until, e.earned, e.xp) for e in hidden]
                for e in hidden:
                    e.hp = 0.05 if e.hp > 0.5 else 0.95
                    e.flash_at = g.t + 300 if e.flash_ready(g.t) else 0.0
                    e.item_gold += 2000
                    e.earned += 3000          # 골드는 점수판으로만 추정(sim/scoreboard.py)
                    e.xp += 400 if e.level < 18 and e.xp + 400 < g.rules.levels[e.level] else 0
                    e.task = {"kind": "objective", "name": "dragon", "started": True}
                agent.known, agent.known_cs, agent._ev_i = st
                after = agent.observe()
                for e, hp, fa, ig, task, bu, ea, xp in saved:
                    e.hp, e.flash_at, e.item_gold, e.task, e.busy_until, e.earned, e.xp = hp, fa, ig, task, bu, ea, xp
                agent.known, agent.known_cs, agent._ev_i = st
                np.testing.assert_array_equal(base, after, err_msg=f"seed {seed} t {g.t}: 안 보이는 적 상태가 관측에 샘")
                checked += 1
        self.assertGreater(checked, 5)

    def test_flash_cooldown_known_only_when_seen(self):
        g, _ = _game(0)
        e = next(x for x in g.champs if x.team == 200 and x.role == "MIDDLE")
        e.node = "R_fountain_far" if "R_fountain_far" in g.map.nodes else "R_raptors"
        if e.node in g.vision(100):
            self.skipTest("시험 노드가 블루 시야 안")
        e.flash_at = g.t + 300
        g.note_flash(e)
        self.assertTrue(g.known_flash_ready(100, e))


if __name__ == "__main__":
    unittest.main()
