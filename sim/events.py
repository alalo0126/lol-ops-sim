"""이벤트 계산 함수(교전 층). 입출력 형식은 설계 문서대로 고정하고, 계산식은 교체 가능한 자리표시다.

입력: 참가자 체력·레벨, 점멸 유무, 웨이브 위치, 상대가 진입을 미리 봤는지, 합류 가능한 아군 수
출력: 성공/실패, 사망자, 점멸 소모, 남은 체력, 소요 시간
"""
from dataclasses import dataclass, field

from .vision import logistic


@dataclass
class GankInput:
    attackers: list          # Champ 목록, 첫 번째가 갱 온 정글러
    defenders: list          # 같은 노드의 적 라이너
    target: object           # defenders 중 노리는 대상
    seen_prob: float         # 상대가 진입을 미리 봤을 확률(0~1)
    target_push: float       # 대상 기준 웨이브 위치(0 = 자기 포탑, 1 = 상대 포탑)
    at_tower: bool           # 대상이 자기 포탑 아래로 물러나 있음


@dataclass
class GankResult:
    success: bool
    deaths: list = field(default_factory=list)
    killer: object = None
    assists: list = field(default_factory=list)
    flash_used: list = field(default_factory=list)
    hp_after: dict = field(default_factory=dict)
    duration_s: float = 0.0
    prob: float = 0.0


def gank_success_prob(inp: GankInput, now, coef):
    t = inp.target
    lvl_att = sum(c.level for c in inp.attackers) / len(inp.attackers)
    lvl_def = sum(c.level for c in inp.defenders) / len(inp.defenders)
    x = (coef["bias"]
         + coef["target_low_hp"] * (0.5 - t.hp) * 2
         + coef["num_adv"] * (len(inp.attackers) - len(inp.defenders))
         + coef["level_diff"] * (lvl_att - lvl_def)
         + coef["target_flash"] * (1.0 if t.flash_ready(now) else 0.0)
         + coef["seen"] * inp.seen_prob
         + coef["target_pushed"] * (inp.target_push - 0.5)
         + coef["at_tower"] * (1.0 if inp.at_tower else 0.0))
    return logistic(x)


def resolve_gank(inp: GankInput, now, rng, p):
    """p = sim/params.json gank 블록."""
    prob = gank_success_prob(inp, now, p["coef"])
    res = GankResult(success=rng.random() < prob, duration_s=p["duration_s"], prob=prob)
    lo, hi = p["hp_cost"]
    for c in inp.attackers + inp.defenders:
        res.hp_after[c.pid] = max(0.05, c.hp - rng.uniform(lo, hi))
    t = inp.target
    if t.flash_ready(now):
        res.flash_used.append(t)
    if res.success:
        res.deaths.append(t)
        jungler = inp.attackers[0]
        laners = inp.attackers[1:]
        if not laners or rng.random() < p["jungler_gets_kill"]:
            res.killer = jungler
        else:
            res.killer = rng.choice(laners)
        res.assists = [c for c in inp.attackers if c is not res.killer]
    elif rng.random() < p["fail_attacker_death"]:
        victim = inp.attackers[0]
        res.deaths.append(victim)
        res.killer = t
        res.assists = [c for c in inp.defenders if c is not t]
    return res
