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
    trades: list = field(default_factory=list)   # 성공했어도 공격 쪽이 잃은 사람 (피해자, 처치자, 도움)


LANE_OF_ROLE = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}


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
         + coef["at_tower"] * (1.0 if inp.at_tower else 0.0)
         + coef.get("lane_bias", {}).get(LANE_OF_ROLE.get(t.role), 0.0)
         + coef.get("item_gold_diff", 0.0) * (sum(c.item_gold for c in inp.attackers) / len(inp.attackers)
                                              - sum(c.item_gold for c in inp.defenders) / len(inp.defenders)) / 1000)
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
        if rng.random() < p.get("success_attacker_death", 0.0):
            others = [d for d in inp.defenders if d is not t]
            k = rng.choice(others) if others else t
            res.trades.append((jungler, k, [d for d in inp.defenders if d is not k]))
    elif rng.random() < p["fail_attacker_death"]:
        victim = inp.attackers[0]
        res.deaths.append(victim)
        res.killer = t
        res.assists = [c for c in inp.defenders if c is not t]
    return res


@dataclass
class SkirmishResult:
    winner: int
    deaths: list = field(default_factory=list)      # (victim, killer, assists)
    hp_after: dict = field(default_factory=dict)
    duration_s: float = 0.0
    prob: float = 0.0


def resolve_skirmish(sides, now, rng, p):
    """정글·강가 조우 교전. sides = {팀: [Champ, ...]}. 첫 번째 원소가 그 팀 정글러.

    승리 확률은 인원·레벨 합·체력 합 차이의 로지스틱. 진 쪽에서 1명이 죽고, 일정 확률로 1명 더,
    이긴 쪽도 일정 확률로 1명 잃는다(맞교환). 계산식은 자리표시이며 킬 이벤트로 보정한다.
    """
    (ta, a), (tb, b) = sides.items()
    c = p["coef"]
    x = (c["num_adv"] * (len(a) - len(b))
         + c["level_diff"] * (sum(m.level for m in a) - sum(m.level for m in b)) / max(len(a), len(b))
         + c["hp_diff"] * (sum(m.hp for m in a) - sum(m.hp for m in b))
         + c.get("item_gold_diff", 0.0) * (sum(m.item_gold for m in a) - sum(m.item_gold for m in b)) / max(len(a), len(b)) / 1000
         + c.get("legendary_diff", 0.0) * (sum(m.legendaries for m in a) - sum(m.legendaries for m in b)) / max(len(a), len(b)))
    prob = logistic(x)
    win, lose = (a, b) if rng.random() < prob else (b, a)
    res = SkirmishResult(winner=(ta if win is a else tb), duration_s=p["duration_s"], prob=prob)
    lo, hi = p["hp_cost"]
    for m in a + b:
        res.hp_after[m.pid] = max(0.05, m.hp - rng.uniform(lo, hi))

    def take(victims, killers):
        v = min(victims, key=lambda m: res.hp_after[m.pid])
        victims.remove(v)
        k = rng.choice(killers)
        res.deaths.append((v, k, [m for m in killers if m is not k]))

    lose_left, win_left = list(lose), list(win)
    if rng.random() < p["loser_death"]:
        take(lose_left, win_left)
        if lose_left and rng.random() < p["second_death"]:
            take(lose_left, win_left)
        if win_left and lose_left and rng.random() < p["trade_death"]:
            take(win_left, lose_left)
    return res


def resolve_group(sides, now, rng, p):
    """여러 명이 모인 교전(한타·오브젝트 싸움). 이기는 확률은 resolve_skirmish와 같은 식, 진 쪽은 한 명씩 p["lose_death"],
    이긴 쪽은 p["win_death"] 확률로 죽는다(인원이 많을수록 사망도 많음). 킬 관여 = 그 교전의 살아남은 상대 전원."""
    (ta, a), (tb, b) = sides.items()
    c = p["coef"]
    x = (c["num_adv"] * (len(a) - len(b))
         + c["level_diff"] * (sum(m.level for m in a) - sum(m.level for m in b)) / max(len(a), len(b))
         + c["hp_diff"] * (sum(m.hp for m in a) - sum(m.hp for m in b))
         + c.get("item_gold_diff", 0.0) * (sum(m.item_gold for m in a) - sum(m.item_gold for m in b)) / max(len(a), len(b)) / 1000
         + c.get("legendary_diff", 0.0) * (sum(m.legendaries for m in a) - sum(m.legendaries for m in b)) / max(len(a), len(b)))
    prob = logistic(x)
    win, lose = (a, b) if rng.random() < prob else (b, a)
    res = SkirmishResult(winner=(ta if win is a else tb), duration_s=p["duration_s"], prob=prob)
    lo, hi = p["hp_cost"]
    for m in a + b:
        res.hp_after[m.pid] = max(0.05, m.hp - rng.uniform(lo, hi))
    dead_l = [m for m in lose if rng.random() < p["lose_death"]]
    dead_w = [m for m in win if rng.random() < p["win_death"]]
    if len(dead_w) >= len(win):
        dead_w = dead_w[:-1]
    alive_w = [m for m in win if m not in dead_w]
    alive_l = [m for m in lose if m not in dead_l]
    for v in dead_l:
        k = rng.choice(alive_w)
        res.deaths.append((v, k, [m for m in alive_w if m is not k]))
    for v in dead_w:
        killers = alive_l or [m for m in lose if m is not v] or lose
        k = rng.choice(killers)
        res.deaths.append((v, k, [m for m in killers if m is not k]))
    return res
