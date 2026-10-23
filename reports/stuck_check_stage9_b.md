# 굳은 행동 점검

- 정책 `models/stage9_b/policy_latest.npz`(바닥 0.0), 확률 조사 100판, 행동마다 강제 200쌍.
- 드문 행동 = 고를 수 있는 결정의 5% 이상에서 열려 있는데 평균 확률 < 0.3%.

| 행동 | 열린 비율 | 평균 확률 | 강제 쌍 | 20분 승률 차(강제 − 기준) | 95% 구간 | 판정 |
|---|---|---|---|---|---|---|
| invade_blue | 100% | 0.2417% | 200 | -0.0171 | -0.0635~+0.0292 | 차이 작음 |
| invade_red | 100% | 0.1986% | 200 | -0.0464 | -0.0911~-0.0017 | 진짜 나쁜 행동 |
| cover_mid | 94% | 0.2860% | 200 | -0.0483 | -0.0917~-0.0048 | 진짜 나쁜 행동 |
| cover_bot | 94% | 0.1043% | 200 | -0.0410 | -0.0855~+0.0036 | 차이 작음 |
| move_own_top_entrance | 99% | 0.1622% | 200 | -0.0267 | -0.0666~+0.0132 | 차이 작음 |
| move_own_bot_entrance | 99% | 0.2027% | 200 | -0.0550 | -0.1004~-0.0095 | 진짜 나쁜 행동 |
| move_enemy_top_entrance | 100% | 0.0759% | 200 | -0.0144 | -0.0579~+0.0290 | 차이 작음 |
| move_enemy_bot_entrance | 100% | 0.0547% | 200 | -0.0486 | -0.0923~-0.0050 | 진짜 나쁜 행동 |
| move_river_top_brush | 99% | 0.1181% | 200 | -0.0129 | -0.0563~+0.0305 | 차이 작음 |

## 모든 주 행동 평균 확률(고를 수 있을 때)

| 행동 | 열린 비율 | 평균 확률 |
|---|---|---|
| obj_dragon | 10% | 81.838% |
| obj_herald | 6% | 55.654% |
| obj_grubs | 4% | 48.342% |
| camp_wolves | 100% | 11.184% |
| gank_bot | 91% | 10.412% |
| camp_gromp | 100% | 9.610% |
| camp_raptors | 100% | 8.893% |
| scuttle_bot | 100% | 8.493% |
| camp_blue | 100% | 6.352% |
| scuttle_top | 100% | 5.569% |
| gank_mid | 91% | 4.809% |
| gank_top | 81% | 3.354% |
| recall | 100% | 3.201% |
| camp_krugs | 100% | 2.917% |
| wait | 100% | 2.877% |
| camp_red | 100% | 2.715% |
| move_dragon_pit | 97% | 1.222% |
| lanefarm_top | 94% | 1.182% |
| invade_krugs | 100% | 1.008% |
| lanefarm_mid | 94% | 0.718% |
| move_baron_pit | 98% | 0.708% |
| cover_top | 94% | 0.522% |
| move_river_bot_brush | 96% | 0.471% |
| invade_gromp | 100% | 0.449% |
| invade_raptors | 100% | 0.445% |
| invade_wolves | 100% | 0.373% |
| lanefarm_bot | 94% | 0.300% |
| cover_mid | 94% | 0.286% |
| invade_blue | 100% | 0.242% |
| move_own_bot_entrance | 99% | 0.203% |
| invade_red | 100% | 0.199% |
| move_own_top_entrance | 99% | 0.162% |
| move_river_top_brush | 99% | 0.118% |
| cover_bot | 94% | 0.104% |
| move_enemy_top_entrance | 100% | 0.076% |
| move_enemy_bot_entrance | 100% | 0.055% |
