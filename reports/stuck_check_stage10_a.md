# 굳은 행동 점검

- 정책 `models/stage10_a/policy_latest.npz`(바닥 0.002), 확률 조사 100판, 행동마다 강제 200쌍.
- 드문 행동 = 고를 수 있는 결정의 5% 이상에서 열려 있는데 평균 확률 < 0.3%.

| 행동 | 열린 비율 | 평균 확률 | 강제 쌍 | 20분 승률 차(강제 − 기준) | 95% 구간 | 판정 |
|---|---|---|---|---|---|---|
| invade_wolves | 100% | 0.1692% | 200 | -0.0049 | -0.0503~+0.0404 | 차이 작음 |
| cover_top | 93% | 0.2231% | 200 | -0.0466 | -0.0928~-0.0005 | 진짜 나쁜 행동 |
| cover_bot | 93% | 0.1600% | 200 | -0.0357 | -0.0853~+0.0139 | 차이 작음 |
| lanefarm_bot | 93% | 0.1900% | 200 | -0.0139 | -0.0591~+0.0314 | 차이 작음 |
| move_own_top_entrance | 99% | 0.2517% | 200 | -0.0398 | -0.0840~+0.0045 | 차이 작음 |

## 모든 주 행동 평균 확률(고를 수 있을 때)

| 행동 | 열린 비율 | 평균 확률 |
|---|---|---|
| obj_dragon | 9% | 82.124% |
| obj_herald | 4% | 76.800% |
| obj_grubs | 4% | 61.659% |
| camp_raptors | 100% | 8.544% |
| gank_bot | 91% | 8.500% |
| camp_wolves | 100% | 8.303% |
| scuttle_bot | 100% | 8.114% |
| camp_gromp | 100% | 7.825% |
| camp_blue | 100% | 7.175% |
| scuttle_top | 100% | 6.485% |
| gank_mid | 90% | 5.142% |
| gank_top | 81% | 3.740% |
| camp_krugs | 100% | 3.037% |
| camp_red | 100% | 2.917% |
| recall | 100% | 2.391% |
| move_dragon_pit | 97% | 2.087% |
| invade_krugs | 100% | 1.557% |
| move_enemy_bot_entrance | 100% | 1.385% |
| lanefarm_mid | 93% | 1.265% |
| invade_raptors | 100% | 1.219% |
| invade_blue | 100% | 1.168% |
| cover_mid | 93% | 1.133% |
| move_river_top_brush | 98% | 1.119% |
| move_baron_pit | 98% | 1.011% |
| wait | 100% | 0.893% |
| move_river_bot_brush | 97% | 0.608% |
| invade_red | 100% | 0.571% |
| invade_gromp | 100% | 0.458% |
| lanefarm_top | 93% | 0.443% |
| move_enemy_top_entrance | 100% | 0.413% |
| move_own_bot_entrance | 99% | 0.314% |
| move_own_top_entrance | 99% | 0.252% |
| cover_top | 93% | 0.223% |
| lanefarm_bot | 93% | 0.190% |
| invade_wolves | 100% | 0.169% |
| cover_bot | 93% | 0.160% |
