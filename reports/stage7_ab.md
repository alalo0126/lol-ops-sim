# 정글러 두 갈래 본 비교: 라이너 학습·한타 보강 뒤 (2026-10-23)

둘 다 `models/stage6c`에서 출발해 PPO 300회(학습 경기 시드 500000~, 같은 시드). 라이너 8명은 라이너 1·2차와 한타 보강을 넣은 규칙 봇, 상대 정글은 규칙 봇.

- 갈래 A `models/stage7_a/policy_latest.npz`(= 300회째 policy_0300): 시뮬 20분 예측 승률만.
- 갈래 B `models/stage7_b/policy_latest.npz`: 승률 + 실제 천상계 정글러 행동 모방(`sim/train.py --real models/stage7_real/real_train.npz --real-coef 0.1`).
  - 모방 표본: 리플레이 70판(26.21), 3~17분 분 시작 + 그 1분 동안의 모든 결정 시각 = 학습 10,941개 / 시험 2,787개(경기 단위로 나눔).
    지난 기준점(stage6d)은 분 시작만 1,205개였다(`sim/imitate.py`, `--minute-start-only`로 예전 방식).
  - 처음에 기준점과 같은 세기 0.3으로 돌렸더니 B가 귀환을 남발해(판당 약 46번) 학습 경기 승률 지표가 0.55 → 0.24로 무너져 0.1로 낮췄다(`models/stage7_b_bad`).
  - 모방 손실은 이제 주 행동 머리만 쓴다(와드·강타 선택은 정답이 없어 손실이 수억으로 튀었음, `sim/teacher.py bc_grads(heads=)`).

| | 갈래 A | 갈래 B | 규칙 봇 | 실제 |
|---|---|---|---|---|
| 규칙 봇 대비 20분 예측 승률(400판, 같은 시드) | 0.580 (+0.047, 95% +0.008~+0.086) | 0.576 (+0.044, +0.004~+0.084) | 0.532 | - |
| 시험 경기 2,787개 실제 행동 일치 / 행동 묶음 | 19% / 30% | 55% / 65% | - | - |
| 10분 / 20분 정글 CS | 66 / 127 | 57 / 102 | 62 / 123 | 70 / 139 |
| 20분 킬 / 데스 | 5.1 / 4.1 | 6.1 / 4.8 | 5.2 / 4.2 | 5 / 3 |
| 1초 동선: 우리 캠프 / 카정 / 바위게 | 31 / 0 / 18% | 23 / 2 / 17% | 33 / 2 / 14% | 30 / 6 / 13% |
| 라인 / 오브젝트 / 귀환 / 사망 | 24 / 9 / 9 / 9% | 26 / 13 / 8 / 10% | 22 / 10 / 9 / 9% | 25 / 7 / 10 / 8% |
| 탑 갱 / 바텀 갱 | 0.3 / 11.7% | 0.9 / 8.9% | 4.8 / 5.1% | 7.4 / 7.1% |

## 읽기

- 시뮬 승률은 A와 B가 거의 같다(차이 0.004, 각자 규칙 봇 대비 신뢰구간 폭이 ±0.04라 구별 불가). 기준점 때와 같은 결론: 시뮬 안에서 천상계 방식을 따라 해도 승률 손해가 거의 없다.
- B는 이제 이동 중 결정까지 배워서 시험 경기 일치가 55%(A 19%)로 기준점 B(분 시작 47%)보다 높다. 경기 전체 행동 비율도 A와 달라졌다: 카정이 0 → 2%, 탑 갱이 조금 생김.
- 하지만 B는 정글 CS가 실제보다 낮고(10분 57, 실제 70) 오브젝트에 머무는 시간이 많다(13%, 실제 7%). 실제처럼 행동을 고르지만 시뮬에서 같은 행동의 결과(캠프 처치 시간·동선)가 달라 CS가 덜 쌓이는 것으로 보인다.
- 두 갈래 모두 탑 갱이 거의 없다(실제 7%). 위쪽(유충·전령·탑 포탑) 가치가 승률 모델에 작다는 기존 가설이 여전히 남아 있다.
- 한타 보강 뒤 학습 정글러의 규칙 봇 대비 이득은 +0.05 정도로, 6단계 때(+0.3)보다 훨씬 작다. 교전이 많아지고 라이너가 실제처럼 움직이면서 정글러 한 명의 판단으로 바꿀 수 있는 몫이 줄었다.

## 다시 돌리기

    python -m sim.imitate --workers 2 --seeds 1 --save-dir models/stage7_real --out reports/imitate_within.md   # data/replay_raw 필요(PC)
    python -m sim.train --init models/stage6c/policy_latest.npz --iters 300 --seed0 500000 --out models/stage7_a
    python -m sim.train --init models/stage6c/policy_latest.npz --iters 300 --seed0 500000 --out models/stage7_b --real models/stage7_real/real_train.npz --real-coef 0.1
    python -m sim.evaluate --policy models/stage7_a/policy_0300.npz --games 400 --out reports/stage7_a_eval.md
    python -m sim.real_actions --policy models/stage7_a/policy_0300.npz --games 200 --out reports/stage7_a_actions.md
