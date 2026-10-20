# LoL 운영 시뮬레이터 (ops-sim)

리그 오브 레전드 정글 운영을 강화학습으로 배우고 천상계 정글러의 선택과 비교하는 시뮬레이터.

- 설계 문서: https://claude.ai/code/artifact/1e43b6f0-d5ff-42ef-ae42-f38face1facc
- 3단계 상태·행동 설계 초안: https://claude.ai/code/artifact/5c5d37f6-6c11-4813-801c-2807f8fb1a48

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `collect/` | 1단계: Riot API 경기 수집(`collect_matches.py`)과 분포 요약(`summarize.py`) |
| `rules/` | 2단계: 패치 26.20 규칙 데이터, 항목마다 출처 태그. 미검증 목록은 `rules/UNVERIFIED.md` |
| `tools/` | 규칙 생성(`build_ddragon_rules.py`, `build_map_graph.py`), 실제 경기 대조(`crosscheck_rules.py`), 4단계 기준값 추출(`stage4_targets.py`) |
| `sim/` | 4단계: 1초 틱 최소 시뮬레이터와 규칙 봇. 모델 값은 `sim/params.json`. 5단계: 학습 정글러(`agent.py`), 승률 보상(`winprob.py`), numpy PPO(`ppo.py`, `train.py`), 모방 시작점(`teacher.py`), 판정(`evaluate.py`) |
| `models/` | 5단계 학습된 정책(`models/stage5/*.npz`)과 학습 기록. git에 올리지 않음 |
| `tests/` | `python -m unittest discover tests` |
| `reports/` | 시뮬레이터와 실제 분포 비교표 |
| `data/` | 수집 원본·Data Dragon 캐시. git에 올리지 않음 |

## 1단계: 실제 데이터 수집

고정 패치 26.20 (match-v5의 `gameVersion` 표기로는 `16.20.*`), KR 솔로랭크,
챌린저·그랜드마스터 선수가 **정글로 뛴** 경기만 모은다.

```bat
:: Windows CMD, 이 폴더(D:\AI\lol\ops-sim)에서
pip install -r requirements.txt
set RIOT_API_KEY=RGAPI-여기에-새-키
python collect\collect_matches.py --patch 26.20 --target 1000
python collect\summarize.py
```

WSL이면 `cd /mnt/d/AI/lol/ops-sim` 후 `export RIOT_API_KEY=...`, 경로 구분자는 `/`.

- 개발자 키는 24시간마다 만료된다. 만료되면 스크립트가 멈추고, 새 키로 같은 명령을 다시 실행하면 이어서 받는다. Ctrl+C로 끊어도 마찬가지.
- 개발자 키 한도(2분 100회) 기준으로 1,000판에 대략 1시간 남짓 걸릴 것으로 본다(정글이 아닌 경기를 거르느라 매치 조회가 더 든다). 실측 필요.
- 원본은 `data/raw/<matchId>.json`(경기당 약 0.9MB, 1,000판이면 약 900MB), 진행 상태는 `data/collect_state.json`.
- `data/`는 git에 올리지 않는다. API 키도 파일에 적지 말고 환경변수로만 넣는다.

### 완료 기준 요약 (`summarize.py`)

정글러 10분 골드·CS·레벨·경험치, 3·4분 정글 CS, 첫 킬·유충·드래곤·전령·포탑 시각 분포를 `data/summary_stage1.json`으로 저장한다.

타임라인으로 확인된 한계 (2026-10-19, 기존 캐시 121판으로 검수):
- 프레임은 60초 간격. 일반 캠프 처치 이벤트가 없어 첫 풀캠 시각은 직접 잴 수 없다.
- `WARD_PLACED`에는 위치가 없다(`WARD_KILL`도 마찬가지). 시야 상태는 수동 입력 대상.
- `CHAMPION_KILL`, `ELITE_MONSTER_KILL`, `BUILDING_KILL`에는 위치·시각이 있고, 참가자 프레임에도 분당 위치가 있다.

## 4단계: 강화학습 없는 최소 시뮬레이터

```bat
python tools\stage4_targets.py          :: data\raw → data\stage4_targets.json (실제 경기 기준값)
copy data\stage4_targets.json sim\targets_stage1.json
python -m sim.run_baseline --games 300  :: 규칙 봇 300판 → reports\stage4_baseline.md
python -m unittest discover tests
```

- 맵: `rules/map_nodes.json` 노드 33개. `tools/build_map_graph.py`로 만든다(블루 진영만 적고 레드는 점대칭). 포탑·용/바론 둥지·우물 좌표는 1단계 이벤트 위치로 확인했다.
- 엔진(`sim/game.py`): 1초 틱, 0~20분. 캠프 생성·리스폰, 처치 시간(체력 ÷ 초당 피해), 골드·경험치·펫 보너스·간식, 강타 충전, 귀환, 처치·부활, 용·유충·전령, 팀별 시야와 인지 제약.
- 규칙 봇(`sim/bots.py`): 정글러는 시작 캠프별 정해진 동선으로 첫 풀캠 → 귀환 → 갱 기회 > 오브젝트 > 캠프 > 리스폰 대기. 라이너는 3단계 kk 기준 중 후퇴·귀환·와드·기본만(웨이브 모듈이 없어서 나머지는 아직).
- 이벤트 계산 함수(`sim/events.py`): 갱 성공 확률은 로지스틱 자리표시. 입출력 형식만 설계 문서대로 고정.
- `sim/targets_stage1.json`은 실제 경기의 집계값(분위수)만 담는다. 1단계 수집이 끝나면 위 두 줄로 다시 만든다.
- 처치 시간·캠프 간 이동·우물 이속·강타 충전·간식 시각·당겨먹기·레벨/템에 따른 처치 속도는 kk 연습 모드 실측(마스터 이), 갱·교전 계수는 실제 10분 킬·데스·어시스트에 맞춘 값이다.
- 판정(2026-10-20, 400판 vs 1,000판 양 팀 정글러): 불일치 없음, 10분 골드·정글 CS·레벨·경험치 모두 실제 p25~p75 안 → 4단계 통과. 알려진 한계: 10분 정글 CS가 중앙값보다 약 6 낮고(1풀캠 뒤 동선 판단이 단순한 규칙 봇), 15분 골드·경험치는 p10~p90 안이지만 낮은 편. 자세한 내용은 `reports/stage4_notes.md`.

## 5단계: 정글러 강화학습

정글러 하나만 PPO로 학습하고 나머지 9명은 규칙 봇. GPU·PyTorch 없이 numpy만 쓴다(정책망이 작아 CPU로 충분).

```bat
python tools\fit_winprob.py              :: data\raw → sim\winprob.json (분별 승률 모델, 보상)
python tools\stage5_targets.py           :: data\raw → sim\targets_stage5.json (20분 실제 분포, 악용 점검용)
python -m sim.teacher --games 3000        :: 규칙 봇 모방 → models\stage5\policy_bc.npz
python -m sim.train --init models\stage5\policy_bc.npz --iters 400 --lr 1e-4 --ent-coef 0.003
python -m sim.evaluate --games 4000       :: 규칙 봇 정글러와 같은 시드로 짝지어 비교 → reports\stage5_eval.md
```

- 보상(3단계 확정): 결정 사이 예측 승률 변화. 합은 20분 예측 승률 − 시작 승률(잠재 기반 보상이라 최적 행동은 20분 승률 기준과 같음).
  승률 모델은 1단계 1,000판의 분별 로지스틱 회귀(골드·경험치·드래곤·유충·전령 팀 차이). 포탑·킬을 넣어도 교차검증 정확도가 거의 같아 시뮬레이터가 만드는 값만 쓴다.
- 행동(3단계 확정): 주 행동 36개(캠프 6, 카정 6, 바위게 2, 갱 3, 카운터 갱·합류 3, 오브젝트 3, 귀환, 라인 미니언 3, 대기, 이동 8) + 와드(없음/지금/다음 노드) + 강타(사용/아낌). 결정은 행동이 끝났을 때, 이동 중 5초마다, 새 정보(적 정글러 발견·킬·오브젝트 생성) 때.
- 학습 정글러 팀은 경기마다 블루/레드 번갈아. 학습 경기 시드는 100000부터, 판정 시드는 0부터라 겹치지 않는다.
- 자세한 판정과 악용 점검은 `reports/stage5_eval.md`, `reports/stage5_notes.md`.
