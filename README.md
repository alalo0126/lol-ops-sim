# LoL 운영 시뮬레이터 (ops-sim)

리그 오브 레전드 정글 운영을 강화학습으로 배우고 천상계 정글러의 선택과 비교하는 시뮬레이터.

- 설계 문서: https://claude.ai/code/artifact/1e43b6f0-d5ff-42ef-ae42-f38face1facc
- 3단계 상태·행동 설계 초안: https://claude.ai/code/artifact/5c5d37f6-6c11-4813-801c-2807f8fb1a48

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `collect/` | 1단계: Riot API 경기 수집(`collect_matches.py`)과 분포 요약(`summarize.py`) |
| `rules/` | 2단계: 패치 26.20 규칙 데이터, 항목마다 출처 태그. 미검증 목록은 `rules/UNVERIFIED.md` |
| `tools/` | 규칙 생성(`build_ddragon_rules.py`)과 실제 경기 대조(`crosscheck_rules.py`) |
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
