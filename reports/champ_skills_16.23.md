# 챔피언 스킬 표 검수 16.23

- 챔피언 173명, 읽기 실패 0

## Data Dragon 쿨 대조: 692/692 일치


## Data Dragon 소모값 대조: 다름 0


## 이동속도 차이 0


## Meraki(25.15) 타게팅 대조: 593/607 일치

- Ambessa E: 게임 파일 Location / Meraki Auto
- Azir W: 게임 파일 TargetOrLocation / Meraki Location
- Azir E: 게임 파일 Area / Meraki Unit
- Bard E: 게임 파일 TerrainLocation / Meraki Direction
- Corki E: 게임 파일 Cone / Meraki Auto
- Ivern W: 게임 파일 TargetOrLocation / Meraki Location
- Kindred W: 게임 파일 TargetOrLocation / Meraki Location
- Qiyana W: 게임 파일 TerrainType / Meraki Location
- Riven E: 게임 파일 TargetOrLocation / Meraki Direction
- Rumble Q: 게임 파일 Cone / Meraki Auto
- Skarner W: 게임 파일 Direction / Meraki Auto
- Zac R: 게임 파일 Area / Meraki Auto
- Zeri W: 게임 파일 WallDetection / Meraki Direction
- Zeri E: 게임 파일 WallDetection / Meraki Direction

## 검수 메모 (Claude, 2026-11-28)

- 원본: CommunityDragon 16.23.8418428 게임 파일 173명. Meraki(25.15에서 멈춤)는 타게팅 교차 확인에만 씀. 타게팅 다른 14개는 대부분 Meraki의 "Auto"/분류 차이(지형 지정·벽 감지 포함)로 게임 파일을 따름.
- 2차 검수에서 고친 것: 쿨은 `Cooldown.values`, 소모는 `manaValues`가 새 칸(옛 칸 `cooldownTime`·`mana`엔 패치 전 값이 남은 스킬 있음: 신드라 W·E, 흐웨이 E). 6칸 배열(소모)은 0번이 1레벨. 투사체 사거리 25000은 자리값이라 뺌.
- 3차 검수에서 고친 것: 쿨 칸이 없는 스킬 12개(럭스 Q, 모르가나 Q 등)는 게임 기본값 10초(Data Dragon도 10) → `defaults`에 표시. 레벨당 AD 0은 세나뿐(실제로 세나는 성장 AD 없음, 맞음).
- 남은 한계(미검증): 시전 시간 없는 스킬 70개(기본값 미확인), 사거리 없는 스킬 55개(전역 궁·자기 강화·하위 스킬에 사거리), 스킬샷 282개 중 74개는 투사체 정보 없음(즉발 판정 또는 하위 스킬), 능력치 번호 7·8·9·18·29·31·4는 이름 미확인. 피해 기본값·계수는 Data Dragon에 숫자가 없어 대조 못 함 → 실제 경기 교전 결과로 시험.
