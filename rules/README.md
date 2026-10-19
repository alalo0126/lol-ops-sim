# rules/ (2단계 규칙 데이터, 패치 26.20)

| 파일 | 내용 |
|---|---|
| meta.json | 패치·Data Dragon 버전, 출처 수준 정의, 출처 URL 목록 |
| positions.json | 포지션 배정, 강타, 정글 펫, 포지션 퀘스트, 서포터 아이템 |
| camps.json | 일반 정글 캠프 생성·리스폰·골드·경험치 |
| objectives.json | 드래곤, 장로, 유충, 전령, 바론(아타칸 제거) |
| minions.json | 미니언 웨이브 일정·골드 |
| items_restrictions.json | 포지션별 아이템 제한, 근거리·원거리 구매 제한과 수치 차이 |
| champions.json, items.json | PC에서 `python tools\build_ddragon_rules.py`로 생성 (Data Dragon 16.20.1) |
| UNVERIFIED.md | 미검증 항목 목록 |

항목 형식: `{value, src, level, note?, crosscheck?}`. `src`는 meta.json의 sources 키, `level`은 verified / consensus / unverified.
