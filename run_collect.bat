@echo off
REM 26.21 리플레이 최대 확보 (다음 패치 전에). 더블클릭 한 번이면 끝까지 돈다. 중간에 꺼도 다시 실행하면 이어서 한다.
REM 실행 중엔 키보드·마우스를 쓰지 말 것(카메라 고정 키를 게임 창에 넣음). 진행 로그: data\replay_tracks\collect_log.txt
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if exist "%~dp0..\riot_key.txt" set /p RIOT_API_KEY=<"%~dp0..\riot_key.txt"
echo [1/4] 천상계 정글 경기 찾기 + 리플레이 받아 두기 (최근 7일, 최대 400판)
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --download-only
echo [2/4] 클라이언트 없이 exe로 열리는지 시험 (되면 이 빌드 Game 폴더 보관)
python -m tools.replay_direct --test --backup
echo [3/4] 기록: 정글·탑·미드, 경기 끝까지 (20분까지 해 둔 경기는 이어서)
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --until 0 --speed 8 --role JUNGLE,TOP,MIDDLE
echo [4/4] 기록: 원딜·서폿, 경기 끝까지
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --until 0 --speed 8 --role BOTTOM,UTILITY
pause
