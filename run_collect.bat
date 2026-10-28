@echo off
REM Collect 26.21 replays before the next patch. Double-click once; rerun to resume.
REM Do not use keyboard/mouse while recording. Log: data\replay_tracks\collect_log.txt
REM Keep this file ASCII only: cmd misreads lines of UTF-8 batch files after chcp 65001.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set RIOT_API_KEY=
if exist "%~dp0..\riot_key.txt" set /p RIOT_API_KEY=<"%~dp0..\riot_key.txt"
echo [1/4] find challenger/GM jungle games (7 days, up to 400) and download replays
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --download-only
echo [2/4] test opening a replay without the client (backup Game folder if it works)
python -m tools.replay_direct --test --backup
echo [3/4] record JUNGLE, TOP, MIDDLE to the end of each game
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --until 0 --speed 8 --role JUNGLE,TOP,MIDDLE
echo [4/4] record BOTTOM, UTILITY to the end of each game
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --until 0 --speed 8 --role BOTTOM,UTILITY
pause
