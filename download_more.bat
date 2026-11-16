@echo off
REM Find more challenger/GM jungle games with the Riot key and download their replays.
REM Safe to run while run_collect2.bat is recording (it only uses the client download API).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set RIOT_API_KEY=
if exist "%~dp0..\riot_key.txt" set /p RIOT_API_KEY=<"%~dp0..\riot_key.txt"
python -m tools.replay_batch --count 400 --hours 168 --per-player 3 --download-only --log data\replay_tracks\download_log.txt
pause
