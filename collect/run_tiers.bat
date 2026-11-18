@echo off
chcp 65001 >nul
cd /d %~dp0..
if "%RIOT_API_KEY%"=="" if exist "D:\AI\lol\riot_key.txt" set /p RIOT_API_KEY=<"D:\AI\lol\riot_key.txt"
if "%RIOT_API_KEY%"=="" set /p RIOT_API_KEY=Riot API 키(RGAPI-...): 
python collect\collect_tiers.py --probe
python collect\collect_tiers.py --target 1000 %*
pause
