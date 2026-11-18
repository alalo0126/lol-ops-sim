@echo off
chcp 65001 >nul
cd /d %~dp0..
if "%RIOT_API_KEY%"=="" if exist "D:\AI\lol\riot_key.txt" set /p RIOT_API_KEY=<"D:\AI\lol\riot_key.txt"
if "%RIOT_API_KEY%"=="" set /p RIOT_API_KEY=Riot API 키(RGAPI-...): 
rem 1) 26.23 리플레이용 경기만 티어당 30판 (몇 분)
python -u collect\collect_tiers.py --patch 26.23 --target 30 --root data/tiers_26.23 --tiers GOLD,SILVER,BRONZE,IRON,PLATINUM,EMERALD,DIAMOND,MASTER
rem 2) 리플레이 기록: 골드는 26.22 18판에 더해 2판, 이어서 실버·브론즈... (하루 받기 한도에 걸리면 멈춤, 다음 날 다시 실행)
python -u -m tools.tier_replays --root data/tiers_26.23 --extra-from data/tiers --tier GOLD --count 20
python -u -m tools.tier_replays --root data/tiers_26.23 --tier SILVER --count 20
python -u -m tools.tier_replays --root data/tiers_26.23 --tier BRONZE --count 20
pause
