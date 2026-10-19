"""Riot API 클라이언트: 개발자 키 호출 한도(1초 20회, 2분 100회)를 지키고 429는 Retry-After만큼 기다린다."""
import os
import sys
import time
from collections import deque

import requests

PLATFORM_TO_REGION = {
    "kr": "asia", "jp1": "asia",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}

# 개발자 키 기본 한도. 개인/프로덕션 키면 --rate 옵션으로 바꾼다.
DEV_KEY_LIMITS = [(20, 1.0), (100, 120.0)]


class KeyExpired(Exception):
    pass


class RateLimiter:
    def __init__(self, limits):
        self.windows = [(n, sec, deque()) for n, sec in limits]

    def wait(self):
        while True:
            now = time.monotonic()
            sleep_for = 0.0
            for n, sec, q in self.windows:
                while q and now - q[0] >= sec:
                    q.popleft()
                if len(q) >= n:
                    sleep_for = max(sleep_for, sec - (now - q[0]) + 0.05)
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        now = time.monotonic()
        for _, _, q in self.windows:
            q.append(now)


class RiotAPI:
    def __init__(self, platform="kr", api_key=None, limits=DEV_KEY_LIMITS):
        self.api_key = api_key or os.environ.get("RIOT_API_KEY", "").strip()
        if not self.api_key.startswith("RGAPI-"):
            sys.exit("RIOT_API_KEY 환경변수가 없거나 형식이 다릅니다 (RGAPI-로 시작). "
                     "developer.riotgames.com에서 발급 후 설정하세요.")
        self.platform = platform
        self.region = PLATFORM_TO_REGION[platform]
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = self.api_key
        self.limiter = RateLimiter(limits)
        self.calls = 0

    def _get(self, url, params=None, retries=5):
        for attempt in range(retries):
            self.limiter.wait()
            self.calls += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                time.sleep(2 ** attempt)
                print(f"  네트워크 오류, 재시도 {attempt + 1}/{retries}: {e}")
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "10"))
                print(f"  호출 한도 초과, {wait}초 대기")
                time.sleep(wait)
                continue
            if resp.status_code in (401, 403):
                raise KeyExpired(f"{resp.status_code}: API 키가 만료됐거나 잘못됐습니다. 새로 발급 후 다시 실행하면 이어서 받습니다.")
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"재시도 {retries}회 실패: {url}")

    def apex_entries(self, tier, queue="RANKED_SOLO_5x5"):
        path = {"challenger": "challengerleagues", "grandmaster": "grandmasterleagues",
                "master": "masterleagues"}[tier]
        data = self._get(f"https://{self.platform}.api.riotgames.com/lol/league/v4/{path}/by-queue/{queue}")
        return (data or {}).get("entries", [])

    def match_ids(self, puuid, start=0, count=100, queue=420, start_time=None):
        params = {"start": start, "count": count, "queue": queue}
        if start_time:
            params["startTime"] = int(start_time)
        return self._get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids",
                         params) or []

    def match(self, match_id):
        return self._get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}")

    def timeline(self, match_id):
        return self._get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline")
