"""롤 공식 패치 노트 전체 받기 (kk PC에서 실행; 클라우드는 leagueoflegends.com 접속 차단).

    python -m tools.patch_notes_fetch            # 목록 찾기 + 받기 (이미 받은 건 건너뜀)
    python -m tools.patch_notes_fetch --list     # 목록만 찾아 index.csv
    python -m tools.patch_notes_fetch --lang ko-kr,en-us

찾는 방법
  1) 언어별 사이트맵(<lang>/sitemap_loc.xml)에서 game-updates 아래 패치 노트 주소
  2) 패치 노트 태그 목록 페이지에 박힌 주소
  3) 번호로 만든 후보 주소(patch-X-Y-notes 등)를 직접 확인
  ko-kr에 없는 패치는 en-us로 채움.

결과 (data/patch_notes/)
  index.csv            버전, 언어, 주소, 제목, 게시일, 글자 수
  raw/<lang>/<slug>.html.gz   원본 HTML (본문 해석을 나중에 다시 할 수 있게)
  text/<version>.md    본문 텍스트(제목 줄 #, 목록 -)
  patch_notes_text.tar.gz   text/ + index.csv 묶음 (클라우드로 옮길 때)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
import sys
import tarfile
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

BASE = "https://www.leagueoflegends.com"
OUT = Path("data/patch_notes")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})

EXCLUDE = re.compile(r"teamfight|tft|wild-rift|valorant|2xko|legends-of-runeterra|"
                     r"arena-|swarm|preview|rundown|mid-patch|midpatch|hotfix|"
                     r"pbe|tldr|tl-dr|summary|highlights", re.I)


def get(url: str, tries: int = 4) -> requests.Response | None:
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code in (200, 404, 410):
                return r
            if r.status_code == 403:
                print(f"  403 {url}", file=sys.stderr)
                return r
        except requests.RequestException as e:
            print(f"  재시도 {i+1}: {e}", file=sys.stderr)
        time.sleep(2 ** (i + 1))
    return None


# ---------- 버전 해석 ----------

def version_from(text: str) -> str | None:
    """주소 slug나 제목에서 패치 번호. 25.S1.1 형식도."""
    t = text.lower().replace("_", "-")
    m = re.search(r"(\d{1,2})[-.\s]s(\d)[-.\s](\d{1,2})", t)
    if m:
        return f"{int(m.group(1))}.S{m.group(2)}.{int(m.group(3))}"
    m = re.search(r"(?:patch|paechi|패치)[-\s]*(\d{1,2})[-.](\d{1,2})(?!\d)", t)
    if not m:
        m = re.search(r"(?<!\d)(\d{1,2})[-.](\d{1,2})(?!\d)[-\s]*(?:patch|paechi|패치)", t)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > 40 or b > 30:
        return None
    return f"{a}.{b:02d}" if a >= 25 else f"{a}.{b}"


def version_key(v: str):
    p = v.replace("S", "").split(".")
    nums = [int(x) for x in p]
    if len(nums) == 3:      # 25.S1.1 -> 25, 0, 1, 1 (25.04보다 앞)
        return (nums[0], 0, nums[1], nums[2])
    return (nums[0], nums[1], 0, 0)


def is_patch_note_url(u: str) -> bool:
    if "/news/game-updates/" not in u and "/news/tags/" not in u:
        return False
    slug = u.rstrip("/").rsplit("/", 1)[-1]
    if not re.search(r"patch|paechi", slug):
        return False
    if not re.search(r"note|noteu", slug):
        return False
    return not EXCLUDE.search(slug)


# ---------- 찾기 ----------

def from_sitemap(lang: str) -> set[str]:
    urls: set[str] = set()
    r = get(f"{BASE}/{lang}/sitemap_loc.xml")
    if r is None or r.status_code != 200:
        print(f"  사이트맵 실패 {lang}")
        return urls
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
    # 사이트맵 안에 하위 사이트맵이 있으면 따라감
    for loc in locs:
        loc = html.unescape(loc)
        if loc.endswith(".xml"):
            r2 = get(loc)
            if r2 is not None and r2.status_code == 200:
                for l2 in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r2.text):
                    urls.add(html.unescape(l2))
        else:
            urls.add(loc)
    found = {u for u in urls if is_patch_note_url(u)}
    print(f"  사이트맵 {lang}: 주소 {len(urls)}개 중 패치 노트 {len(found)}개")
    return found


def from_tag_page(lang: str) -> set[str]:
    r = get(f"{BASE}/{lang}/news/tags/patch-notes/")
    if r is None or r.status_code != 200:
        return set()
    hrefs = re.findall(r'(/' + lang + r'/news/game-updates/[a-z0-9\-]+/?)', r.text)
    found = {BASE + h for h in hrefs if is_patch_note_url(BASE + h)}
    print(f"  태그 목록 {lang}: {len(found)}개")
    return found


def candidates(lang: str, have: set[str]) -> list[str]:
    """번호로 만든 후보 주소 (이미 찾은 버전은 제외)."""
    out = []
    vers = []
    for a in range(4, 16):
        for b in range(1, 25):
            vers.append((a, b, None))
    for s in (1, 2, 3):
        for b in range(1, 4):
            vers.append((25, b, s))
    for b in range(4, 25):
        vers.append((25, b, None))
    for b in range(1, 25):
        vers.append((26, b, None))
    for a, b, s in vers:
        v = f"{a}.S{s}.{b}" if s else (f"{a}.{b:02d}" if a >= 25 else f"{a}.{b}")
        if v in have:
            continue
        if s:
            slugs = [f"patch-{a}-s{s}-{b}-notes", f"league-of-legends-patch-{a}-s{s}-{b}-notes"]
        else:
            bs = {str(b), f"{b:02d}"}
            slugs = []
            for x in bs:
                slugs += [f"patch-{a}-{x}-notes", f"league-of-legends-patch-{a}-{x}-notes",
                          f"{a}-{x}-paechi-noteu"]
        out += [(v, f"{BASE}/{lang}/news/game-updates/{sl}/") for sl in slugs]
    return out


# ---------- 본문 텍스트 ----------

class TextOut(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button"}
    BLOCK = {"p", "div", "section", "article", "br", "tr", "table", "blockquote", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.cur: list[str] = []
        self.skip = 0
        self.prefix = ""

    def flush(self):
        s = re.sub(r"\s+", " ", "".join(self.cur)).strip()
        if s:
            self.lines.append(self.prefix + s)
        self.cur = []
        self.prefix = ""

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5"):
            self.flush()
            self.prefix = "#" * int(tag[1]) + " "
        elif tag == "li":
            self.flush()
            self.prefix = "- "
        elif tag in self.BLOCK:
            self.flush()
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.cur.append(f" [{alt}] ")
        elif tag in ("td", "th"):
            self.cur.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "li") or tag in self.BLOCK:
            self.flush()

    def handle_data(self, data):
        if not self.skip:
            self.cur.append(data)


def article_html(page: str) -> str:
    """본문 부분만. 못 찾으면 <main>, 그것도 없으면 전체."""
    for pat in (r'<div[^>]+data-testid="rich-text-html"[^>]*>', r'<div[^>]+id="patch-notes-container"[^>]*>',
                r'<article[^>]*>', r'<main[^>]*>'):
        m = re.search(pat, page)
        if m:
            return page[m.start():]
    return page


def page_meta(page: str) -> dict:
    meta = {}
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', page)
    meta["title"] = html.unescape(m.group(1)) if m else ""
    m = (re.search(r'<time[^>]+datetime="([^"]+)"', page)
         or re.search(r'"(?:datePublished|publishedAt|date)"\s*:\s*"(\d{4}-\d\d-\d\dT[^"]+)"', page))
    meta["date"] = m.group(1)[:10] if m else ""
    m = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', page)
    meta["canonical"] = m.group(1) if m else ""
    return meta


def html_to_text(page: str) -> str:
    p = TextOut()
    p.feed(article_html(page))
    p.flush()
    lines, prev = [], None
    for ln in p.lines:
        if ln != prev:
            lines.append(ln)
        prev = ln
    return "\n".join(lines)


# ---------- 실행 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ko-kr,en-us")
    ap.add_argument("--list", action="store_true", help="목록만")
    ap.add_argument("--no-guess", action="store_true", help="후보 주소 확인 생략")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    langs = args.lang.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "text").mkdir(exist_ok=True)

    # 버전 -> (언어 순위, url)
    chosen: dict[str, tuple[int, str, str]] = {}
    unknown: list[str] = []
    for rank, lang in enumerate(langs):
        print(f"[{lang}] 목록 찾기")
        urls = from_sitemap(lang) | from_tag_page(lang)
        for u in sorted(urls):
            v = version_from(u.rstrip("/").rsplit("/", 1)[-1])
            if v is None:
                unknown.append(u)
                continue
            if v not in chosen or chosen[v][0] > rank:
                chosen[v] = (rank, lang, u)
        if not args.no_guess:
            have = {v for v, c in chosen.items() if c[0] <= rank}
            cands = candidates(lang, have)
            print(f"  후보 주소 확인 {len(cands)}개 (느림, --no-guess로 생략)")
            hit = set()
            for v, u in cands:
                if v in hit:
                    continue
                r = get(u)
                time.sleep(args.sleep / 2)
                if r is not None and r.status_code == 200 and len(r.text) > 5000:
                    hit.add(v)
                    if v not in chosen or chosen[v][0] > rank:
                        chosen[v] = (rank, lang, u)
                        print(f"   + {v} {u}")
    (OUT / "unknown_urls.txt").write_text("\n".join(unknown), encoding="utf-8")
    print(f"버전 {len(chosen)}개, 번호 못 읽은 주소 {len(unknown)}개 (unknown_urls.txt)")

    rows = []
    idx_path = OUT / "index.csv"
    old = {}
    if idx_path.exists():
        with open(idx_path, encoding="utf-8") as f:
            old = {r["version"]: r for r in csv.DictReader(f)}

    for v in sorted(chosen, key=version_key):
        _, lang, u = chosen[v]
        slug = u.rstrip("/").rsplit("/", 1)[-1]
        raw = OUT / "raw" / lang / f"{slug}.html.gz"
        txt = OUT / "text" / f"{v}.md"
        if args.list:
            rows.append({"version": v, "lang": lang, "url": u, "title": "", "date": "", "chars": ""})
            continue
        if raw.exists() and txt.exists() and v in old and old[v]["url"] == u:
            rows.append(old[v])
            continue
        r = get(u)
        time.sleep(args.sleep)
        if r is None or r.status_code != 200:
            print(f"  실패 {v} {u} {getattr(r, 'status_code', None)}")
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw, "wt", encoding="utf-8") as f:
            f.write(r.text)
        meta = page_meta(r.text)
        body = html_to_text(r.text)
        head = f"<!-- version={v} lang={lang} date={meta['date']} url={u} -->\n# {meta['title']}\n\n"
        txt.write_text(head + body, encoding="utf-8")
        rows.append({"version": v, "lang": lang, "url": u, "title": meta["title"],
                     "date": meta["date"], "chars": len(body)})
        print(f"  {v:>8} {lang} {meta['date']} {len(body):>7}자 {meta['title'][:40]}")

    with open(idx_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["version", "lang", "url", "title", "date", "chars"])
        w.writeheader()
        w.writerows(rows)
    if not args.list:
        tgz = OUT / "patch_notes_text.tar.gz"
        with tarfile.open(tgz, "w:gz") as t:
            t.add(idx_path, arcname="index.csv")
            t.add(OUT / "text", arcname="text")
            t.add(OUT / "unknown_urls.txt", arcname="unknown_urls.txt")
        print(f"끝: {len(rows)}개 -> {tgz}")


if __name__ == "__main__":
    main()
