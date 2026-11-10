"""패치 노트에서 정글·오브젝트·시스템 변경 부분만 뽑기.

    python -m tools.patch_notes_extract   # knowledge/patch_notes/*.md -> knowledge/patch_system_extract.md

챔피언·스킨·버그·아레나·칼바람 등 섹션은 빼고, 남은 섹션 중 주제어가 있는 줄을
'버전 | 섹션 경로 | 줄' 형태로 모음. 연표(knowledge/patch_timeline.md)를 손으로 쓸 때 근거 자료.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "knowledge" / "patch_notes"
OUT = ROOT / "knowledge" / "patch_system_extract.md"

TOPICS = {
    "정글": r"정글|몬스터|캠프|바위게|칼날부리|늑대|돌거북|두꺼비|골렘|파수꾼|덩굴정령|강타|동반자|정글 펫|정글링",
    "오브젝트": r"드래곤|용의 영혼|장로|바론|내셔 남작|전령|공허 유충|공허충|아타칸|에픽 몬스터|오브젝트",
    "건물": r"포탑|방호물|방패|억제기|넥서스|채굴|철거",
    "미니언·라인": r"미니언|웨이브|대포|공성",
    "보상": r"경험치|현상금|골드|처치 보상|어시스트|퍼스트 블러드|따라잡기",
    "이동·시작": r"민병대|귀환|부활|산뜻한 출발|이동 속도 증가|게임 시작|우물|분수",
    "시야": r"와드|시야|렌즈|투명|제어 와드|장신구",
    "맵": r"맵|지형|수풀|부쉬|협곡|정글 길|벽",
    "주문·역할": r"점멸|순간이동|소환사 주문|점화|탈진|정화|유체화|회복 주문|방어막 주문|역할 퀘스트|퀘스트",
}
ANY = re.compile("|".join(f"(?:{p})" for p in TOPICS.values()))

SKIP_HEAD = re.compile(
    r"스킨|크로마|버그|아레나|무작위 총력전|칼바람|아수라장|증강|신화급|상점 판매|URF|우르프|"
    r"돌격! 넥서스|전략적 팀 전투|TFT|이벤트|패스|프레스티지|모드|체력바|편의성|아이콘|감정 표현|"
    r"롤토체스|와일드|Swarm|스웜|도전 과제|랭크|순위|보상 상점|행동 관련|면죄|클래식 모드 아님", re.I)


def champion_names() -> set[str]:
    d = json.loads((ROOT / "rules" / "champions.json").read_text(encoding="utf-8"))
    c = d["champions"]
    vals = c.values() if isinstance(c, dict) else c
    return {x["name_ko"] for x in vals} | {"니달리", "누누와 윌럼프", "누누", "레나타 글라스크", "레나타"}


def main():
    champs = champion_names()
    rows = list(csv.DictReader(open(NOTES / "index.csv", encoding="utf-8")))
    out = ["# 패치 노트 시스템 변경 발췌 (자동)", "",
           "tools/patch_notes_extract.py로 생성. 챔피언·아이템 개별 조정 섹션 제외. 원문은 knowledge/patch_notes/<버전>.md.", ""]
    total = 0
    for r in rows:
        lines = (NOTES / r["file"]).read_text(encoding="utf-8").splitlines()
        stack: list[tuple[int, str]] = []
        picked: list[str] = []
        last_path = None
        for ln in lines[4:]:
            m = re.match(r"^(#+)\s*(.*)$", ln)
            if m:
                lvl, h = len(m.group(1)), m.group(2).strip()
                stack = [(l, x) for l, x in stack if l < lvl] + [(lvl, h)]
                continue
            heads = [x for _, x in stack if x]
            if any(h in champs or SKIP_HEAD.search(h) for h in heads):
                continue
            hp = " ".join(heads)
            if re.search(r"아이템|룬|스킬 신호", hp) and not re.search(r"정글|서포터|시작 아이템|와드|장신구|시야", hp):
                continue
            if ln.strip() in ("맨 위로 돌아가기", "Back to top"):
                continue
            # 챔피언 이름으로 시작하는 줄(소제목 없는 형식)도 제외
            if any(ln.lstrip("- ").startswith(c + " ") for c in champs):
                continue
            path = " > ".join(heads[-3:])
            if not (ANY.search(ln) or ANY.search(path)):
                continue
            if len(ln) < 4:
                continue
            if path != last_path:
                picked.append(f"**[{path}]**" if path else "**[-]**")
                last_path = path
            picked.append(ln if ln.startswith("- ") else "  " + ln)
        if picked:
            out += [f"## {r['version']} ({r['date']})", ""] + picked + [""]
            total += len(picked)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"{len(rows)}개 패치, {total}줄 -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
