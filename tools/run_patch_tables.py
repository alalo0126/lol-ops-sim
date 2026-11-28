"""새 패치 표 한 번에 만들기(kk PC). run_patch_tables.bat이 부른다.

데이터 받기 → 아이템 벡터(+이전 버전과 비교) → 챔피언 스킬 표. 결과 경로를 data/patch_tables_log.txt에 남긴다.
"""
import glob
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = open(os.path.join("data", "patch_tables_log.txt"), "w", encoding="utf-8")


def run(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8", env=env)
    log.write("$ " + " ".join(args) + "\n" + r.stdout + r.stderr + "\n")
    log.flush()
    return r.stdout


out = run(["tools/fetch_patch_data.py"])
ver = out.strip().splitlines()[-1].split()[-1]
prev = sorted(p for p in glob.glob("rules/item_vectors_*.json") if ver not in p)
run(["tools/build_item_vectors.py", f"data/ddragon/{ver}/en_US/item.json"] + (["--prev", prev[-1]] if prev else []))
run(["tools/build_champ_kits.py", f"data/ddragon/{ver}/en_US/championFull.json"])
log.write("끝\n")
print("끝, data\\patch_tables_log.txt 확인")
