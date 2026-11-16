"""라이너 체력 분포 검수: 리플레이(data/replay_raw) 2~10분 분 프레임 vs 규칙 봇 시뮬.

    python tools/laner_hp_check.py 200      (시뮬 판 수)

1,000판(data/raw)으로 보려면 real()의 경로를 data/raw로 바꾼다. 결과 정리는 reports/laner_hp_notes.md.
"""
import sys,os,glob,json,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); os.chdir(ROOT)
from multiprocessing import Pool
from sim.game import Game
ROLES=['TOP','MIDDLE','BOTTOM','UTILITY']
M=range(2,11)
def real():
    out={(r,m):[] for r in ROLES for m in M}
    for p in glob.glob('data/replay_raw/*.json'):
        d=json.load(open(p)); fr=d['timeline']['info']['frames']
        for q in d['detail']['info']['participants']:
            if q['teamPosition'] not in ROLES: continue
            for m in M:
                if m>=len(fr): continue
                cs=fr[m]['participantFrames'][str(q['participantId'])]['championStats']
                if cs['health']>0: out[(q['teamPosition'],m)].append(cs['health']/max(1,cs['healthMax']))
    return out
def one(seed):
    g=Game(seed=seed); out=[]
    while g.t<=600:
        if g.t%60==0 and g.t//60 in M:
            for c in g.champs:
                if c.role in ROLES and c.alive(g.t): out.append((c.role,g.t//60,c.hp))
        g.step()
    return out
if __name__=='__main__':
    R=real()
    with Pool(4) as pl: rs=[x for r in pl.map(one,range(int(sys.argv[1]))) for x in r]
    S={(r,m):[] for r in ROLES for m in M}
    for r,m,h in rs: S[(r,m)].append(h)
    print('role  m  real mean/med/p25/<0.5  |  sim mean/med/p25/<0.5')
    for r in ROLES:
        for m in M:
            a,b=np.array(R[(r,m)]),np.array(S[(r,m)])
            f=lambda v:f"{v.mean():.2f}/{np.median(v):.2f}/{np.percentile(v,25):.2f}/{(v<0.5).mean():.0%}"
            print(r[:3],m,f(a),'|',f(b))
    a=np.concatenate([np.array(R[k]) for k in R]); b=np.concatenate([np.array(S[k]) for k in S]); print('all',a.mean().round(3),b.mean().round(3))
