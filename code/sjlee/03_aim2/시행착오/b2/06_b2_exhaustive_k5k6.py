"""
Aim 2 · B2 소문항 전수조사 (k=5,6 전 조합) (sjlee)
===================================================
작은 문항수는 전 조합 탐색 가능: k=5(20,349), k=6(54,264).
목적: (1) 최고 조합(단 승자의 저주로 MAE는 낙관적), (2) **상위 조합에 자주 나오는
문항 = 진짜 핵심 문항**(이게 신뢰할 유용 결과), (3) 둘다이김 통과 조합 비율.

모형: 부분PO 엘라스틱넷(max_iter=500 스크리닝) + **중증우선 τ튜닝**(제약=중증놓침≤훈련A0,
목적=MAE최소; τ격자 0.05 스크리닝), leave-one-trial-out. (통일 표준과 정렬; 구판은 고정임계·MAE순.)

주의: 전수조사 최고 = 승자의 저주(2만~5만 중 운좋은 것) → MAE 낙관적. 정직 성능은
중첩CV값(5~6문항은 A1보다 나빴음)이 기준. 여기선 '핵심 문항'과 '상한 감' 파악용.

산출: aim2/시행착오/b2_exhaustive_report.md
"""
import sys
from pathlib import Path
from itertools import combinations
from math import comb
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

KS = [int(x) for x in sys.argv[1:]] or [5, 6]     # 인자로 k 지정(병렬용), 없으면 5,6
OUTDIR=Path(__file__).parent
REPORT=OUTDIR/(f"b2_exhaustive_k{'_'.join(map(str,KS))}_report.md")
ITEMS=D.ITEMS; TRIALS=D.TRIALS; NI=len(ITEMS)
LAB={"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b옷입기","ADL0106A":"Q6a옷고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자있기","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
_lines=[]
def log(s=""):
    print(s.encode("ascii","replace").decode("ascii")); _lines.append(s)

m=D.build()
X=SimpleImputer(strategy="median").fit_transform(m[ITEMS])
trial=m.STUDYID.values; y=m.ds_stage.astype(int).values; ds=m.ds_stage.values
a0=m.A0_harmonized.values
rows={t:np.where(trial==t)[0] for t in TRIALS}
a1_mae=D.mae(ds,m.A1_2015_stage.values); a0_miss=D.miss(ds,m.A0_harmonized.values)*100

TAU=np.round(np.arange(0.10,0.55,0.05),3)   # 스크리닝용 격자(중증우선 τ)
def _P(g,n):
    P=np.zeros((n,6));P[:,0]=1-g[1]
    for k in range(1,5):P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5];P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True);return P
def _tune_sev(Ptr,ds_tr,a0t):
    """제약=중증놓침≤A0목표, 목적=MAE최소(도달불가 시 중증최소)."""
    best=None
    for t4 in TAU:
        for t5 in TAU:
            if t5<t4:continue
            p=D.asym(Ptr,t4,t5)
            if D.miss(ds_tr,p)<=a0t+1e-9:
                key=D.mae(ds_tr,p)
                if best is None or key<best[0]:best=(key,t4,t5)
    if best is None:
        c=[(D.miss(ds_tr,D.asym(Ptr,a,b)),D.mae(ds_tr,D.asym(Ptr,a,b)),a,b) for a in TAU for b in TAU if b>=a]
        _,_,a,b=min(c);return a,b
    return best[1],best[2]

def evalc(cols):
    """중증우선: fold마다 τ를 '중증놓침≤훈련A0' 제약 하 MAE최소로 튜닝(누출0)."""
    pred=np.zeros(len(m))
    for ho in TRIALS:
        tr=np.concatenate([rows[x] for x in TRIALS if x!=ho]); te=rows[ho]
        sc=StandardScaler().fit(X[np.ix_(tr,cols)])
        Ztr=sc.transform(X[np.ix_(tr,cols)]); Zte=sc.transform(X[np.ix_(te,cols)])
        md={k:LogisticRegression(solver="saga",l1_ratio=0.5,C=0.1,max_iter=500,tol=1e-3,
              random_state=0).fit(Ztr,(y[tr]>=k).astype(int)) for k in range(1,6)}
        gtr={k:md[k].predict_proba(Ztr)[:,1] for k in range(1,6)}
        gte={k:md[k].predict_proba(Zte)[:,1] for k in range(1,6)}
        Ptr=_P(gtr,len(tr)); Pte=_P(gte,len(te))
        a0t=D.miss(ds[tr],a0[tr])                    # 훈련 A0 중증놓침 목표
        t4,t5=_tune_sev(Ptr,ds[tr],a0t)
        pred[te]=D.asym(Pte,t4,t5)
    return D.mae(ds,pred), D.miss(ds,pred)*100, D.kap(ds,pred)

def run_k(k):
    log(f"## k={k} 전수조사 ({comb(NI,k):,}조합)")
    results=[]  # (mae, miss, kap, cols)
    for cols in combinations(range(NI),k):
        mae,miss,kap=evalc(list(cols)); results.append((mae,miss,kap,cols))
    results.sort(key=lambda r:r[0])   # 중증우선: 중증 A0수준 통제됨 → MAE순=최선
    n=len(results); champ=[r for r in results if r[0]<a1_mae and r[1]<=a0_miss+2.0]
    log(f"- 지배+안전(A1 MAE↓·중증≤A0+2%p) 조합: {len(champ):,}/{n:,} ({len(champ)/n*100:.1f}%)")
    log(f"- 최고 MAE {results[0][0]:.3f} (⚠️낙관: {n:,}중 최고) / 중앙 {results[n//2][0]:.3f} / 최악 {results[-1][0]:.3f}")
    log(f"\n### 최고 10조합 (MAE순, ⚠️낙관적)")
    log("| MAE | 중증 | 카파 | 문항 |")
    log("|---|---|---|---|")
    for mae,miss,kap,cols in results[:10]:
        log(f"| {mae:.3f} | {miss:.1f}% | {kap:.3f} | {', '.join(LAB[ITEMS[i]] for i in cols)} |")
    # 상위 1% 문항 빈도 = 진짜 핵심
    top=results[:max(50,n//100)]
    from collections import Counter
    cnt=Counter();
    for _,_,_,cols in top: cnt.update(cols)
    log(f"\n### 상위 {len(top):,}조합(상위 1%)에 자주 나오는 문항 = 핵심")
    log("| 문항 | 상위조합 등장률 |")
    log("|---|---|")
    for i,c in cnt.most_common(k+4):
        log(f"| {LAB[ITEMS[i]]} | {c/len(top)*100:.0f}% |")
    log("")

def main():
    log("# Aim 2 · B2 소문항 전수조사 k=5,6 (sjlee)\n")
    log(f"공통표본 {len(m)}. 기준 A1 MAE {a1_mae:.3f}/A0 중증 {a0_miss:.1f}%.")
    log("⚠️ 전수 최고 MAE=승자의 저주(낙관). 정직 성능은 중첩CV(5·6문항 A1보다 나빴음)가 기준. "
        "여기선 **핵심 문항 파악**용.\n")
    for k in KS:
        run_k(k); log(f"[진행] k={k} 완료")
    REPORT.write_text("\n".join(_lines),encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__=="__main__":
    main()
