# -*- coding: utf-8 -*-
"""
2단계: 파이프라인 검증 (sjlee) — dev(1,652) LOTO 로 공표 후보 수치 재현. test 미접근.
  모형: 절단별 L2 로지스틱(부분비례오즈) + 조건부중앙값 + 비대칭 τ4/τ5
  결측: listwise deletion(조합 문항 중 하나라도 결측이면 제외)
  τ튜닝: A0-목표(중증우선) — 제약 중증놓침 ≤ train A0, 목적 MAE 최소, 그리드 0.02
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")

HERE=Path(__file__).parent
REC=Path(r"C:/Users/user/OneDrive - 이화여자대학교/문서/urp-AD/03_aim2/보조검증")
sys.path.insert(0,str(REC)); import _recon_from_raw as R
DL=Path.home()/"Downloads"
TRIALS=["AD-1061","AD-1063","AD-1064"]; STAGES=np.arange(6)
GRID=np.round(np.arange(0.02,0.60,0.02),2)

FIXED6=["ADL0103","ADL0104","ADL0105","ADL0106A","ADL0115A","ADL0116A"]
CAND={
 "7문항 주(+Q17금전)":            FIXED6+["ADL0117A"],
 "8문항 주(+Q7전화,Q17금전)":     FIXED6+["ADL0107A","ADL0117A"],   # ← 사전지정 PRIMARY
 "8문항 보조1(+Q1먹기,Q7전화)":   FIXED6+["ADL0101","ADL0107A"],
 "8문항 보조2(+Q1먹기,Q17금전)":  FIXED6+["ADL0101","ADL0117A"],
 "9문항 주(+Q6b,Q17,Q23)":       FIXED6+["ADL0106B","ADL0117A","ADL0123L"],
}
PUB={"7문항 주(+Q17금전)":(0.561,14.55,0.552),
     "8문항 주(+Q7전화,Q17금전)":(0.561,14.55,0.553),
     "8문항 보조1(+Q1먹기,Q7전화)":(0.556,14.55,0.554),
     "8문항 보조2(+Q1먹기,Q17금전)":(0.559,14.55,0.550),
     "9문항 주(+Q6b,Q17,Q23)":(0.553,13.64,0.561)}

def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);hi=t>=4
    return (p[hi]<=3).mean() if hi.sum() else np.nan
def cmed(P): return STAGES[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,t4,t5):
    p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>t5,5,np.where(p4>t4,4,cmed(P)))
def fit(Xtr,ytr):
    sc=StandardScaler().fit(Xtr);Z=sc.transform(Xtr)
    md={k:LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,max_iter=4000).fit(Z,(ytr>=k).astype(int)) for k in range(1,6)}
    return sc,md
def P_(sc,md,X):
    Z=sc.transform(X);g={k:md[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((len(X),6));P[:,0]=1-g[1]
    for k in range(1,5):P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5];P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True);return P
def tune(Ptr,ytr,budget):
    best=None
    for t4 in GRID:
        for t5 in GRID:
            if t5<t4:continue
            p=asym(Ptr,t4,t5)
            if miss(ytr,p)<=budget+1e-9:
                key=mae(ytr,p)
                if best is None or key<best[0]:best=(key,t4,t5)
    if best is None:
        c=[(miss(ytr,asym(Ptr,a,b)),mae(ytr,asym(Ptr,a,b)),a,b) for a in GRID for b in GRID if b>=a]
        _,_,a,b=min(c);return a,b
    return best[1],best[2]

# ---- 데이터: dev 행에 ADL문항 + A0 병합 ----
rec=R.build_from_raw(TRIALS)
bs=pd.read_csv(DL/"baseline_sample.csv")
com=bs[bs["in_common_comparison_sample"]==True][["STUDYID","USUBJID","ds_stage","A0_harmonized","A1_2015_stage"]].copy()
from sklearn.model_selection import train_test_split
com["ds_stage"]=com["ds_stage"].astype(int)
strata=com["STUDYID"].astype(str)+"_"+com["ds_stage"].astype(str)
_,test_idx=train_test_split(com.index,test_size=0.25,stratify=strata,random_state=20260814)
com["split"]=np.where(com.index.isin(test_idx),"test","dev")
dev=com[com.split=="dev"].merge(rec.drop(columns=[c for c in ["ds_stage"] if c in rec]),
                                on=["STUDYID","USUBJID"],how="left").reset_index(drop=True)
print("dev 병합 n =",len(dev))

print("\n=== dev LOTO 재현 vs 공표값 ===")
print(f"{'후보':<26}{'n채점':>7}{'MAE':>8}{'놓침%':>8}{'카파':>8}   {'공표(MAE/놓침/카파)':>22}")
for name,cols in CAND.items():
    sub=dev.dropna(subset=cols).reset_index(drop=True)
    rows={t:sub.index[sub.STUDYID==t] for t in TRIALS}
    pred=np.full(len(sub),np.nan)
    for ho in TRIALS:
        tri=sub.index[sub.STUDYID!=ho]; tei=rows[ho]
        if len(tei)==0: continue
        sc,md=fit(sub.loc[tri,cols].values,sub.loc[tri,"ds_stage"].values)
        Ptr=P_(sc,md,sub.loc[tri,cols].values); Pte=P_(sc,md,sub.loc[tei,cols].values)
        budget=miss(sub.loc[tri,"ds_stage"].values,sub.loc[tri,"A0_harmonized"].values)
        t4,t5=tune(Ptr,sub.loc[tri,"ds_stage"].values,budget)
        pred[tei]=asym(Pte,t4,t5)
    m=mae(sub["ds_stage"],pred); ms=miss(sub["ds_stage"],pred)*100
    kp=cohen_kappa_score(sub["ds_stage"],pred.astype(int),weights="quadratic")
    pm,pmi,pk=PUB[name]
    flag="✔" if abs(m-pm)<=0.02 and abs(ms-pmi)<=2.0 else "≈"
    print(f"{name:<26}{len(sub):>7}{m:>8.3f}{ms:>8.2f}{kp:>8.3f}   {pm:>6.3f}/{pmi:>5.2f}/{pk:.3f} {flag}")
print("\n✔=MAE±0.02·놓침±2%p 이내 재현 / ≈=근사(스펙 미세차). 재현되면 test 진행 가능.")
