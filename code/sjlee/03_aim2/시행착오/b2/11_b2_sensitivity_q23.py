# -*- coding: utf-8 -*-
"""
B2 민감도: 확정 7문항(고정6+Q7) vs +Q23가전사용(8문항) (sjlee)
============================================================
Q23은 요인적재상 F3(집안일·설거지·빨래 축)에 강하게 실림(F3=-0.63).
"F3 축 커버 강화" 목적으로 Q23을 추가하면 어떻게 되는지 dev LOTO + test 1회로 확인.
결과 요약: Q23 추가 시 MAE는 개선되나 중증놓침이 A0(20.8%) 위로 올라가 안전기준 위반
          → 중증우선 원칙상 7문항(고정6+Q7) 유지가 정답.
프로토콜/스펙은 b2_test_and_scoretable.py 와 동일(재현 분할, L2, 조건부중앙값+비대칭 τ, A0-목표).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")

REC=Path(r"C:/Users/user/OneDrive - 이화여자대학교/문서/urp-AD/03_aim2/보조검증")
sys.path.insert(0, str(REC)); import _recon_from_raw as R
DL=Path.home()/"Downloads"
TR=["AD-1061","AD-1063","AD-1064"]; ST=np.arange(6); GRID=np.round(np.arange(0.02,0.60,0.02),2)
F6=["ADL0103","ADL0104","ADL0105","ADL0106A","ADL0115A","ADL0116A"]
CAND={"7문항 확정(고정6+Q7)": F6+["ADL0107A"],
      "8문항(고정6+Q7+Q23)":  F6+["ADL0107A","ADL0123L"]}

def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);h=t>=4;return (p[h]<=3).mean() if h.sum() else np.nan
def cmed(P): return ST[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,a,b): p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>b,5,np.where(p4>a,4,cmed(P)))
def fit(X,y):
    s=StandardScaler().fit(X);Z=s.transform(X)
    return s,{k:LogisticRegression(penalty="l2",C=1.0,max_iter=5000).fit(Z,(y>=k).astype(int)) for k in range(1,6)}
def P_(s,m,X):
    Z=s.transform(X);g={k:m[k].predict_proba(Z)[:,1] for k in range(1,6)};P=np.zeros((len(X),6));P[:,0]=1-g[1]
    for k in range(1,5):P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5];P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True);return P
def tune(P,y,b):
    best=None
    for a in GRID:
        for c in GRID:
            if c<a: continue
            p=asym(P,a,c)
            if miss(y,p)<=b+1e-9:
                k=mae(y,p)
                if best is None or k<best[0]: best=(k,a,c)
    if best is None:
        cc=[(miss(y,asym(P,a,c)),mae(y,asym(P,a,c)),a,c) for a in GRID for c in GRID if c>=a]
        _,_,a,c=min(cc);return a,c
    return best[1],best[2]

rec=R.build_from_raw(TR); bs=pd.read_csv(DL/"baseline_sample.csv")
com=bs[bs.in_common_comparison_sample==True][["STUDYID","USUBJID","ds_stage","A0_harmonized","A1_2015_stage"]].copy()
com["ds_stage"]=com.ds_stage.astype(int)
st=com.STUDYID.astype(str)+"_"+com.ds_stage.astype(str)
_,ti=train_test_split(com.index,test_size=0.25,stratify=st,random_state=20260814)
com["split"]=np.where(com.index.isin(ti),"test","dev")
full=com.merge(rec.drop(columns=[c for c in ["ds_stage"] if c in rec]),on=["STUDYID","USUBJID"],how="left")
dev=full[full.split=="dev"]; test=full[full.split=="test"]

print(f"{'조합':<22}{'구간':>8}{'n':>6}{'MAE':>8}{'놓침%':>8}{'카파':>8}")
for nm,cols in CAND.items():
    sub=dev.dropna(subset=cols); rows={t:sub.index[sub.STUDYID==t] for t in TR}
    pr=pd.Series(np.full(len(sub),np.nan),index=sub.index)
    for ho in TR:
        tri=sub.index[sub.STUDYID!=ho]; tei=rows[ho]; s,m=fit(sub.loc[tri,cols].values,sub.loc[tri,"ds_stage"].values)
        a,c=tune(P_(s,m,sub.loc[tri,cols].values),sub.loc[tri,"ds_stage"].values,
                 miss(sub.loc[tri,"ds_stage"].values,sub.loc[tri,"A0_harmonized"].values))
        pr.loc[tei]=asym(P_(s,m,sub.loc[tei,cols].values),a,c)
    print(f"{nm:<22}{'devLOTO':>8}{len(sub):>6}{mae(sub.ds_stage,pr):>8.3f}{miss(sub.ds_stage,pr)*100:>8.2f}"
          f"{cohen_kappa_score(sub.ds_stage,pr.astype(int),weights='quadratic'):>8.3f}")
    dtr=dev.dropna(subset=cols); dte=test.dropna(subset=cols); s,m=fit(dtr[cols].values,dtr.ds_stage.values)
    a,c=tune(P_(s,m,dtr[cols].values),dtr.ds_stage.values,miss(dtr.ds_stage.values,dtr.A0_harmonized.values))
    pt=asym(P_(s,m,dte[cols].values),a,c)
    print(f"{'':<22}{'TEST':>8}{len(dte):>6}{mae(dte.ds_stage,pt):>8.3f}{miss(dte.ds_stage,pt)*100:>8.2f}"
          f"{cohen_kappa_score(dte.ds_stage,pt.astype(int),weights='quadratic'):>8.3f}")
for nm,ref in [("(참고)A1 test","A1_2015_stage"),("(참고)A0 test","A0_harmonized")]:
    print(f"{nm:<22}{'':>8}{len(test):>6}{mae(test.ds_stage,test[ref]):>8.3f}{miss(test.ds_stage,test[ref])*100:>8.2f}"
          f"{cohen_kappa_score(test.ds_stage,test[ref].astype(int),weights='quadratic'):>8.3f}")
print("\n결론: Q23 추가 시 test MAE 0.538→0.515(개선)이나 중증놓침 19.4→22.2%로 A0(20.8%) 초과 → 안전기준 위반.")
print("      중증우선 원칙상 7문항(고정6+Q7) 유지가 정답.")
