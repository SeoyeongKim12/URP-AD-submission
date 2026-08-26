# -*- coding: utf-8 -*-
"""
3단계: test 1회 검증 + 원눈금 채점표 (sjlee)
  프로토콜: dev(1652) 전체로 적합 + τ4/τ5 튜닝(A0-목표) → test(551) 딱 1회 예측.
  사전지정 primary=8문항 주였으나 팀 최종 확정=7문항(고정6+Q7전화, Q17은 Q7과 같은 축이라 제외).
  나머지 후보는 참고(선택은 test로 안 함). 채점표: 확정 7문항 dev+test 재학습 → 원눈금 계수·절편 + τ4/τ5 CSV.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")

HERE=Path(__file__).parent
REC=Path(r"C:/Users/user/OneDrive - 이화여자대학교/문서/urp-AD/03_aim2/보조검증")
sys.path.insert(0,str(REC)); import _recon_from_raw as R
DL=Path.home()/"Downloads"
TRIALS=["AD-1061","AD-1063","AD-1064"]; STAGES=np.arange(6)
GRID=np.round(np.arange(0.02,0.60,0.02),2)
LAB={"ADL0101":"Q1먹기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
     "ADL0106A":"Q6a옷고르기","ADL0106B":"Q6b옷입기","ADL0107A":"Q7전화","ADL0112A":"Q12집안일",
     "ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑","ADL0117A":"Q17금전","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
FIXED6=["ADL0103","ADL0104","ADL0105","ADL0106A","ADL0115A","ADL0116A"]
PRIMARY="7문항 확정(+Q7전화)"   # 팀 최종 확정 = 고정6 + Q7전화 (Q17은 Q7과 같은 축이라 제외)
CAND={  # 추가검증 페이지 8개 후보 전부 (7문항×2, 8문항×3, 9문항×3)
 "7문항 주(+Q17금전)":            FIXED6+["ADL0117A"],
 "7문항 확정(+Q7전화)":           FIXED6+["ADL0107A"],
 "8문항 주(+Q7전화,Q17금전)":     FIXED6+["ADL0107A","ADL0117A"],
 "8문항 보조1(+Q1먹기,Q7전화)":   FIXED6+["ADL0101","ADL0107A"],
 "8문항 보조2(+Q1먹기,Q17금전)":  FIXED6+["ADL0101","ADL0117A"],
 "9문항 주(+Q6b,Q17,Q23)":       FIXED6+["ADL0106B","ADL0117A","ADL0123L"],
 "9문항 보조1(+Q6b,Q12,Q17)":    FIXED6+["ADL0106B","ADL0112A","ADL0117A"],
 "9문항 보조2(+Q1,Q6b,Q22)":     FIXED6+["ADL0101","ADL0106B","ADL0122Q"],
}
def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);hi=t>=4
    return (p[hi]<=3).mean() if hi.sum() else np.nan
def exact(t,p): return (np.asarray(t,int)==np.asarray(p,int)).mean()
def within1(t,p): return (np.abs(np.asarray(t,float)-np.asarray(p,float))<=1).mean()
def cmed(P): return STAGES[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,t4,t5):
    p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>t5,5,np.where(p4>t4,4,cmed(P)))
def fit_std(Xtr,ytr):
    sc=StandardScaler().fit(Xtr);Z=sc.transform(Xtr)
    md={k:LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,max_iter=5000).fit(Z,(ytr>=k).astype(int)) for k in range(1,6)}
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

# ---- 데이터 + 공식 재현 분할 ----
rec=R.build_from_raw(TRIALS)
bs=pd.read_csv(DL/"baseline_sample.csv")
com=bs[bs["in_common_comparison_sample"]==True][["STUDYID","USUBJID","ds_stage","A0_harmonized","A1_2015_stage"]].copy()
com["ds_stage"]=com["ds_stage"].astype(int)
strata=com["STUDYID"].astype(str)+"_"+com["ds_stage"].astype(str)
_,test_idx=train_test_split(com.index,test_size=0.25,stratify=strata,random_state=20260814)
com["split"]=np.where(com.index.isin(test_idx),"test","dev")
full=com.merge(rec.drop(columns=[c for c in ["ds_stage"] if c in rec]),on=["STUDYID","USUBJID"],how="left")
dev=full[full.split=="dev"].reset_index(drop=True); test=full[full.split=="test"].reset_index(drop=True)
print(f"dev={len(dev)}  test={len(test)}")
print(f"dev A0 놓침(예산)={miss(dev.ds_stage,dev.A0_harmonized)*100:.2f}%  "
      f"test A0 놓침={miss(test.ds_stage,test.A0_harmonized)*100:.2f}%")
print(f"test 전체 A1: MAE={mae(test.ds_stage,test.A1_2015_stage):.3f} 놓침={miss(test.ds_stage,test.A1_2015_stage)*100:.2f}% "
      f"카파={cohen_kappa_score(test.ds_stage,test.A1_2015_stage.astype(int),weights='quadratic'):.3f}")
print(f"test 전체 A0: MAE={mae(test.ds_stage,test.A0_harmonized):.3f} 놓침={miss(test.ds_stage,test.A0_harmonized)*100:.2f}% "
      f"카파={cohen_kappa_score(test.ds_stage,test.A0_harmonized.astype(int),weights='quadratic'):.3f}")

print("\n"+"="*94)
print("TEST 1회 검증 — dev 전체 적합+τ튜닝 → test 예측 (★=사전지정 primary)")
print("="*94)
print(f"{'후보':<26}{'n':>5}{'MAE':>7}{'놓침%':>7}{'카파':>7}{'±1%':>7}{'정확%':>7}  {'τ4/τ5':>9}")
results={}
for name,cols in CAND.items():
    dtr=dev.dropna(subset=cols); dte=test.dropna(subset=cols)
    sc,md=fit_std(dtr[cols].values,dtr["ds_stage"].values)
    Ptr=P_(sc,md,dtr[cols].values)
    budget=miss(dtr["ds_stage"].values,dtr["A0_harmonized"].values)
    t4,t5=tune(Ptr,dtr["ds_stage"].values,budget)
    Pte=P_(sc,md,dte[cols].values); pte=asym(Pte,t4,t5)
    r=dict(n=len(dte),mae=mae(dte.ds_stage,pte),miss=miss(dte.ds_stage,pte)*100,
           kappa=cohen_kappa_score(dte.ds_stage,pte.astype(int),weights="quadratic"),
           w1=within1(dte.ds_stage,pte)*100,ex=exact(dte.ds_stage,pte)*100,t4=t4,t5=t5,
           a1_mae=mae(dte.ds_stage,dte.A1_2015_stage),a1_miss=miss(dte.ds_stage,dte.A1_2015_stage)*100,
           a0_miss=miss(dte.ds_stage,dte.A0_harmonized)*100)
    results[name]=r
    star="★" if name==PRIMARY else " "
    print(f"{star}{name:<25}{r['n']:>5}{r['mae']:>7.3f}{r['miss']:>7.2f}{r['kappa']:>7.3f}{r['w1']:>7.1f}{r['ex']:>7.1f}  {t4:>4.2f}/{t5:<4.2f}")

p=results[PRIMARY]
print(f"\n── 최종 확정({PRIMARY}) test 판정 ──")
print(f"  vs A1(동일 test행): MAE {p['mae']:.3f} {'<' if p['mae']<p['a1_mae'] else '≥'} {p['a1_mae']:.3f} | "
      f"놓침 {p['miss']:.1f}% {'≤' if p['miss']<=p['a1_miss'] else '>'} A1 {p['a1_miss']:.1f}%")
print(f"  vs A0(동일 test행) 놓침: {p['miss']:.1f}% {'≤' if p['miss']<=p['a0_miss']+1e-9 else '>'} A0 {p['a0_miss']:.1f}%")

# ================= 원눈금 채점표 (확정 7문항, dev+test 재학습) =================
print("\n"+"="*94)
print(f"원눈금 채점표 — 최종 확정 7문항(고정6+Q7전화), dev+test 재학습")
print("="*94)
cols=CAND[PRIMARY]
allrows=full.dropna(subset=cols).reset_index(drop=True)
X=allrows[cols].values.astype(float); y=allrows["ds_stage"].values
sc=StandardScaler().fit(X); mu=sc.mean_; sd=sc.scale_; Z=sc.transform(X)
rows=[]
for k in range(1,6):
    m=LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,max_iter=5000).fit(Z,(y>=k).astype(int))
    cstd=m.coef_[0]; b_std=m.intercept_[0]
    craw=cstd/sd                                   # 원눈금 계수
    b_raw=b_std-np.sum(cstd*mu/sd)                 # 원눈금 절편
    for j,it in enumerate(cols):
        rows.append(dict(cutpoint=f"P(Y>={k})",item_code=it,item_label=LAB[it],
                         ridge_coef_raw=round(craw[j],4),ridge_coef_standardized=round(cstd[j],4)))
    rows.append(dict(cutpoint=f"P(Y>={k})",item_code="(intercept)",item_label="(절편)",
                     ridge_coef_raw=round(b_raw,4),ridge_coef_standardized=round(b_std,4)))
score=pd.DataFrame(rows)
# 최종 τ4/τ5 (dev+test 재학습 in-sample, A0-목표)
Pall=P_(sc,{k:LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,max_iter=5000).fit(Z,(y>=k).astype(int)) for k in range(1,6)},X)
budget_all=miss(y,allrows["A0_harmonized"].values)
T4,T5=tune(Pall,y,budget_all)
score.to_csv(HERE/"b2_scoretable_7item_raw.csv",index=False,encoding="utf-8-sig")
rule=pd.DataFrame([dict(rule="asymmetric_threshold",tau4=T4,tau5=T5,budget_A0_miss=round(budget_all*100,2),
    decision="p5=P(Y=5); p4=P(Y>=4); 예측=5 if p5>tau5 elif 4 if p4>tau4 else 조건부중앙값",
    n_train=len(allrows),penalty="L2(C=1.0, standardized)")])
rule.to_csv(HERE/"b2_scoretable_7item_rule.csv",index=False,encoding="utf-8-sig")
print(score.to_string(index=False))
print(f"\nτ4={T4:.2f}  τ5={T5:.2f}  (A0예산 {budget_all*100:.2f}%, n={len(allrows)})")
print("저장: b2_scoretable_7item_raw.csv / b2_scoretable_7item_rule.csv")
