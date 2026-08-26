"""
Aim 2 · Q6(옷 고르기·입기) 처리 방식 비교 (sjlee)
==================================================
VIF에서 Q6a(5.05)·Q6b(5.00)만 높음 → Q6 처리를 4가지로 CV 비교.
(a) 둘 다 유지(21) / (b) 합치기 Q6=Q6a+Q6b(20) / (c1) Q6a만(20) / (c2) Q6b만(20)

모형: B1(부분PO 엘라스틱넷 + A0-목표 임계), leave-one-trial-out. 예측 거의 같으면
'해석·계획서19·개념'으로 결정하면 됨.

산출: aim2/시행착오/q6_variants_report.md
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "q6_variants_report.md"
TRIALS = D.TRIALS
_lines=[]
def log(s=""):
    print(s.encode("ascii","replace").decode("ascii")); _lines.append(s)

def fit_cuts(Ztr,y):
    return {k:LogisticRegression(solver="saga",l1_ratio=0.5,C=0.1,max_iter=3000,tol=1e-3,
            random_state=0).fit(Ztr,(y>=k).astype(int)) for k in range(1,6)}
def probs(mdl,Z):
    g={k:mdl[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((Z.shape[0],6)); P[:,0]=1-g[1]
    for k in range(1,5): P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5]; P=np.clip(P,1e-9,None); P/=P.sum(1,keepdims=True); return P
def tune_tau(Ptr,ds_tr,a0m):
    grid=np.round(np.arange(0.05,0.55,0.02),3); best=None
    for t4 in grid:
        for t5 in grid:
            if t5<t4: continue
            p=D.asym(Ptr,t4,t5)
            if D.miss(ds_tr,p)<=a0m+1e-9:
                key=D.mae(ds_tr,p)
                if best is None or key<best[0]: best=(key,t4,t5)
    if best is None:
        c=[(D.miss(ds_tr,D.asym(Ptr,a,b)),D.mae(ds_tr,D.asym(Ptr,a,b)),a,b) for a in grid for b in grid if b>=a]
        _,_,t4,t5=min(c); best=(0,t4,t5)
    return best[1],best[2]

def cv_eval(m, feats):
    t=m["ds_stage"].values; pred=np.zeros(len(m))
    for ho in TRIALS:
        tri=np.array(m.index[m.STUDYID!=ho]); tei=np.array(m.index[m.STUDYID==ho])
        imp=SimpleImputer(strategy="median").fit(m.loc[tri,feats])
        Xtr=imp.transform(m.loc[tri,feats]); Xte=imp.transform(m.loc[tei,feats])
        sc=StandardScaler().fit(Xtr)
        y=m.loc[tri,"ds_stage"].astype(int).values; ds_tr=m.loc[tri,"ds_stage"].values
        a0m=D.miss(ds_tr,m.loc[tri,"A0_harmonized"].values)
        mdl=fit_cuts(sc.transform(Xtr),y)
        Ptr=probs(mdl,sc.transform(Xtr)); Pte=probs(mdl,sc.transform(Xte))
        t4,t5=tune_tau(Ptr,ds_tr,a0m)
        pred[tei]=D.asym(Pte,t4,t5)
    return D.mae(t,pred), D.miss(t,pred)*100, D.kap(t,pred)

def main():
    log("# Aim 2 · Q6 처리 방식 비교 (sjlee)\n")
    m=D.build()
    m["Q6_sum"]=m["ADL0106A"]+m["ADL0106B"]           # 합치기용
    a1_mae=D.mae(m.ds_stage.values,m.A1_2015_stage.values)
    a0_miss=D.miss(m.ds_stage.values,m.A0_harmonized.values)*100
    base19=[c for c in D.ITEMS if c not in ("ADL0106A","ADL0106B")]   # Q6 뺀 19개
    variants={
        "(a) 둘 다 유지 [21문항]": base19+["ADL0106A","ADL0106B"],
        "(b) 합치기 Q6=a+b [20문항]": base19+["Q6_sum"],
        "(c1) Q6a(고르기)만 [20문항]": base19+["ADL0106A"],
        "(c2) Q6b(입기)만 [20문항]": base19+["ADL0106B"],
    }
    log(f"공통표본 {len(m)}. 기준 A1 MAE {a1_mae:.3f} / A0 중증 {a0_miss:.1f}%.\n")
    log("| Q6 처리 | 문항수 | MAE | 중증놓침 | 카파 | A1대비MAE | 둘다이김 |")
    log("|---|---|---|---|---|---|---|")
    res={}
    for name,feats in variants.items():
        mae,miss,kap=cv_eval(m,feats); res[name]=(mae,miss,kap)
        champ="✅" if (mae<a1_mae and miss<=a0_miss+1.0) else "❌"
        log(f"| {name} | {len(feats)} | {mae:.3f} | {miss:.1f}% | {kap:.3f} | {a1_mae-mae:+.3f} | {champ} |")
    maes=[v[0] for v in res.values()]
    log(f"\n- MAE 최대-최소 차: **{max(maes)-min(maes):.3f}** (작을수록 '예측 차이 없음').")
    log("- 예측 거의 같으면 → 해석 깔끔함/계획서 19문항/개념(Q6a=iADL,Q6b=bADL)으로 선택.")
    log("- 개념 주의: Chandler는 Q6a=iADL, Q6b=bADL로 다른 도메인 → (b)합치기는 두 도메인 혼합.\n")
    REPORT.write_text("\n".join(_lines),encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__=="__main__":
    main()
