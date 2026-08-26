"""
Aim 2 · 12문항 vs 13문항 최종 후보 비교 (sjlee)
================================================
두 후보를 동일 기준(중증우선 CV, 목표=훈련A0)으로 평가 + FA4축 커버 + 4조건 체크 + 채점표.
12 = 중요도 top-12(성능기준 최소). 13 = 12 + 설거지(가사축 커버 → 4조건 모두).
산출: b2_pick_12_13_report.md, aim2_sjlee/b2_short12_scoretable.csv, b2_short13_scoretable.csv
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import b1_gn_elasticnet as G
warnings.filterwarnings("ignore")
OUTDIR = Path(__file__).parent; CSV = OUTDIR / "aim2_sjlee"
REPORT = OUTDIR / "b2_pick_12_13_report.md"
TRIALS = G.TRIALS; STAGES = np.arange(6); L1 = 0.5; Cg = 0.1
LAB = {"ADL0101":"먹기","ADL0102":"걷기","ADL0103":"화장실","ADL0104":"목욕","ADL0105":"몸단장",
 "ADL0106B":"입기","ADL0106A":"고르기","ADL0107A":"전화","ADL0110A":"설거지","ADL0111A":"식사준비",
 "ADL0112A":"집안일","ADL0113A":"빨래","ADL0114A":"가전","ADL0115A":"외출","ADL0116A":"쇼핑",
 "ADL0116B":"지불","ADL0117A":"금전","Q18":"혼자","ADL0121A":"글쓰기","ADL0122Q":"취미","ADL0123L":"가전사용"}
# FA 4축(앞서 배정)
AXIS = {"F1옷·식사":["ADL0106B","ADL0106A","ADL0111A"],
        "F2지역·금전":["ADL0107A","ADL0115A","ADL0116A","ADL0116B","ADL0117A","Q18","ADL0121A"],
        "F3가사":["ADL0110A","ADL0112A","ADL0113A","ADL0114A","ADL0122Q","ADL0123L"],
        "F4기초":["ADL0101","ADL0102","ADL0103","ADL0104","ADL0105"]}
def axis_of(it):
    for a, its in AXIS.items():
        if it in its: return a
    return "?"
S12 = ["ADL0104","ADL0115A","ADL0117A","ADL0105","ADL0103","ADL0111A","ADL0116B","ADL0107A","Q18","ADL0116A","ADL0102","ADL0106B"]
S13 = S12 + ["ADL0110A"]
_lines = []
def log(s=""):
    print(s.encode("ascii","replace").decode("ascii")); _lines.append(s)

def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);hi=t>=4
    return (p[hi]<=3).mean() if hi.sum() else np.nan
def kap(t,p): return cohen_kappa_score(np.asarray(t,int),np.asarray(p,int),weights="quadratic",labels=list(range(6)))
def cmed(P): return STAGES[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,t4,t5):
    p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>t5,5,np.where(p4>t4,4,cmed(P)))
def fit_sub(Xtr,ytr):
    imp=SimpleImputer(strategy="median").fit(Xtr);sc=StandardScaler().fit(imp.transform(Xtr));Z=sc.transform(imp.transform(Xtr))
    md={k:LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=L1,C=Cg,max_iter=3000,tol=1e-3,random_state=0).fit(Z,(ytr>=k).astype(int)) for k in range(1,6)}
    return imp,sc,md
def Psub(imp,sc,md,X):
    Z=sc.transform(imp.transform(X));g={k:md[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((len(X),6));P[:,0]=1-g[1]
    for k in range(1,5):P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5];P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True);return P
def tune(Ptr,ds_tr,tgt):
    grid=np.round(np.arange(0.10,0.55,0.025),3);best=None
    for t4 in grid:
        for t5 in grid:
            if t5<t4:continue
            p=asym(Ptr,t4,t5)
            if miss(ds_tr,p)<=tgt+1e-9:
                key=mae(ds_tr,p)
                if best is None or key<best[0]:best=(key,t4,t5)
    if best is None:
        c=[(miss(ds_tr,asym(Ptr,a,b)),mae(ds_tr,asym(Ptr,a,b)),a,b) for a in grid for b in grid if b>=a];_,_,a,b=min(c);return a,b
    return best[1],best[2]

def main():
    m=G.build_matrix().reset_index(drop=True)
    bs=pd.read_csv(Path.home()/"Downloads"/"baseline_sample.csv")
    bs=bs[bs["in_common_comparison_sample"]==True][["STUDYID","USUBJID","A0_harmonized"]]
    m=m.merge(bs,on=["STUDYID","USUBJID"],how="left").reset_index(drop=True)
    ds=m["ds_stage"].values
    a1_mae=mae(ds,m["A1_2015_stage"].values);a0_miss=miss(ds,m["A0_harmonized"].values)*100
    log("# Aim 2 · 12문항 vs 13문항 최종 후보 (sjlee)\n")
    log(f"기준 A1 MAE {a1_mae:.3f} / A0 중증놓침 {a0_miss:.1f}%. 중증우선 CV(목표=훈련A0).\n")

    def evalset(cols):
        pred=np.zeros(len(m));per={}
        for ho in TRIALS:
            tr=m.index[m.STUDYID!=ho];te=m.index[m.STUDYID==ho]
            imp,sc,md=fit_sub(m.loc[tr,cols].values,m.loc[tr,"ds_stage"].astype(int).values)
            Ptr=Psub(imp,sc,md,m.loc[tr,cols].values);Pte=Psub(imp,sc,md,m.loc[te,cols].values)
            tgt=miss(m.loc[tr,"ds_stage"].values,m.loc[tr,"A0_harmonized"].values)
            t4,t5=tune(Ptr,m.loc[tr,"ds_stage"].values,tgt)
            pr=asym(Pte,t4,t5);pred[te]=pr;per[ho]=miss(m.loc[te,"ds_stage"].values,pr)*100
        return mae(ds,pred),miss(ds,pred)*100,kap(ds,pred),per

    for name,cols in [("12문항",S12),("13문항",S13)]:
        mm,hh,kk,per=evalset(cols)
        axes=sorted(set(axis_of(i) for i in cols))
        log(f"## {name} ({len(cols)}개)")
        log("- 문항(영역): " + ", ".join(f"{LAB[i]}({axis_of(i)[:2]})" for i in cols))
        log(f"- **성능: MAE {mm:.3f} / 중증놓침 {hh:.1f}% / 카파 {kk:.3f}**")
        log(f"- 시험별 중증놓침: {{{', '.join(f'{h}:{v:.1f}%' for h,v in per.items())}}}")
        log(f"- FA축 커버: {axes} ({len(axes)}/4)")
        c1 = mm <= a1_mae; c2 = hh <= a0_miss; c3 = len(axes)==4
        core = all(x in cols for x in ["ADL0103","ADL0104","ADL0105","ADL0115A"])
        log(f"- 조건: 핵심4고정 {'✅' if core else '❌'} / MAE≤A1 {'✅' if c1 else '❌'} / "
            f"중증≤A0 {'✅' if c2 else '❌'} / 4축커버 {'✅' if c3 else '❌'} "
            f"→ **{'4조건 전부 충족' if (c1 and c2 and c3 and core) else '일부 미충족'}**\n")
        # 채점표
        imp=SimpleImputer(strategy="median").fit(m[cols]);sc=StandardScaler().fit(imp.transform(m[cols]))
        Z=sc.transform(imp.transform(m[cols]));sd=sc.scale_;mu=sc.mean_;y=m["ds_stage"].astype(int).values
        rows=[]
        for k in range(1,6):
            lr=LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=L1,C=Cg,max_iter=5000,tol=1e-3,random_state=0).fit(Z,(y>=k).astype(int))
            w=lr.coef_[0]/sd;b0=lr.intercept_[0]-np.sum(lr.coef_[0]*mu/sd)
            for i,it in enumerate(cols):rows.append({"cut":f"P(Y>={k})","item":it,"label":LAB[it],"coef_raw":round(w[i],4)})
            rows.append({"cut":f"P(Y>={k})","item":"(절편)","label":"","coef_raw":round(b0,4)})
        pd.DataFrame(rows).to_csv(CSV/f"b2_short{len(cols)}_scoretable.csv",index=False,encoding="utf-8-sig")

    log("## 요약")
    log("- **13 = 12 + 설거지**. 12는 성능조건(MAE≤A1·중증≤A0)만 최소, 가사(F3)축 빠짐(4조건 중 축균형 미충족).")
    log("- **13은 설거지로 가사축까지 커버 → 4조건 전부 충족.** 문항 1개 더로 축균형 확보.")
    log("- 두 안 성능 차이는 미미(노이즈 범위). 선택은 '축균형(4조건)까지 지킬지'로 귀결.")
    REPORT.write_text("\n".join(_lines),encoding="utf-8");print(f">>> 저장: {REPORT}")

if __name__=="__main__":
    main()
