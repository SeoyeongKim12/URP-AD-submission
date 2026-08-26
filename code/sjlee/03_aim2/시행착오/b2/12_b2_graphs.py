"""
Aim 2 · B2 결정용 그림 2종 (sjlee)
===================================
1) 라쏘 경로: 순수 L1로 λ 스윕 → 문항별 |계수|합 경로 + 문항 죽는 순서(fold 안정성).
2) 성능-문항수: 중증우선 중첩CV의 MAE·중증놓침 vs K + A0/A1/안전경계 → 엘보.

산출: aim2/시행착오/fig_lasso_path.png, fig_perf_vs_k.png
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import b1_gn_elasticnet as G

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OUTDIR = Path(__file__).parent
ITEMS = G.ITEMS; TRIALS = G.TRIALS; STAGES = np.arange(6); L1 = 0.5; Cg = 0.1
LAB = {"ADL0101":"먹기","ADL0102":"걷기","ADL0103":"화장실","ADL0104":"목욕","ADL0105":"몸단장",
 "ADL0106B":"입기","ADL0106A":"고르기","ADL0107A":"전화","ADL0110A":"설거지","ADL0111A":"식사준비",
 "ADL0112A":"집안일","ADL0113A":"빨래","ADL0114A":"가전","ADL0115A":"외출","ADL0116A":"쇼핑",
 "ADL0116B":"지불","ADL0117A":"금전","Q18":"혼자","ADL0121A":"글쓰기","ADL0122Q":"취미","ADL0123L":"가전사용"}
CORE = ["ADL0103","ADL0104","ADL0105","ADL0115A"]

def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);hi=t>=4
    return (p[hi]<=3).mean() if hi.sum() else np.nan
def cmed(P): return STAGES[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,t4,t5):
    p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>t5,5,np.where(p4>t4,4,cmed(P)))

m=G.build_matrix().reset_index(drop=True)
bs=pd.read_csv(Path.home()/"Downloads"/"baseline_sample.csv")
bs=bs[bs["in_common_comparison_sample"]==True][["STUDYID","USUBJID","A0_harmonized"]]
m=m.merge(bs,on=["STUDYID","USUBJID"],how="left").reset_index(drop=True)
ds=m["ds_stage"].values
a1_mae=mae(ds,m["A1_2015_stage"].values); a0_miss=miss(ds,m["A0_harmonized"].values)*100

# ─────────────────────────── 1) 라쏘 경로 ───────────────────────────
Cgrid=np.logspace(-2.3,1.0,22)
def lasso_item_abssum(Xtr,ytr,Cs):
    imp=SimpleImputer(strategy="median").fit(Xtr); sc=StandardScaler().fit(imp.transform(Xtr))
    Z=sc.transform(imp.transform(Xtr))
    out=np.zeros((len(Cs),len(ITEMS)))
    for ci,C in enumerate(Cs):
        s=np.zeros(len(ITEMS))
        for k in range(1,6):
            lr=LogisticRegression(penalty="l1",solver="liblinear",C=C,max_iter=2000).fit(Z,(ytr>=k).astype(int))
            s+=np.abs(lr.coef_[0])
        out[ci]=s
    return out

Xall=m[ITEMS].values; yall=m["ds_stage"].astype(int).values
path_full=lasso_item_abssum(Xall,yall,Cgrid)
# 문항 죽는 순서(fold별): C 작은쪽부터 마지막까지 0 아닌 순서 = 생존 순위
drop_order={}
for ho in ["FULL"]+TRIALS:
    if ho=="FULL": P=path_full
    else:
        idx=m.index[m.STUDYID!=ho]; P=lasso_item_abssum(m.loc[idx,ITEMS].values,m.loc[idx,"ds_stage"].astype(int).values,Cgrid)
    # 각 문항이 처음으로 0 아니게 되는 C(=생존 강도). 작을수록 오래 생존(중요)
    surv=[]
    for j in range(len(ITEMS)):
        nz=np.where(P[:,j]>1e-6)[0]
        surv.append(Cgrid[nz[0]] if len(nz) else np.inf)
    drop_order[ho]=[ITEMS[j] for j in np.argsort(surv)]   # 오래 생존순

fig,ax=plt.subplots(figsize=(9,6))
for j,it in enumerate(ITEMS):
    y=path_full[:,j]; core=it in CORE
    ax.plot(Cgrid,y,lw=2.4 if core else 1.0,alpha=0.95 if core else 0.4,
            color=None if core else "gray",zorder=3 if core else 1,
            label=LAB[it] if core else None)
    if y[-1]>1e-6:
        ax.text(Cgrid[-1]*1.02,y[-1],LAB[it],fontsize=8,va="center",
                fontweight="bold" if core else "normal",color="black" if core else "gray")
ax.set_xscale("log"); ax.set_xlabel("C (← 벌점 강함 / 벌점 약함 →)")
ax.set_ylabel("문항 |계수|합 (5개 절단)")
ax.set_title("라쏘 경로 — 문항 생존(굵은선=고정 핵심4)")
ax.legend(loc="upper left",fontsize=9,title="핵심4")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR/"fig_lasso_path.png",dpi=130); plt.close()

# 생존순위 안정성 표(콘솔)
print("=== 라쏘 생존순위(오래 살아남는=중요) 상위 13 ===")
for ho in ["FULL"]+TRIALS:
    print(f"[{ho}] "+", ".join(LAB[i] for i in drop_order[ho][:13]))

# ─────────────────────────── 2) 성능-문항수 ───────────────────────────
# 중요도 순위(elastic-net |계수|)로 top-K, 중증우선 CV(target=훈련A0)
imp0=SimpleImputer(strategy="median").fit(m[ITEMS]); sc0=StandardScaler().fit(imp0.transform(m[ITEMS]))
Z0=sc0.transform(imp0.transform(m[ITEMS]))
imps=np.zeros(len(ITEMS))
for k in range(1,6):
    imps+=np.abs(LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=L1,C=Cg,max_iter=3000,tol=1e-3,random_state=0).fit(Z0,(yall>=k).astype(int)).coef_[0])
order=[ITEMS[i] for i in np.argsort(-imps)]

def fit_sub(Xtr,ytr):
    imp=SimpleImputer(strategy="median").fit(Xtr);sc=StandardScaler().fit(imp.transform(Xtr))
    Z=sc.transform(imp.transform(Xtr))
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
        c=[(miss(ds_tr,asym(Ptr,a,b)),mae(ds_tr,asym(Ptr,a,b)),a,b) for a in grid for b in grid if b>=a]
        _,_,a,b=min(c);return a,b
    return best[1],best[2]
def cvK(cols):
    pred=np.zeros(len(m))
    for ho in TRIALS:
        tr=m.index[m.STUDYID!=ho];te=m.index[m.STUDYID==ho]
        imp,sc,md=fit_sub(m.loc[tr,cols].values,m.loc[tr,"ds_stage"].astype(int).values)
        Ptr=Psub(imp,sc,md,m.loc[tr,cols].values);Pte=Psub(imp,sc,md,m.loc[te,cols].values)
        tgt=miss(m.loc[tr,"ds_stage"].values,m.loc[tr,"A0_harmonized"].values)
        t4,t5=tune(Ptr,m.loc[tr,"ds_stage"].values,tgt)
        pred[te]=asym(Pte,t4,t5)
    return mae(ds,pred),miss(ds,pred)*100

Ks=[4,5,6,7,8,9,10,11,12,13,15,17,19,21]
maes=[];misses=[]
for K in Ks:
    mm,hh=cvK(order[:K]);maes.append(mm);misses.append(hh)
    print(f"K={K}: MAE {mm:.3f}, 중증 {hh:.1f}%")

fig,(a1,a2)=plt.subplots(2,1,figsize=(9,7),sharex=True)
a1.plot(Ks,maes,"o-",color="#2166ac",lw=2)
a1.axhline(a1_mae,color="red",ls="--",lw=1.5,label=f"A1 MAE {a1_mae:.3f}")
a1.set_ylabel("MAE (낮을수록 좋음)"); a1.set_title("성능 vs 문항수 (중증우선 중첩CV)")
a1.legend(fontsize=9);a1.grid(alpha=0.3)
a2.plot(Ks,misses,"o-",color="#b2182b",lw=2)
a2.axhline(a0_miss,color="green",ls="--",lw=1.5,label=f"A0 중증 {a0_miss:.1f}% (엄격 기준)")
a2.axhline(a0_miss+2,color="orange",ls=":",lw=1.5,label=f"A0+2%p {a0_miss+2:.1f}% (관대 기준)")
a2.set_ylabel("중증놓침 % (낮을수록 좋음)");a2.set_xlabel("문항 수 K")
a2.legend(fontsize=9);a2.grid(alpha=0.3)
plt.tight_layout();plt.savefig(OUTDIR/"fig_perf_vs_k.png",dpi=130);plt.close()
print(">>> 저장: fig_lasso_path.png, fig_perf_vs_k.png")
