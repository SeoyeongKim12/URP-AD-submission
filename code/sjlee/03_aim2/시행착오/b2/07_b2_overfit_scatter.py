"""
B2 과적합 실증 산점도 (내부 vs AD-1062 외부) (sjlee)
=====================================================
fixed6 + 무작위 추가로 조합 생성 → 내부 LOTO 성능 vs AD-1062 외부 성능 상관.
내부 종합점수(낮을수록 좋음)가 외부 중증놓침을 예측 못 하면 = 전수탐색 과적합 실증.
산출: aim2/시행착오/fig_b2_overfit_scatter.png, b2_overfit_scatter.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False

HERE=Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent/"보조검증"))
import _recon_from_raw as R
ITEMS=R.ITEMS; TRIALS=["AD-1061","AD-1063","AD-1064"]; STAGES=np.arange(6)
FIXED=["ADL0103","ADL0104","ADL0105","ADL0106A","ADL0115A","ADL0116A"]
POOL=[i for i in ITEMS if i not in FIXED]
DL=Path.home()/"Downloads"

def mae(t,p): return np.abs(np.asarray(t,float)-np.asarray(p,float)).mean()
def miss(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);hi=t>=4
    return (p[hi]<=3).mean() if hi.sum() else np.nan
def cmed(P): return STAGES[(np.cumsum(P,1)>=0.5).argmax(1)]
def asym(P,t4,t5):
    p5=P[:,5];p4=P[:,4]+P[:,5];return np.where(p5>t5,5,np.where(p4>t4,4,cmed(P)))
def fit(Xtr,ytr):
    imp=SimpleImputer(strategy="median").fit(Xtr);sc=StandardScaler().fit(imp.transform(Xtr))
    Z=sc.transform(imp.transform(Xtr))
    md={k:LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,max_iter=2000).fit(Z,(ytr>=k).astype(int)) for k in range(1,6)}
    return imp,sc,md
def P_(imp,sc,md,X):
    Z=sc.transform(imp.transform(X));g={k:md[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((len(X),6));P[:,0]=1-g[1]
    for k in range(1,5):P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5];P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True);return P
def tune(Ptr,ds_tr,tgt):
    grid=np.round(np.arange(0.10,0.55,0.05),3);best=None
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

print("AD-1062 재구성...")
ex=R.build_from_raw(["AD-1062"]).dropna(subset=["ds_stage"]).reset_index(drop=True)
print("3시험 재구성...")
tr=R.build_from_raw(TRIALS)
bs=pd.read_csv(DL/"baseline_sample.csv")
bs=bs[bs["in_common_comparison_sample"]==True][["STUDYID","USUBJID","ds_stage","A1_2015_stage","A0_harmonized"]]
tr=tr.drop(columns=["ds_stage"]).merge(bs,on=["STUDYID","USUBJID"],how="inner").dropna(subset=["ds_stage"]).reset_index(drop=True)
A1_MAE=mae(tr["ds_stage"],tr["A1_2015_stage"]); A0_MISS=miss(tr["ds_stage"],tr["A0_harmonized"])
rows={t:tr.index[tr.STUDYID==t] for t in TRIALS}
dsx=ex["ds_stage"].values; Xex=ex[ITEMS].values

rng=np.random.default_rng(20260814)
combos=set()
while len(combos)<130:
    k=int(rng.integers(1,9)); add=tuple(sorted(rng.choice(len(POOL),k,replace=False)))
    combos.add(add)
combos=[FIXED+[POOL[i] for i in c] for c in combos]

res=[]
for ci,cols in enumerate(combos):
    # 내부 LOTO
    pred=np.zeros(len(tr))
    for ho in TRIALS:
        tri=tr.index[tr.STUDYID!=ho]; tei=rows[ho]
        imp,sc,md=fit(tr.loc[tri,cols].values,tr.loc[tri,"ds_stage"].astype(int).values)
        Ptr=P_(imp,sc,md,tr.loc[tri,cols].values);Pte=P_(imp,sc,md,tr.loc[tei,cols].values)
        tgt=miss(tr.loc[tri,"ds_stage"].values,tr.loc[tri,"A0_harmonized"].values)
        t4,t5=tune(Ptr,tr.loc[tri,"ds_stage"].values,tgt)
        pred[tei]=asym(Pte,t4,t5)
    in_mae=mae(tr["ds_stage"],pred); in_miss=miss(tr["ds_stage"],pred)*100
    score=0.5*(in_mae/A1_MAE)+0.5*((in_miss/100)/A0_MISS)
    # 외부 AD-1062
    imp,sc,md=fit(tr[cols].values,tr["ds_stage"].astype(int).values)
    Ptr=P_(imp,sc,md,tr[cols].values);Pex=P_(imp,sc,md,ex[cols].values)
    t4,t5=tune(Ptr,tr["ds_stage"].values,miss(tr["ds_stage"],tr["A0_harmonized"].values))
    pex=asym(Pex,t4,t5)
    ex_mae=mae(dsx,pex); ex_miss=miss(dsx,pex)*100
    res.append(dict(k=len(cols),in_mae=in_mae,in_miss=in_miss,score=score,ex_mae=ex_mae,ex_miss=ex_miss))
    if ci%20==0: print(f"{ci}/{len(combos)}")

df=pd.DataFrame(res); df["in_rank"]=df["score"].rank()
df.to_csv(HERE/"b2_overfit_scatter.csv",index=False,encoding="utf-8-sig")
r1,p1=spearmanr(df["score"],df["ex_miss"])
r2,p2=spearmanr(df["in_miss"],df["ex_miss"])
print(f"내부종합점수 vs 외부중증놓침 Spearman r={r1:.3f} (p={p1:.3g})")
print(f"내부중증놓침 vs 외부중증놓침 Spearman r={r2:.3f} (p={p2:.3g})")

NAVY="#1b2a4a"; RED="#c0392b"
fig,ax=plt.subplots(figsize=(7.4,5.2))
ax.scatter(df["in_rank"],df["ex_miss"],s=42,color=NAVY,alpha=.65,edgecolor="white",lw=.5)
# 추세선
z=np.polyfit(df["in_rank"],df["ex_miss"],1);xx=np.array([df["in_rank"].min(),df["in_rank"].max()])
ax.plot(xx,z[0]*xx+z[1],color=RED,lw=2,ls="--",label=f"추세 (Spearman r={r1:+.2f})")
ax.set_xlabel("내부 성능 순위 (1=내부에서 최고 조합)")
ax.set_ylabel("AD-1062 외부 중증놓침 (%)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(alpha=.25)
ax.set_title("B2 · 내부 순위는 외부 성능을 예측하지 못함 (전수탐색 과적합)",fontsize=12.5,fontweight="bold",color=NAVY,pad=12)
ax.legend(frameon=False,fontsize=11)
ax.text(0.02,0.02,f"조합 {len(df)}개 · 상관 거의 0 → 내부 1위가 외부 1위 아님",transform=ax.transAxes,fontsize=9,color="#555")
plt.tight_layout(); plt.savefig(HERE/"fig_b2_overfit_scatter.png",dpi=150,facecolor="white"); plt.close()
print(">>> 저장: fig_b2_overfit_scatter.png")
