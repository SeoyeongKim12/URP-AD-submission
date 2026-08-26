"""
Aim 2 · B2 문항 축약 v2 — 1단위 곡선 + 후진제거 + 무작위조합 (sjlee)
====================================================================
팀 피드백 반영:
(1) 문항 수를 1단위(21→4)로 변화 관찰.
(2) '계수 작은 순' 한 조합만 보지 말고, 삭제 조합을 여러 개 테스트:
    - 후진제거(성능 기준 탐욕적 경로) — 계수순위와 다른 경로
    - 무작위 조합 샘플링 — 조합 바꾸면 얼마나 흔들리나 + 우리 선택이 최선 근처인가

모형/전제: b1_dominate_a0의 부분PO 엘라스틱넷 + A0-목표 임계, leave-one-trial-out.
- 곡선/무작위: 고정 문항집합의 CV 평가 = 누출 없음.
- 후진제거: CV 성능으로 경로 선택 → 낙관적(탐색용, 라벨 명시).

산출: aim2/시행착오/b2_item_reduction_v2_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b2_item_reduction_v2_report.md"
ITEMS = D.ITEMS; TRIALS = D.TRIALS; NI = len(ITEMS)
LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b옷입기","ADL0106A":"Q6a옷고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자있기","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
_lines=[]
def log(s=""):
    print(s.encode("ascii","replace").decode("ascii")); _lines.append(s)

def fit_cuts(Ztr, y):
    return {k: LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, max_iter=2000, tol=1e-3,
                                  random_state=0).fit(Ztr,(y>=k).astype(int)) for k in range(1,6)}
def probs(models, Z):
    g={k:models[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((Z.shape[0],6)); P[:,0]=1-g[1]
    for k in range(1,5): P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5]; P=np.clip(P,1e-9,None); P/=P.sum(1,keepdims=True); return P
def tune_tau(Ptr, ds_tr, a0m):
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

def prep(m):
    folds={}
    for ho in TRIALS:
        tri=np.array(m.index[m.STUDYID!=ho]); tei=np.array(m.index[m.STUDYID==ho])
        imp=SimpleImputer(strategy="median").fit(m.loc[tri,ITEMS])
        Xtr=imp.transform(m.loc[tri,ITEMS]); Xte=imp.transform(m.loc[tei,ITEMS])
        y=m.loc[tri,"ds_stage"].astype(int).values; ds_tr=m.loc[tri,"ds_stage"].values
        a0m=D.miss(ds_tr, m.loc[tri,"A0_harmonized"].values)
        sc0=StandardScaler().fit(Xtr); mdl0=fit_cuts(sc0.transform(Xtr),y)
        imp_ord=np.argsort(-sum(np.abs(mdl0[k].coef_[0]) for k in range(1,6)))
        folds[ho]=dict(tei=tei,Xtr=Xtr,Xte=Xte,y=y,ds_tr=ds_tr,a0m=a0m,order=imp_ord)
    return folds

def eval_subset(m, folds, keep):
    t=m["ds_stage"].values; pred=np.zeros(len(m))
    for ho in TRIALS:
        f=folds[ho]
        sc=StandardScaler().fit(f["Xtr"][:,keep])
        mdl=fit_cuts(sc.transform(f["Xtr"][:,keep]), f["y"])
        Ptr=probs(mdl, sc.transform(f["Xtr"][:,keep])); Pte=probs(mdl, sc.transform(f["Xte"][:,keep]))
        t4,t5=tune_tau(Ptr, f["ds_tr"], f["a0m"])
        pred[f["tei"]]=D.asym(Pte,t4,t5)
    return D.mae(t,pred), D.miss(t,pred)*100, D.kap(t,pred)

def curve_point(m, folds, k):  # 계수순위, per-fold(누출0)
    t=m["ds_stage"].values; pred=np.zeros(len(m))
    for ho in TRIALS:
        f=folds[ho]; keep=sorted(f["order"][:k].tolist())
        sc=StandardScaler().fit(f["Xtr"][:,keep])
        mdl=fit_cuts(sc.transform(f["Xtr"][:,keep]), f["y"])
        Ptr=probs(mdl, sc.transform(f["Xtr"][:,keep])); Pte=probs(mdl, sc.transform(f["Xte"][:,keep]))
        t4,t5=tune_tau(Ptr, f["ds_tr"], f["a0m"])
        pred[f["tei"]]=D.asym(Pte,t4,t5)
    return D.mae(t,pred), D.miss(t,pred)*100, D.kap(t,pred)


def main():
    log("# Aim 2 · B2 문항축약 v2 — 1단위·후진제거·무작위조합 (sjlee)\n")
    m=D.build(); a1_mae=D.mae(m.ds_stage.values, m.A1_2015_stage.values)
    a0_miss=D.miss(m.ds_stage.values, m.A0_harmonized.values)*100
    folds=prep(m)
    log(f"공통표본 {len(m)}. 기준 A1 MAE {a1_mae:.3f} / A0 중증 {a0_miss:.1f}%. 문항 {NI}개.\n")

    # 1) 1단위 곡선 (계수순위, 누출0)
    log("## (1) 1문항 단위 곡선 — 계수순위 선택 (누출 없음)")
    log("| 문항수 | MAE | 중증놓침 | 카파 | A1대비MAE(앞섬) | 둘다이김 |")
    log("|---|---|---|---|---|---|")
    curve={}
    for k in range(NI,3,-1):
        mae,miss,kap=curve_point(m,folds,k); curve[k]=(mae,miss,kap)
        champ="✅" if (mae<a1_mae and miss<=a0_miss+1.0) else "❌"
        log(f"| {k} | {mae:.3f} | {miss:.1f}% | {kap:.3f} | {a1_mae-mae:+.3f} | {champ} |")
    log("")

    # 2) 후진제거 (성능 기준, 낙관적 탐색)
    log("## (2) 후진 제거 경로 — 성능 기준 탐욕적 (탐색·낙관적)")
    log("- 매 단계 '빼도 MAE 제일 안 나빠지는' 문항 제거. 계수순위와 다른 경로.")
    keep=list(range(NI)); bwd={}
    mae,miss,kap=eval_subset(m,folds,sorted(keep)); bwd[NI]=(sorted(keep),mae,miss,kap)
    log("| 문항수 | 제거된 문항 | MAE | 중증놓침 | (계수순위 MAE) |")
    log("|---|---|---|---|---|")
    while len(keep)>4:
        best=None
        for i in keep:
            cand=[j for j in keep if j!=i]
            mm=eval_subset(m,folds,sorted(cand))[0]
            if best is None or mm<best[0]: best=(mm,i)
        keep=[j for j in keep if j!=best[1]]
        mae,miss,kap=eval_subset(m,folds,sorted(keep)); bwd[len(keep)]=(sorted(keep),mae,miss,kap)
        log(f"| {len(keep)} | {LAB[ITEMS[best[1]]]} | {mae:.3f} | {miss:.1f}% | ({curve[len(keep)][0]:.3f}) |")
    log("")

    # 3) 무작위 조합 (고정 부분집합, 누출0) at k=12,10,8,6
    log("## (3) 무작위 조합 샘플링 — 조합 민감도 (k=12/10/8/6, 각 150개)")
    log("| 문항수 | 최고MAE | 중앙MAE | 최저MAE | 계수순위(백분위) | 후진제거(백분위) |")
    log("|---|---|---|---|---|---|")
    rng=np.random.default_rng(20260805)
    gorder=folds[TRIALS[0]]["order"]  # 참고용 전역 계수순위(첫 fold)
    for k in [12,10,8,6]:
        maes=[]
        for _ in range(150):
            keep=sorted(rng.choice(NI,k,replace=False).tolist())
            maes.append(eval_subset(m,folds,keep)[0])
        maes=np.array(maes)
        coef_mae=curve[k][0]; bwd_mae=bwd[k][1]
        pc=(maes<coef_mae).mean()*100; pb=(maes<bwd_mae).mean()*100
        log(f"| {k} | {maes.min():.3f} | {np.median(maes):.3f} | {maes.max():.3f} | "
            f"{coef_mae:.3f}({pc:.0f}%) | {bwd_mae:.3f}({pb:.0f}%) |")
    log("\n- 백분위 = 무작위 조합 중 그보다 나은(MAE 낮은) 비율. 낮을수록 우리 선택이 상위(좋음).")
    log("- 최저MAE(최고 조합)가 계수순위·후진제거보다 훨씬 낮으면 → 더 나은 조합 존재 = 탐색 여지.\n")

    # 4) 대표 문항수의 후진제거 문항군
    log("## (4) 후진제거 선택 문항군 (참고)")
    for k in [10,8,6]:
        names=[LAB[ITEMS[i]] for i in bwd[k][0]]
        log(f"- {k}문항: {', '.join(names)}")
    log("")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__=="__main__":
    main()
