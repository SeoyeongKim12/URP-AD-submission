"""
Aim 2 · B2 중첩CV 랜덤서치 (정직판) (sjlee)
============================================
팀 제안: 각 문항수(5~12)에서 무작위 200조합을 중첩CV로 정직하게 평가해
'몇 문항이면 충분한가'를 낙관 없이 확인.

구조(누출 없음):
- 바깥: leave-one-trial-out(3). 평가시험은 선택에 절대 미사용.
- 안쪽: 남은 2시험 중 leave-one-inner-trial(2). 여기서 200조합 중 최고 선택.
- 선택된 조합을 바깥 훈련(2시험)으로 재학습 → 바깥 평가시험 예측.
→ 각 k의 정직한 성능 = 바깥 예측 취합.

주의: 시험 3개라 안쪽 선택 노이즈 큼 → 조합최적화 이득이 작게/불안정하게 나올 수 있음(정직 결과).
결측 대치는 전역 중앙값 1회(결측 71칸/46,263, 무시 수준 → 누출 미미).

산출: aim2/시행착오/b2_nested_randomsearch_report.md
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b2_nested_randomsearch_report.md"
TRIALS = D.TRIALS; ITEMS = D.ITEMS; NI = len(ITEMS)
K_RANGE = list(range(5, 13))     # 5..12
N_RANDOM = 200
LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b옷입기","ADL0106A":"Q6a옷고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자있기","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
_lines=[]
def log(s=""):
    print(s.encode("ascii","replace").decode("ascii")); _lines.append(s)

def fit_cuts(Ztr,y):
    return {k:LogisticRegression(solver="saga",l1_ratio=0.5,C=0.1,max_iter=1500,tol=1e-3,
            random_state=0).fit(Ztr,(y>=k).astype(int)) for k in range(1,6)}
def probs(mdl,Z):
    g={k:mdl[k].predict_proba(Z)[:,1] for k in range(1,6)}
    P=np.zeros((Z.shape[0],6)); P[:,0]=1-g[1]
    for k in range(1,5): P[:,k]=g[k]-g[k+1]
    P[:,5]=g[5]; P=np.clip(P,1e-9,None); P/=P.sum(1,keepdims=True); return P
def tune_tau(Ptr,ds_tr,a0m):
    grid=np.round(np.arange(0.05,0.55,0.03),3); best=None
    for t4 in grid:
        for t5 in grid:
            if t5<t4: continue
            p=D.asym(Ptr,t4,t5)
            if D.miss(ds_tr,p)<=a0m+1e-9:
                key=D.mae(ds_tr,p)
                if best is None or key<best[0]: best=(key,t4,t5)
    if best is None: return 0.3,0.5
    return best[1],best[2]

def predict(X, tr, te, cols, y, ds, a0):
    sc=StandardScaler().fit(X[np.ix_(tr,cols)])
    Ztr=sc.transform(X[np.ix_(tr,cols)]); Zte=sc.transform(X[np.ix_(te,cols)])
    mdl=fit_cuts(Ztr, y[tr])
    Ptr=probs(mdl,Ztr); Pte=probs(mdl,Zte)
    t4,t5=tune_tau(Ptr, ds[tr], D.miss(ds[tr], a0[tr]))
    return D.asym(Pte,t4,t5)

def main():
    log("# Aim 2 · B2 중첩CV 랜덤서치 (정직판) (sjlee)\n")
    m=D.build()
    X=SimpleImputer(strategy="median").fit_transform(m[ITEMS])   # 전역 대치 1회
    trial=m["STUDYID"].values; y=m["ds_stage"].astype(int).values
    ds=m["ds_stage"].values; a0=m["A0_harmonized"].values
    a1_mae=D.mae(ds, m["A1_2015_stage"].values); a0_miss=D.miss(ds, a0)*100
    rows={t:np.where(trial==t)[0] for t in TRIALS}
    rng=np.random.default_rng(20260805)
    log(f"공통표본 {len(m)}. 기준 A1 MAE {a1_mae:.3f}/A0 중증 {a0_miss:.1f}%. "
        f"k={K_RANGE[0]}~{K_RANGE[-1]}, 각 {N_RANDOM}조합, 중첩CV.\n")

    outer_pred={k:np.full(len(m),np.nan) for k in K_RANGE}
    sel={k:{} for k in K_RANGE}
    for ho in TRIALS:
        otr=np.concatenate([rows[t] for t in TRIALS if t!=ho]); ote=rows[ho]
        inner=[t for t in TRIALS if t!=ho]
        for k in K_RANGE:
            best=(np.inf,None)
            for _ in range(N_RANDOM):
                cols=sorted(rng.choice(NI,k,replace=False).tolist())
                im=[]
                for iv in inner:
                    it=[t for t in inner if t!=iv][0]
                    pv=predict(X, rows[it], rows[iv], cols, y, ds, a0)
                    im.append(D.mae(ds[rows[iv]], pv))
                mi=np.mean(im)
                if mi<best[0]: best=(mi, cols)
            cols=best[1]; sel[k][ho]=cols
            outer_pred[k][ote]=predict(X, otr, ote, cols, y, ds, a0)
        log(f"[진행] outer={ho} 완료")

    # 정직판 곡선 + 참고: 계수순위 곡선(v2)과 비교는 별도
    log("\n## 중첩CV 랜덤서치 정직 성능 (문항수별)")
    log("| 문항수 | MAE | 중증놓침 | 카파 | A1대비MAE | 둘다이김 |")
    log("|---|---|---|---|---|---|")
    for k in K_RANGE:
        p=outer_pred[k]; mae=D.mae(ds,p); miss=D.miss(ds,p)*100; kap=D.kap(ds,p)
        champ="✅" if (mae<a1_mae and miss<=a0_miss+1.0) else "❌"
        log(f"| {k} | {mae:.3f} | {miss:.1f}% | {kap:.3f} | {a1_mae-mae:+.3f} | {champ} |")

    log("\n## fold별 선택 조합 (안쪽 선택, 참고)")
    for k in [10,8,6]:
        log(f"### {k}문항")
        for ho in TRIALS:
            names=[LAB[ITEMS[i]] for i in sel[k][ho]]
            log(f"- outer={ho}: {', '.join(names)}")
        common=set(sel[k][TRIALS[0]])
        for ho in TRIALS[1:]: common&=set(sel[k][ho])
        log(f"- **3fold 공통: {[LAB[ITEMS[i]] for i in sorted(common)]}**")
    log("\n- 정직판이 v2(계수순위/후진제거)보다 낮으면 = 낙관 제거된 진짜 성능. "
        "fold별 선택이 많이 다르면 = 소표본서 조합 최적화 불안정(정직 결론).")
    REPORT.write_text("\n".join(_lines),encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__=="__main__":
    main()
