"""
Aim 2 · ④ B2 문항축약 정식 (표준화 elastic-net PPOM = Gn) (sjlee)
==================================================================
확정 Gn(부분비례오즈 elastic-net + 비대칭임계 + 마진) 위에서 문항을 줄여도
성능이 유지되는지 정직하게(누출 0) 확인하고 축약 채점표를 낸다.

문항선택 규칙(사전고정): 절단(1..5) 전부에서 계수 0이면 '제거', 하나라도 0 아니면 '유지'.
두 경로: (a) λ상향(벌점↑→자연 사멸), (b) 안정성선택 top-K(fold내 |계수|합 순위→재적합).
평가: leave-one-trial-out 3-fold. 표준화·선택·λ·τ 전부 fold 훈련 내부에서만(누출0).

2겹 허용기준:
  (내부) 전체문항 대비 MAE 증가 ≤0.10 + 2↑·(4·5→≤3) 각 증가 ≤2%p.
  (외부, 진짜 관문) A1 대비 4조건 유지(MAE감소≥0.10 / 2↑증가≤2%p / 중증증가≤2%p / MAE감소 ≥2/3시험).

주의(승자의 저주): 전수조사 4핵심(외출·목욕·화장실·몸단장)은 전체데이터서 나온 것.
여기선 그 앵커에 의존하지 않고 fold내부 순위로 선택 → 낙관 편향 제거.

산출: b2_item_selection_report.md, aim2_sjlee/b2_reduced_scoretable.csv,
      aim2_sjlee/b2_selection_frequency.csv
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_gn_elasticnet as G

warnings.filterwarnings("ignore")
OUTDIR = Path(__file__).parent
CSV = OUTDIR / "aim2_sjlee"; CSV.mkdir(exist_ok=True)
REPORT = OUTDIR / "b2_item_selection_report.md"
TRIALS = G.TRIALS; ITEMS = G.ITEMS; STAGES = np.arange(6)
L1, MARGIN = 0.5, 0.14
C_MAIN = 0.1
KS = [21, 15, 12, 10, 8, 7, 6, 5]
LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b입기","ADL0106A":"Q6a고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

def mae(t, p): return np.abs(np.asarray(t, float) - np.asarray(p, float)).mean()
def miss(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan
def u2(t, p): return (np.asarray(t, float) - np.asarray(p, float) >= 2).mean()
def kap(t, p): return cohen_kappa_score(np.asarray(t, int), np.asarray(p, int), weights="quadratic", labels=list(range(6)))
def cmed(P): return STAGES[(np.cumsum(P, 1) >= 0.5).argmax(1)]
def asym(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cmed(P)))

def fit_items(df, items, C):
    imp = SimpleImputer(strategy="median").fit(df[items])
    sc = StandardScaler().fit(imp.transform(df[items]))
    Z = sc.transform(imp.transform(df[items])); y = df["ds_stage"].astype(int).values
    md = {k: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=L1, C=C,
          max_iter=3000, tol=1e-3, random_state=0).fit(Z, (y >= k).astype(int)) for k in range(1, 6)}
    return imp, sc, md
def P_items(imp, sc, md, df, items):
    Z = sc.transform(imp.transform(df[items]))
    g = {k: md[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    P = np.zeros((len(df), 6)); P[:, 0] = 1 - g[1]
    for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
    P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
def imp_scores(md, items):        # 문항별 절단간 |표준화계수| 합
    s = np.zeros(len(items))
    for k in range(1, 6): s += np.abs(md[k].coef_[0])
    return s
def n_alive(md):                  # 하나라도 계수!=0인 문항 수
    S = np.zeros(md[1].coef_[0].shape)
    for k in range(1, 6): S += np.abs(md[k].coef_[0])
    return int((S > 1e-8).sum())
def tune_tau(Ptr, ds_tr, a0_target):
    """중증놓침 우선: 중증놓침 ≤ A0목표를 만족하는 τ 중 MAE 최소.
    (제약=중증놓침, 목적=MAE. 목표 도달 불가 시 중증놓침 최소.)"""
    grid = np.round(np.arange(0.10, 0.55, 0.025), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym(Ptr, t4, t5)
            if miss(ds_tr, p) <= a0_target + 1e-9:
                key = mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    if best is None:
        cand = [(miss(ds_tr, asym(Ptr, a, b)), mae(ds_tr, asym(Ptr, a, b)), a, b)
                for a in grid for b in grid if b >= a]
        _, _, a, b = min(cand); return a, b
    return best[1], best[2]

def four_conditions(m, pred):
    """A1 대비 4조건(추가가치). pred: np array 전체표본."""
    dfp = pd.Series(pred, index=m.index)
    def g(t):
        idx = m.STUDYID == t
        return m.loc[idx, "ds_stage"].values, dfp[idx].values, m.loc[idx, "A1_2015_stage"].values
    bm = {t: mae(g(t)[0], g(t)[1]) for t in TRIALS}; am = {t: mae(g(t)[0], g(t)[2]) for t in TRIALS}
    bh = {t: miss(g(t)[0], g(t)[1]) for t in TRIALS}; ah = {t: miss(g(t)[0], g(t)[2]) for t in TRIALS}
    bu = {t: u2(g(t)[0], g(t)[1]) for t in TRIALS}; au = {t: u2(g(t)[0], g(t)[2]) for t in TRIALS}
    mean = lambda d: np.mean([d[t] for t in TRIALS])
    mae_drop = mean(am) - mean(bm); d_hi = mean(bh) - mean(ah); d_u2 = mean(bu) - mean(au)
    nb = sum(bm[t] < am[t] for t in TRIALS)
    c1, c2, c3, c4 = mae_drop >= 0.10, d_u2 <= 0.02, d_hi <= 0.02, nb >= 2
    return dict(mae_drop=mae_drop, d_hi=d_hi, d_u2=d_u2, nb=nb,
                c1=c1, c2=c2, c3=c3, c4=c4, allpass=c1 and c2 and c3 and c4)

def curve_stability(m):
    """안정성선택 경로: fold내 |계수|순위 top-K 재적합 → 문항수별 정직 성능 + 선택빈도."""
    ds = m["ds_stage"].values
    preds = {K: np.zeros(len(m)) for K in KS}
    freq = {K: {} for K in KS}
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        _, _, md_full = fit_items(tr, ITEMS, C_MAIN)
        order = np.argsort(-imp_scores(md_full, ITEMS))
        for K in KS:
            sel = [ITEMS[i] for i in order[:K]]
            freq[K][ho] = sel
            imp, sc, md = fit_items(tr, sel, C_MAIN)
            Ptr = P_items(imp, sc, md, tr, sel)
            a0t = miss(tr["ds_stage"], tr["A0_harmonized"])   # 중증놓침 목표=훈련 A0
            t4, t5 = tune_tau(Ptr, tr["ds_stage"].values, a0t)
            Pte = P_items(imp, sc, md, te, sel)
            preds[K][te.index] = asym(Pte, t4, t5)
    return ds, preds, freq

def main():
    log("# Aim 2 · ④ B2 문항축약 (표준화 elastic-net PPOM=Gn, 누출0) (sjlee)\n")
    m = G.build_matrix().reset_index(drop=True)
    # A0_harmonized 병합(중증놓침 목표용) — baseline_sample.csv
    bs = pd.read_csv(Path.home() / "Downloads" / "baseline_sample.csv")
    bs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "A0_harmonized"]]
    m = m.merge(bs, on=["STUDYID", "USUBJID"], how="left").reset_index(drop=True)
    ds = m["ds_stage"].values
    a1_mae = mae(ds, m["A1_2015_stage"].values); a1_miss = miss(ds, m["A1_2015_stage"].values) * 100
    a0_miss = miss(ds, m["A0_harmonized"].values) * 100
    log(f"공통표본 {len(m)} × 21문항. 기준 A1 MAE {a1_mae:.3f}/중증 {a1_miss:.1f}%, A0 중증 {a0_miss:.1f}%. "
        f"모형=Gn(elastic-net l1={L1}, C={C_MAIN}), LOTO 3-fold.\n")
    log("**τ 튜닝(수정)**: 중증놓침을 **훈련 A0 수준으로 고정**하고 그 안에서 **MAE 최소화** "
        "(제약=중증놓침 우선, 목적=MAE). ← 이전판은 제약/목적이 뒤바뀌어 있어 수정함.\n")
    log("**문항선택 규칙**: 절단 전부 계수0=제거, 하나라도≠0=유지(문항단위 drop). "
        "표준화·선택·τ 전부 fold 훈련 내부(누출0).\n")

    # 경로 (b) 안정성선택 --------------------------------------------------------
    ds, preds, freq = curve_stability(m)
    full = preds[21]
    base_mae, base_u2, base_hi = mae(ds, full), u2(ds, full) * 100, miss(ds, full) * 100
    log("## 경로(b) 안정성선택 top-K — 문항수별 성능 + 정렬된 2겹 관문")
    log("판정(중증우선 튜닝에 정렬): **내부**(전체대비 MAE↑≤0.10·중증↑≤2%p·2↑↑≤2%p) + "
        "**외부 지배**(A1을 MAE·중증 둘 다 이김) + **안전**(중증 ≤ A0+2%p).")
    log("| 문항수 | MAE | 카파 | 4·5→≤3 | 2↑과소 | 내부 | A1지배 | 안전(≤A0+2) | 최종 |")
    log("|---|---|---|---|---|---|---|---|---|")
    passK = []
    for K in KS:
        p = preds[K]
        mk, kk, hk, uk = mae(ds, p), kap(ds, p), miss(ds, p) * 100, u2(ds, p) * 100
        internal = (mk - base_mae <= 0.10) and (uk - base_u2 <= 2.0) and (hk - base_hi <= 2.0)
        fc = four_conditions(m, p)
        dominate = (mk < a1_mae) and (hk < a1_miss) and (fc["nb"] >= 2)   # A1 두 지표 지배
        safe = hk <= a0_miss + 2.0                                        # 중증 A0 수준 유지
        both = internal and dominate and safe
        if both: passK.append(K)
        log(f"| {K} | {mk:.3f} | {kk:.3f} | {hk:.1f}% | {uk:.1f}% | "
            f"{'✅' if internal else '❌'} | {'✅' if dominate else '❌'} | "
            f"{'✅' if safe else '❌'} | {'★' if both else ''} |")
    kmin = min(passK) if passK else None
    log(f"\n- **정렬된 3겹 동시 통과 최소 문항수: {kmin if kmin else '없음'}** "
        f"(내부보존 + A1 양지표 지배 + 중증 A0수준).")
    log("- 참고: 발표가 쓴 'MAE 감소≥0.10' 4조건은 *MAE우선* 튜닝 기준이라 중증우선판과 불일치 → "
        "여기선 지배+안전 기준으로 정렬.\n")

    # 경로 (a) λ상향 -------------------------------------------------------------
    log("## 경로(a) λ상향 — 벌점↑에 따른 생존 문항수·성능")
    log("| C(λ) | 평균 생존문항 | MAE | 4·5→≤3 | A1-4조건 |")
    log("|---|---|---|---|---|")
    for C in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
        pr = np.zeros(len(m)); alive = []
        for ho in TRIALS:
            tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
            imp, sc, md = fit_items(tr, ITEMS, C); alive.append(n_alive(md))
            Ptr = P_items(imp, sc, md, tr, ITEMS)
            t4, t5 = tune_tau(Ptr, tr["ds_stage"].values, miss(tr["ds_stage"], tr["A0_harmonized"]))
            pr[te.index] = asym(P_items(imp, sc, md, te, ITEMS), t4, t5)
        fc = four_conditions(m, pr)
        log(f"| {C} | {np.mean(alive):.1f} | {mae(ds,pr):.3f} | {miss(ds,pr)*100:.1f}% | {'✅' if fc['allpass'] else '❌'} |")
    log("\n- λ만으로는 문항이 잘 안 죽음(elastic-net L2성분). 축약 주경로는 안정성선택.\n")

    # 선택빈도표 -----------------------------------------------------------------
    log("## 선택빈도(안정성) — 각 K에서 fold 3개 공통 문항")
    freq_rows = []
    for K in [10, 8, 6]:
        sets = [set(freq[K][ho]) for ho in TRIALS]
        common = set.intersection(*sets)
        cnt = {}
        for s in sets:
            for it in s: cnt[it] = cnt.get(it, 0) + 1
        log(f"### {K}문항 — 3/3 공통: {sorted(LAB[i] for i in common)}")
        for it, c in sorted(cnt.items(), key=lambda x: -x[1]):
            if c >= 2: freq_rows.append({"K": K, "item": it, "label": LAB[it], "folds": f"{c}/3"})
    pd.DataFrame(freq_rows).to_csv(CSV / "b2_selection_frequency.csv", index=False, encoding="utf-8-sig")

    # 최종 축약 채점표 (전체자료 재적합, 원눈금) ----------------------------------
    Kfinal = kmin if kmin else 10
    _, _, md_all = fit_items(m, ITEMS, C_MAIN)
    order = np.argsort(-imp_scores(md_all, ITEMS))
    sel = [ITEMS[i] for i in order[:Kfinal]]
    imp = SimpleImputer(strategy="median").fit(m[sel]); Xi = imp.transform(m[sel])
    sc = StandardScaler().fit(Xi); sd = sc.scale_; mu = sc.mean_
    Z = sc.transform(Xi); y = m["ds_stage"].astype(int).values
    rows = []
    for k in range(1, 6):
        lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=L1, C=C_MAIN,
                                max_iter=5000, tol=1e-3, random_state=0).fit(Z, (y >= k).astype(int))
        w = lr.coef_[0] / sd; b0 = lr.intercept_[0] - np.sum(lr.coef_[0] * mu / sd)
        for i, it in enumerate(sel):
            rows.append({"cut": f"P(Y>={k})", "item": it, "label": LAB[it], "coef_raw": round(w[i], 4)})
        rows.append({"cut": f"P(Y>={k})", "item": "(절편)", "label": "", "coef_raw": round(b0, 4)})
    pd.DataFrame(rows).to_csv(CSV / "b2_reduced_scoretable.csv", index=False, encoding="utf-8-sig")
    log(f"## 최종 축약 채점표: **{Kfinal}문항** (전체자료 재적합·원눈금)")
    log(f"- 선택문항: {', '.join(LAB[i] for i in sel)}")
    log(f"- 계수표 CSV: b2_reduced_scoretable.csv (절단별 원눈금 coef+절편). "
        f"임계 τ는 배포모형(gn_final_hyperparams.csv) 적용.")
    log("\n## 한계")
    log("- 이건 **독립검증된 단축형이 아니라**, 본 자료에서 성능을 유지하며 줄인 후보. 외부검증 전 낙관적.")
    log("- 4핵심(외출·목욕·화장실·몸단장)은 전수조사(전체데이터) 산물 → 여기선 앵커로 강제 안 하고 "
        "fold내부 순위로만 선택(승자의 저주 회피). 그럼에도 상위에 일관 등장하면 신뢰도 높음.\n")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__ == "__main__":
    main()
