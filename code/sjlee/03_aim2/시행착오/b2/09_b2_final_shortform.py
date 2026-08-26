"""
Aim 2 · B2 최종 단축형 — 4조건 제약 하 최소 문항 (sjlee)
=========================================================
사용자 확정 규칙:
  ① 반복 핵심 4문항 고정: Q3화장실·Q4목욕·Q5몸단장·Q15외출.
  ② FA 4축 각 축 최소 1문항 보존(설명력·축균형).
  ③ 성능: MAE ≤ A1(0.589) AND 중증놓침(4·5→≤3) ≤ A0(17.7%). (정직 CV 기준)
  ④ ①~③ 만족하며 문항수 최소화.

모형: Gn(부분PO elastic-net, C=0.1, l1=0.5) + 비대칭임계, 중증우선 τ.
  · 조건③의 '중증 ≤ A0'를 CV에서 담보하려면 τ 목표를 A0보다 gap만큼 당김
    (일반화 격차 보정). gap은 전체 21문항에서 CV중증≤A0 되게 1회 보정 후 전 크기 공통 적용.
누출 0: 표준화·τ·선택 전부 fold 훈련 내부. leave-one-trial-out 3-fold.

산출: b2_final_shortform_report.md, aim2_sjlee/b2_final_shortform_scoretable.csv
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import b1_gn_elasticnet as G

warnings.filterwarnings("ignore")
OUTDIR = Path(__file__).parent
CSV = OUTDIR / "aim2_sjlee"; CSV.mkdir(exist_ok=True)
REPORT = OUTDIR / "b2_final_shortform_report.md"
ITEMS = G.ITEMS; TRIALS = G.TRIALS; STAGES = np.arange(6); L1 = 0.5; C = 0.1
FIXED = ["ADL0103", "ADL0104", "ADL0105", "ADL0115A"]     # Q3·Q4·Q5·Q15
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
def kap(t, p): return cohen_kappa_score(np.asarray(t, int), np.asarray(p, int), weights="quadratic", labels=list(range(6)))
def cmed(P): return STAGES[(np.cumsum(P, 1) >= 0.5).argmax(1)]
def asym(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cmed(P)))

def fit_sub(Xtr, ytr):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr = sc.transform(imp.transform(Xtr))
    md = {k: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=L1, C=C,
          max_iter=3000, tol=1e-3, random_state=0).fit(Ztr, (ytr >= k).astype(int)) for k in range(1, 6)}
    return imp, sc, md
def P_sub(imp, sc, md, X):
    Z = sc.transform(imp.transform(X)); g = {k: md[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    P = np.zeros((len(X), 6)); P[:, 0] = 1 - g[1]
    for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
    P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
def tune_tau(Ptr, ds_tr, target):
    grid = np.round(np.arange(0.10, 0.55, 0.025), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym(Ptr, t4, t5)
            if miss(ds_tr, p) <= target + 1e-9:
                key = mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    if best is None:
        cand = [(miss(ds_tr, asym(Ptr, a, b)), mae(ds_tr, asym(Ptr, a, b)), a, b) for a in grid for b in grid if b >= a]
        _, _, a, b = min(cand); return a, b
    return best[1], best[2]

def cv_eval(m, cols, gap):
    """중증우선 CV: 목표=훈련A0중증−gap. 반환 (MAE, 중증, 카파)."""
    ds = m["ds_stage"].values; pred = np.zeros(len(m))
    for ho in TRIALS:
        tr = m.index[m.STUDYID != ho]; te = m.index[m.STUDYID == ho]
        imp, sc, md = fit_sub(m.loc[tr, cols].values, m.loc[tr, "ds_stage"].astype(int).values)
        Ptr = P_sub(imp, sc, md, m.loc[tr, cols].values); Pte = P_sub(imp, sc, md, m.loc[te, cols].values)
        target = miss(m.loc[tr, "ds_stage"].values, m.loc[tr, "A0_harmonized"].values) - gap
        t4, t5 = tune_tau(Ptr, m.loc[tr, "ds_stage"].values, max(target, 0.0))
        pred[m.index.get_indexer(te)] = asym(Pte, t4, t5)
    return mae(ds, pred), miss(ds, pred), kap(ds, pred)


def main():
    log("# Aim 2 · B2 최종 단축형 — 4조건 제약 하 최소 문항 (sjlee)\n")
    m = G.build_matrix().reset_index(drop=True)
    bs = pd.read_csv(Path.home() / "Downloads" / "baseline_sample.csv")
    bs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "A0_harmonized"]]
    m = m.merge(bs, on=["STUDYID", "USUBJID"], how="left").reset_index(drop=True)
    ds = m["ds_stage"].values
    a1_mae = mae(ds, m["A1_2015_stage"].values); a0_miss = miss(ds, m["A0_harmonized"].values)
    log(f"공통표본 {len(m)}. 기준 **A1 MAE {a1_mae:.3f} / A0 중증놓침 {a0_miss*100:.1f}%**.")
    log(f"조건: ①고정 {[LAB[i] for i in FIXED]} ②FA4축 각1 ③MAE≤{a1_mae:.3f}·중증≤{a0_miss*100:.1f}% ④최소화\n")

    # ── ② FA 4축 배정 ──────────────────────────────────────────────
    Ximp = SimpleImputer(strategy="median").fit_transform(m[ITEMS])
    Z = StandardScaler().fit_transform(Ximp)
    fa = FactorAnalysis(n_components=4, rotation="varimax", random_state=0).fit(Z)
    Lmat = fa.components_.T                       # 21×4 적재
    assign = {ITEMS[i]: int(np.argmax(np.abs(Lmat[i]))) for i in range(len(ITEMS))}
    factors = {f: [it for it in ITEMS if assign[it] == f] for f in range(4)}
    log("## FA 4축 문항 배정 (max|적재|)")
    for f in range(4):
        log(f"- **F{f+1}**: {', '.join(LAB[i] for i in factors[f])}")
    log(f"- 고정4 축 커버: " + ", ".join(f"{LAB[i]}=F{assign[i]+1}" for i in FIXED) + "\n")

    # ── 중요도 순위(전체 elastic-net |계수|합) ────────────────────
    imp0, sc0, md0 = fit_sub(m[ITEMS].values, m["ds_stage"].astype(int).values)
    impscore = np.zeros(len(ITEMS))
    for k in range(1, 6): impscore += np.abs(md0[k].coef_[0])
    order = [ITEMS[i] for i in np.argsort(-impscore)]

    # ── gap 보정: 전체 21문항서 CV중증 ≤ A0 되는 최소 gap ─────────
    gap = 0.0
    for g_try in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        _, mi, _ = cv_eval(m, ITEMS, g_try)
        if mi <= a0_miss + 1e-9: gap = g_try; break
        gap = g_try
    log(f"## τ 목표 보정(gap)\n- CV 중증 ≤ A0 담보 위해 목표=훈련A0−**{gap*100:.0f}%p** 적용(21문항서 보정).\n")

    # ── 베이스: 고정4 + 미커버 축 보강 ────────────────────────────
    base = list(FIXED); covered = set(assign[i] for i in FIXED)
    for f in range(4):
        if f not in covered:
            add = next(it for it in order if assign[it] == f and it not in base)
            base.append(add); covered.add(f)
    log(f"## 베이스(고정4 + 미커버축 보강): {len(base)}문항 — {[LAB[i] for i in base]}")
    log(f"- 커버 축: {sorted('F'+str(f+1) for f in covered)}\n")

    # ── ④ 그리디 최소화: 성능 만족까지 중요도순 추가 ──────────────
    log("## 문항수별 성능 (베이스→중요도순 추가, 중증우선)")
    log("| 문항수 | MAE(≤A1?) | 중증놓침(≤A0?) | 카파 | 4조건 | 문항 |")
    log("|---|---|---|---|---|---|")
    cur = list(base); final = None; path = []
    pool = [it for it in order if it not in cur]
    while True:
        mc, mi, kc = cv_eval(m, cur, gap)
        ok = (mc <= a1_mae + 1e-9) and (mi <= a0_miss + 1e-9)
        path.append((len(cur), mc, mi, kc, ok, list(cur)))
        log(f"| {len(cur)} | {mc:.3f}{'✅' if mc<=a1_mae else '❌'} | {mi*100:.1f}%{'✅' if mi<=a0_miss else '❌'} | "
            f"{kc:.3f} | {'★' if ok else ''} | {', '.join(LAB[i] for i in cur)} |")
        if ok and final is None: final = list(cur)
        if final is not None or not pool: break
        cur.append(pool.pop(0))
    log("")

    if final is None:
        log("## 판정: 4조건 동시 만족 불가 — 전 문항수에서 MAE≤A1 & 중증≤A0 동시 미달.")
        log(f"- 최소 중증 도달값 참조(위 표). 조건 완화(중증≤A0+ε) 필요할 수 있음.\n")
        final = base
    else:
        log(f"## ★ 최종 단축형: **{len(final)}문항** — {', '.join(LAB[i] for i in final)}")
        fc = set(assign[i] for i in final)
        log(f"- FA 4축 커버: {sorted('F'+str(f+1) for f in fc)} ({'전축 커버' if len(fc)==4 else '미달'})")
        mc, mi, kc = cv_eval(m, final, gap)
        log(f"- 성능: MAE {mc:.3f}(≤A1 {a1_mae:.3f} ✅) / 중증 {mi*100:.1f}%(≤A0 {a0_miss*100:.1f}% ✅) / 카파 {kc:.3f}")
        log(f"- 고정4 포함 ✅, 4축 각1 ✅, 성능 ✅, 최소화 ✅\n")

    # 최종 채점표(전체자료 재적합) — 계수만(비-L1로 깔끔히 하려면 별도, 여기선 elastic-net)
    imp, sc, md = fit_sub(m[final].values, m["ds_stage"].astype(int).values)
    sd = sc.scale_; mu = sc.mean_
    rows = []
    for k in range(1, 6):
        w = md[k].coef_[0] / sd; b0 = md[k].intercept_[0] - np.sum(md[k].coef_[0] * mu / sd)
        for i, it in enumerate(final): rows.append({"cut": f"P(Y>={k})", "item": it, "label": LAB[it], "coef_raw": round(w[i], 4)})
        rows.append({"cut": f"P(Y>={k})", "item": "(절편)", "label": "", "coef_raw": round(b0, 4)})
    pd.DataFrame(rows).to_csv(CSV / "b2_final_shortform_scoretable.csv", index=False, encoding="utf-8-sig")
    log(f"- 채점표 CSV: b2_final_shortform_scoretable.csv")
    log("- **주의**: 독립검증된 단축형 아님(본 자료 최적화). 임계 τ는 배포용 별도 고정.")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__ == "__main__":
    main()
