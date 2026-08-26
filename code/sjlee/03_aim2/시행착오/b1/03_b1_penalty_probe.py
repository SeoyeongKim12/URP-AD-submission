"""
Aim 2 · B1 개선 보강 — 부분비례오즈 벌점(정규화) 강도·유형 탐색 (sjlee)
=====================================================================
질문: C(부분비례오즈=비평행 누적로짓)의 벌점을 제대로 걸면(세기 튜닝·엘라스틱넷)
Gn의 4조건 통과(특히 조건3 마진 +1.9%p)가 더 안정적/유리해지는가.

기존 b1_improve의 C는 L2 릿지 C=1.0 고정이었음. 여기선 벌점 유형·세기를 바꿔가며
'C+임계(mae_thr=0.10)'와 '예산마진 스윕에서 4조건 통과 여부·조건3 마진'을 비교.

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/b1_penalty_probe_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b1_penalty_probe_report.md"
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)
BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

def build_matrix():
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 2.0].copy()
    def col(c): return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {}
    for c in ITEMS:
        data[c] = (col("ADL0118A") + col("ADL0118B") + col("ADL0118C")) if c == "Q18" else col(c)
    X = pd.DataFrame(data); X["STUDYID"] = b["STUDYID"].values; X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]]
    return cs.merge(X, on=["STUDYID", "USUBJID"], how="left")

def cond_median(P): return STAGES[(np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)]
def asym_rule(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cond_median(P)))
def m_mae(t, p): return np.abs(np.asarray(t) - np.asarray(p)).mean()
def m_u2(t, p):  return (np.asarray(t) - np.asarray(p) >= 2).mean()
def m_hi(t, p):
    t = np.asarray(t); p = np.asarray(p); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan

def make_clf(pen):
    if pen[0] == "l2":
        return LogisticRegression(penalty="l2", C=pen[1], max_iter=5000)
    return LogisticRegression(penalty="elasticnet", solver="saga",
                              l1_ratio=pen[2], C=pen[1], max_iter=8000)

def nonparallel_probs(m, pen):
    folds = {}
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        sc = StandardScaler().fit(imp.transform(tr[ITEMS]))
        Ztr = sc.transform(imp.transform(tr[ITEMS])); Zte = sc.transform(imp.transform(te[ITEMS]))
        y = tr["ds_stage"].astype(int).values
        def build(Z):
            geq = {k: make_clf(pen).fit(Ztr, (y >= k).astype(int)).predict_proba(Z)[:, 1] for k in range(1, 6)}
            P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - geq[1]
            for k in range(1, 5): P[:, k] = geq[k] - geq[k + 1]
            P[:, 5] = geq[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
        folds[ho] = dict(te_idx=te.index, ytr=y, Ptr=build(Ztr), Pte=build(Zte))
    return folds

def tune_apply(m, folds, mae_thr):
    grid = np.round(np.arange(0.10, 0.55, 0.025), 3)
    pred = pd.Series(index=m.index, dtype=float)
    for ho in TRIALS:
        f = folds[ho]; yt = f["ytr"]; tr = m[m.STUDYID != ho]
        budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - mae_thr
        best = None
        for t4 in grid:
            for t5 in grid:
                if t5 < t4: continue
                p = asym_rule(f["Ptr"], t4, t5); mae = m_mae(yt, p)
                if mae <= budget:
                    key = (m_hi(yt, p), mae)
                    if best is None or key < best[0]: best = (key, float(t4), float(t5))
        _, t4, t5 = best
        pred.loc[f["te_idx"]] = asym_rule(f["Pte"], t4, t5)
    return pred

def eval4(m, pred, mae_thr=0.10):
    per = lambda fn, col: {t: fn(m.loc[m.STUDYID == t, "ds_stage"], col.loc[m.STUDYID == t] if hasattr(col, "loc") else m.loc[m.STUDYID == t, col]) for t in TRIALS}
    b_mae = {t: m_mae(m.loc[m.STUDYID == t, "ds_stage"], pred.loc[m.STUDYID == t]) for t in TRIALS}
    a_mae = {t: m_mae(m.loc[m.STUDYID == t, "ds_stage"], m.loc[m.STUDYID == t, "A1_2015_stage"]) for t in TRIALS}
    b_hi = {t: m_hi(m.loc[m.STUDYID == t, "ds_stage"], pred.loc[m.STUDYID == t]) for t in TRIALS}
    a_hi = {t: m_hi(m.loc[m.STUDYID == t, "ds_stage"], m.loc[m.STUDYID == t, "A1_2015_stage"]) for t in TRIALS}
    b_u2 = {t: m_u2(m.loc[m.STUDYID == t, "ds_stage"], pred.loc[m.STUDYID == t]) for t in TRIALS}
    a_u2 = {t: m_u2(m.loc[m.STUDYID == t, "ds_stage"], m.loc[m.STUDYID == t, "A1_2015_stage"]) for t in TRIALS}
    mean = lambda d: np.mean([d[t] for t in TRIALS])
    mae_drop = mean(a_mae) - mean(b_mae); d_hi = mean(b_hi) - mean(a_hi); d_u2 = mean(b_u2) - mean(a_u2)
    n_better = sum(b_mae[t] < a_mae[t] for t in TRIALS)
    c1, c2, c3, c4 = mae_drop >= mae_thr, d_u2 <= 0.02, d_hi <= 0.02, n_better >= 2
    return dict(mae_drop=mae_drop, hi=mean(b_hi), d_hi=d_hi, allpass=c1 and c2 and c3 and c4)

def main():
    log("# Aim 2 · 부분비례오즈 벌점 강도·유형 탐색 (sjlee)\n")
    m = build_matrix()
    log("각 벌점에서: (1) C+임계(mae_thr=0.10) 결과, (2) 예산마진 스윕에서 4조건(주0.10) 통과하는")
    log("train감소목표 구간과 그때 조건3 여유. **조건3 여유(+2%p 한도까지 거리)가 클수록 견고.**\n")

    pens = [("l2", 0.1, None), ("l2", 0.25, None), ("l2", 0.5, None),
            ("l2", 1.0, None), ("l2", 2.0, None),
            ("en", 0.5, 0.5), ("en", 0.25, 0.5)]
    names = {("l2", 0.1, None): "L2 C=0.1(강벌점)", ("l2", 0.25, None): "L2 C=0.25",
             ("l2", 0.5, None): "L2 C=0.5", ("l2", 1.0, None): "L2 C=1.0(기존)",
             ("l2", 2.0, None): "L2 C=2.0(약벌점)", ("en", 0.5, 0.5): "ElasticNet C=0.5,l1=.5",
             ("en", 0.25, 0.5): "ElasticNet C=0.25,l1=.5"}
    margins = [0.10, 0.12, 0.13, 0.14, 0.16]

    log("| 벌점 | C+임계 4·5→≤3 | 4조건통과 마진구간 | 그중 조건3(여유) |")
    log("|---|---|---|---|")
    for pen in pens:
        folds = nonparallel_probs(m, pen)
        p10 = tune_apply(m, folds, 0.10); r10 = eval4(m, p10, 0.10)
        pass_margins = []; best_slack = None
        for mg in margins:
            pr = tune_apply(m, folds, mg); rr = eval4(m, pr, 0.10)
            if rr["allpass"]:
                pass_margins.append(mg)
                slack = 0.02 - rr["d_hi"]      # +2%p 한도까지 여유
                if best_slack is None or slack > best_slack[0]:
                    best_slack = (slack, rr["hi"], mg)
        pm = f"{min(pass_margins):.2f}~{max(pass_margins):.2f}" if pass_margins else "없음"
        slk = f"{best_slack[1]*100:.1f}% (여유 {best_slack[0]*100:+.1f}%p)" if best_slack else "—"
        log(f"| {names[pen]} | {r10['hi']*100:.1f}% | {pm} | {slk} |")
    log("\n- '4조건통과 마진구간'이 넓고 '조건3 여유'가 클수록 그 벌점이 Gn을 견고하게 함.")
    log("- 기존(L2 C=1.0) 대비 개선되는 벌점이 있으면 그걸로 Gn 재확정 권장.\n")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__ == "__main__":
    main()
