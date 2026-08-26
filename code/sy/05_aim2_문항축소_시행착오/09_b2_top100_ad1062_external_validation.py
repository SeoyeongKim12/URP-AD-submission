"""
Aim 2 · B2 종합순위 상위 100개 조합 — AD-1062 진단용 외부검증 (listwise) (sjlee)
====================================================================================
목적: b2_top16_ad1062_external_validation.py(16개)는 "내부CV 순위와 외부성능이 얼마나
일치하는가"를 보기엔 표본이 작음(상관계수 하나 계산하기에도 불안정). 상위 100개로
넓혀서 이 상관관계·분포를 더 안정적으로 확인함.

**중요 — 이 스크립트의 결과로 "새 최종 조합"을 뽑지 않음.** 100개 중 AD-1062 성능이
제일 좋은 걸 또 골라버리면, 그 선택 자체가 AD-1062라는 표본에 대해 또 다른 규모의
"승자의 저주"(다중비교)를 만들어냄 — AD-1062를 최종 확인용으로 다시 쓸 수 없게
됨(이미 100번 비교에 오염됨). 이 스크립트는 순수하게 "내부CV가 얼마나 낙관적인지,
그 경향이 얼마나 일관적인지"를 진단하는 용도로만 씀. 최종 조합 확정은 여전히
FA/4대 규칙으로 좁힌 소수 후보 + 이번 진단 결과(패턴 이해) + 임상적 해석을 종합해서
사람이 판단해야 함.

절차: b2_top16 스크립트와 동일 원칙(3개 시험 풀 전체로 최종모형 학습 → τ A0-목표
1회 튜닝 → AD-1062에 1회만 적용, listwise deletion 유지). COMBOS는 하드코딩 대신
b2_fixed6_exhaustive_listwise_{EN,L2}.csv에서 종합점수(score) 기준 상위 100개를
그때그때 다시 뽑아옴(별도 CSV로 저장도 해둠 — top100_combos.csv).

산출: aim2/시행착오/b2_top100_ad1062_external_report.md
      aim2/시행착오/b2_top100_ad1062_predictions.csv (조합별 AD-1062 예측치, 롱포맷)
      aim2/시행착오/b2_top100_combo_list.csv (이번에 검증한 100개 조합 목록)
"""
from pathlib import Path
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b2_top100_ad1062_external_report.md"
EXT_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study_csv\extra_validation_ad1062")

TAU_GRID = np.round(np.arange(0.05, 0.55, 0.02), 3)
PENALTY_KW = {
    "EN": dict(penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=3000, tol=1e-3),
    "L2": dict(penalty="l2", solver="lbfgs", max_iter=3000, tol=1e-3),
}
C_MAIN = 0.1
N_TOP = 100


def build_top_n_combos(n=N_TOP):
    """listwise 전수조사 결과에서 종합점수(score) 기준 상위 n개 (penalty, items) 조합을 뽑음.
    b2_fixed6_EN_L2_종합비교_보고서_listwise판.md 2절과 동일한 정의(가중치 0.5:0.5,
    기준통과 필터 먼저 적용)."""
    en = pd.read_csv(OUTDIR / "b2_fixed6_exhaustive_listwise_EN.csv"); en["penalty"] = "EN"
    l2 = pd.read_csv(OUTDIR / "b2_fixed6_exhaustive_listwise_L2.csv"); l2["penalty"] = "L2"
    df = pd.concat([en, l2], ignore_index=True)
    a1 = df["a1_mae_ref"].iloc[0]; a0 = df["a0_miss_pct_ref"].iloc[0]
    df["pass"] = (df["mae"] < a1) & (df["hi_missed_pct"] <= a0)
    df["score"] = 0.5 * (df["mae"] / a1) + 0.5 * (df["hi_missed_pct"] / a0)
    strict = df[df["pass"]].sort_values("score").reset_index(drop=True)
    top = strict.head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    top.to_csv(OUTDIR / "b2_top100_combo_list.csv", index=False, encoding="utf-8-sig")
    combos = [(int(r["rank"]), r["penalty"], r["items_code"].split(","),
               r["score"], r["mae"], r["hi_missed_pct"]) for _, r in top.iterrows()]
    return combos


def load_ad1062():
    adl = pd.read_csv(EXT_DIR / "adl_wide_ad1062.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 1.0].copy()

    def col(name):
        r = f"{name}__resolved"
        return b[r] if r in b.columns else b[name]

    data = {}
    for item in D.ITEMS:
        if item == "Q18":
            data[item] = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
        else:
            data[item] = col(item)
    X = pd.DataFrame(data)
    X["STUDYID"] = b["STUDYID"].values
    X["USUBJID"] = b["USUBJID"].values
    X["A0_harmonized"] = b["A0_harmonized"].values
    X["A1_2015_stage"] = b["A1_2015_stage"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])

    ds = pd.read_csv(EXT_DIR / "ds_wide_ad1062.csv", low_memory=False)
    ds_bl = ds[ds["VISITNUM"] == 1.0][["STUDYID", "USUBJID", "ds_stage"]]

    m = ds_bl.merge(X, on=["STUDYID", "USUBJID"], how="left")
    m = m.dropna(subset=["ds_stage", "A0_harmonized", "A1_2015_stage"])
    return m.reset_index(drop=True)


def fit_final(train, items, penalty_tag, C=C_MAIN):
    kw = PENALTY_KW[penalty_tag]
    sc = StandardScaler().fit(train[items].to_numpy(dtype=float))
    Z = sc.transform(train[items].to_numpy(dtype=float))
    y = train["ds_stage"].astype(int).values
    models = {k: LogisticRegression(C=C, random_state=0, **kw).fit(Z, (y >= k).astype(int))
              for k in range(1, 6)}

    def probs(df):
        Zx = sc.transform(df[items].to_numpy(dtype=float))
        g = {k: models[k].predict_proba(Zx)[:, 1] for k in range(1, 6)}
        P = np.zeros((len(df), 6)); P[:, 0] = 1 - g[1]
        for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
        P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)
        return P
    return probs, y


def tune_tau(Ptr, ytr, target):
    best = None
    for t4 in TAU_GRID:
        for t5 in TAU_GRID:
            if t5 < t4:
                continue
            p = D.asym(Ptr, t4, t5)
            if D.miss(ytr, p) <= target + 1e-9:
                key = D.mae(ytr, p)
                if best is None or key < best[0]:
                    best = (key, float(t4), float(t5))
    if best is None:
        return 0.3, 0.5
    return best[1], best[2]


def main():
    _lines = []
    def log(s=""):
        print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    log("# Aim 2 · B2 종합순위 상위 100개 조합 -- AD-1062 진단용 외부검증 (listwise, sjlee)\n")
    log("**주의**: 이 결과로 새 최종조합을 고르지 않음(고르면 AD-1062가 또 다른 다중비교에 "
        "오염됨). 내부CV가 얼마나 낙관적인지 패턴만 확인하는 진단용.\n")

    combos = build_top_n_combos(N_TOP)
    train_full = D.build()
    ext_full = load_ad1062()
    log(f"검증 대상: 내부CV 종합순위 상위 {len(combos)}개 조합")
    log(f"훈련풀(결측필터 전): {len(train_full)}명 / AD-1062(결측필터 전): {len(ext_full)}명\n")

    t0 = time.time()
    rows = []
    pred_rows = []
    for i, (rank, ptag, items, in_score, in_mae, in_miss) in enumerate(combos):
        tr = train_full.loc[train_full[items].notna().all(axis=1)].reset_index(drop=True)
        ex = ext_full.loc[ext_full[items].notna().all(axis=1)].reset_index(drop=True)

        probs_fn, ytr = fit_final(tr, items, ptag, C=C_MAIN)
        Ptr = probs_fn(tr)
        target = D.miss(ytr, tr["A0_harmonized"].values)
        t4, t5 = tune_tau(Ptr, tr["ds_stage"].values, target)

        pred = D.asym(probs_fn(ex), t4, t5)
        t = ex["ds_stage"].values
        a0 = ex["A0_harmonized"].values; a1 = ex["A1_2015_stage"].values
        b_mae, b_miss, b_k = D.mae(t, pred), D.miss(t, pred) * 100, D.kap(t, pred)
        a0_mae, a0_miss = D.mae(t, a0), D.miss(t, a0) * 100
        a1_mae, a1_miss = D.mae(t, a1), D.miss(t, a1) * 100
        dom_a0 = (b_miss <= a0_miss + 1.0) and (b_mae < a0_mae)
        dom_a1 = (b_miss < a1_miss) and (b_mae < a1_mae)

        rows.append({"내부CV_순위": rank, "penalty": ptag, "문항수": len(items),
                     "내부CV_score": round(in_score, 4), "내부CV_MAE": in_mae, "내부CV_놓침%": in_miss,
                     "AD1062_n": len(ex), "AD1062_MAE": round(b_mae, 4), "AD1062_놓침%": round(b_miss, 2),
                     "AD1062_카파": round(b_k, 4), "A0지배": dom_a0, "A1지배": dom_a1})
        for uid, tt, pp in zip(ex["USUBJID"], t, pred):
            pred_rows.append({"내부CV_순위": rank, "penalty": ptag, "n_items": len(items),
                               "USUBJID": uid, "ds_stage": tt, "B2_pred": pp})
        if (i + 1) % 20 == 0:
            log(f"[진행] {i+1}/{len(combos)} 완료 ({time.time()-t0:.0f}초 경과)")

    sdf = pd.DataFrame(rows)
    pd.DataFrame(pred_rows).to_csv(OUTDIR / "b2_top100_ad1062_predictions.csv", index=False, encoding="utf-8-sig")

    log(f"\n총 {len(combos)}개 조합 외부검증 완료 ({time.time()-t0:.0f}초)\n")

    # ---- 진단 요약 ----
    corr_score = sdf["내부CV_순위"].corr(sdf["AD1062_MAE"])
    corr_mae = sdf["내부CV_MAE"].corr(sdf["AD1062_MAE"])
    corr_miss = sdf["내부CV_놓침%"].corr(sdf["AD1062_놓침%"])
    n_a0 = sdf["A0지배"].sum(); n_a1 = sdf["A1지배"].sum()
    gap_mae = (sdf["AD1062_MAE"] - sdf["내부CV_MAE"]).mean()
    gap_miss = (sdf["AD1062_놓침%"] - sdf["내부CV_놓침%"]).mean()

    log("## 진단 요약 (100개 조합 기준)")
    log(f"- **내부CV순위 vs AD-1062 MAE 상관계수: {corr_score:.3f}** (음수면 '내부 순위가 높을수록"
        "(숫자가 작을수록) 외부 MAE도 낮다'는 뜻 — 방향과 크기를 같이 봐야 함)")
    log(f"- 내부CV MAE vs AD-1062 MAE 상관계수: {corr_mae:.3f}")
    log(f"- 내부CV 놓침% vs AD-1062 놓침% 상관계수: {corr_miss:.3f}")
    log(f"- **평균 성능갭(외부-내부)**: MAE {gap_mae:+.4f}, 놓침% {gap_miss:+.2f}%p "
        "(양수면 외부에서 더 나빠진다는 뜻 — 이게 승자의 저주로 인한 낙관편향의 평균 크기)")
    log(f"- 100개 중 AD-1062에서 A0 지배: {n_a0}개 / A1 지배: {n_a1}개\n")

    log("## 참고: AD-1062 MAE 기준 상위 15개 (진단용 — 이걸 '새 1등'으로 확정하지 말 것)")
    log("| 내부CV순위 | penalty | 문항수 | 내부MAE | 내부놓침% | AD1062 MAE | AD1062 놓침% | A0지배 | A1지배 |")
    log("|---|---|---|---|---|---|---|---|---|")
    for _, r in sdf.sort_values("AD1062_MAE").head(15).iterrows():
        log(f"| {r['내부CV_순위']} | {r['penalty']} | {r['문항수']} | {r['내부CV_MAE']} | "
            f"{r['내부CV_놓침%']}% | {r['AD1062_MAE']} | {r['AD1062_놓침%']}% | "
            f"{'예' if r['A0지배'] else '아니오'} | {'예' if r['A1지배'] else '아니오'} |")

    sdf.to_csv(OUTDIR / "b2_top100_ad1062_summary.csv", index=False, encoding="utf-8-sig")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
    print(f">>> 저장: {OUTDIR / 'b2_top100_ad1062_summary.csv'}")
    print(f">>> 저장: {OUTDIR / 'b2_top100_ad1062_predictions.csv'}")
    print(f">>> 저장: {OUTDIR / 'b2_top100_combo_list.csv'}")


if __name__ == "__main__":
    main()
