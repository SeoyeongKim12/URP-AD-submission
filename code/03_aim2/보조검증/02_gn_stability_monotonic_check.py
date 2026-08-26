"""
Aim 2 · Gn 보조검증 — 비단조 확률 진단 + fold별 계수 안정성 (sjlee)
============================================================
체크리스트 2·3번:
  2) 비단조 확률: 절단별로 독립 모형을 쓰다 보니 P(Y≥k)가 k에 따라
     항상 감소한다는 보장이 없음. 위반 비율<5%, 클립전 음수확률질량
     인당평균<0.01 이면 안심 가능(사전 기준).
  3) fold별 계수 안정성: 특히 이전에 부호반전으로 의심됐던 3문항
     (ADL0101, ADL0110A, ADL0112A)이 leave-one-trial-out 3개 fold에서
     절단마다 일관되게(같은 부호로) 살아남는지, 아니면 fold마다
     들쭉날쭉(노이즈성)한지 확인.

범위: AD-1061/1063/1064 3개 훈련시험 내부 leave-one-trial-out만 사용
(AD-1062 외부검증 코드가 최종 채택한 C=0.1을 그대로 씀 — 같은 파이프라인의
후속 진단이라 하이퍼파라미터를 다르게 쓰면 비교 의미가 없어짐).

입력   {TRAIN_DIR}/adl_wide.parquet, baseline_sample.parquet
출력   {OUT_DIR}/gn_monotonic_diagnostic.csv     (fold별 비단조 발생률·음수질량)
       {OUT_DIR}/gn_coef_stability_by_fold.csv   (fold×문항×절단 계수 전부)
       {OUT_DIR}/gn_coef_stability_summary.csv   (문항×절단: 평균·SD·부호일관성)
       터미널에 의심문항 3개 요약 + 판정 출력
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =================================================================
# 설정
# =================================================================
TRAIN_DIR = Path.home() / "Downloads" / "preprocessed"  # 01_전처리/01_preprocess_dependence_study.py의 OUT_DIR과 동일
OUT_DIR = Path(__file__).parent

TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
L1_RATIO = 0.5
FINAL_C = 0.1  # gn_external_validation_ad1062.py 에서 최종 채택된 값과 동일하게 고정

BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = [
    "ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
    "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
    "ADL0121A", "ADL0122Q", "ADL0123L",
]
ITEMS = BADL + IADL
SUSPECT_ITEMS = ["ADL0101", "ADL0110A", "ADL0112A"]  # 이전에 부호반전으로 의심됐던 문항


def hr(title: str = "", ch: str = "="):
    print(ch * 70)
    if title:
        print(title)
        print(ch * 70)


# =================================================================
# step 0. 데이터 로드 (gn_external_validation_ad1062.py 와 동일)
# =================================================================
def load_train_pool() -> pd.DataFrame:
    adl = pd.read_parquet(TRAIN_DIR / "adl_wide.parquet")
    b = adl[adl["VISITNUM"] == 2.0].copy()

    def col(name: str) -> pd.Series:
        resolved = f"{name}__resolved"
        return b[resolved] if resolved in b.columns else b[name]

    data = {}
    for item in ITEMS:
        if item == "Q18":
            data[item] = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
        else:
            data[item] = col(item)
    X = pd.DataFrame(data)
    X["STUDYID"] = b["STUDYID"].values
    X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])

    baseline = pd.read_parquet(TRAIN_DIR / "baseline_sample.parquet")
    common = baseline[baseline["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]
    ]
    m = common.merge(X, on=["STUDYID", "USUBJID"], how="left")
    return m.dropna(subset=["ds_stage"])


def fit_en(train_df: pd.DataFrame, C: float):
    imp = SimpleImputer(strategy="median").fit(train_df[ITEMS])
    sc = StandardScaler().fit(imp.transform(train_df[ITEMS]))
    Ztr = sc.transform(imp.transform(train_df[ITEMS]))
    y = train_df["ds_stage"].astype(int).values
    models = {
        k: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=L1_RATIO,
            C=C, max_iter=5000, tol=1e-3,
        ).fit(Ztr, (y >= k).astype(int))
        for k in range(1, 6)
    }

    def geq(rows: pd.DataFrame):
        Z = sc.transform(imp.transform(rows[ITEMS]))
        return {k: models[k].predict_proba(Z)[:, 1] for k in range(1, 6)}

    return geq, models


# =================================================================
# step 1. 비단조 확률 진단
# =================================================================
def monotonic_diagnostic(m: pd.DataFrame) -> pd.DataFrame:
    hr("STEP 1. 비단조 확률 진단 (P(Y>=k)가 k에서 증가하는 위반 사례)")
    rows = []
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]
        te = m[m.STUDYID == ho]
        geq, _ = fit_en(tr, FINAL_C)
        g = geq(te)
        G = np.column_stack([g[k] for k in range(1, 6)])  # n x 5, P(Y>=1..5)

        # 단조감소 위반: G[:,k] >= G[:,k+1] 이 항상 성립해야 하는데 안 그런 행
        dec_ok = np.all(np.diff(G, axis=1) <= 1e-9, axis=1)
        n_viol = int((~dec_ok).sum())

        # 클리핑 전 원래 P(Y=k) 계산해서 음수질량 확인
        P0 = np.zeros((len(te), 6))
        P0[:, 0] = 1 - G[:, 0]
        for k in range(1, 5):
            P0[:, k] = G[:, k - 1] - G[:, k]
        P0[:, 5] = G[:, 4]
        neg_mass = float(np.clip(-P0, 0, None).sum())

        rows.append({
            "held_out_trial": ho, "n": len(te),
            "n_violation": n_viol, "violation_rate": n_viol / len(te),
            "neg_mass_total": neg_mass, "neg_mass_per_capita": neg_mass / len(te),
        })
        print(f"  {ho}: n={len(te)}  비단조 {n_viol}명({n_viol/len(te)*100:.1f}%)  "
              f"음수질량 인당평균 {neg_mass/len(te):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gn_monotonic_diagnostic.csv", index=False)

    overall_rate = df["n_violation"].sum() / df["n"].sum()
    overall_neg = df["neg_mass_total"].sum() / df["n"].sum()
    print(f"\n전체 비단조 발생률: {overall_rate*100:.1f}%  (기준 <5%)")
    print(f"전체 음수질량 인당평균: {overall_neg:.4f}  (기준 <0.01)")
    verdict = overall_rate < 0.05 and overall_neg < 0.01
    print(f"판정: {'PASS — 비단조 미미, 클립·정규화로 충분히 신뢰 가능' if verdict else 'FAIL — 비단조 상당함, 결과 해석에 주의 필요'}")
    return df


# =================================================================
# step 2. fold별 계수 안정성
# =================================================================
def coef_stability(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hr("STEP 2. fold별 절단별 계수 안정성 (21문항 전체 + 의심문항 3개 강조)")
    rows = []
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]
        _, models = fit_en(tr, FINAL_C)
        for k in range(1, 6):
            coefs = models[k].coef_[0]
            for item, c in zip(ITEMS, coefs):
                rows.append({"held_out_trial": ho, "cutpoint": f"P(Y>={k})", "item": item, "coef": c})

    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(OUT_DIR / "gn_coef_stability_by_fold.csv", index=False)

    def summarize(g: pd.DataFrame) -> pd.Series:
        vals = g["coef"].values
        n_nonzero = int((np.abs(vals) > 1e-9).sum())
        signs = np.sign(vals[np.abs(vals) > 1e-9])
        sign_consistent = bool(len(signs) == 0 or np.all(signs == signs[0]))
        return pd.Series({
            "mean_coef": vals.mean(), "std_coef": vals.std(),
            "n_folds_nonzero": n_nonzero, "sign_consistent": sign_consistent,
        })

    summary = by_fold.groupby(["item", "cutpoint"], sort=False).apply(
        summarize, include_groups=False
    ).reset_index()
    summary.to_csv(OUT_DIR / "gn_coef_stability_summary.csv", index=False)

    hr("의심문항 3개(ADL0101, ADL0110A, ADL0112A) — fold별 계수 상세")
    sus = by_fold[by_fold["item"].isin(SUSPECT_ITEMS)].pivot_table(
        index=["item", "cutpoint"], columns="held_out_trial", values="coef"
    )
    print(sus.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n판정 기준: 0/3(항상 제외) 또는 3/3+부호일관(항상 포함)이면 '일관적',")
    print("           1/3·2/3(어떤 fold엔 있고 없고 왔다갔다)이면 '불안정(노이즈 가능성)'\n")
    sus_summary = summary[summary["item"].isin(SUSPECT_ITEMS)]
    for _, r in sus_summary.iterrows():
        if r["n_folds_nonzero"] == 0:
            tag = "일관적(항상 제외)"
        elif r["n_folds_nonzero"] == 3 and r["sign_consistent"]:
            tag = "일관적(항상 포함, 진짜 신호 가능성)"
        else:
            tag = "불안정(노이즈 가능성)"
        print(f"  {r['item']:<10} {r['cutpoint']:<12} "
              f"비0 fold수={r['n_folds_nonzero']}/3  부호일관={r['sign_consistent']}  → {tag}")

    return by_fold, summary


# =================================================================
# main
# =================================================================
def main():
    hr("Aim 2 · Gn 보조검증 — 비단조 확률 + fold별 계수 안정성 (sjlee)")
    m = load_train_pool()
    print(f"훈련풀 {len(m)}명, 최종 채택 C={FINAL_C}\n")

    monotonic_diagnostic(m)
    print()
    coef_stability(m)

    hr("완료")
    print(f"저장: {OUT_DIR / 'gn_monotonic_diagnostic.csv'}")
    print(f"저장: {OUT_DIR / 'gn_coef_stability_by_fold.csv'}")
    print(f"저장: {OUT_DIR / 'gn_coef_stability_summary.csv'}")


if __name__ == "__main__":
    main()
