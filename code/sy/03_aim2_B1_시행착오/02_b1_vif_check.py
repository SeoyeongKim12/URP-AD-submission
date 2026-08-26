"""
Aim 2 · B1 — 다중공선성(VIF) 점검 + ADL0110A 단독회귀 확인 (sjlee)
============================================================
목적
  1) 21개 문항점수 간 VIF(분산팽창지수) 계산 → 다중공선성 심각도 정량화
  2) ADL0110A만 단독(bivariate)으로 ds_stage에 회귀 → 다변량에서 나온
     부호반전(+0.128, p=.016)이 억제효과(suppression)인지 실제 반전인지 판정

범위
  진단(diagnostic) 목적이라 CV 외부검증 없이 전체 공통표본에서 수행함
  (po_assumption()과 동일 범위). fold 분리·정보누출 통제는 여기선 해당 없음.

입력   {DATA_DIR}/adl_wide.parquet, {DATA_DIR}/baseline_sample.parquet
출력   {OUT_DIR}/b1_vif.csv                (문항별 VIF)
       {OUT_DIR}/b1_adl0110a_corr.csv      (ADL0110A 상관 top5)
       {OUT_DIR}/b1_adl0110a_bivariate.csv (단독 vs 다변량 계수 비교)
       터미널에 핵심 계수 + 해석 출력
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =================================================================
# 설정
# =================================================================
DATA_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study")
OUT_DIR = Path(__file__).parent

ADL_WIDE_PATH = DATA_DIR / "adl_wide.parquet"
BASELINE_PATH = DATA_DIR / "baseline_sample.parquet"

BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = [
    "ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
    "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
    "ADL0121A", "ADL0122Q", "ADL0123L",
]
ITEMS = BADL + IADL  # 21문항

VIF_HIGH = 10   # VIF 판정 임계값(심각)
VIF_MID = 5     # VIF 판정 임계값(주의)

MULTIVARIATE_REF = {  # 직전 21문항 다변량 모형에서 나온 ADL0110A 결과(비교용, 하드코딩)
    "coef": 0.128, "OR": 1.14, "z": 2.41, "p": 1.6e-02,
}


def hr(title: str = "", ch: str = "="):
    """터미널 구분선 출력용."""
    print(ch * 70)
    if title:
        print(title)
        print(ch * 70)


# =================================================================
# step 0. 데이터 로드
# =================================================================
def load_matrix() -> pd.DataFrame:
    adl = pd.read_parquet(ADL_WIDE_PATH)
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

    baseline = pd.read_parquet(BASELINE_PATH)
    common = baseline[baseline["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]
    ]
    m = common.merge(X, on=["STUDYID", "USUBJID"], how="left")
    return m


def impute_items(m: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median").fit(m[items])
    return pd.DataFrame(imputer.transform(m[items]), columns=items, index=m.index)


# =================================================================
# step 1. VIF
# =================================================================
def compute_vif(m: pd.DataFrame) -> pd.DataFrame:
    X = impute_items(m, ITEMS)
    Xc = sm.add_constant(X)

    rows = []
    for i, item in enumerate(Xc.columns):
        if item == "const":
            continue
        vif = variance_inflation_factor(Xc.values, i)
        flag = "high" if vif > VIF_HIGH else ("mid" if vif > VIF_MID else "low")
        rows.append({"item": item, "VIF": vif, "flag": flag})

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_df.to_csv(OUT_DIR / "b1_vif.csv", index=False)
    return vif_df


def print_vif(vif_df: pd.DataFrame):
    hr("STEP 1. VIF (다중공선성)")
    print(vif_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    n_high = (vif_df["flag"] == "high").sum()
    n_mid = (vif_df["flag"] == "mid").sum()
    print(f"\nVIF>{VIF_HIGH} (심각) {n_high}개 / {VIF_MID}~{VIF_HIGH} (주의) {n_mid}개 "
          f"/ <{VIF_MID} (낮음) {len(vif_df) - n_high - n_mid}개")
    top = vif_df.iloc[0]
    print(f"해석: 최고 VIF는 {top['item']} ({top['VIF']:.2f}). "
          f"{'다중공선성이 심각한 수준' if top['VIF'] > VIF_HIGH else '심각하진 않으나 일부 문항 간 중복 존재'}.")
    print(f"저장: {OUT_DIR / 'b1_vif.csv'}")


# =================================================================
# step 2. ADL0110A 상관 top5 (VIF 원인 추적)
# =================================================================
def compute_top_correlations(m: pd.DataFrame, target: str = "ADL0110A", topn: int = 5) -> pd.DataFrame:
    X = impute_items(m, ITEMS)
    corr = X.corr()[target].drop(target)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(topn)
    corr_df = corr.reset_index()
    corr_df.columns = ["item", "corr_with_" + target]
    corr_df.to_csv(OUT_DIR / f"b1_{target.lower()}_corr.csv", index=False)
    return corr_df


def print_top_correlations(corr_df: pd.DataFrame, target: str):
    hr(f"STEP 2. {target} 상관 top {len(corr_df)} (VIF 원인 추적)")
    print(corr_df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"저장: {OUT_DIR / f'b1_{target.lower()}_corr.csv'}")


# =================================================================
# step 3. ADL0110A 단독(bivariate) 순서형 회귀
# =================================================================
def bivariate_check(m: pd.DataFrame, item: str = "ADL0110A") -> pd.DataFrame:
    x = impute_items(m, [item]).values
    y = m["ds_stage"].astype(int).values

    res = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False, maxiter=200)
    coef, se = res.params[0], res.bse[0]
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    out = pd.DataFrame([
        {"model": "bivariate", "item": item, "coef": coef, "OR": np.exp(coef), "z": z, "p": p},
        {"model": "multivariate_21items", "item": item,
         "coef": MULTIVARIATE_REF["coef"], "OR": MULTIVARIATE_REF["OR"],
         "z": MULTIVARIATE_REF["z"], "p": MULTIVARIATE_REF["p"]},
    ])
    out.to_csv(OUT_DIR / f"b1_{item.lower()}_bivariate.csv", index=False)
    return out


def print_bivariate(out: pd.DataFrame, item: str):
    hr(f"STEP 3. {item} 단독(bivariate) vs 다변량 회귀 비교")
    print(out.to_string(index=False, formatters={
        "coef": lambda v: f"{v:+.3f}", "OR": lambda v: f"{v:.2f}",
        "z": lambda v: f"{v:+.2f}", "p": lambda v: f"{v:.3g}",
    }))

    biv_coef = out.loc[out["model"] == "bivariate", "coef"].iloc[0]
    if biv_coef > 0:
        verdict = (
            f"단독에서도 부호가 양(+)으로 유지됨 → 억제효과(suppression)가 아니라 "
            f"{item} 자체의 실제(marginal) 연관 방향이 다른 문항과 반대일 가능성이 큼. "
            f"채점방향(coding) 재확인 + 임상적 해석 필요."
        )
    else:
        verdict = (
            f"단독에서는 부호가 음(-)으로 바뀜 → 다변량에서만 양(+)으로 나온 건 "
            f"다른 문항들과 얽힌 억제효과(suppression)일 가능성이 큼. 통계적 인공물에 가까움."
        )
    print(f"\n해석: {verdict}")
    print(f"저장: {OUT_DIR / f'b1_{item.lower()}_bivariate.csv'}")


# =================================================================
# main
# =================================================================
def main():
    hr("Aim 2 · B1 — VIF + ADL0110A 단독회귀 점검 (sjlee)")
    print(f"입력: {ADL_WIDE_PATH}")
    print(f"      {BASELINE_PATH}")

    m = load_matrix()
    print(f"표본 {len(m)}명 x 문항 {len(ITEMS)}개 (결측은 진단용 중앙값 대치, CV 아님)\n")

    vif_df = compute_vif(m)
    print_vif(vif_df)
    print()

    corr_df = compute_top_correlations(m, "ADL0110A", topn=5)
    print_top_correlations(corr_df, "ADL0110A")
    print()

    biv_df = bivariate_check(m, "ADL0110A")
    print_bivariate(biv_df, "ADL0110A")

    hr("완료")
    print(f"CSV 산출물: {OUT_DIR}/b1_vif.csv, b1_adl0110a_corr.csv, b1_adl0110a_bivariate.csv")


if __name__ == "__main__":
    main()