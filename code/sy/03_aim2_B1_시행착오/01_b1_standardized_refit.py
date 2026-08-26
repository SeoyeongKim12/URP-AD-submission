"""
Aim 2 · B1 — 표준화(z-score) 입력변수로 비례오즈모형 재적합 (sjlee)
============================================================
목적
  21개 문항점수를 표준화(평균0, 표준편차1)한 뒤 비례오즈(OrderedModel)를
  다시 적합함. 원점수 계수는 문항마다 척도가 달라(단일문항 0~3/0~4 vs
  Q18처럼 세 문항 합산 0~9) 크기 비교가 공정하지 않았음 — 표준화하면
  "1SD 증가당 로그오즈 변화"로 전 문항이 동일 단위가 돼 비교 가능해짐.
  참고로 표준화는 선형 재척도라 z값/p값/모형적합도는 원점수 모형과
  이론상 동일하고(오즈비만 "1점당"→"1SD당"으로 재해석됨), 여기서는
  직접 재적합해 수치를 그대로 검증함.

범위
  진단/보고 목적으로 전체 공통표본에서 전체자료 재적합(final_scoretable()
  과 동일 범위). CV 외부검증이 아니므로 fold 분리는 하지 않음.

입력   {DATA_DIR}/adl_wide.parquet, {DATA_DIR}/baseline_sample.parquet
출력   {OUT_DIR}/b1_standardized_coefs.csv   (문항별: 표준화계수/원점수계수/
                                              SE/OR/95%CI/z/p 전부)
       {OUT_DIR}/b1_standardized_thresholds.csv (절단점 5개)
       터미널에 유의문항·부호반전·해석 출력
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

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

ALPHA = 0.05  # 유의수준


def hr(title: str = "", ch: str = "="):
    print(ch * 70)
    if title:
        print(title)
        print(ch * 70)


# =================================================================
# step 0. 데이터 로드 (b1_vif_check.py 와 동일 로직)
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
    # 주의: dependence_study 원본엔 in_common_comparison_sample이 없어
    # in_aim1_2_sample + ds_stage 결측 제거로 동일 취지의 표본을 구성함.
    common = baseline[
        (baseline["in_aim1_2_sample"] == True) & baseline["ds_stage"].notna()
    ][["STUDYID", "USUBJID", "ds_stage"]]
    m = common.merge(X, on=["STUDYID", "USUBJID"], how="left")
    return m


def impute_items(m: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median").fit(m[items])
    return pd.DataFrame(imputer.transform(m[items]), columns=items, index=m.index)


# =================================================================
# step 1. 표준화(z-score)
# =================================================================
def standardize_items(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """평균0/표준편차1 표준화. scaler 통계(mean, std)도 같이 반환해
    CSV에 원점수<->표준화 환산 정보를 남긴다."""
    scaler = StandardScaler().fit(X)
    Xz = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    scale_info = pd.DataFrame({
        "item": X.columns,
        "mean": scaler.mean_,
        "std": scaler.scale_,
    })
    return Xz, scale_info


# =================================================================
# step 2. 비례오즈모형 적합 (원점수 vs 표준화 나란히)
# =================================================================
def fit_ordered_model(X: pd.DataFrame, y: np.ndarray):
    res = OrderedModel(y, X.values, distr="logit").fit(method="bfgs", disp=False, maxiter=300)
    return res


def summarize_items(res, items: list[str]) -> pd.DataFrame:
    n_items = len(items)
    coef = res.params[:n_items]
    se = res.bse[:n_items]
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    return pd.DataFrame({
        "item": items,
        "coef": coef,
        "se": se,
        "z": z,
        "p": p,
        "OR": np.exp(coef),
        "OR_ci_low": np.exp(ci_lo),
        "OR_ci_high": np.exp(ci_hi),
        "sig": np.where(p < ALPHA, "*", ""),
    })


def summarize_thresholds(res, n_items: int, n_cats: int) -> pd.DataFrame:
    """cutpoint(theta_1..theta_{K-1}) 요약. OrderedModel은 이를 누적확률
    절편으로 내부적으로 재파라미터화(threshold[0], 이후 log-diff)하므로
    res.model.transform_threshold_params 로 원 스케일 절편을 복원한다.
    반환 배열은 [-inf, theta_1, ..., theta_{K-1}, +inf] 형태(길이 K+1,
    실측 확인됨)라서 양끝(-inf, +inf, 둘 다 실제 절단값 아님)을 버리고
    가운데 K-1개만 쓴다."""
    raw = res.params[n_items:]
    theta = res.model.transform_threshold_params(raw)
    theta = theta[1:-1]  # 양끝(-inf, +inf) 제외, 실제 절단값 K-1개만
    return pd.DataFrame({
        "cutpoint": [f"{k}|{k+1}" for k in range(len(theta))],
        "theta": theta,
    })


# =================================================================
# main
# =================================================================
def main():
    hr("Aim 2 · B1 — 표준화 입력으로 비례오즈모형 재적합 (sjlee)")
    print(f"입력: {ADL_WIDE_PATH}")
    print(f"      {BASELINE_PATH}")

    m = load_matrix()
    y = m["ds_stage"].astype(int).values
    n_cats = len(np.unique(y))
    print(f"표본 {len(m)}명 x 문항 {len(ITEMS)}개, 단계 {n_cats}개 (전체자료 재적합, CV 아님)\n")

    X_raw = impute_items(m, ITEMS)
    Xz, scale_info = standardize_items(X_raw)

    hr("STEP 1. 원점수 모형 (참고, raw)")
    res_raw = fit_ordered_model(X_raw, y)
    raw_df = summarize_items(res_raw, ITEMS)
    print(raw_df.sort_values("p").to_string(index=False, formatters={
        "coef": lambda v: f"{v:+.3f}", "se": lambda v: f"{v:.3f}",
        "z": lambda v: f"{v:+.2f}", "p": lambda v: f"{v:.3g}",
        "OR": lambda v: f"{v:.2f}", "OR_ci_low": lambda v: f"{v:.2f}",
        "OR_ci_high": lambda v: f"{v:.2f}",
    }))

    hr("STEP 2. 표준화(z-score) 모형 — 1SD당 로그오즈/오즈비")
    res_z = fit_ordered_model(Xz, y)
    z_df = summarize_items(res_z, ITEMS)
    z_df = z_df.rename(columns={
        "coef": "coef_per_1SD", "OR": "OR_per_1SD",
        "OR_ci_low": "OR_per_1SD_ci_low", "OR_ci_high": "OR_per_1SD_ci_high",
    })
    z_df_sorted = z_df.sort_values("p").reset_index(drop=True)
    print(z_df_sorted.to_string(index=False, formatters={
        "coef_per_1SD": lambda v: f"{v:+.3f}", "se": lambda v: f"{v:.3f}",
        "z": lambda v: f"{v:+.2f}", "p": lambda v: f"{v:.3g}",
        "OR_per_1SD": lambda v: f"{v:.2f}",
        "OR_per_1SD_ci_low": lambda v: f"{v:.2f}",
        "OR_per_1SD_ci_high": lambda v: f"{v:.2f}",
    }))

    n_sig = (z_df["p"] < ALPHA).sum()
    n_flip = (z_df["coef_per_1SD"] > 0).sum()
    flip_items = ", ".join(z_df.loc[z_df["coef_per_1SD"] > 0, "item"])
    print(f"\n유의(p<{ALPHA}) {n_sig}/{len(z_df)}개, 부호반전(양수) {n_flip}개: {flip_items}")

    # 원점수 vs 표준화 순위(영향력) 비교 — |z|값은 척도 불변이라 원래도 같은
    # 순위여야 하나, 표준화된 coef 크기 자체의 순위가 raw 계수 크기 순위와
    # 얼마나 다른지 보여줌(척도 문제로 인한 왜곡 확인용)
    rank_raw = raw_df.assign(rank_raw=raw_df["coef"].abs().rank(ascending=False))[["item", "rank_raw"]]
    rank_z = z_df.assign(rank_z=z_df["coef_per_1SD"].abs().rank(ascending=False))[["item", "rank_z"]]
    rank_cmp = rank_raw.merge(rank_z, on="item")
    rank_cmp["rank_shift"] = (rank_cmp["rank_raw"] - rank_cmp["rank_z"]).abs()
    biggest_shift = rank_cmp.sort_values("rank_shift", ascending=False).head(3)
    hr("STEP 3. 원점수 vs 표준화 — 영향력 순위 변화 (척도 왜곡 확인)")
    print(rank_cmp.sort_values("rank_shift", ascending=False).to_string(index=False))
    print(f"\n순위가 가장 많이 바뀐 문항: "
          f"{', '.join(biggest_shift['item'] + '(' + biggest_shift['rank_shift'].astype(int).astype(str) + '계단)')}")

    # 절단점(cutpoint)
    thr_df = summarize_thresholds(res_z, len(ITEMS), n_cats)
    hr("STEP 4. 절단점(cutpoint, 표준화 모형 기준)")
    print(thr_df.to_string(index=False, formatters={"theta": lambda v: f"{v:+.3f}"}))

    # ---- CSV 저장 ----
    out = raw_df.rename(columns={
        "coef": "coef_raw", "se": "se_raw", "z": "z_raw", "p": "p_raw",
        "OR": "OR_raw", "OR_ci_low": "OR_raw_ci_low", "OR_ci_high": "OR_raw_ci_high",
        "sig": "sig_raw",
    }).merge(
        z_df.rename(columns={"se": "se_z", "z": "z_z", "p": "p_z", "sig": "sig_z"}),
        on="item",
    ).merge(scale_info, on="item")
    out = out.sort_values("p_z").reset_index(drop=True)
    out.to_csv(OUT_DIR / "b1_standardized_coefs.csv", index=False)
    thr_df.to_csv(OUT_DIR / "b1_standardized_thresholds.csv", index=False)

    hr("완료")
    print(f"저장: {OUT_DIR / 'b1_standardized_coefs.csv'} (문항별 원점수/표준화 계수·SE·z·p·OR·95%CI 전부)")
    print(f"저장: {OUT_DIR / 'b1_standardized_thresholds.csv'} (절단점 5개)")


if __name__ == "__main__":
    main()
