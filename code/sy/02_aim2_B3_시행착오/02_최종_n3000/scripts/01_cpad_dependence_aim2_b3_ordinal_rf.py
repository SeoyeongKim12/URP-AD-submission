"""
Aim 2 — B3: 순서형 랜덤포레스트(Ordered Forest), 비선형 성능 상한 확인 (보조)
================================================================
연구계획서 "ADCS-ADL 기반 파생 의존도 지표의 실측 Dependence Scale 대비
타당성 및 반응성 검증" 2.4절(Aim 2, 대안 채점법 개발) B3 구현.

B3 정의 (계획서 원문 요약, 2.4절)
---------------------------------
- 입력변수: 2025 개정판이 쓰는 ADCS-ADL 문항의 문항별 최종점수(A0와 동일 입력,
  19개 문항 후보). 시험(STUDYID) 변수는 투입하지 않는다.
- 출력: 예측분포의 조건부 중앙값을 1차 예측단계로 사용(연구 전체 사전 고정 규칙).
- B3는 희소 범주(기저 0단계=29명)에서 불안정하므로 0·1 단계를 통합한 5범주에서만
  보조적으로 평가한다 (B1의 6범주 주분석과는 분리된 범위).
- 검증 구조: AD-1061/1063/1064를 하나씩 남기는 leave-one-trial-out 3-fold.
  각 외부 fold에서 평가시험을 완전히 제외하고 학습 2개 시험 내부에서만
  결측처리·문항선택·튜닝·최종 재적합을 수행한다.
- 성능지표: MAE, 가중 카파(quadratic), ±1단계 정확도, 단계별 calibration,
  두 가지 과소평가 지표(원래 0–5 척도 정의; B3는 0·1 통합 5범주라 근사치로만
  계산 — 아래 6번 함수 docstring 참고).

데이터 연결
-----------
urp-AD/dependence_study/ 아래 기존 전처리 산출물(parquet)을 그대로 재사용한다:
    - baseline_sample.parquet : STUDYID, USUBJID, ds_stage, in_aim1_2_sample 등
    - adl_wide.parquet        : 기저(VISITNUM==2.0) ADCS-ADL 문항 원점수 + gate 상태
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

try:
    import orf  # Ordered Forest (Lechner & Okasa) — 순서형 랜덤포레스트
except ImportError as e:
    raise ImportError("pip install orf --break-system-packages 로 설치할 것") from e

# orf 0.2.0 내부(_BaseOrderedForest._performance)는 sklearn<1.2의
# OneHotEncoder(sparse=...) 호출부를 그대로 쓰고 있다. 최신 sklearn은
# sparse_output=...으로 개명되어 TypeError가 난다. orf 소스를 고치는 대신
# orf 모듈 네임스페이스의 OneHotEncoder 참조만 호환 함수로 바꿔치기한다.
from sklearn.preprocessing import OneHotEncoder as _OrigOHE


def _ohe_compat(**kwargs):
    sparse = kwargs.pop("sparse", None)
    if sparse is not None:
        kwargs.setdefault("sparse_output", sparse)
    return _OrigOHE(**kwargs)


orf._BaseOrderedForest.OneHotEncoder = _ohe_compat

# orf 0.2.0은 sklearn._forest._generate_sample_indices를 자체 함수
# (_generate_sample_indices_bootstrap / _generate_sample_indices_subsampling,
# 둘 다 random_state·n_samples·n_samples_bootstrap 3개 인자만 받음)로
# 바꿔치기해서 쓴다. 최신 sklearn은 이 내부함수 호출부에 sample_weight
# 인자를 하나 더 추가했는데, orf의 대체 함수는 이를 받지 못해
# "takes 3 positional arguments but 4 were given" TypeError가 난다.
# orf 함수들이 sample_weight를 받아도 무시하도록 감싸서 양쪽 sklearn
# 버전(3개 인자만 넘기는 구버전 / 4개 넘기는 신버전) 모두에서 동작하게 만든다.
def _ignore_extra_arg(fn):
    def wrapped(random_state, n_samples, n_samples_bootstrap, sample_weight=None):
        return fn(random_state, n_samples, n_samples_bootstrap)
    return wrapped


orf._BaseOrderedForest._generate_sample_indices_bootstrap = _ignore_extra_arg(
    orf._BaseOrderedForest._generate_sample_indices_bootstrap
)
orf._BaseOrderedForest._generate_sample_indices_subsampling = _ignore_extra_arg(
    orf._BaseOrderedForest._generate_sample_indices_subsampling
)


DATA_DIR = "/sessions/eloquent-serene-mccarthy/mnt/urp-AD/dependence_study"
TARGET_STUDIES = ["AD-1061", "AD-1063", "AD-1064"]

# preprocess_dependence_study.py 와 동일한 매핑 (재사용, 임의 변경 없음)
ADL_GATE_MAP = {
    "ADL0106": ["ADL0106A", "ADL0106B"],
    "ADL0107": ["ADL0107A"],
    "ADL0110": ["ADL0110A"],
    "ADL0111": ["ADL0111A"],
    "ADL0112": ["ADL0112A"],
    "ADL0113": ["ADL0113A"],
    "ADL0114": ["ADL0114A"],
    "ADL0115": ["ADL0115A"],
    "ADL0116": ["ADL0116A", "ADL0116B"],
    "ADL0117": ["ADL0117A"],
    "ADL0118": ["ADL0118A", "ADL0118B", "ADL0118C"],
    "ADL0121": ["ADL0121A"],
    "ADL0122": ["ADL0122Q"],
    "ADL0123": ["ADL0123L"],
}

ADL_2025_EXCLUDE = [
    "ADL0108", "ADL0108A", "ADL0108B", "ADL0108C",
    "ADL0109", "ADL0109A",
    "ADL0119", "ADL0119A", "ADL0119B", "ADL0119C",
    "ADL0120", "ADL0120A", "ADL0120B",
]
ADL_NOT_A_PERFORMANCE_ITEM = ["ADL0118N", "ADL0124", "ADL0125"]


def load_baseline_features() -> pd.DataFrame:
    baseline = pd.read_parquet(f"{DATA_DIR}/baseline_sample.parquet")
    adl = pd.read_parquet(f"{DATA_DIR}/adl_wide.parquet")
    adl_bl = adl[adl["VISITNUM"] == 2.0].copy()

    for gate, subs in ADL_GATE_MAP.items():
        gate_col = gate + "__gate"
        if gate_col not in adl_bl.columns:
            continue
        not_gen = adl_bl[gate_col] == "not_generated"
        for sub in subs:
            if sub in adl_bl.columns:
                adl_bl[sub] = adl_bl[sub].where(~not_gen, 0)

    all_gate_cols = [
        c for c in adl_bl.columns
        if c.startswith("ADL01") and not c.endswith("__gate")
    ]
    input_items = sorted(
        set(all_gate_cols) - set(ADL_2025_EXCLUDE) - set(ADL_NOT_A_PERFORMANCE_ITEM)
    )

    keep_cols = ["STUDYID", "USUBJID"] + input_items
    merged = baseline[baseline["in_aim1_2_sample"]][
        ["STUDYID", "USUBJID", "ds_stage", "ARM"]
    ].merge(adl_bl[keep_cols], on=["STUDYID", "USUBJID"], how="left")

    merged.attrs["input_items"] = input_items
    return merged


def collapse_stage_for_b3(ds_stage: pd.Series) -> pd.Series:
    collapsed = ds_stage.clip(lower=1)
    return collapsed.astype(int)


def impute_train_median(train_X: pd.DataFrame, apply_to: pd.DataFrame):
    medians = train_X.median()
    return train_X.fillna(medians), apply_to.fillna(medians), medians


def median_class_from_probs(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)
    idx = (cum >= 0.5).argmax(axis=1)
    return classes[idx]


def per_stage_calibration(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in classes:
        mask = y_pred == c
        rows.append({
            "predicted_stage": c,
            "n": int(mask.sum()),
            "observed_mean_stage": y_true[mask].mean() if mask.sum() > 0 else np.nan,
            "exact_match_rate": (y_true[mask] == c).mean() if mask.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def underestimation_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_two_classes: tuple) -> dict:
    under2 = ((y_true - y_pred) >= 2).mean()
    top_mask = np.isin(y_true, top_two_classes)
    if top_mask.sum() > 0:
        top_under = (y_pred[top_mask] <= (min(top_two_classes) - 1)).mean()
    else:
        top_under = np.nan
    return {
        "underest_ge2_rate": under2,
        "top2class_underclassified_rate": top_under,
    }


def run_b3_ordered_forest(df: pd.DataFrame, input_items: list[str],
                           n_estimators: int = 3000, min_samples_leaf: int = 5,
                           random_state: int = 42):
    # n_estimators=3000을 최종값으로 채택함(2026-08-10 확정).
    # 근거: (1) 200~3000 수렴 확인 결과 3000까지 MAE·카파가 계속 개선되고
    # (b3_n_estimators_convergence.csv) (2) 학습-held-out 성능 격차가 트리
    # 수와 무관하게 일정해 과적합이 아님을 확인함(b3_overfit_check.csv)
    # (3) min_samples_leaf/max_features/sample_fraction을 중첩CV로 튜닝한
    # v2(cpad_dependence_aim2_b3_ordinal_rf_v2.py) 결과 및 v2+3000그루 결합
    # 실험과 비교했을 때도 "나머지 기본값 + n_estimators=3000"이 가장
    # 낫거나 동등했음(B3_하이퍼파라미터_탐색_보고서.txt 5절 참고).
    df = df.dropna(subset=["ds_stage"]).copy()
    df["y5"] = collapse_stage_for_b3(df["ds_stage"])
    classes = np.array([1, 2, 3, 4, 5])

    fold_metrics = []
    calib_tables = []
    raw_predictions = []

    for held_out in TARGET_STUDIES:
        train_df = df[df["STUDYID"] != held_out].reset_index(drop=True)
        test_df = df[df["STUDYID"] == held_out].reset_index(drop=True)

        Xtr_raw = train_df[input_items]
        Xte_raw = test_df[input_items]
        Xtr, Xte, _ = impute_train_median(Xtr_raw, Xte_raw)

        model = orf.OrderedForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            replace=True,
            honesty=False,
            random_state=random_state,
        )
        model.fit(X=Xtr.to_numpy(dtype=float), y=train_df["y5"].to_numpy(dtype=int))

        pred = model.predict(X=Xte.to_numpy(dtype=float), prob=True)
        probs = pred.get("predictions")
        y_pred = median_class_from_probs(probs, classes)
        y_true = test_df["y5"].to_numpy(dtype=int)

        mae = mean_absolute_error(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        acc_pm1 = (np.abs(y_true - y_pred) <= 1).mean()
        under = underestimation_metrics(y_true, y_pred, top_two_classes=(4, 5))

        fold_metrics.append({
            "held_out_STUDYID": held_out,
            "n": len(test_df),
            "MAE": mae,
            "weighted_kappa": kappa,
            "acc_within_1stage": acc_pm1,
            **under,
        })

        calib = per_stage_calibration(y_true, y_pred, classes)
        calib["held_out_STUDYID"] = held_out
        calib_tables.append(calib)

        raw_predictions.append(pd.DataFrame({
            "STUDYID": held_out,
            "USUBJID": test_df["USUBJID"].to_numpy(),
            "y_true_5cat": y_true,
            "y_pred_5cat": y_pred,
        }))

    metrics_df = pd.DataFrame(fold_metrics)
    calib_df = pd.concat(calib_tables, ignore_index=True)
    pred_df = pd.concat(raw_predictions, ignore_index=True)
    return metrics_df, calib_df, pred_df


if __name__ == "__main__":
    data = load_baseline_features()
    input_items = data.attrs["input_items"]
    print(f"입력 문항 수: {len(input_items)}")
    print(f"분석표본(Aim1/2, DS13 완전+ADCS-ADL 기저 보유): {len(data)}")
    print(data.groupby("STUDYID").size())

    metrics_df, calib_df, pred_df = run_b3_ordered_forest(data, input_items)

    print("\n=== B3 leave-one-trial-out 성능 (0·1 통합 5범주) ===")
    print(metrics_df.to_string(index=False))
    print(f"\n3개 시험 동일가중 평균 MAE: {metrics_df['MAE'].mean():.3f}")
    print(f"3개 시험 동일가중 평균 가중카파: {metrics_df['weighted_kappa'].mean():.3f}")

    print("\n=== 단계별 calibration ===")
    print(calib_df.to_string(index=False))

    metrics_df.to_csv("b3_ordered_forest_metrics.csv", index=False, encoding="utf-8-sig")
    calib_df.to_csv("b3_ordered_forest_calibration.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv("b3_ordered_forest_predictions.csv", index=False, encoding="utf-8-sig")
