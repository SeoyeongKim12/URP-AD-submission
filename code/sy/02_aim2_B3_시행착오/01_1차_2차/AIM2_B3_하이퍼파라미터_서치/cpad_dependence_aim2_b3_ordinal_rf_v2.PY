"""
Aim 2 — B3 v2: 중첩 교차검증(nested CV)으로 하이퍼파라미터 튜닝
====================================================================
v1(cpad_dependence_aim2_b3_ordinal_rf.py)은 min_samples_leaf·max_features·
sample_fraction을 orf 기본값 그대로 썼다. 계획서 2.4절이 원래 요구하는
절차는 다음과 같은 중첩(nested) 구조다.

    바깥 루프(모형 평가) : AD-1061/1063/1064 leave-one-trial-out 3-fold
        └ 안쪽 루프(하이퍼파라미터 선택) : held-out 시험을 제외한 학습
          2개 시험 내부에서만 stratified 5-fold 교차검증

즉 하이퍼파라미터는 반드시 "그 fold의 학습 2개 시험 데이터"만 보고
정해야 하고, held-out 시험 정보는 최종 평가 단계 전까지 전혀 노출되면
안 된다. v1처럼 n_estimators를 held-out 성능을 직접 보면서 스윕하는 건
n_estimators 자체가 과적합을 일으키지 않는 성질(지난 대화에서 확인) 때문에
위험이 적지만, min_samples_leaf 등 진짜 편향-분산 트레이드오프가 있는
파라미터를 그렇게 고르면 정보누출로 성능이 낙관적으로 부풀려진다.

이 스크립트는 v1과 완전히 분리된 v2로, v1 파일·산출물은 전혀 건드리지
않는다. 최종 산출물 파일명도 b3_v2_*로 구분해서 저장한다.

튜닝 대상과 탐색범위
--------------------
- min_samples_leaf : [5, 10, 20, 40]   (작을수록 유연/과적합 위험↑)
- max_features     : [None, 0.33, 0.66] (orf는 'sqrt' 같은 문자열을 받지
                                          않고 int/float/None만 허용함을
                                          직접 확인함 — 23개 입력문항 기준
                                          0.33≈8개, 0.66≈15개 후보)
- sample_fraction  : [0.3, 0.5, 0.7]   (트리 하나당 쓰는 표본 비율)
n_estimators는 v1에서 이미 "과적합 위험이 사실상 없고 많을수록 안정적"임을
확인했으므로 탐색 대상에서 제외한다. 탐색 단계에서는 속도를 위해
n_estimators=300으로 상대순위만 비교하고, 선택된 조합으로 최종 재적합할
때는 v1과 동일한 n_estimators=1000을 써서 공정하게 비교 가능하게 한다.

선택 기준
---------
안쪽 5-fold CV의 평균 MAE가 가장 낮은 조합을 선택한다(이 연구의 1차
평가지표가 MAE이므로 — 2.4절 "일차 평가지표가 MAE이므로 이론적으로
대응"과 동일 원칙). 카파도 함께 기록해 참고한다.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from cpad_dependence_aim2_b3_ordinal_rf import (
    load_baseline_features, collapse_stage_for_b3, impute_train_median,
    median_class_from_probs, orf, TARGET_STUDIES, per_stage_calibration,
    underestimation_metrics,
)

RANDOM_STATE = 42
CLASSES = np.array([1, 2, 3, 4, 5])

PARAM_GRID = {
    "min_samples_leaf": [5, 10, 20, 40],
    "max_features": [None, 0.33, 0.66],
    "sample_fraction": [0.3, 0.5, 0.7],
}
N_ESTIMATORS_SEARCH = 300   # 탐색 단계(속도 우선, 상대순위 비교용)
N_ESTIMATORS_FINAL = 1000   # 최종 재적합(v1과 동일 — 공정 비교용)
N_INNER_SPLITS = 5


def _fit_predict_mae_kappa(Xtr, ytr, Xte, yte, params, n_estimators, random_state):
    model = orf.OrderedForest(
        n_estimators=n_estimators,
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        sample_fraction=params["sample_fraction"],
        replace=True, honesty=False, random_state=random_state,
    )
    model.fit(X=Xtr, y=ytr)
    pred = model.predict(X=Xte, prob=True)
    y_pred = median_class_from_probs(pred.get("predictions"), CLASSES)
    mae = mean_absolute_error(yte, y_pred)
    kappa = cohen_kappa_score(yte, y_pred, weights="quadratic")
    return mae, kappa


def inner_cv_tune(train_df: pd.DataFrame, input_items: list[str],
                   param_grid: dict | None = None,
                   n_splits: int = N_INNER_SPLITS,
                   n_estimators_search: int = N_ESTIMATORS_SEARCH,
                   random_state: int = RANDOM_STATE) -> tuple[dict, pd.DataFrame]:
    """held-out 시험을 완전히 제외한 학습 2개 시험 데이터(train_df)
    내부에서만 stratified 5-fold CV를 돌려 하이퍼파라미터를 고른다.

    주의(구현 메모): param_grid 기본값을 함수 정의부에서 `=PARAM_GRID`로
    바로 바인딩하면 파이썬의 "가변 기본인자는 함수 정의 시점에 딱 한 번만
    평가된다"는 규칙 때문에, 이후 모듈 전역 PARAM_GRID를 바꿔도 이 함수의
    기본값에는 반영되지 않는다(실제로 이 버그로 테스트 중 그리드가 의도치
    않게 36콤보 그대로 쓰여 시간을 두 번 허비했다). 그래서 None을 기본값으로
    받고 함수 본문에서 전역값을 다시 읽도록 했다."""
    param_grid = param_grid if param_grid is not None else PARAM_GRID
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_all = train_df["y5"].to_numpy(int)

    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]

    rows = []
    for params in combos:
        fold_maes, fold_kappas = [], []
        for tr_idx, va_idx in skf.split(train_df, y_all):
            tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
            Xtr, Xva, _ = impute_train_median(tr[input_items], va[input_items])
            mae, kappa = _fit_predict_mae_kappa(
                Xtr.to_numpy(dtype=float), tr["y5"].to_numpy(int),
                Xva.to_numpy(dtype=float), va["y5"].to_numpy(int),
                params, n_estimators_search, random_state,
            )
            fold_maes.append(mae)
            fold_kappas.append(kappa)
        rows.append({**params, "inner_cv_MAE": np.mean(fold_maes),
                     "inner_cv_kappa": np.mean(fold_kappas)})

    grid_result = pd.DataFrame(rows).sort_values("inner_cv_MAE")
    best_params = {k: grid_result.iloc[0][k] for k in keys}

    # DataFrame을 거치며 min_samples_leaf 등이 numpy.float64로 바뀌는데,
    # orf/sklearn은 min_samples_leaf에 float(0~1 사이 비율 의미)이 오면
    # "정수 1 이상 또는 (0,1) 사이 실수"라는 규칙에 걸려 에러를 낸다.
    # 원래 int로 넣은 값은 다시 int로 되돌리고, max_features는 None
    # 복원, sample_fraction만 float로 명확히 캐스팅한다.
    if "min_samples_leaf" in best_params:
        best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    if pd.isna(best_params.get("max_features")):
        best_params["max_features"] = None
    elif "max_features" in best_params:
        best_params["max_features"] = float(best_params["max_features"])
    if "sample_fraction" in best_params:
        best_params["sample_fraction"] = float(best_params["sample_fraction"])

    return best_params, grid_result


def run_b3_v2_nested_cv(df: pd.DataFrame, input_items: list[str],
                         param_grid: dict | None = None,
                         random_state: int = RANDOM_STATE):
    param_grid = param_grid if param_grid is not None else PARAM_GRID
    df = df.dropna(subset=["ds_stage"]).copy()
    df["y5"] = collapse_stage_for_b3(df["ds_stage"])

    fold_metrics, calib_tables, tuning_tables, chosen_params_rows = [], [], [], []

    for held_out in TARGET_STUDIES:
        train_df = df[df["STUDYID"] != held_out].reset_index(drop=True)
        test_df = df[df["STUDYID"] == held_out].reset_index(drop=True)

        n_combos = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"[{held_out} held-out] 안쪽 5-fold CV로 하이퍼파라미터 탐색 중"
              f"(조합 {n_combos}개 x {N_INNER_SPLITS} folds)...")
        best_params, grid_result = inner_cv_tune(train_df, input_items,
                                                  param_grid=param_grid, random_state=random_state)
        grid_result["held_out_STUDYID"] = held_out
        tuning_tables.append(grid_result)
        chosen_params_rows.append({"held_out_STUDYID": held_out, **best_params})
        print(f"  선택된 파라미터: {best_params} "
              f"(inner CV MAE={grid_result.iloc[0]['inner_cv_MAE']:.4f})")

        # 선택된 하이퍼파라미터로 학습 2개 시험 전체 재적합(n_estimators는 v1과 동일하게 1000)
        Xtr, Xte, _ = impute_train_median(train_df[input_items], test_df[input_items])
        model = orf.OrderedForest(
            n_estimators=N_ESTIMATORS_FINAL,
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            sample_fraction=best_params["sample_fraction"],
            replace=True, honesty=False, random_state=random_state,
        )
        model.fit(X=Xtr.to_numpy(dtype=float), y=train_df["y5"].to_numpy(int))

        pred = model.predict(X=Xte.to_numpy(dtype=float), prob=True)
        probs = pred.get("predictions")
        y_pred = median_class_from_probs(probs, CLASSES)
        y_true = test_df["y5"].to_numpy(int)

        mae = mean_absolute_error(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        acc_pm1 = (np.abs(y_true - y_pred) <= 1).mean()
        under = underestimation_metrics(y_true, y_pred, top_two_classes=(4, 5))

        fold_metrics.append({
            "held_out_STUDYID": held_out, "n": len(test_df),
            "MAE": mae, "weighted_kappa": kappa, "acc_within_1stage": acc_pm1,
            **under, **{f"chosen_{k}": v for k, v in best_params.items()},
        })

        calib = per_stage_calibration(y_true, y_pred, CLASSES)
        calib["held_out_STUDYID"] = held_out
        calib_tables.append(calib)

    metrics_df = pd.DataFrame(fold_metrics)
    calib_df = pd.concat(calib_tables, ignore_index=True)
    tuning_df = pd.concat(tuning_tables, ignore_index=True)
    chosen_df = pd.DataFrame(chosen_params_rows)
    return metrics_df, calib_df, tuning_df, chosen_df


if __name__ == "__main__":
    n_combos_total = 1
    for v in PARAM_GRID.values():
        n_combos_total *= len(v)
    print(f"전체 그리드 {n_combos_total}콤보 x {N_INNER_SPLITS}fold x 3개 held-out 시험 "
          f"= 약 {n_combos_total * N_INNER_SPLITS * 3}회 적합(n_estimators={N_ESTIMATORS_SEARCH}) "
          f"+ 최종 재적합 3회(n_estimators={N_ESTIMATORS_FINAL}). "
          f"실측 기준 콤보당 약 1.1~1.2초 소요 — 전체 약 10~12분 예상.")

    data = load_baseline_features()
    input_items = data.attrs["input_items"]

    metrics_df, calib_df, tuning_df, chosen_df = run_b3_v2_nested_cv(data, input_items)

    print("\n=== B3 v2(중첩CV 튜닝) leave-one-trial-out 성능 ===")
    print(metrics_df[["held_out_STUDYID", "n", "MAE", "weighted_kappa",
                       "acc_within_1stage"]].to_string(index=False))
    print(f"\n3개 시험 동일가중 평균 MAE: {metrics_df['MAE'].mean():.3f}")
    print(f"3개 시험 동일가중 평균 가중카파: {metrics_df['weighted_kappa'].mean():.3f}")

    print("\n=== fold별 선택된 하이퍼파라미터 ===")
    print(chosen_df.to_string(index=False))

    print("\n=== v1(튜닝 없음, n_estimators=1000) 대비 참고 ===")
    print("v1: MAE=0.388, 가중카파=0.564 (cpad_dependence_aim2_b3_ordinal_rf.py 결과)")

    metrics_df.to_csv("b3_v2_ordered_forest_metrics.csv", index=False, encoding="utf-8-sig")
    calib_df.to_csv("b3_v2_ordered_forest_calibration.csv", index=False, encoding="utf-8-sig")
    tuning_df.to_csv("b3_v2_tuning_grid_results.csv", index=False, encoding="utf-8-sig")

    # max_features=None이 DataFrame/CSV에서는 NaN으로 보여 "결측"과 헷갈릴 수
    # 있어, 저장용 사본에서만 문자열로 명시한다(반환되는 chosen_df 자체는 그대로 둠).
    chosen_df_csv = chosen_df.copy()
    chosen_df_csv["max_features"] = chosen_df_csv["max_features"].apply(
        lambda x: "all_items(None)" if pd.isna(x) else x
    )
    chosen_df_csv.to_csv("b3_v2_chosen_params.csv", index=False, encoding="utf-8-sig")
