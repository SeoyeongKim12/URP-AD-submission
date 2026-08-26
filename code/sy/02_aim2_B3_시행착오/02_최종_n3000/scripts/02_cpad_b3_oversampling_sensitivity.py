"""
B3 — 희소 범주(1·5단계) 오버샘플링 민감도분석
================================================
지난 대화에서 확인한 문제: 예측분포 요약 규칙(median/mode/expected_round)을
바꿔도 1·5단계(0·1통합 5범주 기준) recall이 거의 개선되지 않았다. 원인이
"규칙"이 아니라 "모형이 애초에 희소 클래스에 낮은 확률만 준다"는 데 있다는
가설을 검증하기 위해, 학습 데이터에서 1·5단계를 오버샘플링해 모형이 그
클래스를 더 자주 보게 만들고 recall이 실제로 개선되는지 확인한다.

원칙
----
- 오버샘플링은 학습 fold(2개 시험) 내부에서만 수행하고 held-out 시험에는
  전혀 손대지 않는다(중복 생성된 행이 평가에 섞이면 성능이 부풀려짐).
- median 규칙은 그대로 둔다 — 지난번 답변대로 규칙을 사후에 바꾸는 게 아니라
  모형(학습 데이터 구성) 쪽을 건드리는 것이므로 별도의 민감도분석으로
  자리매김한다. B3 자체가 이미 보조/탐색적 분석이라 이 위에 얹는 것도
  탐색적 성격이 유지된다.
- multiplier(오버샘플링 배수)를 여러 단계로 스윕해서, recall이 얼마나
  오르고 그 대가로 전체 MAE·카파가 얼마나 나빠지는지 트레이드오프 곡선으로
  보여준다 — 특정 배수 하나를 "정답"으로 고르지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from cpad_dependence_aim2_b3_ordinal_rf import (
    load_baseline_features, collapse_stage_for_b3, impute_train_median,
    median_class_from_probs, orf, TARGET_STUDIES,
)

RANDOM_STATE = 42


def oversample_train(train_df: pd.DataFrame, target_classes=(1, 5),
                      multiplier: float = 1.0, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """target_classes(기본 1·5단계)의 행을 복원추출로 늘려 최종 개수가
    원래 개수 * multiplier(반올림)가 되도록 만든다. multiplier=1.0이면
    원본 그대로(오버샘플링 없음). 나머지 클래스는 손대지 않는다."""
    if multiplier <= 1.0:
        return train_df

    rng = np.random.default_rng(random_state)
    extra_parts = []
    for c in target_classes:
        cls_rows = train_df[train_df["y5"] == c]
        n_now = len(cls_rows)
        if n_now == 0:
            continue
        n_target = int(round(n_now * multiplier))
        n_extra = max(0, n_target - n_now)
        if n_extra > 0:
            idx = rng.choice(cls_rows.index.to_numpy(), size=n_extra, replace=True)
            extra_parts.append(train_df.loc[idx])

    if not extra_parts:
        return train_df
    return pd.concat([train_df] + extra_parts, ignore_index=True)


def run_b3_with_oversampling(df: pd.DataFrame, input_items: list[str],
                              multiplier: float = 1.0,
                              target_classes=(1, 5),
                              n_estimators: int = 3000, min_samples_leaf: int = 5,
                              random_state: int = RANDOM_STATE):
    # n_estimators=3000: v1 최종 결정(2026-08-10)과 통일
    df = df.dropna(subset=["ds_stage"]).copy()
    df["y5"] = collapse_stage_for_b3(df["ds_stage"])
    classes = np.array([1, 2, 3, 4, 5])

    all_true, all_pred = [], []
    for held_out in TARGET_STUDIES:
        train_df = df[df["STUDYID"] != held_out].reset_index(drop=True)
        test_df = df[df["STUDYID"] == held_out].reset_index(drop=True)

        train_df_os = oversample_train(train_df, target_classes=target_classes,
                                        multiplier=multiplier, random_state=random_state)

        Xtr_raw, Xte_raw = train_df_os[input_items], test_df[input_items]
        # 결측 대치 기준(중앙값)은 오버샘플링 "이전" 원본 학습 데이터에서 구해야
        # 중복행이 중앙값 계산에 과대 반영되지 않는다.
        _, _, medians = impute_train_median(train_df[input_items], test_df[input_items])
        Xtr = Xtr_raw.fillna(medians)
        Xte = Xte_raw.fillna(medians)

        model = orf.OrderedForest(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            replace=True, honesty=False, random_state=random_state,
        )
        model.fit(X=Xtr.to_numpy(dtype=float), y=train_df_os["y5"].to_numpy(dtype=int))

        pred = model.predict(X=Xte.to_numpy(dtype=float), prob=True)
        probs = pred.get("predictions")
        y_pred = median_class_from_probs(probs, classes)
        y_true = test_df["y5"].to_numpy(dtype=int)

        all_true.append(y_true)
        all_pred.append(y_pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def summarize(y_true, y_pred, multiplier) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    row = {"multiplier": multiplier, "MAE": mae, "weighted_kappa": kappa}
    for c, label in [(1, "stage1"), (5, "stage5")]:
        n_pred = (y_pred == c).sum()
        n_true = (y_true == c).sum()
        precision = (y_true[y_pred == c] == c).mean() if n_pred > 0 else np.nan
        recall = ((y_pred == c) & (y_true == c)).sum() / n_true if n_true > 0 else np.nan
        row[f"{label}_n_predicted"] = n_pred
        row[f"{label}_precision"] = precision
        row[f"{label}_recall"] = recall
    return row


if __name__ == "__main__":
    data = load_baseline_features()
    input_items = data.attrs["input_items"]

    multipliers = [1.0, 3.0, 6.0, 10.0]  # 1.0 = 오버샘플링 없음(기존 결과 재현용 베이스라인)
    results = []
    for m in multipliers:
        print(f"오버샘플링 배수 x{m} 실행 중...")
        y_true, y_pred = run_b3_with_oversampling(data, input_items, multiplier=m)
        row = summarize(y_true, y_pred, m)
        results.append(row)
        print(f"  MAE={row['MAE']:.3f}, kappa={row['weighted_kappa']:.3f} | "
              f"1단계: n={row['stage1_n_predicted']}, recall={row['stage1_recall']:.1%}, "
              f"precision={row['stage1_precision']:.1%} | "
              f"5단계: n={row['stage5_n_predicted']}, recall={row['stage5_recall']:.1%}, "
              f"precision={row['stage5_precision']:.1%}")

    result_df = pd.DataFrame(results)
    print("\n=== 오버샘플링 배수별 트레이드오프 요약 ===")
    print(result_df.to_string(index=False))
    result_df.to_csv("b3_oversampling_sensitivity.csv", index=False, encoding="utf-8-sig")
