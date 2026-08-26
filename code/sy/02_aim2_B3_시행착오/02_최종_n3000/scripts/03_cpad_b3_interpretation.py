"""
Aim 2 — B3(순서형 랜덤포레스트) 해석 도구
================================================
RF 앙상블 자체에는 회귀계수 같은 게 없으므로, 문항별 "기여·영향"을 보려면
아래 세 가지 사후(post-hoc) / 모형-특화 도구를 따로 계산해야 한다.

1. formal marginal effects (orf.margins) — 각 문항 점수를 평가점 근방에서
   조금 움직였을 때 각 단계 확률 P(Y=m|X)이 얼마나 바뀌는지, 표준오차·p-value와
   함께 추정한다. RF 기반 방법 중에서는 회귀계수에 가장 가까운 형태의 산출물이다.
2. permutation importance — 문항 하나의 값을 무작위로 섞었을 때 held-out
   예측성능(MAE)이 얼마나 나빠지는지로 "그 문항이 없으면 얼마나 못 맞히는가"를
   본다. 방향(+/-)은 안 주고 중요도 크기만 준다.
3. partial dependence(PDP) — 한 문항의 값을 관측범위 전체로 훑으며(나머지
   문항은 각 개인의 실제값 고정) 예측 기댓값이 어떻게 변하는지 곡선으로 그린다.
   margins()가 국소적(evaluation point 근방) 효과라면, PDP는 전체 구간의
   비선형 형태까지 보여준다.

주의: 세 가지 모두 "이 데이터·이 모형에서 관측된 연관성"이지 인과효과가
아니다. ADCS-ADL 문항들은 서로 강하게 상관돼 있어(같은 IADL/BADL 기능을
다른 각도로 묻는 문항이 많음), 한 문항만 실제로 바꾸는 개입을 했을 때도
같은 크기의 변화가 나타난다고 해석해서는 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from cpad_dependence_aim2_b3_ordinal_rf import (
    load_baseline_features, collapse_stage_for_b3, impute_train_median,
    median_class_from_probs, orf,
)

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 0. 해석용 모형 — 전체 표본으로 재학습(성능평가용 아님, 해석 전용)
# ----------------------------------------------------------------------
def fit_full_model(df: pd.DataFrame, input_items: list[str], honesty: bool = True,
                    n_estimators: int | None = None):
    """margins()의 표준오차 추정에는 honesty=True(+ subsampling)가 필요하므로
    해석용 모형은 성능평가(run_b3_ordered_forest, honesty=False)와 설정을
    다르게 둔다. 두 모형은 목적이 다르며 서로 대체하지 않는다.

    honesty=True + inference=True는 트리 수·평가 문항 수에 거의 선형으로
    비례해 느려진다(2,208명 기준 문항당 약 9~25초, 트리 수에 비례). 그래서
    margins() 계산용 모형은 기본 트리 수를 200으로 낮췄고(성능평가용 모형의
    1000보다 적어 개별 marginal effect의 표준오차가 다소 크게 나올 수 있음),
    margins()도 permutation importance 상위 문항으로만 제한해서 부른다
    (아래 main 참고). 계산자원이 충분하면 n_estimators를 올려 재현할 것."""
    df = df.dropna(subset=["ds_stage"]).copy()
    df["y5"] = collapse_stage_for_b3(df["ds_stage"])
    X = df[input_items].fillna(df[input_items].median())
    y = df["y5"].to_numpy(dtype=int)

    n_estimators = n_estimators or (200 if honesty else 3000)  # honesty=False는 v1과 통일
    model = orf.OrderedForest(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        replace=False,
        sample_fraction=0.5,
        honesty=honesty,
        honesty_fraction=0.5,
        inference=honesty,
        random_state=RANDOM_STATE,
    )
    model.fit(X=X.to_numpy(dtype=float), y=y)
    return model, X, y


# ----------------------------------------------------------------------
# 1. Formal marginal effects (orf 내장) — RF 기반 방법 중 계수에 가장 가까움
# ----------------------------------------------------------------------
def compute_marginal_effects(model, X: pd.DataFrame, input_items: list[str],
                              eval_items: list[str] | None = None) -> pd.DataFrame:
    """eval_items를 주면 그 문항들만 계산한다(문항당 약 20~25초 소요되므로
    전체 23개를 다 돌리면 8분 이상 걸림 — permutation importance 상위
    문항으로만 제한해 부르는 것을 권장)."""
    eval_items = eval_items or input_items
    eval_idx = [input_items.index(it) for it in eval_items]

    result = model.margins(X_eval=eval_idx, eval_point="mean", window=0.1, verbose=False)
    effects = np.asarray(result.get("effects"))      # shape (n_eval_items, n_classes)
    pvalues = result.get("p-values")
    pvalues = np.asarray(pvalues) if pvalues is not None else None

    rows = []
    for i, item in enumerate(eval_items):
        row = {"item": item}
        if effects.ndim == 2:
            for c in range(effects.shape[1]):
                row[f"effect_class{c+1}"] = effects[i, c]
                if pvalues is not None:
                    row[f"pval_class{c+1}"] = pvalues[i, c]
        else:
            row["effect"] = effects[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. Permutation importance — 문항을 섞었을 때 held-out MAE 악화 정도
# ----------------------------------------------------------------------
def permutation_importance_b3(df: pd.DataFrame, input_items: list[str],
                               n_repeats: int = 20, test_size: float = 0.2,
                               random_state: int = RANDOM_STATE) -> pd.DataFrame:
    df = df.dropna(subset=["ds_stage"]).copy()
    df["y5"] = collapse_stage_for_b3(df["ds_stage"])
    classes = np.array([1, 2, 3, 4, 5])

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["y5"],
    )
    Xtr_raw, Xte_raw = train_df[input_items], test_df[input_items]
    Xtr, Xte, _ = impute_train_median(Xtr_raw, Xte_raw)
    ytr, yte = train_df["y5"].to_numpy(int), test_df["y5"].to_numpy(int)

    model = orf.OrderedForest(
        n_estimators=3000, min_samples_leaf=5,   # v1 최종 결정(2026-08-10)과 통일
        replace=True, honesty=False, random_state=random_state,
    )
    model.fit(X=Xtr.to_numpy(dtype=float), y=ytr)

    def predict_mae(Xmat: np.ndarray) -> float:
        pred = model.predict(X=Xmat, prob=True)
        probs = pred.get("predictions")
        y_pred = median_class_from_probs(probs, classes)
        return mean_absolute_error(yte, y_pred)

    baseline_mae = predict_mae(Xte.to_numpy(dtype=float))

    # orf.predict()는 호출당 고정 오버헤드가 커서(트리 3000개 기준 약 0.45초),
    # 문항당 n_repeats번 따로 부르면 매우 느려진다. n_repeats개의 순열버전을
    # 세로로 이어붙여 문항당 predict() 호출을 1번으로 줄인다(반복 횟수가
    # 아니라 호출 횟수를 줄이는 최적화 — 결과는 수학적으로 동일함).
    rng = np.random.default_rng(random_state)
    n_test = len(Xte)
    rows = []
    for item in input_items:
        Xte_rep = pd.concat([Xte] * n_repeats, ignore_index=True)
        perm_col = np.concatenate([
            rng.permutation(Xte[item].to_numpy()) for _ in range(n_repeats)
        ])
        Xte_rep[item] = perm_col

        pred = model.predict(X=Xte_rep.to_numpy(dtype=float), prob=True)
        probs = pred.get("predictions")
        y_pred_all = median_class_from_probs(probs, classes)

        deltas = []
        for r in range(n_repeats):
            y_pred_r = y_pred_all[r * n_test:(r + 1) * n_test]
            deltas.append(mean_absolute_error(yte, y_pred_r) - baseline_mae)

        rows.append({
            "item": item,
            "importance_mae_increase_mean": np.mean(deltas),
            "importance_mae_increase_std": np.std(deltas),
        })

    out = pd.DataFrame(rows).sort_values("importance_mae_increase_mean", ascending=False)
    out.attrs["baseline_mae"] = baseline_mae
    return out


# ----------------------------------------------------------------------
# 3. Partial dependence — 문항값 전 구간에 대한 예측 기댓값 곡선
# ----------------------------------------------------------------------
def partial_dependence_b3(model, X: pd.DataFrame, item: str,
                           n_grid: int = 10, n_sample: int = 300,
                           random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """item 값을 관측 분위수(0~100%, n_grid개 지점)로 고정하고 나머지 문항은
    무작위 표본 n_sample명의 실제값을 그대로 둔 채 예측해, 예측단계 기댓값
    (E[Y] = sum(m * P(Y=m)))의 평균을 곡선 한 점으로 삼는다."""
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    classes = np.arange(1, 6)

    grid = np.unique(np.quantile(X[item].dropna(), np.linspace(0, 1, n_grid)))
    rows = []
    for val in grid:
        X_mod = X_sample.copy()
        X_mod[item] = val
        pred = model.predict(X=X_mod.to_numpy(dtype=float), prob=True)
        probs = pred.get("predictions")
        expected_stage = (probs * classes).sum(axis=1)
        rows.append({
            "item": item, "value": val,
            "expected_stage_mean": expected_stage.mean(),
            "expected_stage_se": expected_stage.std() / np.sqrt(len(expected_stage)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. 실행
# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = load_baseline_features()
    input_items = data.attrs["input_items"]

    print("1) Permutation importance (held-out 20%, MAE 증가량 기준)...")
    perm_imp = permutation_importance_b3(data, input_items)
    print(f"   baseline MAE: {perm_imp.attrs['baseline_mae']:.3f}")
    print(perm_imp.to_string(index=False))
    perm_imp.to_csv("b3_permutation_importance.csv", index=False, encoding="utf-8-sig")

    top_items_for_margins = perm_imp["item"].head(6).tolist()
    print(f"\n2) 전체 표본 재학습(해석 전용, honesty=True, 500 trees)... "
          f"margins()는 permutation importance 상위 6개 문항만 계산함: {top_items_for_margins}")
    model_full, X_full, y_full = fit_full_model(data, input_items, honesty=True)

    print("3) Formal marginal effects (orf.margins, 상위 6개 문항)...")
    try:
        margins_df = compute_marginal_effects(model_full, X_full, input_items,
                                               eval_items=top_items_for_margins)
        print(margins_df.to_string(index=False))
        margins_df.to_csv("b3_marginal_effects.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"   [경고] margins() 실패: {e} — permutation importance/PDP 결과로 대체 해석할 것")

    print("\n4) Partial dependence — permutation importance 상위 5개 문항...")
    top5_items = perm_imp["item"].head(5).tolist()
    # PDP는 honesty=False(예측 성능 위주) 모형으로도 계산 가능하므로 재사용
    model_pred, X_pred, y_pred_ = fit_full_model(data, input_items, honesty=False)
    pdp_all = []
    for item in top5_items:
        pdp = partial_dependence_b3(model_pred, X_pred, item)
        pdp_all.append(pdp)
        print(f"\n[{item}]")
        print(pdp.to_string(index=False))
    pd.concat(pdp_all, ignore_index=True).to_csv(
        "b3_partial_dependence_top5.csv", index=False, encoding="utf-8-sig"
    )
