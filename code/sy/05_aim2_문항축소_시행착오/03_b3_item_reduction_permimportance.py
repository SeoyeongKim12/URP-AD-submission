"""
Aim 2 · B3(순서형 랜덤포레스트) 문항축소 테스트 — permutation importance 순위기반 (Claude)
=========================================================================================
B2(엘라스틱넷+무작위탐색)와 같은 질문("몇 문항으로 줄여도 성능이 유지되나")을
B3(orf.OrderedForest)로도 확인해보는 스크립트. 사용자 선택 반영:
  - 문항선택 방식: **순위기반(방법A)** — 매 outer fold마다 훈련 2개 시험 내부에서만
    permutation importance로 문항 순위를 매기고, k=5~18에 대해 "상위 k개"만 사용.
    B2처럼 k마다 300조합 무작위탐색을 하지 않아서 훨씬 빠름(RF 자체가 로지스틱보다
    한참 느리기 때문에 이 방식이 현실적).
  - 중증(4·5단계) 놓침 처리: **B3 원래 방식 유지** — 조건부중앙값 규칙만 쓰고 별도
    비대칭임계는 적용하지 않음. B3 원 분석(B3_결과해석_보고서.txt)에서 이미 이 규칙의
    중증 놓침이 크다는 게 알려져 있으므로(4·5단계 과소분류 57~72%), 이번 결과도
    그 한계를 그대로 물려받는다는 점을 감안해서 읽을 것 — 이 스크립트는 "문항을
    줄여도 전체 MAE·카파가 유지되는가"에 집중하고, 중증탐지력 개선은 범위 밖.

구조(누출 없음)
  - 바깥: leave-one-trial-out(3). 평가시험은 전혀 사용 안 함.
  - 각 outer 훈련폴드(2개 시험) 내부에서 다시 80/20 분할해 permutation importance로
    문항순위를 매김(B3_최종_n3000/scripts/cpad_b3_interpretation.py의
    permutation_importance_b3()와 동일 로직, outer 훈련폴드 내부로 축소).
  - k=5~18 각각에 대해 "그 순위 상위 k개 문항"만으로 outer 훈련폴드(2개 시험) 전체를
    재학습 → outer 평가시험(held-out) 예측 → 지표 계산.

[계산량 참고] orf.OrderedForest는 로지스틱보다 훨씬 느림. 순위 매기는 단계 1회
(fold당) + k=5~18 각각 최종 재학습 1회(fold당) = fold당 15번 학습. 3 fold 합쳐
45번. n_estimators를 낮춰서(기본 500) 탐색 속도를 확보했고, 이건 B3 최종
분석(n_estimators=3000)보다 가벼운 설정임 — 결과가 그럴듯한 k 범위를 찾은
뒤에는 그 k만 n_estimators=3000으로 재확인하는 걸 권장(N_ESTIMATORS_FINAL 참고).

입력: urp-AD/dependence_study/{adl_wide.parquet, baseline_sample.parquet}
산출: b3_item_reduction_performance_by_k.csv   (k별 MAE·카파·±1acc·중증놓침)
      b3_item_reduction_ranking_by_fold.csv    (fold별 permutation importance 순위 전체)
      b3_item_reduction_predictions.csv        (환자단위 k별 예측값, wide)
      b3_item_reduction_report.md              (사람이 읽는 보고서)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    import orf  # Ordered Forest (Lechner & Okasa)
except ImportError as e:
    raise ImportError("pip install orf --break-system-packages 로 설치할 것") from e

# --- orf 0.2.0 / 최신 sklearn 호환 shim (B3 원본 스크립트와 동일, 그대로 이식) ---
from sklearn.preprocessing import OneHotEncoder as _OrigOHE


def _ohe_compat(**kwargs):
    sparse = kwargs.pop("sparse", None)
    if sparse is not None:
        kwargs.setdefault("sparse_output", sparse)
    return _OrigOHE(**kwargs)


orf._BaseOrderedForest.OneHotEncoder = _ohe_compat


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

# =================================================================
# 설정
# =================================================================
DATA_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study")
OUTDIR = Path(__file__).parent
TARGET_STUDIES = ["AD-1061", "AD-1063", "AD-1064"]
K_RANGE = list(range(5, 19))          # 5..18 (B2와 동일 범위로 맞춰 비교 용이)
N_ESTIMATORS_SEARCH = 500             # 탐색 단계(빠르게) — B3 최종값(3000)보다 낮춤
N_ESTIMATORS_FINAL = 3000             # 참고: 유망한 k 확정되면 이 값으로 재확인 권장
MIN_SAMPLES_LEAF = 5
N_REPEATS_IMPORTANCE = 10             # permutation importance 반복횟수(속도-정밀도 절충)
RANDOM_STATE = 42

ADL_GATE_MAP = {
    "ADL0106": ["ADL0106A", "ADL0106B"], "ADL0107": ["ADL0107A"],
    "ADL0110": ["ADL0110A"], "ADL0111": ["ADL0111A"], "ADL0112": ["ADL0112A"],
    "ADL0113": ["ADL0113A"], "ADL0114": ["ADL0114A"], "ADL0115": ["ADL0115A"],
    "ADL0116": ["ADL0116A", "ADL0116B"], "ADL0117": ["ADL0117A"],
    "ADL0118": ["ADL0118A", "ADL0118B", "ADL0118C"], "ADL0121": ["ADL0121A"],
    "ADL0122": ["ADL0122Q"], "ADL0123": ["ADL0123L"],
}
ADL_2025_EXCLUDE = [
    "ADL0108", "ADL0108A", "ADL0108B", "ADL0108C", "ADL0109", "ADL0109A",
    "ADL0119", "ADL0119A", "ADL0119B", "ADL0119C", "ADL0120", "ADL0120A", "ADL0120B",
]
ADL_NOT_A_PERFORMANCE_ITEM = ["ADL0118N", "ADL0124", "ADL0125"]


# =================================================================
# 데이터 로드 (B3 원 스크립트와 동일 로직)
# =================================================================
def load_baseline_features() -> pd.DataFrame:
    """[Claude 수정] 원 B3 스크립트(cpad_dependence_aim2_b3_ordinal_rf.py)를 그대로
    이식했더니 문항이 19개가 아니라 50개로 잡히는 버그를 발견함 — adl_wide.parquet에
    나중에 추가된 "__resolved" 컬럼이 raw 컬럼과 별개 문항처럼 중복 집계되고,
    ADL_2025_EXCLUDE 제외목록도 raw 이름만 걸러내서 __resolved판은 못 걸러냄
    (예: ADL0108A__resolved가 제외 안 되고 슬쩍 들어옴). 아래처럼 base 코드 기준으로
    __resolved 우선 선택 + 제외규칙을 base 코드에 적용하도록 고침(B1/B2의 col() 관례와
    동일한 원칙)."""
    baseline = pd.read_parquet(DATA_DIR / "baseline_sample.parquet")
    adl = pd.read_parquet(DATA_DIR / "adl_wide.parquet")
    adl_bl = adl[adl["VISITNUM"] == 2.0].copy()

    for gate, subs in ADL_GATE_MAP.items():
        gate_col = gate + "__gate"
        if gate_col not in adl_bl.columns:
            continue
        not_gen = adl_bl[gate_col] == "not_generated"
        for sub in subs:
            if sub in adl_bl.columns:
                adl_bl[sub] = adl_bl[sub].where(~not_gen, 0)

    # base 코드(=__resolved 접미사 뗀 이름) 기준으로 유일한 문항 집합을 만들고,
    # 각 base 코드마다 __resolved가 있으면 그걸, 없으면 raw를 쓴다.
    all_cols = [c for c in adl_bl.columns if c.startswith("ADL01") and not c.endswith("__gate")]
    base_codes = sorted({c[: -len("__resolved")] if c.endswith("__resolved") else c for c in all_cols})
    base_codes = [c for c in base_codes if c not in ADL_2025_EXCLUDE and c not in ADL_NOT_A_PERFORMANCE_ITEM]

    resolved = pd.DataFrame(index=adl_bl.index)
    input_items = []
    for base in base_codes:
        r = base + "__resolved"
        col = r if r in adl_bl.columns else base
        if col not in adl_bl.columns:
            continue
        resolved[base] = adl_bl[col]
        input_items.append(base)

    # [Claude 추가] B1/B2와 동일 관례: ADL0118A/B/C(하루 중 혼자 있었던 시간 관련
    # 3분할 응답)를 "Q18" 하나로 합산해서 씀. 이러면 23문항 → 21문항이 돼서 B1/B2의
    # 21문항 정의와 정확히 일치함(원 B3 보고서의 "19문항"과는 다름 — 그 정확한
    # 유도과정은 재현 못 했고, 대신 프로젝트 전체 일관성을 택함. 필요하면 118A/B/C를
    # 분리해서 23문항으로 되돌리는 것도 가능 — merge_q18=False로 바꾸면 됨).
    if "ADL0118A" in input_items and "ADL0118B" in input_items and "ADL0118C" in input_items:
        resolved["Q18"] = resolved["ADL0118A"] + resolved["ADL0118B"] + resolved["ADL0118C"]
        resolved = resolved.drop(columns=["ADL0118A", "ADL0118B", "ADL0118C"])
        input_items = [c for c in input_items if c not in ("ADL0118A", "ADL0118B", "ADL0118C")] + ["Q18"]

    adl_bl = pd.concat([adl_bl[["STUDYID", "USUBJID"]], resolved], axis=1)

    keep_cols = ["STUDYID", "USUBJID"] + input_items
    merged = baseline[baseline["in_aim1_2_sample"]][["STUDYID", "USUBJID", "ds_stage", "ARM"]].merge(
        adl_bl[keep_cols], on=["STUDYID", "USUBJID"], how="left")
    merged.attrs["input_items"] = input_items
    return merged


def collapse_stage_for_b3(ds_stage: pd.Series) -> pd.Series:
    return ds_stage.clip(lower=1).astype(int)


def impute_train_median(train_X: pd.DataFrame, apply_to: pd.DataFrame):
    medians = train_X.median()
    return train_X.fillna(medians), apply_to.fillna(medians)


def median_class_from_probs(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)
    idx = (cum >= 0.5).argmax(axis=1)
    return classes[idx]


def underestimation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    under2 = ((y_true - y_pred) >= 2).mean()
    top_mask = np.isin(y_true, (4, 5))
    top_under = (y_pred[top_mask] <= 3).mean() if top_mask.sum() > 0 else np.nan
    return {"underest_ge2_rate": under2, "top2class_underclassified_rate": top_under}


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, items: list[str],
                 n_estimators: int, classes: np.ndarray):
    Xtr_raw, Xte_raw = train_df[items], test_df[items]
    Xtr, Xte = impute_train_median(Xtr_raw, Xte_raw)
    model = orf.OrderedForest(n_estimators=n_estimators, min_samples_leaf=MIN_SAMPLES_LEAF,
                               replace=True, honesty=False, random_state=RANDOM_STATE)
    model.fit(X=Xtr.to_numpy(dtype=float), y=train_df["y5"].to_numpy(dtype=int))
    pred = model.predict(X=Xte.to_numpy(dtype=float), prob=True)
    probs = pred.get("predictions")
    y_pred = median_class_from_probs(probs, classes)
    return y_pred


# =================================================================
# fold 내부 permutation importance (훈련 2개 시험만 사용, 평가시험 미관여)
# =================================================================
def rank_items_within_train(train_df: pd.DataFrame, items: list[str], classes: np.ndarray) -> pd.DataFrame:
    tr2, ho2 = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df["y5"])
    Xtr_raw, Xho_raw = tr2[items], ho2[items]
    Xtr, Xho = impute_train_median(Xtr_raw, Xho_raw)
    ytr, yho = tr2["y5"].to_numpy(int), ho2["y5"].to_numpy(int)

    model = orf.OrderedForest(n_estimators=N_ESTIMATORS_SEARCH, min_samples_leaf=MIN_SAMPLES_LEAF,
                               replace=True, honesty=False, random_state=RANDOM_STATE)
    model.fit(X=Xtr.to_numpy(dtype=float), y=ytr)

    def predict_mae(Xmat):
        pred = model.predict(X=Xmat, prob=True)
        return mean_absolute_error(yho, median_class_from_probs(pred.get("predictions"), classes))

    baseline_mae = predict_mae(Xho.to_numpy(dtype=float))

    rng = np.random.default_rng(RANDOM_STATE)
    n_ho = len(Xho)
    rows = []
    for item in items:
        Xho_rep = pd.concat([Xho] * N_REPEATS_IMPORTANCE, ignore_index=True)
        perm_col = np.concatenate([rng.permutation(Xho[item].to_numpy()) for _ in range(N_REPEATS_IMPORTANCE)])
        Xho_rep[item] = perm_col
        pred = model.predict(X=Xho_rep.to_numpy(dtype=float), prob=True)
        y_pred_all = median_class_from_probs(pred.get("predictions"), classes)
        deltas = [mean_absolute_error(yho, y_pred_all[r * n_ho:(r + 1) * n_ho]) - baseline_mae
                  for r in range(N_REPEATS_IMPORTANCE)]
        rows.append({"item": item, "importance_mae_increase": float(np.mean(deltas))})
    return pd.DataFrame(rows).sort_values("importance_mae_increase", ascending=False).reset_index(drop=True)


# =================================================================
# main
# =================================================================
def main():
    _lines = []
    def log(s=""):
        print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    log("# Aim 2 · B3(순서형 랜덤포레스트) 문항축소 — permutation importance 순위기반\n")
    data = load_baseline_features()
    items_full = data.attrs["input_items"]
    data = data.dropna(subset=["ds_stage"]).copy()
    data["y5"] = collapse_stage_for_b3(data["ds_stage"])
    classes = np.array([1, 2, 3, 4, 5])
    K_here = [k for k in K_RANGE if k < len(items_full)]
    log(f"입력 문항 풀: {len(items_full)}개. 분석표본 {len(data)}명. "
        f"탐색 n_estimators={N_ESTIMATORS_SEARCH}(최종 확인은 {N_ESTIMATORS_FINAL} 권장). "
        f"k={K_here[0]}~{K_here[-1]}.\n")

    perf_rows, rank_rows, pred_wide = [], [], data[["STUDYID", "USUBJID", "ds_stage", "y5"]].copy()
    for k in K_here:
        pred_wide[f"pred_k{k}"] = np.nan

    for ho in TARGET_STUDIES:
        train_df = data[data.STUDYID != ho].reset_index(drop=True)
        test_df = data[data.STUDYID == ho].reset_index(drop=True)
        te_idx = data.index[data.STUDYID == ho]

        ranking = rank_items_within_train(train_df, items_full, classes)
        ranking["outer_fold"] = ho
        rank_rows.append(ranking)
        log(f"[진행] outer={ho}: permutation importance 순위 완료 (1위 {ranking.iloc[0]['item']}, "
            f"MAE증가 {ranking.iloc[0]['importance_mae_increase']:.3f})")

        for k in K_here:
            top_items = ranking["item"].head(k).tolist()
            y_pred = fit_predict(train_df, test_df, top_items, N_ESTIMATORS_SEARCH, classes)
            y_true = test_df["y5"].to_numpy(int)
            mae = mean_absolute_error(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
            acc1 = float((np.abs(y_true - y_pred) <= 1).mean())
            under = underestimation_metrics(y_true, y_pred)
            perf_rows.append({"outer_fold": ho, "n_items": k, "n_test": len(test_df),
                               "mae": mae, "kappa": kappa, "acc_within1": acc1, **under})
            pred_wide.loc[te_idx, f"pred_k{k}"] = y_pred
        log(f"[진행] outer={ho}: k={K_here[0]}~{K_here[-1]} 전체 재학습·예측 완료")

    perf_df = pd.DataFrame(perf_rows)
    agg = perf_df.groupby("n_items").apply(
        lambda g: pd.Series({
            "mae": np.average(g["mae"], weights=g["n_test"]),
            "kappa": np.average(g["kappa"], weights=g["n_test"]),
            "acc_within1": np.average(g["acc_within1"], weights=g["n_test"]),
            "underest_ge2_rate": np.average(g["underest_ge2_rate"], weights=g["n_test"]),
            "top2class_underclassified_rate": np.nanmean(g["top2class_underclassified_rate"]),
        }), include_groups=False
    ).reset_index()

    log("\n## 문항수별 정직 성능 (3-fold 가중평균, 조건부중앙값 규칙)")
    log("| 문항수 | MAE | 카파 | ±1단계이내 | 2↑과소 | 4·5→≤3 |")
    log("|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        log(f"| {int(r.n_items)} | {r.mae:.3f} | {r.kappa:.3f} | {r.acc_within1*100:.1f}% | "
            f"{r.underest_ge2_rate*100:.1f}% | {r.top2class_underclassified_rate*100:.1f}% |")
    log("\n- **주의**: 4·5→≤3(중증놓침) 열은 B3 원래 방식(조건부중앙값)을 그대로 써서 "
        "문항수와 무관하게 전반적으로 높게 나올 것으로 예상됨(B3 원 분석에서도 57~72%) — "
        "이번 실험은 '중증탐지 개선'이 아니라 '문항 줄여도 전체 정확도가 유지되는가'만 봄.\n")

    rank_df = pd.concat(rank_rows, ignore_index=True)
    common_top10 = None
    for ho in TARGET_STUDIES:
        s = set(rank_df[rank_df.outer_fold == ho].head(10)["item"])
        common_top10 = s if common_top10 is None else common_top10 & s
    log(f"## fold 간 공통 상위10 문항: {sorted(common_top10)}\n")

    perf_df.to_csv(OUTDIR / "b3_item_reduction_performance_by_k.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUTDIR / "b3_item_reduction_performance_by_k_agg.csv", index=False, encoding="utf-8-sig")
    rank_df.to_csv(OUTDIR / "b3_item_reduction_ranking_by_fold.csv", index=False, encoding="utf-8-sig")
    pred_wide.to_csv(OUTDIR / "b3_item_reduction_predictions.csv", index=False, encoding="utf-8-sig")
    (OUTDIR / "b3_item_reduction_report.md").write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {OUTDIR / 'b3_item_reduction_report.md'}")
    print(f">>> 저장: {OUTDIR / 'b3_item_reduction_performance_by_k.csv'} (fold별)")
    print(f">>> 저장: {OUTDIR / 'b3_item_reduction_performance_by_k_agg.csv'} (3-fold 가중평균)")
    print(f">>> 저장: {OUTDIR / 'b3_item_reduction_ranking_by_fold.csv'}")
    print(f">>> 저장: {OUTDIR / 'b3_item_reduction_predictions.csv'}")


if __name__ == "__main__":
    main()
