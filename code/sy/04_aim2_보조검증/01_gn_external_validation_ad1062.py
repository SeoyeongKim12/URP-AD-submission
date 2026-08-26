"""
Aim 2 · Gn 독립 외부검증 — AD-1062 (sjlee)
============================================================
목적
  지금까지 4조건 통과 판정은 전부 AD-1061/1063/1064 3개 시험 내부
  leave-one-trial-out CV 결과였음 — 같은 3개 시험을 계속 재사용한 것이라
  "우연히 이 조합에 잘 맞은 것"일 위험이 남아있음.
  AD-1062(공개연장시험)는 학습·튜닝·CV 어디에도 한 번도 쓰인 적 없는
  진짜 독립표본이라, 여기서 4조건이 유지되는지가 최종 관문임.

절차 (정보누출 0)
  1) AD-1061/1063/1064 3개 시험 전체를 훈련풀로 묶어 "최종 배포용" 단일
     Gn 모형을 만듦 — (C, margin)은 3개 시험 안에서 leave-one-trial-out으로
     한 번 더 검증해 다수결로 고르고(과거 nested_gn_en과 동일 원리),
     τ4·τ5도 이 풀 전체 데이터로 한 번만 튜닝함.
  2) 이렇게 "완전히 확정된" 모형·계수·임계값을 AD-1062에는 딱 한 번만
     적용해서 예측함 — AD-1062는 하이퍼파라미터 선택에 절대 관여하지 않음.
  3) AD-1062의 실측 ds_stage, 그리고 이미 계산돼 있는 A1_2015_stage와
     비교해 동일한 4조건(MAE감소·2↑과소·4·5→≤3·MAE감소시험수)을 재평가.

입력
  훈련(3개 시험): {TRAIN_DIR}/adl_wide.parquet, baseline_sample.parquet
  외부검증(AD-1062): {EXT_DIR}/adl_wide_ad1062.csv, ds_wide_ad1062.csv
출력
  {OUT_DIR}/gn_ad1062_predictions.csv   (환자단위 예측)
  {OUT_DIR}/gn_ad1062_metrics.csv       (Gn vs A1, 4조건 평가표)
  {OUT_DIR}/gn_final_hyperparams.csv    (최종 확정 C·margin·tau4·tau5)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# =================================================================
# 설정
# =================================================================
TRAIN_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study")
EXT_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study_csv\extra_validation_ad1062")
OUT_DIR = Path(__file__).parent

TRAIN_TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)

L1_RATIO = 0.5
C_GRID = [0.1, 0.25, 0.5, 1.0]
MARGIN_GRID = [0.10, 0.12, 0.14]
TAU_GRID = np.round(np.arange(0.10, 0.55, 0.025), 3)

BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = [
    "ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
    "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
    "ADL0121A", "ADL0122Q", "ADL0123L",
]
ITEMS = BADL + IADL


def hr(title: str = "", ch: str = "="):
    print(ch * 70)
    if title:
        print(title)
        print(ch * 70)


# =================================================================
# step 0. 데이터 로드
# =================================================================
def load_train_pool() -> pd.DataFrame:
    """AD-1061/1063/1064 — 훈련·하이퍼파라미터선택용 (parquet, __resolved 포함)."""
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
    m = m.dropna(subset=["ds_stage"])
    return m


def load_ad1062() -> pd.DataFrame:
    """AD-1062 — 학습에 전혀 관여하지 않는 순수 외부검증 표본 (CSV).
    baseline = VISITNUM 1.0 ("Baseline II", 계획서·전처리 스크립트 주석 확인).
    adl_wide_ad1062.csv엔 이미 __resolved / A1_2015_stage가 계산돼 저장돼 있음."""
    adl = pd.read_csv(EXT_DIR / "adl_wide_ad1062.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 1.0].copy()

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
    X["A1_2015_stage"] = b["A1_2015_stage"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])

    ds = pd.read_csv(EXT_DIR / "ds_wide_ad1062.csv", low_memory=False)
    ds_bl = ds[ds["VISITNUM"] == 1.0][["STUDYID", "USUBJID", "ds_stage", "ds_complete13"]]

    m = ds_bl.merge(X, on=["STUDYID", "USUBJID"], how="left")
    m = m.dropna(subset=["ds_stage", "A1_2015_stage"] + ITEMS, how="any")
    # ds_stage는 dropna로 이미 걸러지지만, 문항 결측까지 전부 없는(완전관측) 행만
    # 남기면 median대치 필요성 자체가 줄어 외부검증이 더 깨끗해짐. 문항결측이
    # 있는 행은 훈련때 imputer로 대치해서 살릴 수도 있으나, 외부검증은 최대한
    # 보수적으로(완전관측만) 가는 편이 해석에 유리해 이렇게 둠.
    return m


# =================================================================
# 공통 함수 (원 Gn 코드와 동일)
# =================================================================
def cond_median(P):
    return STAGES[(np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)]


def asym_rule(P, t4, t5):
    p5 = P[:, 5]
    p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cond_median(P)))


def m_mae(t, p):
    return np.abs(np.asarray(t) - np.asarray(p)).mean()


def m_u2(t, p):
    return (np.asarray(t) - np.asarray(p) >= 2).mean()


def m_hi(t, p):
    t = np.asarray(t)
    p = np.asarray(p)
    hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan


def fit_en(train_df: pd.DataFrame, C: float):
    """절단별 elastic-net 이진로지스틱. 표준화는 train_df 내부에서만 fit."""
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

    def probs(rows: pd.DataFrame):
        g = geq(rows)
        P = np.zeros((len(rows), 6))
        P[:, 0] = 1 - g[1]
        for k in range(1, 5):
            P[:, k] = g[k] - g[k + 1]
        P[:, 5] = g[5]
        P = np.clip(P, 1e-9, None)
        P /= P.sum(1, keepdims=True)
        return P

    return probs, geq, y, models


def tune_tau(Ptr, ytr, budget):
    best = None
    for t4 in TAU_GRID:
        for t5 in TAU_GRID:
            if t5 < t4:
                continue
            p = asym_rule(Ptr, t4, t5)
            mae = m_mae(ytr, p)
            if mae <= budget:
                key = (m_hi(ytr, p), mae)
                if best is None or key < best[0]:
                    best = (key, float(t4), float(t5))
    if best is None:
        # 예산 만족 조합이 하나도 없으면 가장 보수적인(=거의 안 건드리는) 조합으로 폴백
        return 1.0, 1.0
    return best[1], best[2]


# =================================================================
# step 1. 최종 하이퍼파라미터 선택 — 3개 훈련시험 안에서만 (AD-1062 미관여)
# =================================================================
def select_final_hyperparams(train_pool: pd.DataFrame):
    hr("STEP 1. 최종 (C, margin) 선택 — AD-1061/63/64 내부 leave-one-trial-out만 사용")
    score = {}
    for C in C_GRID:
        for mg in MARGIN_GRID:
            npass = 0
            for ho in TRAIN_TRIALS:
                tr = train_pool[train_pool.STUDYID != ho]
                te = train_pool[train_pool.STUDYID == ho]
                probs, _, ytr, _ = fit_en(tr, C)
                budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - mg
                t4, t5 = tune_tau(probs(tr), ytr, budget)
                pv = asym_rule(probs(te), t4, t5)
                a1m = m_mae(te["ds_stage"], te["A1_2015_stage"])
                a1h = m_hi(te["ds_stage"], te["A1_2015_stage"])
                cond1 = (a1m - m_mae(te["ds_stage"], pv)) >= 0.10
                cond3 = (m_hi(te["ds_stage"], pv) - a1h) <= 0.02
                if cond1 and cond3:
                    npass += 1
            score[(C, mg)] = npass
            print(f"  C={C:<5} margin={mg:<5}  통과 {npass}/3")
    best = max(score, key=lambda k: (score[k], -k[0], k[1]))
    print(f"\n선택된 (C, margin) = {best}  (동률시 작은 C·큰 margin 우선, 보수적 선택)")
    return best[0], best[1]


# =================================================================
# step 2. 최종 모형 확정 — 3개 훈련시험 전체로 재학습
# =================================================================
def fit_final_model(train_pool: pd.DataFrame, C: float, margin: float):
    hr("STEP 2. 최종 모형 확정 — AD-1061+1063+1064 전체 풀로 재학습")
    probs_fn, geq_fn, y, models = fit_en(train_pool, C)
    budget = m_mae(train_pool["ds_stage"], train_pool["A1_2015_stage"]) - margin
    t4, t5 = tune_tau(probs_fn(train_pool), y, budget)
    print(f"최종 tau4={t4}, tau5={t5}  (budget={budget:.4f})")
    return probs_fn, t4, t5


# =================================================================
# step 3. AD-1062 외부검증 (단 1회 적용)
# =================================================================
def evaluate_external(ext: pd.DataFrame, probs_fn, t4: float, t5: float) -> dict:
    hr("STEP 3. AD-1062 외부검증 — 확정 모형을 단 1회만 적용")
    P = probs_fn(ext)
    pred = asym_rule(P, t4, t5)
    ext = ext.copy()
    ext["Gn_pred"] = pred

    t = ext["ds_stage"].astype(int)
    gn = ext["Gn_pred"].astype(int)
    a1 = ext["A1_2015_stage"].astype(int)

    gn_mae, a1_mae = m_mae(t, gn), m_mae(t, a1)
    gn_u2, a1_u2 = m_u2(t, gn), m_u2(t, a1)
    gn_hi, a1_hi = m_hi(t, gn), m_hi(t, a1)
    gn_kappa = cohen_kappa_score(t, gn, weights="quadratic", labels=list(range(6)))
    a1_kappa = cohen_kappa_score(t, a1, weights="quadratic", labels=list(range(6)))

    mae_drop = a1_mae - gn_mae
    d_u2 = gn_u2 - a1_u2
    d_hi = gn_hi - a1_hi
    c1 = mae_drop >= 0.10
    c2 = d_u2 <= 0.02
    c3 = d_hi <= 0.02
    allpass = c1 and c2 and c3

    print(f"n = {len(ext)}")
    print(f"MAE        Gn={gn_mae:.3f}  A1={a1_mae:.3f}  (감소 {mae_drop:+.3f})")
    print(f"가중카파    Gn={gn_kappa:.3f}  A1={a1_kappa:.3f}")
    print(f"2↑과소     Gn={gn_u2*100:.1f}%  A1={a1_u2*100:.1f}%  (증가 {d_u2*100:+.1f}%p)")
    print(f"4·5→≤3     Gn={gn_hi*100:.1f}%  A1={a1_hi*100:.1f}%  (증가 {d_hi*100:+.1f}%p)")
    print(f"\n조건1(MAE감소≥0.10) {'PASS' if c1 else 'FAIL'}")
    print(f"조건2(2↑과소 증가≤2%p) {'PASS' if c2 else 'FAIL'}")
    print(f"조건3(4·5→≤3 증가≤2%p) {'PASS' if c3 else 'FAIL'}")
    print(f"\n>>> AD-1062 외부검증 종합: {'PASS — 내부CV 결과가 재현됨' if allpass else 'FAIL — 내부CV 결과가 외부검증에서 재현 안 됨, 과최적화 의심'}")

    ext.to_csv(OUT_DIR / "gn_ad1062_predictions.csv", index=False)

    metrics = pd.DataFrame([
        {"model": "Gn", "n": len(ext), "mae": gn_mae, "kappa": gn_kappa, "under2": gn_u2, "hi_missed": gn_hi},
        {"model": "A1", "n": len(ext), "mae": a1_mae, "kappa": a1_kappa, "under2": a1_u2, "hi_missed": a1_hi},
    ])
    metrics["mae_drop_vs_A1"] = [mae_drop, np.nan]
    metrics["cond1_pass"] = [c1, np.nan]
    metrics["cond2_pass"] = [c2, np.nan]
    metrics["cond3_pass"] = [c3, np.nan]
    metrics["allpass"] = [allpass, np.nan]
    metrics.to_csv(OUT_DIR / "gn_ad1062_metrics.csv", index=False)

    return dict(mae_drop=mae_drop, d_u2=d_u2, d_hi=d_hi, allpass=allpass)


# =================================================================
# main
# =================================================================
def main():
    hr("Aim 2 · Gn 독립 외부검증 — AD-1062 (sjlee)")

    train_pool = load_train_pool()
    print(f"훈련풀(3개 시험): {len(train_pool)}명")
    ext = load_ad1062()
    print(f"외부검증(AD-1062): {len(ext)}명 (완전관측만 사용)\n")

    C, margin = select_final_hyperparams(train_pool)
    probs_fn, t4, t5 = fit_final_model(train_pool, C, margin)

    pd.DataFrame([{"C": C, "margin": margin, "tau4": t4, "tau5": t5}]).to_csv(
        OUT_DIR / "gn_final_hyperparams.csv", index=False
    )

    result = evaluate_external(ext, probs_fn, t4, t5)

    hr("완료")
    print(f"저장: {OUT_DIR / 'gn_ad1062_predictions.csv'}")
    print(f"저장: {OUT_DIR / 'gn_ad1062_metrics.csv'}")
    print(f"저장: {OUT_DIR / 'gn_final_hyperparams.csv'}")
    return result


if __name__ == "__main__":
    main()
