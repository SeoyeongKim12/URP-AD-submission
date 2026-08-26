"""
Aim 2 · Gn 비단조 확률 보정 — 등위회귀(PAVA) (sjlee)
============================================================
목적
  gn_stability_monotonic_check.py에서 확인된 비단조 위반(전체 20.9%,
  음수질량 인당평균 0.0005)을 클리핑 대신 등위회귀(isotonic regression,
  PAVA)로 정식 보정. 환자별 g=(P(Y>=1)..P(Y>=5))를 "단조감소 제약을
  만족하는 가장 가까운 값"으로 투영(min squared error projection).

절차 (정보누출 0, 기존 스크립트와 동일한 leave-one-trial-out 사용)
  STEP 1. 등위회귀 전/후 비단조 위반률·음수질량 비교 (보정이 실제로
          위반을 없애는지 확인)
  STEP 2. 등위회귀 적용이 하위 성능지표(MAE·가중카파·4·5→≤3·2↑과소)를
          훼손하지 않는지 확인 — raw(클립) 파이프라인 vs isotonic
          파이프라인을 같은 fold·같은 C로 나란히 평가
  STEP 3. 최종 배포용(3개 훈련시험 전체 풀) isotonic 파이프라인의
          tau4·tau5 재확정 — gn_external_validation_ad1062.py 이어서
          쓸 수 있도록 저장

입력   {TRAIN_DIR}/adl_wide.parquet, baseline_sample.parquet
       보조 검증/gn_final_hyperparams.csv (있으면 C·margin 재사용, 없으면 기본값)
출력   {OUT_DIR}/gn_monotonic_before_after.csv     (fold별 위반률·음수질량, raw vs isotonic)
       {OUT_DIR}/gn_isotonic_performance_check.csv (fold별 성능지표, raw vs isotonic)
       {OUT_DIR}/gn_isotonic_final_hyperparams.csv (최종 C·margin·tau4·tau5, isotonic 버전)
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
OUT_DIR = Path(__file__).parent

TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)
L1_RATIO = 0.5
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
# step 0. 데이터 로드 + 기존 확정 하이퍼파라미터 재사용
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


def load_prior_hyperparams() -> tuple[float, float]:
    """gn_external_validation_ad1062.py가 저장해둔 최종 (C, margin) 재사용.
    파일 없으면 그 스크립트의 최종 채택값(C=0.1, margin=0.14)으로 폴백."""
    path = OUT_DIR / "gn_final_hyperparams.csv"
    if path.exists():
        row = pd.read_csv(path).iloc[0]
        return float(row["C"]), float(row["margin"])
    print("경고: gn_final_hyperparams.csv 없음 → 기본값 C=0.1, margin=0.14 사용")
    return 0.1, 0.14


# =================================================================
# step 0-1. 등위회귀 (PAVA, 단조감소)
# =================================================================
def _pava_nondecreasing(x: np.ndarray) -> np.ndarray:
    """1차원 배열을 비감소(non-decreasing)로 투영하는 표준 PAVA.
    최소제곱 기준 등위회귀 해를 정확히 구함(근사 아님)."""
    n = len(x)
    stack: list[list[float]] = []  # [합, 가중치, 개수]
    for xi in x:
        stack.append([float(xi), 1.0, 1])
        while len(stack) >= 2 and (stack[-2][0] / stack[-2][1]) > (stack[-1][0] / stack[-1][1]):
            s2, w2, c2 = stack.pop()
            s1, w1, c1 = stack.pop()
            stack.append([s1 + s2, w1 + w2, c1 + c2])
    out = np.empty(n)
    idx = 0
    for s, w, c in stack:
        out[idx: idx + c] = s / w
        idx += c
    return out


def isotonic_decreasing(x: np.ndarray) -> np.ndarray:
    """비증가(non-increasing) 등위회귀. 부호 뒤집어 표준 PAVA 재사용."""
    return -_pava_nondecreasing(-x)


def isotonic_correct_matrix(G: np.ndarray) -> np.ndarray:
    """G: n x 5 (P(Y>=1..5), 환자별 행). 행마다 독립적으로 단조감소 투영."""
    return np.vstack([isotonic_decreasing(row) for row in G])


# =================================================================
# 공통 함수 (기존 스크립트와 동일)
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
    """절단별 elastic-net 이진로지스틱. 표준화는 train_df 내부에서만 fit(누출 0)."""
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

    def geq(rows: pd.DataFrame) -> np.ndarray:
        Z = sc.transform(imp.transform(rows[ITEMS]))
        return np.column_stack([models[k].predict_proba(Z)[:, 1] for k in range(1, 6)])  # n x 5

    return geq, y, models


def probs_raw(G: np.ndarray) -> np.ndarray:
    """원 파이프라인: 클립 + 재정규화만 적용 (등위회귀 없음, 기존 방식)."""
    n = len(G)
    P = np.zeros((n, 6))
    P[:, 0] = 1 - G[:, 0]
    for k in range(1, 5):
        P[:, k] = G[:, k - 1] - G[:, k]
    P[:, 5] = G[:, 4]
    P = np.clip(P, 1e-9, None)
    P /= P.sum(1, keepdims=True)
    return P


def probs_isotonic(G: np.ndarray) -> np.ndarray:
    """새 파이프라인: 등위회귀로 G 자체를 단조감소로 보정한 뒤 계단형 확률 계산.
    이론상 이미 단조라 음수질량이 없어야 하지만, 부동소수 오차 대비 안전클립은 유지."""
    Gc = isotonic_correct_matrix(G)
    n = len(Gc)
    P = np.zeros((n, 6))
    P[:, 0] = 1 - Gc[:, 0]
    for k in range(1, 5):
        P[:, k] = Gc[:, k - 1] - Gc[:, k]
    P[:, 5] = Gc[:, 4]
    P = np.clip(P, 1e-9, None)
    P /= P.sum(1, keepdims=True)
    return P


def violation_stats(G: np.ndarray) -> tuple[int, float]:
    """단조감소 위반 인원수 + 클립 전 음수확률질량."""
    dec_ok = np.all(np.diff(G, axis=1) <= 1e-9, axis=1)
    n_viol = int((~dec_ok).sum())
    n = len(G)
    P0 = np.zeros((n, 6))
    P0[:, 0] = 1 - G[:, 0]
    for k in range(1, 5):
        P0[:, k] = G[:, k - 1] - G[:, k]
    P0[:, 5] = G[:, 4]
    neg_mass = float(np.clip(-P0, 0, None).sum())
    return n_viol, neg_mass


def tune_tau(Ptr: np.ndarray, ytr: np.ndarray, budget: float) -> tuple[float, float]:
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
        return 1.0, 1.0
    return best[1], best[2]


# =================================================================
# step 1. 등위회귀 전/후 — 비단조 위반률·음수질량
# =================================================================
def compare_monotonic(m: pd.DataFrame, C: float) -> pd.DataFrame:
    hr("STEP 1. 등위회귀 전/후 — 비단조 위반률·음수질량 (leave-one-trial-out)")
    rows = []
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]
        te = m[m.STUDYID == ho]
        geq, _, _ = fit_en(tr, C)
        G = geq(te)

        v_raw, neg_raw = violation_stats(G)
        Gc = isotonic_correct_matrix(G)
        v_iso, neg_iso = violation_stats(Gc)

        n = len(te)
        rows.append({
            "held_out_trial": ho, "n": n,
            "raw_violation_rate": v_raw / n, "raw_neg_mass_per_capita": neg_raw / n,
            "iso_violation_rate": v_iso / n, "iso_neg_mass_per_capita": neg_iso / n,
        })
        print(f"  {ho}: n={n}  "
              f"raw 위반 {v_raw/n*100:5.1f}%(음수질량 {neg_raw/n:.4f})  →  "
              f"iso 위반 {v_iso/n*100:5.1f}%(음수질량 {neg_iso/n:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gn_monotonic_before_after.csv", index=False)

    tot_n = df["n"].sum()
    raw_rate = (df["raw_violation_rate"] * df["n"]).sum() / tot_n
    iso_rate = (df["iso_violation_rate"] * df["n"]).sum() / tot_n
    print(f"\n전체 위반률: raw {raw_rate*100:.1f}%  →  isotonic {iso_rate*100:.1f}%")
    if iso_rate < 1e-6:
        print("판정: 등위회귀 적용 후 비단조 위반 완전히 해소됨 (수학적으로 당연 — 투영 자체가 단조 보장)")
    else:
        print("판정: 위반이 남아있음 — PAVA 구현 재점검 필요")
    return df


# =================================================================
# step 2. 등위회귀가 하위 성능지표를 훼손하는지 확인
# =================================================================
def compare_performance(m: pd.DataFrame, C: float, margin: float) -> pd.DataFrame:
    hr("STEP 2. raw vs isotonic — 하위 성능지표(MAE·가중카파·4·5→≤3·2↑과소) 비교")
    rows = []
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]
        te = m[m.STUDYID == ho]
        geq, ytr, _ = fit_en(tr, C)
        Gtr = geq(tr)
        Gte = geq(te)
        yte = te["ds_stage"].astype(int).values

        for tag, probs_fn in (("raw", probs_raw), ("isotonic", probs_isotonic)):
            budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - margin
            t4, t5 = tune_tau(probs_fn(Gtr), ytr, budget)
            pred = asym_rule(probs_fn(Gte), t4, t5)
            kappa = cohen_kappa_score(yte, pred, weights="quadratic", labels=list(range(6)))
            rows.append({
                "held_out_trial": ho, "pipeline": tag, "tau4": t4, "tau5": t5,
                "mae": m_mae(yte, pred), "kappa": kappa,
                "hi_missed": m_hi(yte, pred), "under2": m_u2(yte, pred),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gn_isotonic_performance_check.csv", index=False)

    piv = df.pivot_table(index="held_out_trial", columns="pipeline",
                          values=["mae", "kappa", "hi_missed", "under2"])
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    mean_raw = df[df.pipeline == "raw"][["mae", "kappa", "hi_missed", "under2"]].mean()
    mean_iso = df[df.pipeline == "isotonic"][["mae", "kappa", "hi_missed", "under2"]].mean()
    print(f"\n평균(raw)      MAE={mean_raw.mae:.3f}  카파={mean_raw.kappa:.3f}  "
          f"4·5→≤3={mean_raw.hi_missed*100:.1f}%  2↑과소={mean_raw.under2*100:.1f}%")
    print(f"평균(isotonic) MAE={mean_iso.mae:.3f}  카파={mean_iso.kappa:.3f}  "
          f"4·5→≤3={mean_iso.hi_missed*100:.1f}%  2↑과소={mean_iso.under2*100:.1f}%")

    d_hi = mean_iso.hi_missed - mean_raw.hi_missed
    d_mae = mean_iso.mae - mean_raw.mae
    print(f"\n차이: 4·5→≤3 {d_hi*100:+.1f}%p, MAE {d_mae:+.3f}")
    if abs(d_hi) <= 0.02 and abs(d_mae) <= 0.02:
        print("판정: 등위회귀가 4조건 성능에 유의미한 영향 없음 — 안전하게 교체 가능")
    else:
        print("판정: 등위회귀 적용 후 성능이 눈에 띄게 변함 — tau 재튜닝·조건 재확인 필요")
    return df


# =================================================================
# step 3. 최종 배포용(3개 훈련시험 전체) isotonic 하이퍼파라미터 재확정
# =================================================================
def finalize_isotonic_model(m: pd.DataFrame, C: float, margin: float):
    hr("STEP 3. 최종 배포용 isotonic 파이프라인 tau4·tau5 재확정 (전체 훈련풀)")
    geq, y, _ = fit_en(m, C)
    G = geq(m)
    budget = m_mae(m["ds_stage"], m["A1_2015_stage"]) - margin
    t4, t5 = tune_tau(probs_isotonic(G), y, budget)
    print(f"C={C}, margin={margin}, tau4={t4}, tau5={t5} (budget={budget:.4f})")

    out = pd.DataFrame([{"C": C, "margin": margin, "tau4": t4, "tau5": t5, "pipeline": "isotonic"}])
    out.to_csv(OUT_DIR / "gn_isotonic_final_hyperparams.csv", index=False)
    print(f"저장: {OUT_DIR / 'gn_isotonic_final_hyperparams.csv'}")
    print("→ gn_external_validation_ad1062.py의 probs_fn을 probs_isotonic 기반으로 교체하고")
    print("  이 tau4·tau5로 AD-1062 재검증하면 최종 확인 끝남.")
    return t4, t5


# =================================================================
# main
# =================================================================
def main():
    hr("Aim 2 · Gn 비단조 확률 보정 — 등위회귀(PAVA) (sjlee)")
    m = load_train_pool()
    C, margin = load_prior_hyperparams()
    print(f"훈련풀 {len(m)}명, 재사용 하이퍼파라미터 C={C}, margin={margin}\n")

    compare_monotonic(m, C)
    print()
    compare_performance(m, C, margin)
    print()
    finalize_isotonic_model(m, C, margin)

    hr("완료")
    print(f"저장: {OUT_DIR / 'gn_monotonic_before_after.csv'}")
    print(f"저장: {OUT_DIR / 'gn_isotonic_performance_check.csv'}")
    print(f"저장: {OUT_DIR / 'gn_isotonic_final_hyperparams.csv'}")


if __name__ == "__main__":
    main()
