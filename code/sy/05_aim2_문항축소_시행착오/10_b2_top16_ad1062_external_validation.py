"""
Aim 2 · B2 종합순위 상위 16개 조합 — AD-1062 독립 외부검증 (listwise deletion) (sjlee)
====================================================================================
b2_fixed6_exhaustive_listwise.py 종합비교 결과(listwise판) 상위 16개 조합이 "4만 5천개
중 뽑혀서 우연히 좋아 보이는 것"인지, AD-1062(학습·튜닝·CV 어디에도 한 번도 안 쓰인
독립표본)에서도 재현되는지 확인함.

절차 (gn_external_validation_ad1062.py와 동일 원칙 — 정보누출 0):
  1) AD-1061+1063+1064 3개 시험을 전부 합쳐 "최종 배포용" 단일 모형을 조합별로 학습함
     (leave-one-trial-out 아님 — 하이퍼파라미터 선택 절차가 없으니 3개 시험 전체를
     그대로 훈련에 씀).
  2) τ4/τ5는 이 훈련풀 전체 기준 A0-목표 방식으로 딱 한 번 튜닝(그리드 0.02, 나머지
     스크립트들과 동일 규칙).
  3) 이렇게 확정된 모형·τ를 AD-1062에 **딱 한 번만** 적용해서 예측함. AD-1062는 학습·
     튜닝 어디에도 관여하지 않음.

결측치 처리: listwise deletion 유지. 조합별 문항 중 하나라도 결측이면 훈련풀·AD-1062
양쪽에서 그 사람 제외(조합마다 걸리는 사람이 다름 — b2_fixed6_exhaustive_listwise.py와
동일 원칙). A0·A1 비교기준값도 "그 조합에서 실제로 채점된 AD-1062 사람들"만으로
다시 계산함(짝지어진/matched 비교 — 조합마다 걸러지는 인원이 다르니, 비교기준도 같은
사람들 기준이어야 공정함. b1_ridge_top3_listwise_test.py와 동일한 선택).

산출: aim2/시행착오/b2_top16_ad1062_external_report.md
      aim2/시행착오/b2_top16_ad1062_predictions.csv (조합별 AD-1062 예측치, 롱포맷)
"""
from pathlib import Path
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import b1_dominate_a0 as D  # build/mae/miss/kap/asym/ITEMS 등 재사용

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b2_top16_ad1062_external_report.md"
EXT_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study_csv\extra_validation_ad1062")

TAU_GRID = np.round(np.arange(0.05, 0.55, 0.02), 3)

PENALTY_KW = {
    "EN": dict(penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=3000, tol=1e-3),
    "L2": dict(penalty="l2", solver="lbfgs", max_iter=3000, tol=1e-3),
}
C_MAIN = 0.1

# b2_fixed6_EN_L2_종합비교_보고서_listwise판.md 3절 상위 16개 조합 (rank, penalty, items)
FIXED = ["ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0115A", "ADL0116A"]
TOP16 = [
    (1,  "EN", FIXED + ["ADL0102", "ADL0107A", "ADL0110A", "Q18"]),
    (2,  "EN", FIXED + ["ADL0101", "ADL0102", "ADL0107A", "ADL0110A", "ADL0117A", "Q18"]),
    (3,  "EN", FIXED + ["ADL0102", "ADL0107A", "ADL0110A", "ADL0117A", "Q18"]),
    (4,  "EN", FIXED + ["ADL0101", "ADL0102", "ADL0106B", "ADL0107A", "ADL0113A", "Q18", "ADL0121A"]),
    (5,  "EN", FIXED + ["ADL0101", "ADL0102", "ADL0106B", "ADL0107A", "ADL0113A", "ADL0117A", "Q18", "ADL0121A"]),
    (6,  "L2", FIXED + ["ADL0102", "ADL0106B", "ADL0107A", "ADL0116B", "ADL0122Q"]),
    (7,  "L2", FIXED + ["ADL0102", "ADL0106B", "ADL0107A", "ADL0116B", "ADL0117A", "ADL0122Q"]),
    (8,  "EN", FIXED + ["ADL0102", "ADL0107A", "Q18"]),
    (9,  "L2", FIXED + ["ADL0110A", "ADL0116B", "ADL0122Q", "ADL0123L"]),
    (10, "L2", FIXED + ["ADL0101", "ADL0102", "ADL0107A", "ADL0111A", "ADL0114A", "ADL0117A", "Q18"]),
    (11, "L2", FIXED + ["ADL0106B", "ADL0111A", "Q18"]),
    (12, "L2", FIXED + ["ADL0101", "ADL0106B", "ADL0107A", "ADL0111A", "ADL0114A", "ADL0117A", "Q18", "ADL0122Q"]),
    (13, "EN", FIXED + ["ADL0107A", "ADL0116B"]),
    (14, "L2", FIXED + ["ADL0101", "ADL0116B"]),
    (15, "L2", FIXED + ["ADL0107A"]),
    (16, "EN", FIXED + ["ADL0106B"]),
]


def load_ad1062():
    """AD-1062 — 학습·튜닝·CV 전부 미관여 순수 외부검증 표본."""
    adl = pd.read_csv(EXT_DIR / "adl_wide_ad1062.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 1.0].copy()

    def col(name):
        r = f"{name}__resolved"
        return b[r] if r in b.columns else b[name]

    data = {}
    for item in D.ITEMS:
        if item == "Q18":
            data[item] = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
        else:
            data[item] = col(item)
    X = pd.DataFrame(data)
    X["STUDYID"] = b["STUDYID"].values
    X["USUBJID"] = b["USUBJID"].values
    X["A0_harmonized"] = b["A0_harmonized"].values
    X["A1_2015_stage"] = b["A1_2015_stage"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])

    ds = pd.read_csv(EXT_DIR / "ds_wide_ad1062.csv", low_memory=False)
    ds_bl = ds[ds["VISITNUM"] == 1.0][["STUDYID", "USUBJID", "ds_stage"]]

    m = ds_bl.merge(X, on=["STUDYID", "USUBJID"], how="left")
    m = m.dropna(subset=["ds_stage", "A0_harmonized", "A1_2015_stage"])  # 결과지표 자체가 없으면 애초에 못 씀
    return m.reset_index(drop=True)


def fit_final(train, items, penalty_tag, C=C_MAIN):
    kw = PENALTY_KW[penalty_tag]
    sc = StandardScaler().fit(train[items].to_numpy(dtype=float))
    Z = sc.transform(train[items].to_numpy(dtype=float))
    y = train["ds_stage"].astype(int).values
    models = {k: LogisticRegression(C=C, random_state=0, **kw).fit(Z, (y >= k).astype(int))
              for k in range(1, 6)}

    def probs(df):
        Zx = sc.transform(df[items].to_numpy(dtype=float))
        g = {k: models[k].predict_proba(Zx)[:, 1] for k in range(1, 6)}
        P = np.zeros((len(df), 6)); P[:, 0] = 1 - g[1]
        for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
        P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)
        return P
    return probs, y


def tune_tau(Ptr, ytr, a0_tr, target):
    best = None
    for t4 in TAU_GRID:
        for t5 in TAU_GRID:
            if t5 < t4:
                continue
            p = D.asym(Ptr, t4, t5)
            if D.miss(ytr, p) <= target + 1e-9:
                key = D.mae(ytr, p)
                if best is None or key < best[0]:
                    best = (key, float(t4), float(t5))
    if best is None:
        return 0.3, 0.5
    return best[1], best[2]


def main():
    _lines = []
    def log(s=""):
        print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    log("# Aim 2 · B2 종합순위 상위 16개 조합 -- AD-1062 독립 외부검증 (listwise, sjlee)\n")

    train_full = D.build()
    ext_full = load_ad1062()
    log(f"훈련풀(AD-1061+1063+1064, 결측 필터 전): {len(train_full)}명")
    log(f"AD-1062(결측 필터 전, ds/A0/A1 있는 사람만): {len(ext_full)}명\n")

    pred_rows = []
    summary_rows = []
    for rank, ptag, items in TOP16:
        tr_valid = train_full[items].notna().all(axis=1)
        tr = train_full.loc[tr_valid].reset_index(drop=True)
        ex_valid = ext_full[items].notna().all(axis=1)
        ex = ext_full.loc[ex_valid].reset_index(drop=True)

        probs_fn, ytr = fit_final(tr, items, ptag, C=C_MAIN)
        Ptr = probs_fn(tr)
        a0_tr = tr["A0_harmonized"].values
        target = D.miss(ytr, a0_tr)
        t4, t5 = tune_tau(Ptr, tr["ds_stage"].values, a0_tr, target)

        Pex = probs_fn(ex)
        pred = D.asym(Pex, t4, t5)
        t = ex["ds_stage"].values
        a0 = ex["A0_harmonized"].values; a1 = ex["A1_2015_stage"].values

        b_mae, b_miss, b_k = D.mae(t, pred), D.miss(t, pred) * 100, D.kap(t, pred)
        a0_mae, a0_miss, a0_k = D.mae(t, a0), D.miss(t, a0) * 100, D.kap(t, a0)
        a1_mae, a1_miss, a1_k = D.mae(t, a1), D.miss(t, a1) * 100, D.kap(t, a1)
        dom_a0 = (b_miss <= a0_miss + 1.0) and (b_mae < a0_mae)
        dom_a1 = (b_miss < a1_miss) and (b_mae < a1_mae)

        tag = f"{ptag}_{len(items)}문항_순위{rank}"
        log(f"## 순위{rank} — {ptag}, {len(items)}문항")
        log(f"- 문항: {', '.join(items)}")
        log(f"- 훈련풀: {len(train_full)}명 중 결측제외 {len(tr)}명 사용 / "
            f"AD-1062: {len(ext_full)}명 중 결측제외 {len(ex)}명 채점 (τ4={t4}, τ5={t5})")
        log(f"- 이 부분표본 기준: A0(놓침 {a0_miss:.1f}%, MAE {a0_mae:.3f}, 카파 {a0_k:.3f}) / "
            f"A1(놓침 {a1_miss:.1f}%, MAE {a1_mae:.3f}, 카파 {a1_k:.3f})")
        log(f"- **B2({tag}) AD-1062 외부검증: 놓침 {b_miss:.1f}%, MAE {b_mae:.3f}, 카파 {b_k:.3f}**")
        log(f"- A0지배?(놓침<=A0+1%p & MAE<A0): {'예' if dom_a0 else '아니오'} / "
            f"A1지배?(놓침<A1 & MAE<A1): {'예' if dom_a1 else '아니오'}\n")

        summary_rows.append({"순위": rank, "penalty": ptag, "문항수": len(items),
                              "훈련n": len(tr), "AD1062_n": len(ex),
                              "AD1062_놓침%": round(b_miss, 2), "AD1062_MAE": round(b_mae, 4),
                              "AD1062_카파": round(b_k, 4),
                              "A0지배": dom_a0, "A1지배": dom_a1,
                              "내부CV_score순위": rank})

        for uid, tt, pp, a0v, a1v in zip(ex["USUBJID"], t, pred, a0, a1):
            pred_rows.append({"rank": rank, "penalty": ptag, "n_items": len(items),
                               "USUBJID": uid, "ds_stage": tt, "B2_pred": pp,
                               "A0_harmonized": a0v, "A1_2015_stage": a1v})

    log("## 16개 조합 요약 — 내부CV 순위 vs AD-1062 실제 성능")
    log("| 내부CV순위 | penalty | 문항수 | AD1062 n | 놓침% | MAE | 카파 | A0지배 | A1지배 |")
    log("|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        log(f"| {r['순위']} | {r['penalty']} | {r['문항수']} | {r['AD1062_n']} | "
            f"{r['AD1062_놓침%']}% | {r['AD1062_MAE']} | {r['AD1062_카파']} | "
            f"{'예' if r['A0지배'] else '아니오'} | {'예' if r['A1지배'] else '아니오'} |")

    sdf = pd.DataFrame(summary_rows)
    ad1062_rank = sdf.sort_values("AD1062_MAE").reset_index(drop=True)
    ad1062_rank["AD1062_MAE순위"] = range(1, len(ad1062_rank) + 1)
    log("\n## 참고: AD-1062 MAE 기준으로 다시 줄세우면")
    log("| AD1062_MAE순위 | 내부CV순위 | penalty | 문항수 | AD1062 MAE | AD1062 놓침% |")
    log("|---|---|---|---|---|---|")
    for _, r in ad1062_rank.iterrows():
        log(f"| {r['AD1062_MAE순위']} | {r['순위']} | {r['penalty']} | {r['문항수']} | "
            f"{r['AD1062_MAE']} | {r['AD1062_놓침%']}% |")

    corr = sdf["내부CV_score순위"].corr(sdf["AD1062_MAE"])
    log(f"\n- 내부CV 순위와 AD-1062 MAE의 상관계수: {corr:.3f} (0에 가까우면 내부CV 순위가 "
        "외부성능을 거의 예측 못 한다는 뜻 — 과최적화/승자의 저주 가능성을 뒷받침)")

    pd.DataFrame(pred_rows).to_csv(OUTDIR / "b2_top16_ad1062_predictions.csv", index=False, encoding="utf-8-sig")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
    print(f">>> 저장: {OUTDIR / 'b2_top16_ad1062_predictions.csv'}")


if __name__ == "__main__":
    main()
