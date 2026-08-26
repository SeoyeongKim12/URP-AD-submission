"""
Aim 2 · B1(A0 지배점) 릿지(L2) 버전 — 종합순위 상위 L2 3개 조합 검증 (sjlee)
====================================================================================
b1_dominate_a0.py 파이프라인(A0-목표 τ튜닝 + 시험별 검증 + 최종 채점표)은 그대로 쓰되,
penalty만 elastic-net에서 순수 L2(릿지)로 바꾸고, b2_fixed6_exhaustive.py 종합비교
보고서의 종합점수(score) 상위 L2 조합 3개(14/11/12문항)를 각각 실제로 적합·검증함.

원본 b1_dominate_a0.py는 건드리지 않고 별도 스크립트로 둠(다른 파이프라인이 그 파일을
import해서 쓰고 있어서, 원본을 릿지로 바꿔버리면 그쪽이 깨짐). 대신 D.build/D.mae/D.miss/
D.kap/D.asym/D.TRIALS처럼 penalty와 무관한 유틸리티는 그대로 재사용함.

바뀐 점(원본 b1_dominate_a0.py 대비):
  - LogisticRegression(solver="saga", l1_ratio=0.5, C=C) → solver="lbfgs", penalty="l2"
    (l1_ratio 없음 = elastic-net 아님. b2_fixed6_exhaustive.py의 L2 설정과 동일)
  - ITEMS 21문항 고정 대신 COMBOS 딕셔너리로 조합별 지정(문항코드는 b2 전수조사 결과의
    items_code를 그대로 가져옴)
  - C값 스윕(0.1/0.25/0.5/1.0)으로 결과가 C에 안정적인지도 같이 확인. 대표 결과·부트스트랩
    없는 대신(3개조합×4C라 시간 절약차 생략) 채점표는 C=0.1(=b2 전수조사와 동일 C) 기준

산출: aim2/시행착오/b1_ridge_top3_report.md
      aim2/시행착오/b1_ridge_top3_scoretable_{tag}.csv (조합별 C=0.1 최종 채점표, 원 눈금 계수)
"""
from pathlib import Path
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import b1_dominate_a0 as D  # build/mae/miss/kap/asym/TRIALS 등 penalty-무관 유틸 재사용

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b1_ridge_top3_report.md"

# b2_fixed6_exhaustive.py 종합비교 보고서(3절) 종합점수 상위 L2 조합 3개.
# 고정 6문항(ADL0103,ADL0104,ADL0105,ADL0106A,ADL0115A,ADL0116A)은 세 조합 모두에 포함됨.
COMBOS = {
    "L2_14문항_종합1위": ["ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0107A", "ADL0110A",
                       "ADL0111A", "ADL0112A", "ADL0113A", "ADL0115A", "ADL0116A", "ADL0117A",
                       "Q18", "ADL0123L"],
    "L2_11문항_종합2위": ["ADL0101", "ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0111A",
                       "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18"],
    "L2_12문항_종합3위": ["ADL0101", "ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0107A",
                       "ADL0111A", "ADL0113A", "ADL0114A", "ADL0115A", "ADL0116A", "Q18"],
}
C_GRID = [0.1, 0.25, 0.5, 1.0]
C_MAIN = 0.1   # b2 전수조사와 동일 C. 대표결과/채점표는 이 값 기준


def ridge_probs_traintest(m, items, C):
    """fold별 train확률·test확률 (τ 훈련내부 튜닝용). 릿지(L2) 고정."""
    folds = {}
    for ho in D.TRIALS:
        tri = m.index[m.STUDYID != ho]; tei = m.index[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(m.loc[tri, items])
        sc = StandardScaler().fit(imp.transform(m.loc[tri, items]))
        Ztr = sc.transform(imp.transform(m.loc[tri, items]))
        Zte = sc.transform(imp.transform(m.loc[tei, items]))
        y = m.loc[tri, "ds_stage"].astype(int).values

        def build(Z, C, Ztr=Ztr, y=y):
            geq = {k: LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=3000,
                                          tol=1e-3, random_state=0).fit(Ztr, (y >= k).astype(int)).predict_proba(Z)[:, 1]
                   for k in range(1, 6)}
            Q = np.zeros((Z.shape[0], 6)); Q[:, 0] = 1 - geq[1]
            for k in range(1, 5): Q[:, k] = geq[k] - geq[k + 1]
            Q[:, 5] = geq[5]; Q = np.clip(Q, 1e-9, None); Q /= Q.sum(1, keepdims=True)
            return Q
        folds[ho] = dict(tri=tri, tei=tei, build=build, Ztr=Ztr, Zte=Zte, y=y)
    return folds


def ridge_run(m, items, C):
    """A0-목표 τ튜닝으로 릿지 외부예측 생성 + fold별 선택 τ. b1_dominate_a0.run()과 동일 로직."""
    folds = ridge_probs_traintest(m, items, C)
    pred = pd.Series(index=m.index, dtype=float)
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3)
    chosen = {}
    for ho in D.TRIALS:
        f = folds[ho]; tri, tei = f["tri"], f["tei"]
        Ptr = f["build"](f["Ztr"], C); Pte = f["build"](f["Zte"], C)
        yt = m.loc[tri, "ds_stage"].values
        target = D.miss(yt, m.loc[tri, "A0_harmonized"].values)
        best = None
        for t4 in grid:
            for t5 in grid:
                if t5 < t4:
                    continue
                p = D.asym(Ptr, t4, t5)
                if D.miss(yt, p) <= target + 1e-9:
                    key = D.mae(yt, p)
                    if best is None or key < best[0]:
                        best = (key, float(t4), float(t5))
        if best is None:
            allc = [(D.miss(yt, D.asym(Ptr, t4, t5)), D.mae(yt, D.asym(Ptr, t4, t5)), t4, t5)
                    for t4 in grid for t5 in grid if t5 >= t4]
            _, _, t4, t5 = min(allc); best = (0, t4, t5)
        _, t4, t5 = best
        chosen[ho] = (round(t4, 3), round(t5, 3), round(target * 100, 1))
        pred.loc[tei] = D.asym(Pte, t4, t5)
    return pred, chosen


def ridge_scoretable(m, items, tag, C=C_MAIN):
    """전체자료 재학습 채점표(원 눈금 계수+절편, 최종 τ). b1_dominate_a0.final_scoretable()과 동일 로직."""
    imp = SimpleImputer(strategy="median").fit(m[items])
    Xi = imp.transform(m[items])
    sc = StandardScaler().fit(Xi); Z = sc.transform(Xi); sd = sc.scale_; mu = sc.mean_
    y = m["ds_stage"].astype(int).values
    coef_raw = {}; intc_raw = {}
    for k in range(1, 6):
        lr = LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=3000, tol=1e-3,
                                 random_state=0).fit(Z, (y >= k).astype(int))
        w = lr.coef_[0]; b0 = lr.intercept_[0]
        coef_raw[k] = w / sd; intc_raw[k] = b0 - np.sum(w * mu / sd)

    def probs():
        geq = {k: LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=3000, tol=1e-3,
                                      random_state=0).fit(Z, (y >= k).astype(int)).predict_proba(Z)[:, 1]
               for k in range(1, 6)}
        P = np.zeros((len(m), 6)); P[:, 0] = 1 - geq[1]
        for k in range(1, 5): P[:, k] = geq[k] - geq[k + 1]
        P[:, 5] = geq[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)
        return P
    P = probs(); target = D.miss(y, m["A0_harmonized"].values)
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4:
                continue
            p = D.asym(P, t4, t5)
            if D.miss(y, p) <= target + 1e-9:
                key = D.mae(y, p)
                if best is None or key < best[0]:
                    best = (key, float(t4), float(t5))
    _, t4f, t5f = best
    tbl = pd.DataFrame({"item": items})
    for k in range(1, 6):
        tbl[f"coef_P(Y>={k})"] = [round(coef_raw[k][i], 3) for i in range(len(items))]
    intercepts = pd.DataFrame([{"item": "(절편)", **{f"coef_P(Y>={k})": round(intc_raw[k], 3) for k in range(1, 6)}}])
    out = pd.concat([tbl, intercepts], ignore_index=True)
    fname = OUTDIR / f"b1_ridge_top3_scoretable_{tag}.csv"
    out.to_csv(fname, index=False, encoding="utf-8-sig")
    return fname, t4f, t5f


def main():
    _lines = []
    def log(s=""):
        print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    log("# Aim 2 · B1 릿지(L2) 버전 -- 종합순위 상위 L2 3개 조합 검증 (sjlee)\n")
    m = D.build()
    t = m["ds_stage"].values
    a0 = m["A0_harmonized"].values; a1 = m["A1_2015_stage"].values
    a0_miss = D.miss(t, a0) * 100; a0_mae = D.mae(t, a0); a0_k = D.kap(t, a0)
    a1_miss = D.miss(t, a1) * 100; a1_mae = D.mae(t, a1); a1_k = D.kap(t, a1)
    log(f"공통표본 {len(m)}. 기준: A0(중증놓침 {a0_miss:.1f}%, MAE {a0_mae:.3f}, 카파 {a0_k:.3f}) / "
        f"A1(중증놓침 {a1_miss:.1f}%, MAE {a1_mae:.3f}, 카파 {a1_k:.3f})\n")

    summary_rows = []
    for tag, items in COMBOS.items():
        log(f"## {tag} ({len(items)}문항)")
        log(f"- 문항: {', '.join(items)}\n")
        log("| C | 중증놓침 | MAE | 카파 | A0지배?(놓침<=A0+1%p & MAE<A0) | A1지배?(놓침<A1 & MAE<A1) |")
        log("|---|---|---|---|---|---|")
        main_pred = None
        for C in C_GRID:
            pred, chosen = ridge_run(m, items, C)
            b_miss = D.miss(t, pred.values) * 100; b_mae = D.mae(t, pred.values); b_k = D.kap(t, pred.values)
            dom_a0 = (b_miss <= a0_miss + 1.0) and (b_mae < a0_mae)
            dom_a1 = (b_miss < a1_miss) and (b_mae < a1_mae)
            log(f"| {C} | {b_miss:.1f}% | {b_mae:.3f} | {b_k:.3f} | {'예' if dom_a0 else '아니오'} | {'예' if dom_a1 else '아니오'} |")
            if C == C_MAIN:
                main_pred = (pred, chosen, b_miss, b_mae, b_k)
        log("")
        pred, chosen, b_miss, b_mae, b_k = main_pred
        log(f"### 대표결과(C={C_MAIN}) 시험별")
        log(f"- fold별 (tau4,tau5,A0목표%): {chosen}")
        log("| 시험 | 중증놓침 B1/A0/A1 | MAE B1/A0/A1 | 카파 B1/A0/A1 |")
        log("|---|---|---|---|")
        for tr in D.TRIALS:
            idx = m.STUDYID == tr; tt = t[idx]
            b = pred[idx].values; a0v = m.loc[idx, "A0_harmonized"].values; a1v = m.loc[idx, "A1_2015_stage"].values
            log(f"| {tr} | {D.miss(tt,b)*100:.1f}/{D.miss(tt,a0v)*100:.1f}/{D.miss(tt,a1v)*100:.1f} | "
                f"{D.mae(tt,b):.3f}/{D.mae(tt,a0v):.3f}/{D.mae(tt,a1v):.3f} | "
                f"{D.kap(tt,b):.3f}/{D.kap(tt,a0v):.3f}/{D.kap(tt,a1v):.3f} |")
        log(f"\n- **{tag} 종합(C={C_MAIN}): 중증놓침 {b_miss:.1f}%, MAE {b_mae:.3f}, 카파 {b_k:.3f}**\n")

        fname, t4f, t5f = ridge_scoretable(m, items, tag, C=C_MAIN)
        log(f"- 최종 채점표(전체자료 재학습, 참고용): {fname.name}, 결정규칙 tau4={t4f}, tau5={t5f}\n")
        summary_rows.append({"조합": tag, "문항수": len(items), "C": C_MAIN,
                              "중증놓침": round(b_miss, 2), "MAE": round(b_mae, 3), "카파": round(b_k, 3)})

    log("## 3개 조합 요약 비교 (C=0.1)")
    log("| 조합 | 문항수 | 중증놓침 | MAE | 카파 |")
    log("|---|---|---|---|---|")
    for r in summary_rows:
        log(f"| {r['조합']} | {r['문항수']} | {r['중증놓침']}% | {r['MAE']} | {r['카파']} |")
    log(f"\n- 참고: b2_fixed6_exhaustive.py 전수조사도 동일하게 A0-목표 tau튜닝·C=0.1을 썼기 때문에, "
        "여기서 나온 수치가 전수조사 표의 값과 거의 같게(±0.01 이내) 재현되면 정상. 크게 다르면 "
        "두 파이프라인 어딘가 불일치가 있다는 뜻이라 확인 필요.")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
