"""
Aim 2 · B1(A0 지배점) 릿지(L2) 버전 — listwise deletion(완전사례분석) 재검증 (sjlee)
====================================================================================
b1_ridge_top3_test.py(중앙값 대치판)와 똑같은 파이프라인·조합·C그리드·τ그리드를 쓰되,
결측치를 중앙값으로 채우는 대신 "그 조합의 문항 중 하나라도 결측이면 그 사람을 통째로
빼는" listwise deletion(완전사례분석)으로 바꿔서 다시 검증함.

배경: 21문항 기준 결측 있는 사람은 2,203명 중 65명(3.0%)뿐이라 다 빼도 표본이 크게
줄지 않고, 프로젝트 자체 결측치 처리 원칙(데이터_명세서_의존도연구.md — "임의 대치
최소화")과도 이쪽이 더 일치함. median 대치 버전과 비교해서 결과가 실질적으로 달라지는지
확인하는 목적.

방식: 조합(문항 리스트)별로 그 문항들에 결측이 하나도 없는 사람만 남긴 완전사례
부분표본을 만든 뒤(문항이 다르면 빠지는 사람도 달라짐 — 조합마다 완전사례 표본이 다름),
그 부분표본 안에서 b1_dominate_a0.py와 동일한 leave-one-trial-out(3) + A0-목표
τ튜닝(그리드 0.02) + 릿지(L2) 파이프라인을 그대로 적용함. 결측치대치 단계 자체가
없어짐(SimpleImputer 미사용).

산출: aim2/시행착오/b1_ridge_top3_listwise_report.md
      aim2/시행착오/b1_ridge_top3_listwise_scoretable_{tag}.csv
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

import b1_dominate_a0 as D  # build/mae/miss/kap/asym/TRIALS 등 재사용

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b1_ridge_top3_listwise_report.md"

# b1_ridge_top3_test.py와 동일 조합(종합비교 보고서 상위 L2 3개)
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
C_MAIN = 0.1


def make_complete_case(m, items):
    """이 조합 문항들에 결측이 하나도 없는 사람만 남긴 부분표본. 결측치대치 없음."""
    valid = m[items].notna().all(axis=1)
    return m.loc[valid].reset_index(drop=True), int((~valid).sum())


def ridge_probs_traintest(m, items, C):
    """완전사례 표본 안에서 fold별 train/test 확률. 결측치대치 단계 없음(이미 결측 0개)."""
    folds = {}
    for ho in D.TRIALS:
        tri = m.index[m.STUDYID != ho]; tei = m.index[m.STUDYID == ho]
        sc = StandardScaler().fit(m.loc[tri, items].to_numpy(dtype=float))
        Ztr = sc.transform(m.loc[tri, items].to_numpy(dtype=float))
        Zte = sc.transform(m.loc[tei, items].to_numpy(dtype=float))
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
    Z_raw = m[items].to_numpy(dtype=float)
    sc = StandardScaler().fit(Z_raw); Z = sc.transform(Z_raw); sd = sc.scale_; mu = sc.mean_
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
    fname = OUTDIR / f"b1_ridge_top3_listwise_scoretable_{tag}.csv"
    out.to_csv(fname, index=False, encoding="utf-8-sig")
    return fname, t4f, t5f


def main():
    _lines = []
    def log(s=""):
        print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    log("# Aim 2 · B1 릿지(L2) 버전 -- listwise deletion(완전사례분석) 재검증 (sjlee)\n")
    m_full = D.build()
    log(f"전체 공통표본(결측 포함) {len(m_full)}명\n")

    summary_rows = []
    for tag, items in COMBOS.items():
        m, n_dropped = make_complete_case(m_full, items)
        t = m["ds_stage"].values
        a0 = m["A0_harmonized"].values; a1 = m["A1_2015_stage"].values
        a0_miss = D.miss(t, a0) * 100; a0_mae = D.mae(t, a0); a0_k = D.kap(t, a0)
        a1_miss = D.miss(t, a1) * 100; a1_mae = D.mae(t, a1); a1_k = D.kap(t, a1)

        log(f"## {tag} ({len(items)}문항)")
        log(f"- 문항: {', '.join(items)}")
        log(f"- listwise deletion: {len(m_full)}명 중 이 문항들에 결측 있는 {n_dropped}명 제외 → "
            f"완전사례 표본 {len(m)}명({len(m)/len(m_full)*100:.1f}%)")
        log(f"- 이 부분표본 기준 기준값: A0(중증놓침 {a0_miss:.1f}%, MAE {a0_mae:.3f}, 카파 {a0_k:.3f}) / "
            f"A1(중증놓침 {a1_miss:.1f}%, MAE {a1_mae:.3f}, 카파 {a1_k:.3f})\n")

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
        log(f"\n- **{tag} 종합(C={C_MAIN}, listwise, n={len(m)}): 중증놓침 {b_miss:.1f}%, MAE {b_mae:.3f}, 카파 {b_k:.3f}**\n")

        fname, t4f, t5f = ridge_scoretable(m, items, tag, C=C_MAIN)
        log(f"- 최종 채점표(완전사례 재학습, 참고용): {fname.name}, 결정규칙 tau4={t4f}, tau5={t5f}\n")
        summary_rows.append({"조합": tag, "문항수": len(items), "완전사례n": len(m), "제외n": n_dropped,
                              "중증놓침": round(b_miss, 2), "MAE": round(b_mae, 3), "카파": round(b_k, 3)})

    log("## 3개 조합 요약 (listwise, C=0.1) — median 대치판과 비교용")
    log("| 조합 | 문항수 | 표본n(제외수) | 중증놓침 | MAE | 카파 |")
    log("|---|---|---|---|---|---|")
    for r in summary_rows:
        log(f"| {r['조합']} | {r['문항수']} | {r['완전사례n']}({r['제외n']}명 제외) | {r['중증놓침']}% | {r['MAE']} | {r['카파']} |")
    log(f"\n- median 대치판(`b1_ridge_top3_report.md`) 수치와 나란히 비교해서, 결측치 처리 방식이 "
        "결과를 실질적으로 바꾸는지(차이가 크면 대치방식이 민감한 선택이라는 뜻, 차이가 작으면 "
        "어느 쪽을 써도 결론은 같다는 뜻) 확인할 것.")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
