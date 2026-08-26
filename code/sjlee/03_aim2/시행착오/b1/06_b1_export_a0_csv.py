"""
Aim 2 · A0 지배점(최신) 결과 CSV 내보내기 (Drive 업로드용, 한글 안깨짐) (sjlee)
================================================================================
b1_dominate_a0의 최신 결과(B1이 A0 지배 + 부트스트랩 + 채점표)를 5개 CSV로.
인코딩 utf-8-sig(BOM) → Excel 한글 정상. ※ 아직 '잠정'(시행착오 단계, 팀 합의 전).

산출: aim2/aim2_sjlee/csv_a0/*.csv  (→ Drive)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUT = Path(__file__).parent.parent / "aim2_sjlee" / "csv_a0"; OUT.mkdir(parents=True, exist_ok=True)
LAB = {"ADL0101": "Q1 먹기", "ADL0102": "Q2 걷기", "ADL0103": "Q3 화장실", "ADL0104": "Q4 목욕",
       "ADL0105": "Q5 몸단장", "ADL0106B": "Q6b 옷입기", "ADL0106A": "Q6a 옷고르기",
       "ADL0107A": "Q7 전화", "ADL0110A": "Q10 설거지", "ADL0111A": "Q11 식사준비",
       "ADL0112A": "Q12 집안일", "ADL0113A": "Q13 빨래", "ADL0114A": "Q14 가전",
       "ADL0115A": "Q15 외출", "ADL0116A": "Q16a 쇼핑", "ADL0116B": "Q16b 지불",
       "ADL0117A": "Q17 금전", "Q18": "Q18 혼자있기", "ADL0121A": "Q21 글쓰기",
       "ADL0122Q": "Q22 취미", "ADL0123L": "Q23 가전사용"}


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f">>> {name}.csv ({len(df)}행)")


def main():
    m = D.build(); t = m["ds_stage"].values
    pred, chosen = D.run(m, 0.1)
    b, a0, a1 = pred.values, m["A0_harmonized"].values, m["A1_2015_stage"].values

    # 01 요약
    save(pd.DataFrame({"항목": ["연구", "상태", "표본", "문항(X)", "타깃(y)", "CV", "기법", "핵심 결론"],
        "내용": ["Aim2 B1 대안 채점법 — ADCS-ADL로 실측 DS 단계(0-5) 예측",
                "잠정(시행착오 단계, 팀 합의 전)", "공통표본 2,203 (AD-1061/1063/1064)",
                "A0 입력과 동일 21문항(iADL15+bADL6)", "실측 DS 단계 ds_stage",
                "leave-one-trial-out 3-fold (누출0, 시험변수 미투입)",
                "부분비례오즈(비평행 엘라스틱넷,표준화) + 비대칭임계 + A0-목표 τ튜닝(fold내부)",
                "B1이 A1을 완전 지배, A0는 안전성 동급+정확도 압도 → 세 알고리즘 중 정확도 최고이면서 A0급 중증안전성"]}), "B1_01_요약")

    # 02 성능비교 (전체 + 시험별)
    rows = []
    for lab, pr in [("B1(A0-목표)", b), ("A0", a0), ("A1", a1)]:
        rows.append({"구분": lab, "범위": "전체", "중증놓침(%)": round(D.miss(t, pr) * 100, 1),
                     "MAE": round(D.mae(t, pr), 3), "카파": round(D.kap(t, pr), 3)})
    for tr in D.TRIALS:
        idx = m.STUDYID == tr; tt = t[idx]
        for lab, pr in [("B1(A0-목표)", b[idx]), ("A0", a0[idx]), ("A1", a1[idx])]:
            rows.append({"구분": lab, "범위": tr, "중증놓침(%)": round(D.miss(tt, pr) * 100, 1),
                         "MAE": round(D.mae(tt, pr), 3), "카파": round(D.kap(tt, pr), 3)})
    save(pd.DataFrame(rows), "B1_02_성능비교")

    # 03 부트스트랩 유의성
    rng = np.random.default_rng(20260805); n = len(m); nb = 2000
    recs = []
    for name, r in [("A0", a0), ("A1", a1)]:
        dm, ds, dk = np.empty(nb), np.empty(nb), np.empty(nb)
        for i in range(nb):
            ix = rng.integers(0, n, n); tt = t[ix]
            dm[i] = D.mae(tt, r[ix]) - D.mae(tt, b[ix])
            ds[i] = D.miss(tt, r[ix]) - D.miss(tt, b[ix])
            dk[i] = D.kap(tt, b[ix]) - D.kap(tt, r[ix])
        for met, d, sc in [("MAE 차", dm, 1), ("중증놓침 차(%p)", ds, 100), ("카파 차", dk, 1)]:
            lo, hi = np.percentile(d, [2.5, 97.5])
            recs.append({"비교": f"B1 vs {name}", "지표": met, "점추정(B1우위)": round(d.mean() * sc, 3),
                         "CI하한": round(lo * sc, 3), "CI상한": round(hi * sc, 3),
                         "유의(0미포함)": "유의" if lo > 0 else "동급(0포함)"})
    save(pd.DataFrame(recs), "B1_03_부트스트랩_유의성")

    # 04 채점표 계수 (전체자료 재학습, 원 눈금)
    imp = SimpleImputer(strategy="median").fit(m[D.ITEMS]); Xi = imp.transform(m[D.ITEMS])
    sc = StandardScaler().fit(Xi); Z = sc.transform(Xi); sd = sc.scale_; mu = sc.mean_
    y = t.astype(int); coef = {}; intc = {}
    for k in range(1, 6):
        lr = LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, max_iter=6000, tol=1e-3, random_state=0).fit(Z, (y >= k).astype(int))
        coef[k] = lr.coef_[0] / sd; intc[k] = lr.intercept_[0] - np.sum(lr.coef_[0] * mu / sd)
    tbl = pd.DataFrame({"문항": [LAB[c] for c in D.ITEMS], "코드": D.ITEMS})
    for k in range(1, 6): tbl[f"coef_P(Y>={k})"] = [round(coef[k][i], 3) for i in range(len(D.ITEMS))]
    tbl = pd.concat([tbl, pd.DataFrame([{"문항": "(절편)", "코드": "", **{f"coef_P(Y>={k})": round(intc[k], 3) for k in range(1, 6)}}])], ignore_index=True)
    tbl = pd.concat([tbl, pd.DataFrame([{"문항": "결정규칙", "코드": "P(Y>=5)>0.51→5 / P(Y>=4)>0.09→4 / else 조건부중앙값"}])], ignore_index=True)
    save(tbl, "B1_04_채점표_계수")

    # 05 한계
    save(pd.DataFrame({"한계·주의": [
        "상태: 잠정(시행착오). 팀 합의 후 최종 확정 예정.",
        "3개 시험 소표본 leave-one-trial-out → 외부검증 전 성능은 낙관적.",
        "중증놓침은 A0를 '완전히' 이기진 못함(통합 동급, 부트스트랩상 A0와 차이 유의하지 않음=동급).",
        "부트스트랩 CI는 3시험이라 시험간 변동 미반영으로 낙관적으로 좁음 → 시험별 3/3 일관성과 병행 판단.",
        "채점표 계수는 전체자료 재학습(in-sample)이라 낙관적. 표준화 후 원 눈금 환산값.",
        "구조: 앙상블(비평행 로짓 + 임계 규칙) → 단일 해석 채점표 취지는 일부 약화.",
        "τ(4·5 판정 임계)는 fold 훈련 내부 튜닝(누출0)이나 'A0-목표' 자체가 설계 선택."]}), "B1_05_한계")

    print(f">>> 폴더: {OUT}")


if __name__ == "__main__":
    main()
