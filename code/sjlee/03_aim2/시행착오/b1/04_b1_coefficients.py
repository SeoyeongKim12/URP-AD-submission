"""
Aim 2 · 계수·p-value 산출 (sjlee)
==================================
종합문서 §3·§4의 근거를 재현 가능한 스크립트로 정리.

(A) B1 비례오즈 채점표: 전체자료 재학습(무벌점, 원 눈금) → 문항 계수·OR·p-value.
    * p-value는 무벌점 모형에서만 타당(벌점 모형은 표준오차 편향).
(B) Gn 엘라스틱넷 절단별 계수: 비평행 5개 절단 P(Y≥k)의 계수(표준화→원 눈금 환산, C=0.1).

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/b1_coefficients_report.md
     aim2/b1_po_coefficients.csv        (B1 비례오즈 계수·p)
     aim2/b1_gn_cut_coefficients.csv    (Gn 절단별 계수)
  * 계수표는 환자 데이터가 아니라 모형 계수 → git 추적 가능(단 .gitignore의 *.csv 규칙상
    제외되므로, 리포트(.md)에 표로도 남김. CSV가 필요하면 규칙 조정).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b1_coefficients_report.md"
BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL
LAB = {"ADL0101": "Q1 먹기", "ADL0102": "Q2 걷기", "ADL0103": "Q3 화장실", "ADL0104": "Q4 목욕",
       "ADL0105": "Q5 몸단장", "ADL0106B": "Q6b 옷입기", "ADL0106A": "Q6a 옷고르기",
       "ADL0107A": "Q7 전화", "ADL0110A": "Q10 설거지", "ADL0111A": "Q11 식사준비",
       "ADL0112A": "Q12 집안일", "ADL0113A": "Q13 빨래", "ADL0114A": "Q14 가전",
       "ADL0115A": "Q15 외출", "ADL0116A": "Q16a 쇼핑", "ADL0116B": "Q16b 지불",
       "ADL0117A": "Q17 금전", "Q18": "Q18 혼자있기", "ADL0121A": "Q21 글쓰기",
       "ADL0122Q": "Q22 취미", "ADL0123L": "Q23 가전사용"}
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


def build():
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 2.0].copy()
    def col(c): return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {c: ((col("ADL0118A") + col("ADL0118B") + col("ADL0118C")) if c == "Q18" else col(c)) for c in ITEMS}
    X = pd.DataFrame(data); X["STUDYID"] = b["STUDYID"].values; X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "ds_stage"]]
    m = cs.merge(X, on=["STUDYID", "USUBJID"], how="left")
    imp = SimpleImputer(strategy="median").fit(m[ITEMS])
    Xi = pd.DataFrame(imp.transform(m[ITEMS]), columns=ITEMS)
    return m, Xi, m["ds_stage"].astype(int).values


def part_A(Xi, y):
    log("## (A) B1 비례오즈 채점표 — 계수·OR·p-value (전체자료, 무벌점, 원 눈금)\n")
    res = OrderedModel(y, Xi.values, distr="logit").fit(method="bfgs", disp=False, maxiter=300)
    rows = []
    for i, c in enumerate(ITEMS):
        p = res.pvalues[i]
        rows.append(dict(item=LAB[c], code=c, coef=res.params[i], OR=np.exp(res.params[i]),
                         se=res.bse[i], z=res.tvalues[i], p=p))
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    df.to_csv(OUTDIR / "b1_po_coefficients.csv", index=False)
    log("| 문항 | 코드 | 계수 β | OR | SE | z | p-value | 유의 |")
    log("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        star = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else ""
        log(f"| {r['item']} | {r.code} | {r.coef:+.3f} | {r.OR:.2f} | {r.se:.3f} | {r.z:+.2f} | {r.p:.1e} | {star} |")
    nsig = int((df.p < .05).sum())
    log(f"\n- 유의(p<0.05): **{nsig}/21** (p<0.001: {int((df.p<.001).sum())}) | 비유의: {21-nsig}")
    log("- 음수 계수 = 문항점수 높을수록(독립적일수록) 저단계. 벌점 없는 모형이라 p-value 타당.")
    log("- CSV: b1_po_coefficients.csv\n")


def part_B(Xi, y):
    log("## (B) Gn 엘라스틱넷 절단별 계수 (비평행 5절단, 표준화→원 눈금, C=0.1)\n")
    sc = StandardScaler().fit(Xi.values); Z = sc.transform(Xi.values); sd = sc.scale_
    coefs = {}
    for k in range(1, 6):
        lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                C=0.1, max_iter=6000, tol=1e-3).fit(Z, (y >= k).astype(int))
        coefs[k] = lr.coef_[0] / sd            # 원 눈금 환산
    df = pd.DataFrame({LAB[c]: [coefs[k][i] for k in range(1, 6)] for i, c in enumerate(ITEMS)},
                      index=[f"P(Y>={k})" for k in range(1, 6)]).T
    df.insert(0, "code", ITEMS)
    df.to_csv(OUTDIR / "b1_gn_cut_coefficients.csv")
    log("| 문항 | P(Y≥1) | P(Y≥2) | P(Y≥3) | P(Y≥4) | P(Y≥5) |")
    log("|---|---|---|---|---|---|")
    for c in ITEMS:
        vals = [coefs[k][ITEMS.index(c)] for k in range(1, 6)]
        log(f"| {LAB[c]} | " + " | ".join(f"{v:+.2f}" if abs(v) > 1e-4 else "0" for v in vals) + " |")
    nz = sum(sum(abs(coefs[k][i]) > 1e-4 for k in range(1, 6)) for i in range(len(ITEMS)))
    log(f"\n- 전체 {len(ITEMS)*5}개 계수 중 비영 **{nz}개 ({nz/(len(ITEMS)*5)*100:.0f}%)** — 엘라스틱넷 희소성.")
    for k in range(1, 6):
        log(f"  - P(Y≥{k}) 비영 문항 {int(sum(abs(coefs[k]) > 1e-4))}/21")
    log("- 상위 절단(P(Y≥4·5))=기본 ADL 주도, 하위 절단(P(Y≥1·2))=iADL 주도 → 중증 검출 메커니즘.")
    log("- CSV: b1_gn_cut_coefficients.csv\n")


def main():
    log("# Aim 2 · B1/Gn 계수·p-value 정리 (sjlee)\n")
    m, Xi, y = build()
    log(f"공통표본 {len(m)} × {len(ITEMS)}문항.\n")
    part_A(Xi, y)
    part_B(Xi, y)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
    print(">>> CSV: b1_po_coefficients.csv, b1_gn_cut_coefficients.csv")


if __name__ == "__main__":
    main()
