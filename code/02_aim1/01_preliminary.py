"""
Aim 1 예비 일치도 — 채점 컬럼 독립 재확인
==================================================
D2의 '통과 기준(= 채점 컬럼 독립 재확인)' 부분을 baseline_sample 없이
ds_wide+adl_wide 기저 조인만으로 근사 재현한다.

목적: 팀원이 산출한 A0_harmonized/A1_2015_stage가 실측 DS(ds_stage)와
명세서에 기록된 일치도(A0 카파~0.448/MAE~0.824, A1 카파~0.498/MAE~0.589)를
재현하는지 '내가 직접' 다시 계산해 확인.

주의: 정식 D2는 baseline_sample의 in_common_comparison_sample(2,203명)로 다시
계산해야 함. 여기 표본은 '기저 & DS완전 & A0(또는A1) 산출가능' 근사 표본.

산출: aim1_preliminary_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
OUTDIR.mkdir(exist_ok=True)
REPORT = OUTDIR / "aim1_preliminary_report.md"
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))
    _lines.append(s)


def concordance(true_s, pred_s, name):
    """true_s(실측 DS) vs pred_s(A0/A1)의 일치도 지표."""
    d = pd.DataFrame({"t": true_s, "p": pred_s}).dropna()
    t, p = d["t"].astype(int), d["p"].astype(int)
    n = len(d)
    kappa = cohen_kappa_score(t, p, weights="quadratic")
    mae = (t - p).abs().mean()
    exact = (t == p).mean()
    within1 = ((t - p).abs() <= 1).mean()
    # 임상 안전성: 2단계 이상 과소평가율 = (DS - pred) >= 2
    under2 = ((t - p) >= 2).mean()
    # 실제 4·5단계 중 pred<=3 비율
    hi = t >= 4
    hi_missed = (p[hi] <= 3).mean() if hi.sum() > 0 else np.nan

    log(f"### {name}  (n={n})")
    log(f"- 가중카파(quadratic): **{kappa:.3f}**")
    log(f"- MAE: **{mae:.3f}**")
    log(f"- 완전일치율: {exact*100:.1f}%  | ±1 이내: {within1*100:.1f}%")
    log(f"- [안전성] 2단계이상 과소평가율((DS-pred)>=2): {under2*100:.1f}%")
    log(f"- [안전성] 실제 4·5단계 중 pred<=3 비율: "
        f"{hi_missed*100:.1f}%" if not np.isnan(hi_missed) else "- [안전성] 고단계 없음")
    # 혼동행렬
    labels = sorted(set(t) | set(p))
    cm = confusion_matrix(t, p, labels=labels)
    log("- 혼동행렬 (행=실측 DS, 열=" + name.split()[0] + "):")
    log("```")
    log("      " + "  ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        log(f"  {l:>3} " + "  ".join(f"{cm[i,j]:>4}" for j in range(len(labels))))
    log("```")
    return kappa, mae, n


def main():
    log("# Aim 1 예비 일치도 — 채점 컬럼 독립 재확인\n")
    ds = pd.read_csv(DOWNLOADS / "ds_wide.csv")
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)

    # 기저만
    ds_b = ds[ds["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "ds_stage", "ds_complete13"]]
    adl_b = adl[adl["VISITNUM"] == 2.0][["STUDYID", "USUBJID",
              "A0_harmonized", "A0_2025_stage", "A1_2015_stage"]]
    m = ds_b.merge(adl_b, on=["STUDYID", "USUBJID"], how="inner")
    log(f"기저 DS×ADL 조인: {len(m)}명\n")

    # 근사 공통표본: DS 완전관측
    m = m[m["ds_complete13"] == True]
    log(f"DS 완전관측(ds_complete13) 표본: {len(m)}명\n")

    log("## A0 조화척도(0-5) vs 실측 DS(0-5)")
    log("명세서 기록값: 가중카파 0.448 / MAE 0.824  → 아래 재계산과 대조\n")
    concordance(m["ds_stage"], m["A0_harmonized"], "A0_harmonized vs DS")
    log("")

    log("## A1(2015판, 0-5) vs 실측 DS(0-5)")
    log("명세서 기록값: 가중카파 0.498 / MAE 0.589  → 아래 재계산과 대조\n")
    concordance(m["ds_stage"], m["A1_2015_stage"], "A1_2015_stage vs DS")
    log("")

    # A0 원척도(0-6) vs DS: Spearman
    from scipy.stats import spearmanr
    d06 = m[["ds_stage", "A0_2025_stage"]].dropna()
    rho, _ = spearmanr(d06["ds_stage"], d06["A0_2025_stage"])
    log("## A0 원척도(0-6) vs 실측 DS — Spearman 순위상관")
    log(f"- Spearman rho = {rho:.3f}  (n={len(d06)})\n")

    log("> 판정: 위 카파/MAE가 명세서 기록값에 근접하면, 팀원 A0/A1 채점 컬럼이")
    log("> 실측 DS와의 관계를 문서대로 재현함 = 채점 로직 독립 재확인 통과.")
    log("> (정식 D2는 baseline_sample.in_common_comparison_sample 2,203명 + 시험별")
    log(">  층화 + MMSE중증도 층화로 재계산 필요 — mmse_wide/baseline_sample 확보 후.)")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
