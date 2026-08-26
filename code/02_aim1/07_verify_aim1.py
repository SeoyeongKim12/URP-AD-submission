"""
Aim 1 결과 독립 검증 코드 (시연)
==================================
목적: 팀원이 산출한 aim1/ 결과물(Aim1_정식결과.md, aim1_*_report.md,
aim1_common_sample_scored.csv 등)을 팀원 코드(aim1_formal.py 등)를 실행하지 않고,
원자료(dependence_study_csv/)에서 처음부터 새로 짠 코드로 핵심 수치를 재계산해 대조한다.

검증 대상: 계획서(연구계획서 PDF, Aim 1) §2.3 "가"(준거타당도)·"나"(감독시간 관련성).

입력(원자료, urp-AD/dependence_study_csv/):
  - baseline_sample.csv : 기저 분석표본 + ds_stage/A0_harmonized/A1_2015_stage
  - mmse_wide.csv        : MMSE 총점(방문별)
  - supervision_time.csv : RUD 감독시간 파생값

대조 대상(팀원 산출물, urp-AD/aim1/):
  - Aim1_정식결과.md, aim1_formal_report.md, aim1_supervision_report.md

주의: 이 코드는 팀원 코드(aim1_formal.py, aim1_supervision.py)와 로직을 공유하지 않고
독립적으로 작성함. 같은 원자료에서 같은 숫자가 나오는지만 확인하는 것이 목적이며,
부트스트랩 CI(B4)·두 부분 모형(N3)·민감도분석(S1/S2/M1/M2/M3)은 이번 검증 범위에
포함하지 않음(로직 검토는 별도로 완료, 수치 재현은 미실시).

결과 대조표는 aim1_검증_결과_sy.md 참고.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

RAW_DIR = Path(__file__).parent / "dependence_study_csv"
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def agreement_metrics(true_s, pred_s):
    """실측 DS(true) vs 예측(A0/A1) 일치도 지표. 가중카파(quadratic)·MAE·완전일치·±1이내."""
    d = pd.DataFrame({"t": true_s, "p": pred_s}).dropna()
    t, p = d["t"].astype(int), d["p"].astype(int)
    return dict(
        n=len(d),
        kappa=cohen_kappa_score(t, p, weights="quadratic"),
        mae=(t - p).abs().mean(),
        exact=(t == p).mean(),
        within1=((t - p).abs() <= 1).mean(),
    )


def main():
    log("# Aim 1 독립 검증 실행 로그\n")

    # -----------------------------------------------------------
    # 1) 공통표본 확인
    # -----------------------------------------------------------
    bs = pd.read_csv(RAW_DIR / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True].copy()
    log("## 1. 공통표본 (in_common_comparison_sample==True)")
    log(f"- n = {len(cs)}")
    log(f"- 시험별: {cs.groupby('STUDYID')['USUBJID'].nunique().to_dict()}\n")

    # -----------------------------------------------------------
    # 2) B2 — 준거타당도 (전체)
    # -----------------------------------------------------------
    log("## 2. 준거타당도 — A1/A0 vs 실측 DS (전체 공통표본)")
    m1 = agreement_metrics(cs["ds_stage"], cs["A1_2015_stage"])
    m0 = agreement_metrics(cs["ds_stage"], cs["A0_harmonized"])
    log(f"- A1: n={m1['n']} kappa={m1['kappa']:.3f} MAE={m1['mae']:.3f} "
        f"완전일치={m1['exact']*100:.1f}% ±1이내={m1['within1']*100:.1f}%")
    log(f"- A0: n={m0['n']} kappa={m0['kappa']:.3f} MAE={m0['mae']:.3f} "
        f"완전일치={m0['exact']*100:.1f}% ±1이내={m0['within1']*100:.1f}%\n")

    # -----------------------------------------------------------
    # 3) B3 — 층화 (시험별 / MMSE 중증도별)
    # -----------------------------------------------------------
    log("## 3. 층화 — 시험별")
    for sid, g in cs.groupby("STUDYID"):
        gm1 = agreement_metrics(g["ds_stage"], g["A1_2015_stage"])
        gm0 = agreement_metrics(g["ds_stage"], g["A0_harmonized"])
        log(f"- {sid} (n={len(g)}): A1 kappa={gm1['kappa']:.3f} MAE={gm1['mae']:.3f} | "
            f"A0 kappa={gm0['kappa']:.3f} MAE={gm0['mae']:.3f}")
    log("")

    log("## 4. 층화 — MMSE 기저 중증도별")
    mmse = pd.read_csv(RAW_DIR / "mmse_wide.csv")
    mm = mmse[mmse["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "mmse_total"]]
    j = cs.merge(mm, on=["STUDYID", "USUBJID"], how="left")

    def sev(x):
        if pd.isna(x):
            return np.nan
        if x >= 21:
            return "경도(21-26)"
        if x >= 15:
            return "중등도(15-20)"
        return "중등도중증↑(<15)"

    j["severity"] = j["mmse_total"].apply(sev)
    for s in ["경도(21-26)", "중등도(15-20)", "중등도중증↑(<15)"]:
        g = j[j["severity"] == s]
        gm1 = agreement_metrics(g["ds_stage"], g["A1_2015_stage"])
        gm0 = agreement_metrics(g["ds_stage"], g["A0_harmonized"])
        log(f"- {s} (n={len(g)}): A1 kappa={gm1['kappa']:.3f} MAE={gm1['mae']:.3f} | "
            f"A0 kappa={gm0['kappa']:.3f} MAE={gm0['mae']:.3f}")
    log("")

    # -----------------------------------------------------------
    # 5) N1 — 감독시간 연관성 (비보정 Spearman)
    # -----------------------------------------------------------
    log("## 5. 감독시간 연관성 — 비보정 Spearman (공통표본 ∩ 감독시간관측 ∩ 비정합제외)")
    sup = pd.read_csv(RAW_DIR / "supervision_time.csv")
    sup_b = sup[sup["VISITNUM"] == 2.0][
        ["STUDYID", "USUBJID", "avg_daily_supervision_min", "inconsistent_flag"]].copy()
    sup_b["inconsistent_flag"] = sup_b["inconsistent_flag"].astype("boolean").fillna(False)

    df = cs[["STUDYID", "USUBJID", "ds_stage", "ds_total",
             "A1_2015_stage", "A0_harmonized"]].merge(
        sup_b, on=["STUDYID", "USUBJID"], how="inner")
    n0 = len(df)
    df = df[df["avg_daily_supervision_min"].notna()]
    df = df[~df["inconsistent_flag"]]
    log(f"- 병합 n={n0} -> 분석표본 n={len(df)}")
    for idx in ["ds_stage", "ds_total", "A1_2015_stage", "A0_harmonized"]:
        rho = spearmanr(df[idx], df["avg_daily_supervision_min"])[0]
        log(f"- {idx}: rho={rho:.3f}")
    log("")

    Path(__file__).with_name("aim1_verify_log.txt").write_text(
        "\n".join(OUT), encoding="utf-8")
    print("\n>>> 저장: aim1_verify_log.txt")


if __name__ == "__main__":
    main()
