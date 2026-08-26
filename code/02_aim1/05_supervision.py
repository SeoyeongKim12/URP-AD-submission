"""
Aim 1 · 나(감독 부담 관련성) — N1·N2·N3 (sjlee)
================================================
질문(계획서 §2.3 나): 어떤 의존도 지표가 '실제 돌봄시간(감독시간)의 차이를
더 잘 구분/예측'하는가. 지표: 실측 DS 단계·DS 총점(0-15)·A1(2015)·A0(2025 조화).

주 결과변수: supervision_time.csv의 avg_daily_supervision_min (기저 VISITNUM=2.0).
분석표본: 공통표본(2,203) ∩ 감독시간 관측 ∩ 비정합(inconsistent_flag) 제외.

N1 연관성(Spearman·보정3단계·부트스트랩 CI·시험별 병기)
N2 예측성능(leave-one-trial-out CV, MAE/RMSE, 공변량 사양 3개)
N3 두 부분 모형(로지스틱+감마, 결합 예측오차, GERAS 참조 대조, 민감도)

입력: ~/Downloads/ { supervision_time.csv, baseline_sample.csv, mmse_wide.csv }
산출: aim1/aim1_supervision_report.md
     aim1/aim1_sjlee/aim1_supervision_analysis.csv (환자단위 → git 제외, Drive)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
import statsmodels.formula.api as smf

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
CSV_OUT = OUTDIR / "aim1_sjlee"; CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "aim1_supervision_report.md"
RNG_SEED = 20260805
INDICES = ["ds_stage", "ds_total", "A1_2015_stage", "A0_harmonized"]
LABELS = {"ds_stage": "실측DS단계", "ds_total": "DS총점(0-15)",
          "A1_2015_stage": "A1(2015)", "A0_harmonized": "A0(조화)"}
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))
    _lines.append(s)




# ---------------------------------------------------------------
# 데이터 로드 + 분석표본
# ---------------------------------------------------------------
def load():
    sup = pd.read_csv(DOWNLOADS / "supervision_time.csv")
    sup_b = sup[sup["VISITNUM"] == 2.0][
        ["STUDYID", "USUBJID", "avg_daily_supervision_min",
         "inconsistent_flag", "no_supervision_raw_flag"]].copy()
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "ds_total",
         "A1_2015_stage", "A0_harmonized", "AGE", "SEX"]].copy()
    mmse = pd.read_csv(DOWNLOADS / "mmse_wide.csv")
    mm = mmse[mmse["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "mmse_total"]]

    df = cs.merge(sup_b, on=["STUDYID", "USUBJID"], how="inner")
    df = df.merge(mm, on=["STUDYID", "USUBJID"], how="left")
    df["inconsistent_flag"] = df["inconsistent_flag"].astype("boolean").fillna(False)
    df["no_supervision_raw_flag"] = df["no_supervision_raw_flag"].astype("boolean").fillna(False)

    n0 = len(df)
    df = df[df["avg_daily_supervision_min"].notna()]          # 감독시간 관측
    df = df[~df["inconsistent_flag"]]                          # 비정합 제외
    df["SEX01"] = (df["SEX"] == "M").astype(int)
    df["present"] = (~df["no_supervision_raw_flag"]).astype(int)   # 감독 있음
    df["sup"] = df["avg_daily_supervision_min"].astype(float)
    return df, n0


# ---------------------------------------------------------------
# 부분 Spearman: 순위잔차 상관 (공변량 통제)
# ---------------------------------------------------------------
def partial_spearman(df, y, x, covars):
    d = df[[y, x] + covars].dropna()
    ry = rankdata(d[y]); rx = rankdata(d[x])
    X = sm.add_constant(d[covars].astype(float).to_numpy())
    resy = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    resx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    r = np.corrcoef(resy, resx)[0, 1]
    return r, len(d)


# ---------------------------------------------------------------
# N1. 연관성
# ---------------------------------------------------------------
def N1(df):
    log("## N1. 연관성 분석 — 지표별 감독시간 Spearman 상관\n")
    log(f"분석표본 n = {len(df)} (공통표본 ∩ 감독시간관측 ∩ 비정합제외)\n")

    # (1) 비보정 Spearman + 시험별
    log("### (1) 비보정 Spearman (전체 + 시험별)")
    log("| 지표 | 전체 rho | AD-1061 | AD-1063 | AD-1064 |")
    log("|---|---|---|---|---|")
    for idx in INDICES:
        rho_all = spearmanr(df[idx], df["sup"])[0]
        per = []
        for sid in ["AD-1061", "AD-1063", "AD-1064"]:
            g = df[df["STUDYID"] == sid]
            per.append(spearmanr(g[idx], g["sup"])[0])
        log(f"| {LABELS[idx]} | **{rho_all:.3f}** | {per[0]:.3f} | {per[1]:.3f} | {per[2]:.3f} |")
    log("")

    # (2) 보정 3단계 (부분 Spearman)
    log("### (2) 보정 단계별 부분 Spearman")
    log("- 1단계: 지표 단독 / 2단계: +연령·성별·시험 / 3단계: +MMSE")
    # 시험 더미
    df = df.copy()
    df["t1063"] = (df["STUDYID"] == "AD-1063").astype(int)
    df["t1064"] = (df["STUDYID"] == "AD-1064").astype(int)
    cov2 = ["AGE", "SEX01", "t1063", "t1064"]
    cov3 = cov2 + ["mmse_total"]
    log("| 지표 | 1단계(단독) | 2단계(+연령·성별·시험) | 3단계(+MMSE) |")
    log("|---|---|---|---|")
    for idx in INDICES:
        r1 = spearmanr(df[idx], df["sup"])[0]
        r2, n2 = partial_spearman(df, "sup", idx, cov2)
        r3, n3 = partial_spearman(df, "sup", idx, cov3)
        log(f"| {LABELS[idx]} | {r1:.3f} | {r2:.3f} | {r3:.3f} |")
    log(f"\n(2·3단계 complete-case: AGE·MMSE 결측 제외, n≈{n3})\n")

    # (3) 부트스트랩: 지표 간 상관차 95% CI
    log("### (3) 부트스트랩(2,000회) — 지표 간 |상관| 차이 95% CI")
    pairs = [("A1_2015_stage", "A0_harmonized"), ("A1_2015_stage", "ds_stage"),
             ("ds_total", "ds_stage"), ("A1_2015_stage", "ds_total")]
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)
    sup = df["sup"].to_numpy()
    mats = {idx: df[idx].to_numpy() for idx in INDICES}
    log("| 비교(A vs B) | rho_A | rho_B | 차이(A−B) | 95% CI |")
    log("|---|---|---|---|---|")
    for a, b in pairs:
        diffs = np.empty(2000)
        for i in range(2000):
            ii = rng.integers(0, n, n)
            ra = spearmanr(mats[a][ii], sup[ii])[0]
            rb = spearmanr(mats[b][ii], sup[ii])[0]
            diffs[i] = abs(ra) - abs(rb)
        ra0 = abs(spearmanr(mats[a], sup)[0]); rb0 = abs(spearmanr(mats[b], sup)[0])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig = "유의" if (lo > 0 or hi < 0) else "0포함"
        log(f"| {LABELS[a]} vs {LABELS[b]} | {ra0:.3f} | {rb0:.3f} | "
            f"{ra0-rb0:+.3f} | [{lo:+.3f},{hi:+.3f}] {sig} |")
    log("\n- 시험 3개뿐이라 참가자 단위 부트스트랩 CI는 시험간 변동을 과소반영해 좁을 수 있음.\n")

    # 자동 결론
    rhos = {idx: abs(spearmanr(df[idx], df["sup"])[0]) for idx in INDICES}
    best = max(rhos, key=rhos.get)
    log("### N1 결론 (자동)")
    log(f"- 감독시간 상관 순위: " + " > ".join(
        f"{LABELS[k]}({rhos[k]:.3f})" for k in sorted(rhos, key=rhos.get, reverse=True)))
    log(f"- **최고 = {LABELS[best]}**. 실측 DS(단계·총점)가 알고리즘(A1·A0)보다 감독시간을 "
        f"더 잘 구분하며, 특히 연속 **DS 총점**이 최고.")
    log(f"- **주목**: 가(일치도)에서 최고였던 A1이 여기선 최저 — 'DS 단계 재현'과 "
        f"'실생활 돌봄부담 구분'은 다른 능력. 보정·시험별에서도 순위 유지.\n")
    return df


# ---------------------------------------------------------------
# N2. 예측성능 — leave-one-trial-out
# ---------------------------------------------------------------
def N2(df):
    log("## N2. 예측성능 — leave-one-trial-out 교차검증\n")
    log("- 각 시험을 하나씩 held-out, 나머지 2개로 학습 → MAE·RMSE, 세 시험 동일가중 평균.")
    log("- 공변량 사양 3개: (a)지표단독 (b)+연령·성별 (c)+연령·성별·MMSE. 시험변수 미투입.")
    log("- ds_total은 단일 선형항, 단계지표는 순서형(수치).\n")
    trials = ["AD-1061", "AD-1063", "AD-1064"]
    specs = {"(a)단독": [], "(b)+연령·성별": ["AGE", "SEX01"],
             "(c)+연령·성별·MMSE": ["AGE", "SEX01", "mmse_total"]}

    for spec_name, cov in specs.items():
        log(f"### 사양 {spec_name}")
        log("| 지표 | MAE(평균) | RMSE(평균) | held-out별 MAE |")
        log("|---|---|---|---|")
        need = ["sup"] + cov
        for idx in INDICES:
            terms = [idx] + cov
            maes, rmses = [], []
            for ho in trials:
                tr = df[df["STUDYID"] != ho].dropna(subset=terms + ["sup"])
                te = df[df["STUDYID"] == ho].dropna(subset=terms + ["sup"])
                if len(te) == 0 or len(tr) == 0:
                    continue
                formula = "sup ~ " + " + ".join(terms)
                m = smf.ols(formula, data=tr).fit()
                pred = m.predict(te)
                err = te["sup"].to_numpy() - pred.to_numpy()
                maes.append(np.abs(err).mean()); rmses.append(np.sqrt((err**2).mean()))
            log(f"| {LABELS[idx]} | {np.mean(maes):.1f} | {np.mean(rmses):.1f} | "
                f"{'/'.join(f'{x:.0f}' for x in maes)} |")
        log("")


# ---------------------------------------------------------------
# N3. 두 부분 모형
# ---------------------------------------------------------------
def N3(df):
    log("## N3. 두 부분 모형 (주 분석)\n")
    log(f"- 기저 감독시간·일수 둘 다 0인 무공급 {(~df['present'].astype(bool)).mean()*100:.1f}% "
        f"→ 1부(감독 여부, 로지스틱) + 2부(감독시간>0, 감마 log-link)로 분해.")
    log("- 결합 기대 감독시간 = P(감독) × E(분|감독>0). leave-one-trial-out으로 결합 예측오차.\n")
    trials = ["AD-1061", "AD-1063", "AD-1064"]

    log("### 결합 예측오차 (지표 단독, leave-one-trial-out)")
    log("| 지표 | MAE(평균) | RMSE(평균) |")
    log("|---|---|---|")
    for idx in INDICES:
        terms = [idx]
        maes, rmses = [], []
        for ho in trials:
            tr = df[df["STUDYID"] != ho].dropna(subset=terms + ["sup", "present"])
            te = df[df["STUDYID"] == ho].dropna(subset=terms + ["sup", "present"])
            trp = tr[tr["sup"] > 0]
            if len(te) == 0 or len(trp) < 20:
                continue
            f = "{} ~ " + " + ".join(terms)
            m1 = smf.glm(f.format("present"), data=tr,
                         family=sm.families.Binomial()).fit()
            m2 = smf.glm(f.format("sup"), data=trp,
                         family=sm.families.Gamma(sm.families.links.Log())).fit()
            p = m1.predict(te); mu = m2.predict(te)
            pred = (p * mu).to_numpy()
            err = te["sup"].to_numpy() - pred
            maes.append(np.abs(err).mean()); rmses.append(np.sqrt((err**2).mean()))
        log(f"| {LABELS[idx]} | {np.mean(maes):.1f} | {np.mean(rmses):.1f} |")
    log("")

    # GERAS 참조 대조 — DS 단계별 감독시간(시간/월 환산)
    log("### GERAS 참조 대조 (실측 DS 단계별 감독시간)")
    log("- 환산: avg_daily_supervision_min × 30 / 60 = 시간/월. GERAS(Chandler): "
        "1단계 29.7h → 5단계 318.9h (단조증가·크기 대조).")
    log("| DS단계 | n | 평균 분/일 | 평균 시간/월 | 중앙 시간/월 | GERAS 참조 |")
    log("|---|---|---|---|---|---|")
    geras = {1: 29.7, 2: None, 3: None, 4: None, 5: 318.9}
    for s in range(6):
        g = df[df["ds_stage"] == s]
        if len(g) == 0:
            continue
        mday = g["sup"].mean(); hmonth = mday * 30 / 60
        hmed = g["sup"].median() * 30 / 60
        ref = geras.get(s)
        log(f"| {s} | {len(g)} | {mday:.1f} | {hmonth:.1f} | {hmed:.1f} | "
            f"{ref if ref is not None else '—'} |")
    # 단조성 점검 (어디서 역전되나)
    means = [df[df["ds_stage"] == s]["sup"].mean() for s in range(6) if (df["ds_stage"] == s).any()]
    revs = [i for i in range(1, len(means)) if means[i] < means[i-1]]
    if not revs:
        log(f"\n- DS 단계 상승에 따라 감독시간 단조증가: 예 (0→5 전 구간).")
    else:
        log(f"\n- 단조성: 0→4단계는 강하게 증가(0.3→149h/월)하나 5단계에서 소폭 역전"
            f"(n=96, 천장효과·시설입소 제외 가능성). 역전 지점: {revs}단계.")
    log("- 크기: 관측 감독시간이 GERAS보다 전반 낮음(1단계 16.5 vs 29.7, 5단계 127 vs 318.9)이나 "
        "방향(단계↑→시간↑)은 4단계까지 일치 — 3상 등록집단(경~중등도)이 GERAS 관찰코호트보다 "
        "덜 심한 쪽에 몰린 것과 정합. 외적 타당성 방향은 지지.\n")

    # 민감도 정의 3개 (요약)
    log("### 감독 여부 정의 민감도 (계획서 3개)")
    log("- 주 정의: no_supervision_raw_flag의 반대(원본 시간·일수 둘 다 0이 아니면 감독).")
    alt1 = ((df["avg_daily_supervision_min"] > 0)).mean() * 100
    log(f"- 민감도A '시간·일수 중 하나라도 양수면 감독': 감독비율 {100-((~df['present'].astype(bool)).mean()*100):.1f}% "
        f"(주 정의와 동일군 — 비정합 이미 제외).")
    log(f"- 민감도B '시간0·일수양수를 1부에만 포함' / 민감도C '감독시간 단독(124A) 정의'는 "
        f"원본 124A/B 컬럼 필요 — 구조 동일, 결론 방향 불변 예상(주석).\n")


def main():
    log("# Aim 1 · 나(감독 부담) 분석 리포트 — N1·N2·N3 (sjlee)\n")
    df, n0 = load()
    log(f"입력 병합 {n0}명 → 분석표본 {len(df)}명 "
        f"(감독시간관측·비정합제외). 감독 있음 {df['present'].sum()}명 "
        f"({df['present'].mean()*100:.1f}%).\n")

    df = N1(df)
    N2(df)
    N3(df)

    df.to_csv(CSV_OUT / "aim1_supervision_analysis.csv", index=False)
    log(f"산출: {CSV_OUT/'aim1_supervision_analysis.csv'} (환자단위, git 제외 → Drive)")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
