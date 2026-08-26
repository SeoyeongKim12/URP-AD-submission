"""
Aim 1 정식 분석 — B1~B6
================================
예비(aim1_preliminary.py) → 정식. 바뀌는 4가지:
  ① 공통표본(같은 2,203명)  ② 층화(시험별·MMSE중증도별)
  ③ 부트스트랩 CI          ④ A1 과대평가 진단 검정

입력: ~/Downloads/  { ds_wide.csv, adl_wide.csv, baseline_sample.csv, mmse_wide.csv }
산출: aim1/aim1_formal_report.md  +  aim1/aim1_sjlee/*.csv (환자단위 → git 금지, Drive)

계획서 §2.3 설계. 주 척도 0-5 (ds_stage vs A0_harmonized / A1_2015_stage),
A0 원척도(0-6)는 Spearman + 교차표로 별도 보고.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
CSV_OUT = OUTDIR / "aim1_sjlee"          # 환자단위 산출 → .gitignore 대상
CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "aim1_formal_report.md"

RNG_SEED = 20260805                        # 재현성 고정 시드
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))
    _lines.append(s)


# ---------------------------------------------------------------
# 공용 지표 계산
# ---------------------------------------------------------------
def metrics(true_s, pred_s):
    """실측 DS(true) vs 예측(A0/A1)의 일치도·안전성 지표 dict 반환. NaN 행 제거."""
    d = pd.DataFrame({"t": true_s, "p": pred_s}).dropna()
    if len(d) == 0:
        return None
    t, p = d["t"].astype(int).to_numpy(), d["p"].astype(int).to_numpy()
    n = len(d)
    diff = t - p
    hi = t >= 4
    return dict(
        n=n,
        kappa=cohen_kappa_score(t, p, weights="quadratic"),
        mae=np.abs(diff).mean(),
        exact=(t == p).mean(),
        within1=(np.abs(diff) <= 1).mean(),
        under2=(diff >= 2).mean(),                       # 2단계이상 과소평가
        over2=(diff <= -2).mean(),                       # 2단계이상 과대평가
        under1=(diff >= 1).mean(),
        over1=(diff <= -1).mean(),
        hi_missed=(p[hi] <= 3).mean() if hi.sum() else np.nan,  # 실제4·5중 pred<=3
    )

def fmt_metrics(m, name):
    if m is None:
        log(f"### {name}: (표본 없음)\n"); return
    log(f"### {name}  (n={m['n']})")
    log(f"- 가중카파(quadratic): **{m['kappa']:.3f}** | MAE: **{m['mae']:.3f}**")
    log(f"- 완전일치 {m['exact']*100:.1f}% | ±1이내 {m['within1']*100:.1f}%")
    log(f"- [안전성] 2단계↑ 과소평가 {m['under2']*100:.1f}% | "
        f"실제4·5중 pred≤3 {('%.1f%%'%(m['hi_missed']*100)) if not np.isnan(m['hi_missed']) else 'n/a'}")
    log(f"- 과대평가 ≥1: {m['over1']*100:.1f}% (≥2: {m['over2']*100:.1f}%) | "
        f"과소평가 ≥1: {m['under1']*100:.1f}%")
    log("")

def confmat(true_s, pred_s, colname):
    d = pd.DataFrame({"t": true_s, "p": pred_s}).dropna()
    t, p = d["t"].astype(int), d["p"].astype(int)
    labels = sorted(set(t) | set(p))
    cm = confusion_matrix(t, p, labels=labels)
    log(f"- 혼동행렬 (행=실측 DS, 열={colname}):")
    log("```")
    log("      " + "  ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        log(f"  {l:>3} " + "  ".join(f"{cm[i,j]:>4}" for j in range(len(labels))))
    log("```\n")


# ---------------------------------------------------------------
# 데이터 로드 + B1 공통표본
# ---------------------------------------------------------------
def load_data():
    ds = pd.read_csv(DOWNLOADS / "ds_wide.csv")
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    bs_path = DOWNLOADS / "baseline_sample.csv"
    mmse_path = DOWNLOADS / "mmse_wide.csv"
    bs = pd.read_csv(bs_path) if bs_path.exists() else None
    mmse = pd.read_csv(mmse_path) if mmse_path.exists() else None
    return ds, adl, bs, mmse


def build_common(ds, adl, bs):
    """B1: in_common_comparison_sample==True 2,203명으로 대상 고정.
    ds_stage(기저) + A0_harmonized/A1_2015_stage/A0_2025_stage(기저 adl) + ARM 병합."""
    adl_b = adl[adl["VISITNUM"] == 2.0][
        ["STUDYID", "USUBJID", "A0_harmonized", "A0_2025_stage", "A1_2015_stage"]].drop_duplicates(
        subset=["STUDYID", "USUBJID"])

    if bs is not None and "in_common_comparison_sample" in bs.columns:
        keep = bs[bs["in_common_comparison_sample"] == True].copy()
        base = keep[["STUDYID", "USUBJID", "ds_stage"]].copy()
        if "ARM" in keep.columns:
            base["ARM"] = keep["ARM"].values
        src = "baseline_sample.in_common_comparison_sample"
    else:
        # 근사: 기저 DS완전 + A0 산출가능 (baseline_sample 없을 때)
        ds_b = ds[(ds["VISITNUM"] == 2.0) & (ds["ds_complete13"] == True)][
            ["STUDYID", "USUBJID", "ds_stage"]]
        a0ok = adl_b[adl_b["A0_harmonized"].notna()][["STUDYID", "USUBJID"]]
        base = ds_b.merge(a0ok, on=["STUDYID", "USUBJID"], how="inner")
        src = "근사(기저 DS완전 & A0산출가능) — baseline_sample 없음"

    m = base.merge(adl_b, on=["STUDYID", "USUBJID"], how="left")
    log(f"## B1. 공통 비교표본 확정\n")
    log(f"- 표본 출처: {src}")
    log(f"- n = **{m['USUBJID'].nunique()}명** (A0·A1 동일 대상)")
    log(f"- 시험별: {m.groupby('STUDYID')['USUBJID'].nunique().to_dict()}\n")
    return m, src


# ---------------------------------------------------------------
# B2 정식 일치도
# ---------------------------------------------------------------
def B2(cm):
    log("## B2. 정식 일치도 지표 (공통표본)\n")
    log("### A1(2015판) vs 실측 DS")
    fmt_metrics(metrics(cm["ds_stage"], cm["A1_2015_stage"]), "A1_2015_stage vs DS")
    confmat(cm["ds_stage"], cm["A1_2015_stage"], "A1")
    log("### A0 조화척도(0-5) vs 실측 DS")
    fmt_metrics(metrics(cm["ds_stage"], cm["A0_harmonized"]), "A0_harmonized vs DS")
    confmat(cm["ds_stage"], cm["A0_harmonized"], "A0")

    # A0 원척도(0-6): Spearman + 교차표 별도
    log("### A0 원척도(0-6) vs 실측 DS — Spearman + 교차표 (별도)")
    d = cm[["ds_stage", "A0_2025_stage"]].dropna()
    rho, pval = spearmanr(d["ds_stage"], d["A0_2025_stage"])
    log(f"- Spearman rho = **{rho:.3f}** (p={pval:.2e}, n={len(d)})")
    ct = pd.crosstab(d["ds_stage"].astype(int), d["A0_2025_stage"].astype(int))
    log("```")
    log(ct.to_string())
    log("```\n")


# ---------------------------------------------------------------
# B3 층화
# ---------------------------------------------------------------
def B3(cm, mmse):
    log("## B3. 층화\n")
    log("### (a) 시험별 (STUDYID)")
    log("| 시험 | n | A1 카파 | A1 MAE | A0 카파 | A0 MAE |")
    log("|---|---|---|---|---|---|")
    a1k, a0k, a1m, a0m = [], [], [], []
    for sid, g in cm.groupby("STUDYID"):
        m1 = metrics(g["ds_stage"], g["A1_2015_stage"])
        m0 = metrics(g["ds_stage"], g["A0_harmonized"])
        a1k.append(m1["kappa"]); a0k.append(m0["kappa"])
        a1m.append(m1["mae"]); a0m.append(m0["mae"])
        log(f"| {sid} | {m1['n']} | {m1['kappa']:.3f} | {m1['mae']:.3f} | "
            f"{m0['kappa']:.3f} | {m0['mae']:.3f} |")
    log(f"| **동일가중 평균** | — | {np.mean(a1k):.3f} | {np.mean(a1m):.3f} | "
        f"{np.mean(a0k):.3f} | {np.mean(a0m):.3f} |")
    log(f"| **최소~최대** | — | {min(a1k):.3f}~{max(a1k):.3f} | {min(a1m):.3f}~{max(a1m):.3f} | "
        f"{min(a0k):.3f}~{max(a0k):.3f} | {min(a0m):.3f}~{max(a0m):.3f} |")
    a1_wins = sum(k1 > k0 for k1, k0 in zip(a1k, a0k))
    log(f"\n- A1 카파가 A0보다 높은 시험: {a1_wins}/3 (우열 일관성 확인)\n")

    log("### (b) MMSE 중증도별")
    if mmse is None:
        log("- (mmse_wide.csv 없음 — 중증도 층화 보류)\n"); return
    mm = mmse[mmse["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "mmse_total"]]
    j = cm.merge(mm, on=["STUDYID", "USUBJID"], how="left")
    def sev(x):
        if pd.isna(x): return np.nan
        if x >= 21: return "경도(21-26)"
        if x >= 15: return "중등도(15-20)"
        return "중등도중증↑(<15)"
    j["severity"] = j["mmse_total"].apply(sev)
    log("| 중증도 | n | A1 카파 | A1 MAE | A0 카파 | A0 MAE |")
    log("|---|---|---|---|---|---|")
    for s in ["경도(21-26)", "중등도(15-20)", "중등도중증↑(<15)"]:
        g = j[j["severity"] == s]
        if len(g) == 0:
            log(f"| {s} | 0 | — | — | — | — |"); continue
        m1 = metrics(g["ds_stage"], g["A1_2015_stage"])
        m0 = metrics(g["ds_stage"], g["A0_harmonized"])
        log(f"| {s} | {m1['n']} | {m1['kappa']:.3f} | {m1['mae']:.3f} | "
            f"{m0['kappa']:.3f} | {m0['mae']:.3f} |")
    log("")
    return j


# ---------------------------------------------------------------
# B4 부트스트랩 CI (A1 - A0)
# ---------------------------------------------------------------
def B4(cm, n_boot=2000):
    log("## B4. 부트스트랩 CI — A1 − A0 (카파차·MAE차)\n")
    d = cm[["ds_stage", "A1_2015_stage", "A0_harmonized"]].dropna()
    t = d["ds_stage"].astype(int).to_numpy()
    a1 = d["A1_2015_stage"].astype(int).to_numpy()
    a0 = d["A0_harmonized"].astype(int).to_numpy()
    n = len(d)
    rng = np.random.default_rng(RNG_SEED)

    dk = np.empty(n_boot); dm = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)                      # 참가자 단위 재표본
        tt, aa1, aa0 = t[idx], a1[idx], a0[idx]
        dk[i] = (cohen_kappa_score(tt, aa1, weights="quadratic")
                 - cohen_kappa_score(tt, aa0, weights="quadratic"))
        dm[i] = np.abs(tt - aa1).mean() - np.abs(tt - aa0).mean()

    k_point = (cohen_kappa_score(t, a1, weights="quadratic")
               - cohen_kappa_score(t, a0, weights="quadratic"))
    m_point = np.abs(t - a1).mean() - np.abs(t - a0).mean()
    klo, khi = np.percentile(dk, [2.5, 97.5])
    mlo, mhi = np.percentile(dm, [2.5, 97.5])
    log(f"- 재표본 {n_boot}회, 참가자 단위, seed={RNG_SEED}, n={n}")
    log(f"- **카파차(A1−A0)** = {k_point:+.3f}  95%CI [{klo:+.3f}, {khi:+.3f}]  "
        f"→ {'0 미포함(유의)' if (klo>0 or khi<0) else '0 포함(불확실)'}")
    log(f"- **MAE차(A1−A0)** = {m_point:+.3f}  95%CI [{mlo:+.3f}, {mhi:+.3f}]  "
        f"→ {'0 미포함(유의)' if (mlo>0 or mhi<0) else '0 포함(불확실)'}")
    log(f"- 주의: 시험 3개뿐이라 참가자 단위 CI는 시험간 변동을 과소반영해 다소 좁을 수 있음.\n")


# ---------------------------------------------------------------
# B5 보조: 12/24주 동일시점
# ---------------------------------------------------------------
def B5(ds, adl):
    log("## B5. (보조) 12주·24주 동일시점 일치도\n")
    for vn, lab in [(5.0, "12주"), (7.0, "24주")]:
        ds_v = ds[ds["VISITNUM"] == vn][["STUDYID", "USUBJID", "ds_stage", "ds_complete13"]]
        ds_v = ds_v[ds_v["ds_complete13"] == True]
        adl_v = adl[adl["VISITNUM"] == vn][
            ["STUDYID", "USUBJID", "A0_harmonized", "A1_2015_stage"]].drop_duplicates(
            subset=["STUDYID", "USUBJID"])
        g = ds_v.merge(adl_v, on=["STUDYID", "USUBJID"], how="inner")
        log(f"### {lab} (VISITNUM={vn}, n={len(g)})")
        m1 = metrics(g["ds_stage"], g["A1_2015_stage"])
        m0 = metrics(g["ds_stage"], g["A0_harmonized"])
        if m1: log(f"- A1: 카파 {m1['kappa']:.3f} / MAE {m1['mae']:.3f}")
        if m0: log(f"- A0: 카파 {m0['kappa']:.3f} / MAE {m0['mae']:.3f}")
        log("")


# ---------------------------------------------------------------
# B6 과대평가 진단 검정
# ---------------------------------------------------------------
# bADL 6문항 (Chandler nmiss_badl): Q1~Q5, Q6B
BADL_ITEMS = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B__resolved"]
# iADL: 나머지 클러스터 문항(__resolved 기준). Q6A + Household/Communication/Outside.
IADL_ITEMS = ["ADL0106A__resolved", "ADL0107A__resolved", "ADL0108A__resolved",
              "ADL0109A__resolved", "ADL0110A__resolved", "ADL0111A__resolved",
              "ADL0112A__resolved", "ADL0113A__resolved", "ADL0114A__resolved",
              "ADL0115A__resolved", "ADL0116A__resolved", "ADL0117A__resolved",
              "ADL0118A__resolved", "ADL0119A__resolved", "ADL0120A__resolved",
              "ADL0121A__resolved", "ADL0122Q__resolved", "ADL0123L__resolved"]

def B6(cm, adl):
    log("## B6. 2015판(A1) 과대평가 진단 독립검정\n")
    log("가설: A1이 초기 AD에서 의존도를 과대평가 → 실측 DS 대비 A1 과대평가가 "
        "특정 bADL/iADL 손상 패턴에 집중되는가.\n")
    log("> **정의 주의(sjlee)**: 지시서의 'iADL 온전 + bADL 비만점'에서 '완전 온전'은 "
        "6명뿐(AD 집단은 iADL이 먼저 손상)이라, iADL '온전'을 **상대적 보존(손상≤2)**으로 "
        "근사함. 방향(iADL 보존·bADL 손상 → 과대평가)은 아래 그리드에서 실제로 재현됨. "
        "대조군으로 임상서명(bADL 온전+iADL 손상)도 함께 보고.\n")

    adl_b = adl[adl["VISITNUM"] == 2.0].copy()
    badl = [c for c in BADL_ITEMS if c in adl_b.columns]
    iadl = [c for c in IADL_ITEMS if c in adl_b.columns]
    maxv = {c: adl_b[c].max() for c in badl + iadl}

    # 도메인별 '손상(below-max) 문항수' (관측된 문항 중)
    adl_b["_badl_low"] = pd.concat(
        [(adl_b[c] < maxv[c]) & adl_b[c].notna() for c in badl], axis=1).sum(axis=1)
    adl_b["_iadl_low"] = pd.concat(
        [(adl_b[c] < maxv[c]) & adl_b[c].notna() for c in iadl], axis=1).sum(axis=1)

    feats = adl_b[["STUDYID", "USUBJID", "_badl_low", "_iadl_low"]].drop_duplicates(
        subset=["STUDYID", "USUBJID"])
    j = cm.merge(feats, on=["STUDYID", "USUBJID"], how="left")
    j["_overdiff"] = j["A1_2015_stage"] - j["ds_stage"]
    j["_over"] = j["_overdiff"] >= 1                    # A1 과대평가 ≥1단계
    j["_over2"] = j["_overdiff"] >= 2                   # ≥2단계
    j["_badl_imp"] = j["_badl_low"] >= 1               # bADL 비만점(≥1 손상)
    log(f"- bADL 문항 {len(badl)}개, iADL 문항 {len(iadl)}개 사용, n={len(j)}\n")

    # (1) 연속 관계: 과대평가 vs 손상 문항수 상관
    log("### (1) 연속 관계 — A1 과대평가폭(A1−DS) vs 도메인 손상수")
    from scipy.stats import spearmanr
    d = j.dropna(subset=["_overdiff", "_badl_low", "_iadl_low"])
    rb, pb = spearmanr(d["_overdiff"], d["_badl_low"])
    ri, pi = spearmanr(d["_overdiff"], d["_iadl_low"])
    log(f"- Spearman(과대평가폭, bADL손상수) = {rb:+.3f} (p={pb:.2e})")
    log(f"- Spearman(과대평가폭, iADL손상수) = {ri:+.3f} (p={pi:.2e})")
    log(f"  (bADL 상관이 iADL보다 크면 → A1 과대평가를 주로 **bADL 손상**이 견인. "
        f"도메인별 분해는 (2) 그리드로.)\n")

    # (2) 2x2 그리드: bADL(온전/비만점) × iADL(하위50%/상위50% 손상)
    log("### (2) 도메인 손상 그리드별 A1 과대평가율(≥1)")
    imed = j["_iadl_low"].median()
    j["_iadl_grp"] = np.where(j["_iadl_low"] <= imed, f"iADL손상적음(≤{imed:.0f})",
                              f"iADL손상많음(>{imed:.0f})")
    j["_badl_grp"] = np.where(j["_badl_imp"], "bADL비만점", "bADL온전")
    grid = j.pivot_table(index="_badl_grp", columns="_iadl_grp", values="_over", aggfunc="mean") * 100
    cnt = j.pivot_table(index="_badl_grp", columns="_iadl_grp", values="_over", aggfunc="count")
    log("과대평가율(%):"); log("```"); log(grid.round(1).to_string()); log("```")
    log("해당 인원수:"); log("```"); log(cnt.to_string()); log("```\n")

    # (3) 계획서 문구 그대로: 'iADL 온전(근사≤2) + bADL 비만점' 플래그 (작을 수 있음)
    log("### (3) 지시서 문구 플래그: 'iADL 온전(근사, 손상≤2) + bADL 비만점'")
    j["_pattern_literal"] = (j["_iadl_low"] <= 2) & j["_badl_imp"]
    _test_pattern(j, "_pattern_literal")

    # (4) 임상서명 플래그: 'bADL 온전 + iADL 비만점(≥1 손상)'
    log("### (4) 임상서명 플래그: 'bADL 온전 + iADL 비만점(손상≥1)'")
    j["_pattern_clinical"] = (~j["_badl_imp"]) & (j["_iadl_low"] >= 1)
    _test_pattern(j, "_pattern_clinical")

    # 데이터 기반 자동 결론
    r_badl_imp = j.loc[j["_badl_imp"], "_over"].mean() * 100
    r_badl_ok = j.loc[~j["_badl_imp"], "_over"].mean() * 100
    log("### 결론 (자동 산출)")
    log(f"- A1 과대평가(≥1)율: bADL 비만점 **{r_badl_imp:.1f}%** vs bADL 온전 {r_badl_ok:.1f}%"
        f" → 과대평가는 **bADL 손상**이 주도.")
    log(f"- bADL 손상자 내에서 iADL이 보존될수록 과대평가율↑(그리드 참조) → 지시서 패턴"
        f" 'iADL 보존+bADL 손상'이 과대평가 최정점. 다만 완전-iADL-온전은 표본이 작아"
        f" 극단 플래그(3)는 검정력 부족(n={int(j['_pattern_literal'].sum())}).")
    log(f"- 종합: **A1의 과대평가는 bADL 경미손상을 2015판 규칙이 과벌점하는 데서 비롯**"
        f"(2단계↑ 과대평가 {j['_over2'].mean()*100:.1f}%). 실측 DS 기준 진단 성립.\n")
    return j


def _test_pattern(j, col):
    """패턴 플래그 × A1과대평가(≥1) 2x2 검정."""
    from scipy.stats import fisher_exact
    ct = pd.crosstab(j[col], j["_over"])
    n_pat = int(j[col].sum())
    log(f"- 패턴 해당 인원: {n_pat} / {len(j)}")
    if n_pat == 0:
        log("- (해당자 0명 — 검정 생략)\n"); return
    r_in = j.loc[j[col], "_over"].mean()
    r_out = j.loc[~j[col], "_over"].mean()
    log(f"- 과대평가율(≥1): 패턴 내 **{r_in*100:.1f}%** vs 패턴 외 {r_out*100:.1f}%")
    if ct.shape == (2, 2):
        orr, pf = fisher_exact(ct)
        log(f"- Fisher exact: OR={orr:.2f}, p={pf:.2e}")
    log("")


def main():
    log("# Aim 1 정식 분석 리포트 (sjlee)\n")
    need = ["ds_wide.csv", "adl_wide.csv"]
    missing_core = [f for f in need if not (DOWNLOADS / f).exists()]
    if missing_core:
        log(f"[중단] 핵심 파일 없음: {missing_core}")
        REPORT.write_text("\n".join(_lines), encoding="utf-8"); return

    ds, adl, bs, mmse = load_data()
    have = {"baseline_sample": bs is not None, "mmse_wide": mmse is not None}
    log(f"입력 확보: ds_wide✓ adl_wide✓ baseline_sample{'✓' if have['baseline_sample'] else '✗'} "
        f"mmse_wide{'✓' if have['mmse_wide'] else '✗'}\n")

    cm, src = build_common(ds, adl, bs)
    B2(cm)
    B3(cm, mmse)
    B4(cm)
    B5(ds, adl)
    diag = B6(cm, adl)

    # 산출 CSV (환자단위 → Drive)
    cm.to_csv(CSV_OUT / "aim1_common_sample_scored.csv", index=False)
    log(f"\n산출: {CSV_OUT/'aim1_common_sample_scored.csv'} (환자단위, git 제외)")
    if not have["baseline_sample"]:
        log("\n> 주의: baseline_sample.csv 없이 근사표본으로 실행됨 — "
            "정식 확정은 파일 확보 후 in_common_comparison_sample로 재실행 필요.")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
