"""
Aim 1 · 결측 민감도 — M1 
=================================
AD-1061의 선택적 DS 결측(기저 완전관측 미보유 ~28.5%)이 주 결론을 왜곡했는지 점검.
두 방식: M1-a(AD-1061 제외) / M1-b(AD-1061 내 안정화 IPW 가중).

판정: (가) A1>A0(단계 일치), (나) DS총점 우위(감독시간)가 유지되면 결론 견고.

입력: ~/Downloads/ { baseline_sample, supervision_time, ds_wide, mmse_wide, adl_wide, dm_filtered }
      dm_filtered.csv는 M1-b(IPW)에만 필요 — 없으면 M1-a만 실행.
산출: aim1/aim1_sensitivity_missing_report.md
      aim1/aim1_sjlee/aim1_ad1061_ipw_weights.csv (환자단위 → git 제외, Drive)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
CSV_OUT = OUTDIR / "aim1_sjlee"; CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "aim1_sensitivity_missing_report.md"
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


# ---------- 가중 지표 ----------
def wkappa(t, p, w, lo, ncat):
    O = np.zeros((ncat, ncat))
    for a, b, wi in zip(t, p, w):
        O[int(a) - lo, int(b) - lo] += wi
    O /= O.sum()
    r, c = O.sum(1), O.sum(0)
    E = np.outer(r, c)
    W = np.array([[(i - j) ** 2 for j in range(ncat)] for i in range(ncat)], float) / (ncat - 1) ** 2
    return 1 - (W * O).sum() / (W * E).sum()

def wmae(t, p, w):
    return np.average(np.abs(np.asarray(t) - np.asarray(p)), weights=w)

def wpearson(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)

def wspearman(x, y, w):
    return wpearson(rankdata(x), rankdata(y), w)


def load_base():
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "ds_total", "A1_2015_stage", "A0_harmonized"]].copy()
    sup = pd.read_csv(DOWNLOADS / "supervision_time.csv")
    s = sup[sup["VISITNUM"] == 2.0][
        ["STUDYID", "USUBJID", "avg_daily_supervision_min", "inconsistent_flag"]].copy()
    s["inconsistent_flag"] = s["inconsistent_flag"].astype("boolean").fillna(False)
    df = cs.merge(s, on=["STUDYID", "USUBJID"], how="left")
    df["sup"] = df["avg_daily_supervision_min"]
    return df


def report_metrics(df, w=None, tag=""):
    """가(카파·MAE) + 나(Spearman) 주지표. w=가중치(None=비가중)."""
    if w is None:
        w = np.ones(len(df))
    # 가: A1/A0 vs ds_stage
    g = df.dropna(subset=["ds_stage", "A1_2015_stage", "A0_harmonized"])
    wg = w[g.index] if hasattr(w, "__len__") else w
    k1 = wkappa(g["ds_stage"], g["A1_2015_stage"], wg, 0, 6)
    k0 = wkappa(g["ds_stage"], g["A0_harmonized"], wg, 0, 6)
    m1 = wmae(g["ds_stage"], g["A1_2015_stage"], wg)
    m0 = wmae(g["ds_stage"], g["A0_harmonized"], wg)
    log(f"- **가**: A1 카파 {k1:.3f} / MAE {m1:.3f} vs A0 카파 {k0:.3f} / MAE {m0:.3f} "
        f"→ A1{'>' if k1 > k0 else '<='}A0 {'유지' if k1 > k0 else '주의'}")
    # 나: 감독시간 Spearman (비정합 제외, sup 관측)
    d = df[(~df["inconsistent_flag"]) & df["sup"].notna()]
    wd = w[d.index] if hasattr(w, "__len__") else w
    rr = {c: wspearman(d[c].to_numpy(), d["sup"].to_numpy(), wd)
          for c in ["ds_total", "ds_stage", "A1_2015_stage", "A0_harmonized"]}
    order = sorted(rr, key=lambda k: -abs(rr[k]))
    lab = {"ds_total": "DS총점", "ds_stage": "DS단계", "A1_2015_stage": "A1", "A0_harmonized": "A0"}
    log(f"- **나**: 감독시간 Spearman " + " > ".join(f"{lab[k]}({rr[k]:.3f})" for k in order)
        + f" → 최고 {lab[order[0]]} {'(DS총점 유지)' if order[0]=='ds_total' else '(주의)'}")
    return k1 > k0, order[0] == "ds_total"


# ---------------------------------------------------------------
# M1-a. AD-1061 제외
# ---------------------------------------------------------------
def M1a(df):
    log("## M1-a. AD-1061 제외 분석 (AD-1063·1064만)\n")
    log("### 참조: 전체 3시험")
    report_metrics(df, tag="전체")
    log("\n### AD-1061 제외 (AD-1063+1064)")
    sub = df[df["STUDYID"] != "AD-1061"].reset_index(drop=True)
    g_ok, n_ok = report_metrics(sub, tag="제외")
    log(f"\n- 판정: 두 시험만으로도 가 A1>A0 {'유지' if g_ok else '주의'}, "
        f"나 DS총점 최고 {'유지' if n_ok else '주의'} → 결론 {'견고' if g_ok and n_ok else '재검토'}.\n")


# ---------------------------------------------------------------
# M1-b. AD-1061 안정화 IPW
# ---------------------------------------------------------------
def adcs_adl_total():
    """adl_wide 기저에서 ADCS-ADL 총점(resolved 하위문항 + 기본문항 합) 계산."""
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 2.0].copy()
    basic = [f"ADL010{i}" for i in range(1, 6)]
    resolved = [c for c in b.columns if c.endswith("__resolved")]
    items = [c for c in basic + resolved if c in b.columns]
    b["adl_total"] = b[items].sum(axis=1, min_count=1)
    return b[["STUDYID", "USUBJID", "adl_total"]].drop_duplicates(subset=["STUDYID", "USUBJID"])


def M1b(df):
    log("## M1-b. AD-1061 안정화 IPW 가중\n")
    dm_path = DOWNLOADS / "dm_filtered.csv"
    if not dm_path.exists():
        log(f"[대기] {dm_path.name} 없음 — Drive에서 받아야 실행됨. (M1-a·M2·M3는 완료)\n")
        return
    dm = pd.read_csv(dm_path)
    dm1 = dm[dm["STUDYID"] == "AD-1061"].copy()
    log(f"- AD-1061 등록자(dm_filtered): {dm1['USUBJID'].nunique()}명")

    # observed = 기저 DS 완전관측
    ds = pd.read_csv(DOWNLOADS / "ds_wide.csv")
    dsb = ds[ds["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "ds_complete13"]]
    mmse = pd.read_csv(DOWNLOADS / "mmse_wide.csv")
    mm = mmse[mmse["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "mmse_total"]]
    adlt = adcs_adl_total()

    d = dm1[["STUDYID", "USUBJID", "AGE", "SEX", "ARM"]].merge(dsb, on=["STUDYID", "USUBJID"], how="left")
    d["observed"] = (d["ds_complete13"] == True).astype(int)
    d = d.merge(mm, on=["STUDYID", "USUBJID"], how="left").merge(adlt, on=["STUDYID", "USUBJID"], how="left")
    d["SEX01"] = (d["SEX"] == "M").astype(int)

    log(f"- 관측(observed=기저 DS완전): {d['observed'].sum()} / {len(d)}")
    # 예측변수 결측 사전 확인
    preds = ["AGE", "SEX01", "mmse_total", "adl_total"]
    miss = d[preds].isna().any(axis=1)
    log(f"- 성향모형 예측변수 결측으로 제외: {int(miss.sum())}명 "
        f"(각 결측: {d[preds].isna().sum().to_dict()})")
    dd = d[~miss].copy()

    # 성향모형: observed ~ AGE+SEX+MMSE+ADL총점+ARM
    m = smf.glm("observed ~ AGE + SEX01 + mmse_total + adl_total + C(ARM)",
                data=dd, family=sm.families.Binomial()).fit()
    dd["p_obs_X"] = m.predict(dd)
    p_marg = dd["observed"].mean()
    # 안정화 가중치(관측자에게만 의미; 분석은 관측자만 들어감)
    dd["sw"] = np.where(dd["observed"] == 1, p_marg / dd["p_obs_X"],
                        (1 - p_marg) / (1 - dd["p_obs_X"]))
    # 1·99 백분위 절단
    obs = dd[dd["observed"] == 1].copy()
    lo, hi = np.percentile(obs["sw"], [1, 99])
    obs["sw_trunc"] = obs["sw"].clip(lo, hi)
    ess = obs["sw_trunc"].sum() ** 2 / (obs["sw_trunc"] ** 2).sum()
    log(f"- 안정화 가중치: 절단 [{lo:.2f}, {hi:.2f}], 관측자 {len(obs)}명 "
        f"→ 유효표본크기 ESS = {ess:.0f} (명목 {len(obs)})")
    log(f"- 가중치 요약: 평균 {obs['sw_trunc'].mean():.3f}, 범위 [{obs['sw_trunc'].min():.2f}, {obs['sw_trunc'].max():.2f}]\n")

    # 가중치표 저장
    obs[["STUDYID", "USUBJID", "observed", "p_obs_X", "sw", "sw_trunc"]].to_csv(
        CSV_OUT / "aim1_ad1061_ipw_weights.csv", index=False)

    # 가중 재계산: AD-1061 관측자에 sw_trunc, AD-1063/1064에 1
    w_map = dict(zip(obs["USUBJID"], obs["sw_trunc"]))
    dfx = df.copy().reset_index(drop=True)
    dfx["w"] = dfx.apply(lambda r: w_map.get(r["USUBJID"], 1.0)
                         if r["STUDYID"] == "AD-1061" else 1.0, axis=1)
    # AD-1061 비관측자는 df(공통표본)에 애초에 없음 — 가중은 관측 분포 보정
    log("### IPW 가중 후 주지표 (전체 3시험, AD-1061만 가중)")
    g_ok, n_ok = report_metrics(dfx, w=dfx["w"].to_numpy())
    log(f"\n- 판정: IPW 가중 후에도 가 A1>A0 {'유지' if g_ok else '주의'}, "
        f"나 DS총점 최고 {'유지' if n_ok else '주의'} → 결론 {'견고' if g_ok and n_ok else '재검토'}.")
    log("- **한계**: MAR 가정 의존(미측정 사유의 선택편향은 못 없앰). 성향모형의 ADCS-ADL "
        "총점이 평가대상 A0/A1의 입력이기도 해 보정 독립성 주장은 그만큼 약함(순환논리는 아님).")
    log(f"- 산출: {CSV_OUT/'aim1_ad1061_ipw_weights.csv'} (환자단위 → Drive)\n")


def main():
    log("# Aim 1 · 결측 민감도 리포트 — M1 (sjlee)\n")
    df = load_base()
    log(f"공통표본 {len(df)}명 (감독시간 결합).\n")
    M1a(df)
    M1b(df)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
