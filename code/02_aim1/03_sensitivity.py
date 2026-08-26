"""
Aim 1 · 가(일치도) 민감도분석 — S1, S2 
================================================
주 분석(aim1_formal.py B2)의 A1>A0 결론이 두 민감도에서 유지되는지 확인.

S1. 0·1단계 통합 5범주: ds_stage·A1·A0를 0→1 clip 후 일치도 재계산.
S2. 비위계(non-hierarchical) 실측 DS 응답자 식별·기술 후 제외 민감도.

입력: ~/Downloads/ { baseline_sample.csv, ds_wide.csv }
산출: aim1/aim1_sensitivity_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "aim1_sensitivity_report.md"
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))
    _lines.append(s)


def metrics(true_s, pred_s):
    d = pd.DataFrame({"t": true_s, "p": pred_s}).dropna()
    if len(d) == 0:
        return None
    t, p = d["t"].astype(int).to_numpy(), d["p"].astype(int).to_numpy()
    diff = t - p
    hi = t >= 4
    return dict(n=len(d),
                kappa=cohen_kappa_score(t, p, weights="quadratic"),
                mae=np.abs(diff).mean(),
                exact=(t == p).mean(), within1=(np.abs(diff) <= 1).mean(),
                under2=(diff >= 2).mean(),
                hi_missed=(p[hi] <= 3).mean() if hi.sum() else np.nan)

def row(m, label):
    if m is None:
        return f"| {label} | 0 | — | — | — | — |"
    return (f"| {label} | {m['n']} | {m['kappa']:.3f} | {m['mae']:.3f} | "
            f"{m['exact']*100:.1f}% | {m['under2']*100:.1f}% |")


def load_common():
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True].copy()
    return cs


# ---------------------------------------------------------------
# S1. 0·1 통합 5범주
# ---------------------------------------------------------------
def S1(cs):
    log("## S1. 0·1단계 통합 5범주 (0→1 clip)\n")
    log("- 0단계·1단계가 희소(합 ~3.6%) — 통합해도 A1>A0 결론이 유지되는지 확인.\n")
    d = cs.copy()
    for c in ["ds_stage", "A1_2015_stage", "A0_harmonized"]:
        d[c + "_c"] = d[c].clip(lower=1)

    log("| 척도 | n | 가중카파 | MAE | 완전일치 | 2단계↑과소 |")
    log("|---|---|---|---|---|---|")
    log(row(metrics(d["ds_stage"], d["A1_2015_stage"]), "A1 (원 6범주)"))
    log(row(metrics(d["ds_stage_c"], d["A1_2015_stage_c"]), "A1 (5범주)"))
    log(row(metrics(d["ds_stage"], d["A0_harmonized"]), "A0 (원 6범주)"))
    log(row(metrics(d["ds_stage_c"], d["A0_harmonized_c"]), "A0 (5범주)"))
    k1 = metrics(d["ds_stage_c"], d["A1_2015_stage_c"])["kappa"]
    k0 = metrics(d["ds_stage_c"], d["A0_harmonized_c"])["kappa"]
    log(f"\n- 5범주에서도 A1 카파({k1:.3f}) {'>' if k1 > k0 else '<='} A0({k0:.3f}) "
        f"→ 결론 {'유지' if k1 > k0 else '주의'}.\n")


# ---------------------------------------------------------------
# S2. 비위계 실측 DS 응답자
# ---------------------------------------------------------------
# DS 단계 티어(경→중): 하위(1·2단계) DS01-04, 3단계 DS05-07, 4단계 DS08-10, 5단계 DS11-13
TIERS = [("T_low", [1, 2, 3, 4]), ("T3", [5, 6, 7]), ("T4", [8, 9, 10]), ("T5", [11, 12, 13])]

def S2(cs):
    log("## S2. 비위계(non-hierarchical) 실측 DS 응답자 제외\n")
    log("- DS 위계 가정: 더 심한(상위 티어) 문항을 의존하면 덜 심한(하위 티어) 문항도 "
        "의존해야 함. 위반 = 상위 티어 문항을 endorse(=1)했는데 그 아래 어떤 티어는 "
        "전부 0(건너뜀).\n")
    log("> 조작적 정의(sjlee): 티어 [DS01-04 / DS05-07 / DS08-10 / DS11-13]에서 각 티어 "
        "endorse 여부를 구해, '최고 endorse 티어보다 낮은 티어 중 하나라도 endorse=0'이면 "
        "비위계로 판정.\n")

    ds = pd.read_csv(DOWNLOADS / "ds_wide.csv")
    ds = ds[(ds["VISITNUM"] == 2.0) & (ds["ds_complete13"] == True)].copy()

    def tier_endorsed(rowv):
        return [int(any(rowv[f"DS{i:02d}"] == 1 for i in items)) for _, items in TIERS]

    te = ds.apply(lambda r: pd.Series(tier_endorsed(r), index=[t for t, _ in TIERS]), axis=1)
    ds = pd.concat([ds, te], axis=1)

    tier_names = [t for t, _ in TIERS]
    def nonhier(r):
        endorsed_idx = [i for i, tn in enumerate(tier_names) if r[tn] == 1]
        if not endorsed_idx:
            return False                      # 전부 0 → 0단계, 위계 위반 아님
        h = max(endorsed_idx)
        return any(r[tier_names[j]] == 0 for j in range(h))   # h 아래 빈 티어 존재
    ds["_nonhier"] = ds.apply(nonhier, axis=1)

    n_total = len(ds)
    n_nh = int(ds["_nonhier"].sum())
    log(f"- 기저 DS 완전관측 {n_total}명 중 비위계 **{n_nh}명 ({n_nh/n_total*100:.1f}%)**\n")

    # 유형 기술: 어떤 티어가 비어서 위반됐나
    log("### 비위계 유형 (최고 endorse 티어 × 건너뛴 하위 티어)")
    types = {}
    for _, r in ds[ds["_nonhier"]].iterrows():
        endorsed_idx = [i for i, tn in enumerate(tier_names) if r[tn] == 1]
        h = max(endorsed_idx)
        skipped = [tier_names[j] for j in range(h) if r[tier_names[j]] == 0]
        key = f"최고={tier_names[h]}, 건너뜀={'+'.join(skipped)}"
        types[key] = types.get(key, 0) + 1
    for k, v in sorted(types.items(), key=lambda x: -x[1])[:10]:
        log(f"- {v:4d}명: {key if False else k}")
    log("")

    # 제외 후 일치도 (공통표본에서 비위계 제외)
    nh_ids = set(map(tuple, ds.loc[ds["_nonhier"], ["STUDYID", "USUBJID"]].values))
    cs2 = cs.copy()
    cs2["_nh"] = cs2.apply(lambda r: (r["STUDYID"], r["USUBJID"]) in nh_ids, axis=1)
    kept = cs2[~cs2["_nh"]]
    log(f"### 비위계 제외 후 일치도 (공통표본 {len(cs2)} → {len(kept)})")
    log("| 척도 | n | 가중카파 | MAE | 완전일치 | 2단계↑과소 |")
    log("|---|---|---|---|---|---|")
    log(row(metrics(kept["ds_stage"], kept["A1_2015_stage"]), "A1 (비위계 제외)"))
    log(row(metrics(kept["ds_stage"], kept["A0_harmonized"]), "A0 (비위계 제외)"))
    m1 = metrics(kept["ds_stage"], kept["A1_2015_stage"])
    m0 = metrics(kept["ds_stage"], kept["A0_harmonized"])
    log(f"\n- 제외 후에도 A1({m1['kappa']:.3f}) {'>' if m1['kappa'] > m0['kappa'] else '<='} "
        f"A0({m0['kappa']:.3f}) → 결론 {'유지' if m1['kappa'] > m0['kappa'] else '주의'}.\n")


def main():
    log("# Aim 1 · 가(일치도) 민감도분석 리포트 (S1·S2, sjlee)\n")
    cs = load_common()
    log(f"공통표본 n = {len(cs)}\n")
    S1(cs)
    S2(cs)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
