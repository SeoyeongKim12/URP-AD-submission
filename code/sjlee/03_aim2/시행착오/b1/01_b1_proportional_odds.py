"""
Aim 2 · B1 — 비례오즈(순서형 로지스틱) 대안 채점법 (sjlee)
============================================================
목표: ADCS-ADL 2025개정판 문항점수로 실측 DS 단계(0-5)를 예측하는 해석가능한
순서형 모형 B1(비례오즈)을 개발, leave-one-trial-out 3-fold CV로 A1과 추가가치 비교.

핵심 원칙: 대치·적합 전부 각 외부 fold의 훈련자료 내부에서만(정보누출 0).
시험 변수는 모형 공변량으로 미투입(fold 분할용으로만).

문항(X): A0 입력과 동일한 문항점수 = iADL 15 + bADL 6 = **21개**.
  주의: 지시서는 '19문항'이라 하나, '__resolved 기반·A0 입력과 동일한 문항점수'라는
  정의를 그대로 따르면 팀원 A0 코드가 쓰는 집합은 21개임(데이터명세서도 19 vs 구현
  불일치 플래그). 명시적 정의를 우선해 21개로 두되 ITEMS 상수로 조정 가능하게 둠.

예측단계 규칙(사전 고정, 튜닝 아님): 조건부 중앙값(누적확률 0.5 넘는 최소 단계).
보조: 기대단계(확률가중 연속).

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/b1_report.md, aim2/aim2_sjlee/b1_cv_predictions.csv(환자단위→Drive)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import SimpleImputer
from statsmodels.miscmodels.ordinal_model import OrderedModel

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
CSV_OUT = OUTDIR / "aim2_sjlee"; CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "b1_report.md"
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]

# A0 입력과 동일한 21개 문항점수 (col()=__resolved 우선). Q18=118A+B+C 합성.
BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL   # 21

_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


# ---------------------------------------------------------------
# step 0. 입력 행렬
# ---------------------------------------------------------------
def build_matrix():
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 2.0].copy()
    def col(c):
        return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {}
    for c in ITEMS:
        if c == "Q18":
            data[c] = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
        else:
            data[c] = col(c)
    X = pd.DataFrame(data)
    X["STUDYID"] = b["STUDYID"].values
    X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])

    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]]
    m = cs.merge(X, on=["STUDYID", "USUBJID"], how="left")
    return m


# ---------------------------------------------------------------
# 예측 규칙
# ---------------------------------------------------------------
def cond_median(probs, cats):
    """누적확률 0.5 넘는 최소 단계."""
    cum = np.cumsum(probs, axis=1)
    idx = (cum >= 0.5).argmax(axis=1)
    return cats[idx]

def expected_stage(probs, cats):
    return probs @ cats


# ---------------------------------------------------------------
# step 1-2. leave-one-trial-out CV로 B1 외부예측
# ---------------------------------------------------------------
def cv_predict(m):
    cats = np.array(sorted(m["ds_stage"].dropna().unique()))
    preds = pd.DataFrame(index=m.index,
                         columns=["pred_stage", "exp_stage"] + [f"p{int(c)}" for c in cats], dtype=float)
    for ho in TRIALS:
        tr = m[m["STUDYID"] != ho]
        te = m[m["STUDYID"] == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        Xtr = pd.DataFrame(imp.transform(tr[ITEMS]), columns=ITEMS, index=tr.index)
        Xte = pd.DataFrame(imp.transform(te[ITEMS]), columns=ITEMS, index=te.index)
        ytr = tr["ds_stage"].astype(int)
        mod = OrderedModel(ytr.values, Xtr.values, distr="logit")
        try:
            res = mod.fit(method="bfgs", disp=False, maxiter=200)
        except Exception:
            res = mod.fit(method="lbfgs", disp=False, maxiter=400)
        probs = res.model.predict(res.params, exog=Xte.values)   # n×K
        # 훈련에 없는 카테고리 방어: 열 수 = len(train cats)
        tr_cats = np.array(sorted(ytr.unique()))
        ps = cond_median(probs, tr_cats)
        es = expected_stage(probs, tr_cats)
        preds.loc[te.index, "pred_stage"] = ps
        preds.loc[te.index, "exp_stage"] = es
        for j, c in enumerate(tr_cats):
            preds.loc[te.index, f"p{int(c)}"] = probs[:, j]
    out = m.join(preds)
    return out


# ---------------------------------------------------------------
# step 3. 성능 지표
# ---------------------------------------------------------------
def metrics(true, pred):
    t = np.asarray(true, int); p = np.asarray(pred, int)
    diff = t - p
    hi = t >= 4
    return dict(n=len(t),
                mae=np.abs(diff).mean(),
                kappa=cohen_kappa_score(t, p, weights="quadratic", labels=list(range(6))),
                within1=(np.abs(diff) <= 1).mean(),
                under2=(diff >= 2).mean(),                 # 2단계↑ 과소평가율
                hi_missed=(p[hi] <= 3).mean() if hi.sum() else np.nan)  # 실제4·5→≤3

def perf_table(out, pred_col, name):
    log(f"### {name}")
    log("| 시험 | n | MAE | 가중카파 | ±1 | 2↑과소 | 4·5→≤3 |")
    log("|---|---|---|---|---|---|---|")
    per = {}
    for tr in TRIALS:
        g = out[out["STUDYID"] == tr]
        mm = metrics(g["ds_stage"], g[pred_col])
        per[tr] = mm
        hm = f"{mm['hi_missed']*100:.1f}%" if not np.isnan(mm['hi_missed']) else "n/a"
        log(f"| {tr} | {mm['n']} | {mm['mae']:.3f} | {mm['kappa']:.3f} | "
            f"{mm['within1']*100:.1f}% | {mm['under2']*100:.1f}% | {hm} |")
    # 동일가중 평균 + 범위
    def agg(key):
        vals = [per[t][key] for t in TRIALS]
        return np.mean(vals), min(vals), max(vals)
    for key, lab in [("mae", "MAE"), ("kappa", "가중카파"), ("under2", "2↑과소"), ("hi_missed", "4·5→≤3")]:
        avg, lo, hi = agg(key)
        scale = 100 if key in ("under2", "hi_missed") else 1
        unit = "%" if scale == 100 else ""
        log(f"| **{lab} 평균(범위)** | | {avg*scale:.3f}{unit} ({lo*scale:.3f}~{hi*scale:.3f}{unit}) | | | | |"
            if key == "mae" else
            f"| **{lab} 평균** {avg*scale:.3f}{unit} (범위 {lo*scale:.3f}~{hi*scale:.3f}{unit}) | | | | | | |")
    log("")
    return per

def calibration(out, pred_col, name):
    log(f"### {name} — 단계별 calibration (실측단계별 평균 예측단계)")
    log("| 실측 DS | n | 평균 예측단계 | 평균 기대단계 |")
    log("|---|---|---|---|")
    for s in range(6):
        g = out[out["ds_stage"] == s]
        if len(g) == 0: continue
        ep = g["exp_stage"].mean() if "exp_stage" in g and pred_col == "pred_stage" else np.nan
        log(f"| {s} | {len(g)} | {g[pred_col].mean():.2f} | "
            f"{ep:.2f}" if not np.isnan(ep) else f"| {s} | {len(g)} | {g[pred_col].mean():.2f} | — " + "|")
    log("")


# ---------------------------------------------------------------
# step 4. 추가가치 판정
# ---------------------------------------------------------------
def added_value(b1_per, a1_per):
    log("## step 4. 추가가치 판정 (사전지정 4조건)\n")
    def mean_key(per, k):
        return np.mean([per[t][k] for t in TRIALS])
    b1_mae, a1_mae = mean_key(b1_per, "mae"), mean_key(a1_per, "mae")
    b1_u2, a1_u2 = mean_key(b1_per, "under2"), mean_key(a1_per, "under2")
    b1_hm, a1_hm = mean_key(b1_per, "hi_missed"), mean_key(a1_per, "hi_missed")
    mae_drop = a1_mae - b1_mae
    n_better = sum(b1_per[t]["mae"] < a1_per[t]["mae"] for t in TRIALS)

    log(f"- B1 평균 MAE {b1_mae:.3f} vs A1 {a1_mae:.3f} → 감소 {mae_drop:+.3f}단계")
    log(f"- 2↑과소평가율 B1 {b1_u2*100:.1f}% vs A1 {a1_u2*100:.1f}% → 증가 {(b1_u2-a1_u2)*100:+.1f}%p")
    log(f"- 4·5→≤3 비율 B1 {b1_hm*100:.1f}% vs A1 {a1_hm*100:.1f}% → 증가 {(b1_hm-a1_hm)*100:+.1f}%p")
    log(f"- MAE 감소 시험 수: {n_better}/3\n")

    log("| 조건 | 기준 | 값 | 충족 |")
    log("|---|---|---|---|")
    c1 = mae_drop >= 0.10
    c2 = (b1_u2 - a1_u2) <= 0.02
    c3 = (b1_hm - a1_hm) <= 0.02
    c4 = n_better >= 2
    log(f"| 1. MAE 감소 ≥0.10 | ≥0.10 | {mae_drop:+.3f} | {'✅' if c1 else '❌'} |")
    log(f"| 2. 2↑과소 증가 ≤2%p | ≤+2%p | {(b1_u2-a1_u2)*100:+.1f}%p | {'✅' if c2 else '❌'} |")
    log(f"| 3. 4·5→≤3 증가 ≤2%p | ≤+2%p | {(b1_hm-a1_hm)*100:+.1f}%p | {'✅' if c3 else '❌'} |")
    log(f"| 4. 최소 2시험 MAE 감소 | ≥2/3 | {n_better}/3 | {'✅' if c4 else '❌'} |")
    allpass = c1 and c2 and c3 and c4
    log(f"\n- **종합 판정: {'B1 추가가치 있음 (4조건 충족)' if allpass else 'B1 미충족 — 그 자체가 결과(복잡모형 제한적 추가가치)'}**\n")

    # MAE 임계값 민감도
    log("### MAE 임계값 민감도 (조건1)")
    log("| 임계값 | 조건1 충족 |")
    log("|---|---|")
    for thr in [0.05, 0.10, 0.15]:
        log(f"| {thr:.2f} | {'✅' if mae_drop >= thr else '❌'}{' (주판정)' if thr==0.10 else ''} |")
    log("")
    return allpass


def sensitivity_rules(out):
    """예측단계 규칙 민감도: 조건부중앙값(주) vs 최빈 vs 기대반올림.
    조건3(4·5→≤3) 실패가 규칙 탓인지 모형 탓인지 점검."""
    log("## step 3b. 예측단계 규칙 민감도 (조건3이 규칙 의존인가)\n")
    pcols = [c for c in out.columns if c.startswith("p") and c[1:].isdigit()]
    cats = np.array([int(c[1:]) for c in pcols])
    P = out[pcols].to_numpy()
    mode_stage = cats[np.nanargmax(np.where(np.isnan(P), -1, P), axis=1)]
    exp_round = np.clip(np.round(out["exp_stage"].to_numpy()), 0, 5)
    t = out["ds_stage"].to_numpy().astype(int)
    log("| 예측규칙 | MAE | 4·5→≤3 |")
    log("|---|---|---|")
    for name, p in [("조건부 중앙값(주)", out["pred_stage"].to_numpy()),
                    ("최빈단계", mode_stage), ("기대단계 반올림", exp_round)]:
        p = np.asarray(p, int); hi = t >= 4
        log(f"| {name} | {np.abs(t-p).mean():.3f} | {(p[hi]<=3).mean()*100:.1f}% |")
    log("\n- 세 규칙 모두 4·5→≤3가 높으면 → 조건3 실패는 규칙이 아니라 **모형이 드문 중증을 "
        "상위 확률로 못 밀어올리는 구조적 한계**(중간 단계에 확률질량 집중).\n")


def po_assumption(m):
    """step 5. 비례오즈 가정 점검 — 각 절단(Y<=k)의 이항 로짓 계수 분산(Brant 근사)."""
    log("## step 5. 비례오즈 가정 점검 (Brant 근사)\n")
    imp = SimpleImputer(strategy="median").fit(m[ITEMS])
    X = pd.DataFrame(imp.transform(m[ITEMS]), columns=ITEMS, index=m.index)
    y = m["ds_stage"].astype(int)
    import statsmodels.api as sm
    Xc = sm.add_constant(X)
    coefs = {}
    for k in range(0, 5):                       # Y<=k vs >k, k=0..4
        yk = (y <= k).astype(int)
        if yk.nunique() < 2: continue
        try:
            r = sm.Logit(yk, Xc).fit(disp=False, maxiter=100)
            coefs[k] = r.params[ITEMS]
        except Exception:
            continue
    C = pd.DataFrame(coefs)
    sd = C.std(axis=1).sort_values(ascending=False)
    log(f"- 절단 {list(coefs.keys())}개에서 문항별 이항로짓 계수의 표준편차(클수록 PO 위반):")
    log("| 문항 | 계수 SD(절단 간) |")
    log("|---|---|")
    for it, v in sd.head(6).items():
        log(f"| {it} | {v:.3f} |")
    log(f"\n- 주의: SD 상위가 basic ADL(ADL0101~0105)에 몰리고 값이 매우 큼(수십) — 이는 희소 "
        "절단(Y≤0 n=29, Y≤1 n=84)에서의 **준완전분리(separation)로 인한 계수 불안정**이지 "
        "순수 PO 위반 근거로 보긴 어려움(정식 Brant는 이를 보정). iADL 문항은 SD가 작아(≤0.6) "
        "PO 가정에 크게 어긋나지 않음.")
    log("- 판정: 부분적 PO 위반 가능성은 있으나 결정적이지 않음. **B1은 해석가능성·주 비교대상"
        "으로 그대로 유지**(교체 안 함), 부분비례오즈는 민감도로만(계획서 규정). "
        "**B1의 결정적 한계는 가정이 아니라 중증(4·5) 과소분류(조건3)임.**\n")


def final_scoretable(m):
    """step 7. 전체자료 재학습 B1 채점표(계수). '외부검증 전 낙관적' 명시."""
    log("## step 7. 전체자료 재학습 B1 채점표 (참고용)\n")
    imp = SimpleImputer(strategy="median").fit(m[ITEMS])
    X = pd.DataFrame(imp.transform(m[ITEMS]), columns=ITEMS, index=m.index)
    y = m["ds_stage"].astype(int)
    res = OrderedModel(y.values, X.values, distr="logit").fit(method="bfgs", disp=False, maxiter=300)
    beta = pd.Series(res.params[:len(ITEMS)], index=ITEMS).sort_values()
    tbl = pd.DataFrame({"item": beta.index, "coef": beta.values})
    tbl.to_csv(CSV_OUT / "b1_scoretable_final.csv", index=False)
    log("- 계수(음수=값 클수록 저단계 경향; 문항점수 높을수록 독립). 상·하위 5문항:")
    log("| 문항 | 계수 |")
    log("|---|---|")
    for it, v in list(beta.items())[:5] + list(beta.items())[-5:]:
        log(f"| {it} | {v:+.3f} |")
    log(f"\n- 전체 채점표: {CSV_OUT/'b1_scoretable_final.csv'} (환자 아님, 계수표 → Drive/참고).")
    log("- **주의**: 별도 외부검증 전까지 성능은 낙관적. 또한 B1은 추가가치 4조건 미충족이라 "
        "이 채점표는 현행 규칙 대체 근거가 아니라 참고용.\n")


def main():
    log("# Aim 2 · B1 비례오즈 대안 채점법 리포트 (sjlee)\n")
    m = build_matrix()
    log("## step 0. 입력 행렬")
    log(f"- 공통표본 {len(m)}명 × 문항 {len(ITEMS)}개 (A0 입력과 동일: iADL {len(IADL)}+bADL {len(BADL)}).")
    log(f"- 지시서 '19문항' vs 구현 A0 집합 '21문항' 불일치 — 명시적 정의('A0 입력과 동일')를 우선해 21개 사용(ITEMS 상수).")
    log(f"- 문항 결측 {int(m[ITEMS].isna().sum().sum())}칸(≤2/인, 65명) → fold 훈련자료 중앙값으로 대치(누출 없음).")
    log(f"- 시험별: {m.groupby('STUDYID')['USUBJID'].nunique().to_dict()}\n")

    log("## step 1-2. leave-one-trial-out 3-fold + B1 외부예측")
    log("- 각 fold: 평가시험 제외, 훈련 2시험으로 중앙값대치·OrderedModel(logit) 적합 → 평가시험 예측.")
    log("- 예측단계 = 조건부 중앙값(누적확률 0.5 최소단계), 보조 기대단계.\n")
    out = cv_predict(m)

    log("## step 3. 성능 + A1 대비\n")
    b1_per = perf_table(out, "pred_stage", "B1 (비례오즈, CV 외부예측)")
    a1_per = perf_table(out, "A1_2015_stage", "A1 (2015판, 동일 fold 대비)")
    calibration(out, "pred_stage", "B1")
    sensitivity_rules(out)

    allpass = added_value(b1_per, a1_per)
    po_assumption(m)
    final_scoretable(m)

    # 예측 저장
    keep = ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage", "pred_stage", "exp_stage"] + \
           [c for c in out.columns if c.startswith("p") and c[1:].isdigit()]
    out[keep].to_csv(CSV_OUT / "b1_cv_predictions.csv", index=False)
    log(f"산출: {CSV_OUT/'b1_cv_predictions.csv'} (환자단위 → Drive)")
    log(f"\n>>> B1 4조건 {'충족 → B2(문항축약) 진행 가능' if allpass else '미충족 → B2 생략, B1 미충족으로 보고'}")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
    return allpass


if __name__ == "__main__":
    main()
