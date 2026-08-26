"""
Aim 2 · Gn 정식 확정 — 엘라스틱넷 부분비례오즈 (sjlee)
======================================================
사전지정: 부분비례오즈=비평행 누적로짓, 절단별 elastic-net(l1_ratio=0.5), 표준화.
결정규칙: 비대칭 임계(τ fold내부). **[중증우선 통일]** 제약=중증놓침 ≤ A0목표, 목적=MAE 최소.
하이퍼파라미터 λ=C: **중첩 CV로 훈련 내부에서만 선택**(inner-val에서 A1 지배+A0안전 통과수 기준).
판정: A1 양지표(MAE·중증) 지배 + 중증놓침 A0수준(≤A0+2%p). (구 4조건 MAE감소≥0.10 골격은 폐기.)

추가 점검: 비평행 모형의 비단조 확률(P(Y≥k)가 k에서 증가) 발생률·클립된 음수질량.

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/b1_gn_elasticnet_report.md, aim2/aim2_sjlee/gn_en_cv_predictions.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
CSV_OUT = OUTDIR / "aim2_sjlee"; CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "b1_gn_elasticnet_report.md"
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)
L1_RATIO = 0.5                       # 사전지정
C_GRID = [0.1, 0.25, 0.5, 1.0]       # λ 후보(내부CV 선택)
MARGIN_GRID = [0.10, 0.12, 0.14]     # MAE 안전마진 후보(내부CV 선택)
BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


def build_matrix():
    adl = pd.read_csv(DOWNLOADS / "adl_wide.csv", low_memory=False)
    b = adl[adl["VISITNUM"] == 2.0].copy()
    def col(c): return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {c: ((col("ADL0118A") + col("ADL0118B") + col("ADL0118C")) if c == "Q18" else col(c)) for c in ITEMS}
    X = pd.DataFrame(data); X["STUDYID"] = b["STUDYID"].values; X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]]
    return cs.merge(X, on=["STUDYID", "USUBJID"], how="left")

def cond_median(P): return STAGES[(np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)]
def asym_rule(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cond_median(P)))
def m_mae(t, p): return np.abs(np.asarray(t) - np.asarray(p)).mean()
def m_u2(t, p):  return (np.asarray(t) - np.asarray(p) >= 2).mean()
def m_hi(t, p):
    t = np.asarray(t); p = np.asarray(p); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan


def fit_en(train_df, C):
    """절단별 elastic-net 로지스틱. probs(rows)와 geq(rows) 반환. 표준화 포함."""
    imp = SimpleImputer(strategy="median").fit(train_df[ITEMS])
    sc = StandardScaler().fit(imp.transform(train_df[ITEMS]))
    Ztr = sc.transform(imp.transform(train_df[ITEMS]))
    y = train_df["ds_stage"].astype(int).values
    models = {k: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=L1_RATIO,
                                    C=C, max_iter=5000, tol=1e-3).fit(Ztr, (y >= k).astype(int))
              for k in range(1, 6)}
    def geq(rows):
        Z = sc.transform(imp.transform(rows[ITEMS]))
        return {k: models[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    def probs(rows):
        g = geq(rows)
        P = np.zeros((len(rows), 6)); P[:, 0] = 1 - g[1]
        for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
        P[:, 5] = g[5]
        P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
    return probs, geq, y, models

def tune_tau(Ptr, ds_tr, a0_target):
    """[중증우선/통일] 제약=중증놓침 ≤ A0목표, 목적=MAE 최소.
    (도달 불가 시 중증놓침 최소.)"""
    grid = np.round(np.arange(0.10, 0.55, 0.025), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym_rule(Ptr, t4, t5)
            if m_hi(ds_tr, p) <= a0_target + 1e-9:
                key = m_mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    if best is None:
        cand = [(m_hi(ds_tr, asym_rule(Ptr, a, b)), m_mae(ds_tr, asym_rule(Ptr, a, b)), a, b)
                for a in grid for b in grid if b >= a]
        _, _, a, b = min(cand); return a, b
    return best[1], best[2]

def eval4(m, pred):
    """[중증우선 판정] A1 양지표 지배 + 중증놓침 A0수준(≤A0+2%p) 안전."""
    g = lambda t: (m.loc[m.STUDYID == t, "ds_stage"], pred.loc[m.STUDYID == t],
                   m.loc[m.STUDYID == t, "A1_2015_stage"], m.loc[m.STUDYID == t, "A0_harmonized"])
    bm = {t: m_mae(g(t)[0], g(t)[1]) for t in TRIALS}; am = {t: m_mae(g(t)[0], g(t)[2]) for t in TRIALS}
    bh = {t: m_hi(g(t)[0], g(t)[1]) for t in TRIALS};  ah = {t: m_hi(g(t)[0], g(t)[2]) for t in TRIALS}
    bu = {t: m_u2(g(t)[0], g(t)[1]) for t in TRIALS};  au = {t: m_u2(g(t)[0], g(t)[2]) for t in TRIALS}
    mean = lambda d: np.mean([d[t] for t in TRIALS])
    mae_drop = mean(am) - mean(bm); d_hi = mean(bh) - mean(ah); d_u2 = mean(bu) - mean(au)
    nb = sum(bm[t] < am[t] for t in TRIALS)
    a0_hi = m_hi(m["ds_stage"], m["A0_harmonized"])                 # 전체 A0 중증놓침
    dominate = (mean(bm) < mean(am)) and (mean(bh) <= mean(ah)) and (nb >= 2)  # A1 양지표 지배
    safe = mean(bh) <= a0_hi + 0.02                                 # 중증 A0수준
    kap = cohen_kappa_score(m["ds_stage"].astype(int), pred.astype(int), weights="quadratic", labels=list(range(6)))
    return dict(mae=mean(bm), mae_drop=mae_drop, hi=mean(bh), d_hi=d_hi, u2=mean(bu), d_u2=d_u2,
                nb=nb, kappa=kap, a0_hi=a0_hi, dominate=dominate, safe=safe,
                allpass=dominate and safe)


def nested_gn_en(m):
    """중첩 CV: C를 outer 훈련 2시험의 inner leave-one-trial로만 선택(중증우선).
    내부기준: 각 C에서 중증우선 τ튜닝 → inner-val에서 'A1 양지표 지배 & 중증 A0수준' 통과수,
    동률이면 inner-val MAE 낮은 C."""
    out = m.copy(); out["Gn"] = np.nan; chosen = {}
    for ho in TRIALS:
        ins = [t for t in TRIALS if t != ho]
        score = {}
        for C in C_GRID:
            npass = 0; maes = []
            for iv in ins:
                it = [t for t in ins if t != iv][0]
                tri = m[m.STUDYID == it]; vai = m[m.STUDYID == iv]
                probs, _, yi, _ = fit_en(tri, C)
                a0t = m_hi(tri["ds_stage"], tri["A0_harmonized"])          # inner-train A0 목표
                t4, t5 = tune_tau(probs(tri), tri["ds_stage"].values, a0t)
                pv = asym_rule(probs(vai), t4, t5)
                vm = m_mae(vai["ds_stage"], pv); vh = m_hi(vai["ds_stage"], pv)
                a1m = m_mae(vai["ds_stage"], vai["A1_2015_stage"]); a1h = m_hi(vai["ds_stage"], vai["A1_2015_stage"])
                a0h = m_hi(vai["ds_stage"], vai["A0_harmonized"])
                if (vm < a1m) and (vh <= a1h) and (vh <= a0h + 0.02):      # 지배 & 안전
                    npass += 1
                maes.append(vm)
            score[C] = (npass, -np.mean(maes))
        # 통과 많은 것, 동률이면 inner MAE 낮은 것
        bestC = max(score, key=lambda k: score[k])
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        probs, _, yt, _ = fit_en(tr, bestC)
        a0t = m_hi(tr["ds_stage"], tr["A0_harmonized"])
        t4, t5 = tune_tau(probs(tr), tr["ds_stage"].values, a0t)
        out.loc[te.index, "Gn"] = asym_rule(probs(te), t4, t5)
        chosen[ho] = dict(C=bestC, inner_pass=score[bestC][0])
    log(f"- 중첩CV 선택(fold별, 중증우선): {chosen}\n")
    return out


def monotonic_check(m):
    """비단조 확률 점검: P(Y≥k)가 k에서 증가(단조감소 위반)한 사람 비율 + 클립된 음수질량."""
    log("## 비단조 확률 점검 (비평행 모형 진단)")
    viol_any, neg_mass, n = 0, 0.0, 0
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        probs, geq, _, _ = fit_en(tr, 0.25)      # 대표 C
        g = geq(te)
        G = np.column_stack([g[k] for k in range(1, 6)])   # n×5, P(Y≥1..5)
        # 단조감소여야: G[:,k] >= G[:,k+1]
        dec_ok = np.all(np.diff(G, axis=1) <= 1e-9, axis=1)
        viol_any += int((~dec_ok).sum())
        # 클립 전 원 P(Y=k) 음수질량
        P0 = np.zeros((len(te), 6)); P0[:, 0] = 1 - G[:, 0]
        for k in range(1, 5): P0[:, k] = G[:, k - 1] - G[:, k]
        P0[:, 5] = G[:, 4]
        neg_mass += float(np.clip(-P0, 0, None).sum())
        n += len(te)
    log(f"- 비단조(P(Y≥k) 증가) 발생 인원: {viol_any} / {n} ({viol_any/n*100:.1f}%)")
    log(f"- 클립된 음수 확률질량 총합: {neg_mass:.3f} (인당 평균 {neg_mass/n:.4f})")
    if viol_any / n < 0.05 and neg_mass / n < 0.01:
        log("- 판정: 비단조 미미(<5% 인원, 인당 음수질량<0.01) → 클립·정규화로 충분, 결과 신뢰 가능.\n")
    else:
        log("- 판정: 비단조 상당함 → 단조 제약(isotonic) 또는 결합 순서형 모형 검토 필요.\n")


def main():
    log("# Aim 2 · Gn 엘라스틱넷 정식 확정 리포트 (sjlee) — 중증우선 통일판\n")
    m = build_matrix()
    # A0 병합(중증우선 목표·판정용) — build_matrix는 미변경(다른 스크립트 보호)
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    bs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "A0_harmonized"]]
    m = m.merge(bs, on=["STUDYID", "USUBJID"], how="left")
    a0_hi = m_hi(m["ds_stage"], m["A0_harmonized"]); a1_hi = m_hi(m["ds_stage"], m["A1_2015_stage"])
    a1_mae = m_mae(m["ds_stage"], m["A1_2015_stage"])
    log(f"공통표본 {len(m)} × {len(ITEMS)}문항. elastic-net(l1_ratio={L1_RATIO}), 표준화.")
    log(f"기준 A1 MAE {a1_mae:.3f}/중증 {a1_hi*100:.1f}%, A0 중증 {a0_hi*100:.1f}%.")
    log(f"**τ 튜닝(통일)**: 제약=중증놓침 ≤ A0, 목적=MAE 최소. C는 중첩CV로 훈련 내부 선택({C_GRID}).\n")

    log("## Gn 정식 (엘라스틱넷 + 중첩CV, 누출 없음, 중증우선)")
    out = nested_gn_en(m)
    r = eval4(m, out["Gn"])
    log(f"- MAE {r['mae']:.3f} | 가중카파 {r['kappa']:.3f} | **4·5→≤3 {r['hi']*100:.1f}%** | 2↑과소 {r['u2']*100:.1f}%")
    log(f"- vs A1: MAE 감소 {r['mae_drop']:+.3f}, 중증 {(-r['d_hi'])*100:+.1f}%p 우위, MAE감소 시험 {r['nb']}/3")
    log(f"- **판정: A1 양지표 지배 {'✅' if r['dominate'] else '❌'} + 중증 A0수준(≤{a0_hi*100+2:.1f}%) "
        f"{'✅' if r['safe'] else '❌'} → {'★통과' if r['allpass'] else '미충족'}**\n")

    log("## 견고성 — 여러 λ(C)에서 지배+안전 유지 (중증우선 τ)")
    log("| C(λ) | MAE | 4·5→≤3 | A1지배 | 안전(≤A0+2%p) |")
    log("|---|---|---|---|---|")
    for C in C_GRID:
        oo = m.copy(); oo["p"] = np.nan
        for ho in TRIALS:
            tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
            probs, _, yt, _ = fit_en(tr, C)
            a0t = m_hi(tr["ds_stage"], tr["A0_harmonized"])
            t4, t5 = tune_tau(probs(tr), tr["ds_stage"].values, a0t)
            oo.loc[te.index, "p"] = asym_rule(probs(te), t4, t5)
        rr = eval4(m, oo["p"])
        log(f"| {C} | {rr['mae']:.3f} | {rr['hi']*100:.1f}% | {'✅' if rr['dominate'] else '❌'} | "
            f"{'✅' if rr['safe'] else '❌'} |")
    log("\n- 넓은 λ 범위에서 지배+안전이 유지되는가(견고성 근거).\n")

    monotonic_check(m)

    out[["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage", "Gn"]].to_csv(
        CSV_OUT / "gn_en_cv_predictions.csv", index=False)
    log(f"산출: {CSV_OUT/'gn_en_cv_predictions.csv'} (환자단위 → Drive)")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
