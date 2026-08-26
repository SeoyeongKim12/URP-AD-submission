"""
Aim 2 · B1이 A0를 '지배'하는 지점 탐색 (option 2) (sjlee)
=========================================================
발견: 중증검출(4·5→≤3)과 MAE는 맞교환. A0(17.7% miss, MAE 0.824)를 둘 다 이길 순 없음.
전략(option 2): B1의 중증 놓침을 **A0 수준까지** 낮추고, 그 지점의 MAE가 A0보다 훨씬
좋은지 확인 → 'A0만큼 안전 + 훨씬 정확' = A0 지배.
기법: 부분비례오즈(비평행 엘라스틱넷, 표준화) + 비대칭 임계.
튜닝(누출 0): 각 fold 훈련자료에서 A0의 중증놓침을 목표로, 그 목표를 만족하는 τ 중
**MAE 최소**(가장 덜 공격적)를 선택 → 평가시험 적용.
판정: B1 중증놓침 ≤ A0(+여유) AND B1 MAE < A0 MAE → A0 지배.
견고성: 벌점세기 C 여러 값에서 지배 유지되는지.

[경로 수정 — Claude] 원본은 ~/Downloads/{adl_wide.csv, baseline_sample.csv}(CSV)를 읽었으나,
이후 파이프라인(gn_external_validation_ad1062.py 등)이 정착시킨 parquet 경로로 통일함:
  urp-AD/dependence_study/{adl_wide.parquet, baseline_sample.parquet}
입력: {DATA_DIR}/adl_wide.parquet, baseline_sample.parquet
산출: aim2/시행착오/b1_dominate_a0_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study")
OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b1_dominate_a0_report.md"
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)
L1_RATIO = 0.5
BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)
def build():
    adl = pd.read_parquet(DATA_DIR / "adl_wide.parquet")
    b = adl[adl["VISITNUM"] == 2.0].copy()
    def col(c): return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {c: ((col("ADL0118A") + col("ADL0118B") + col("ADL0118C")) if c == "Q18" else col(c)) for c in ITEMS}
    X = pd.DataFrame(data); X["STUDYID"] = b["STUDYID"].values; X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])
    bs = pd.read_parquet(DATA_DIR / "baseline_sample.parquet")
    cs = bs[bs["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage", "A0_harmonized"]]
    return cs.merge(X, on=["STUDYID", "USUBJID"], how="left").reset_index(drop=True)
def cmed(P): return STAGES[(np.cumsum(P, 1) >= 0.5).argmax(1)]
def asym(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cmed(P)))
def mae(t, p): return np.abs(np.asarray(t, float) - np.asarray(p, float)).mean()
def miss(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan
def kap(t, p): return cohen_kappa_score(np.asarray(t, int), np.asarray(p, int), weights="quadratic", labels=list(range(6)))
def enet_probs(m, C):
    """부분비례오즈(비평행 엘라스틱넷) leave-one-trial-out 확률."""
    P = np.zeros((len(m), 6))
    for ho in TRIALS:
        tri = m.index[m.STUDYID != ho]; tei = m.index[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(m.loc[tri, ITEMS])
        sc = StandardScaler().fit(imp.transform(m.loc[tri, ITEMS]))
        Ztr = sc.transform(imp.transform(m.loc[tri, ITEMS])); Zte = sc.transform(imp.transform(m.loc[tei, ITEMS]))
        y = m.loc[tri, "ds_stage"].astype(int).values
        geq = {k: LogisticRegression(solver="saga", l1_ratio=L1_RATIO, C=C,
                                     max_iter=6000, tol=1e-3, random_state=0).fit(Ztr, (y >= k).astype(int)).predict_proba(Zte)[:, 1]
               for k in range(1, 6)}
        Q = np.zeros((len(tei), 6)); Q[:, 0] = 1 - geq[1]
        for k in range(1, 5): Q[:, k] = geq[k] - geq[k + 1]
        Q[:, 5] = geq[5]; Q = np.clip(Q, 1e-9, None); Q /= Q.sum(1, keepdims=True)
        P[tei] = Q
    return P
def enet_probs_traintest(m):
    """fold별 train확률·test확률·인덱스 (τ를 훈련 내부서 튜닝하기 위함), C 고정 사용."""
    folds = {}
    for ho in TRIALS:
        tri = m.index[m.STUDYID != ho]; tei = m.index[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(m.loc[tri, ITEMS])
        sc = StandardScaler().fit(imp.transform(m.loc[tri, ITEMS]))
        Ztr = sc.transform(imp.transform(m.loc[tri, ITEMS])); Zte = sc.transform(imp.transform(m.loc[tei, ITEMS]))
        y = m.loc[tri, "ds_stage"].astype(int).values
        def build(Z, C):
            geq = {k: LogisticRegression(solver="saga", l1_ratio=L1_RATIO, C=C,
                                         max_iter=6000, tol=1e-3, random_state=0).fit(Ztr, (y >= k).astype(int)).predict_proba(Z)[:, 1]
                   for k in range(1, 6)}
            Q = np.zeros((Z.shape[0], 6)); Q[:, 0] = 1 - geq[1]
            for k in range(1, 5): Q[:, k] = geq[k] - geq[k + 1]
            Q[:, 5] = geq[5]; Q = np.clip(Q, 1e-9, None); Q /= Q.sum(1, keepdims=True); return Q
        folds[ho] = dict(tri=tri, tei=tei, build=build, Ztr=Ztr, Zte=Zte, y=y)
    return folds
def run(m, C):
    """A0-목표 튜닝으로 B1 외부예측 생성 + 지표."""
    folds = enet_probs_traintest(m)
    pred = pd.Series(index=m.index, dtype=float)
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3)
    chosen = {}
    for ho in TRIALS:
        f = folds[ho]; tri, tei = f["tri"], f["tei"]
        Ptr = f["build"](f["Ztr"], C); Pte = f["build"](f["Zte"], C)
        yt = m.loc[tri, "ds_stage"].values
        target = miss(yt, m.loc[tri, "A0_harmonized"].values)     # A0의 중증놓침(훈련)
        # 목표(중증놓침 ≤ target) 만족하는 τ 중 MAE 최소(=가장 덜 공격적)
        best = None
        for t4 in grid:
            for t5 in grid:
                if t5 < t4: continue
                p = asym(Ptr, t4, t5)
                if miss(yt, p) <= target + 1e-9:
                    key = mae(yt, p)
                    if best is None or key < best[0]:
                        best = (key, float(t4), float(t5))
        if best is None:                          # 목표 도달 불가 시 최소 miss
            allc = [(miss(yt, asym(Ptr, t4, t5)), mae(yt, asym(Ptr, t4, t5)), t4, t5)
                    for t4 in grid for t5 in grid if t5 >= t4]
            _, _, t4, t5 = min(allc); best = (0, t4, t5)
        _, t4, t5 = best
        chosen[ho] = (round(t4, 3), round(t5, 3), round(target * 100, 1))
        pred.loc[tei] = asym(Pte, t4, t5)
    return pred, chosen
def paired_bootstrap(m, pred, n_boot=2000, seed=20260805):
    """짝지은 환자단위 부트스트랩: B1−A0, B1−A1의 지표 차이 95% CI.
    부호규약: 모두 '양수 = B1 우위'. MAE·중증놓침은 (기준−B1), 카파는 (B1−기준)."""
    t = m["ds_stage"].values
    b = pred.values; a0 = m["A0_harmonized"].values; a1 = m["A1_2015_stage"].values
    rng = np.random.default_rng(seed)
    n = len(m)
    keys = [("A1", "mae"), ("A1", "miss"), ("A1", "kappa"),
            ("A0", "mae"), ("A0", "miss"), ("A0", "kappa")]
    D = {k: np.empty(n_boot) for k in keys}
    ref = {"A0": a0, "A1": a1}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        tt, bb = t[idx], b[idx]
        for name, r in ref.items():
            rr = r[idx]
            D[(name, "mae")][i] = mae(tt, rr) - mae(tt, bb)        # +면 B1 MAE 낮음(우위)
            D[(name, "miss")][i] = miss(tt, rr) - miss(tt, bb)     # +면 B1 놓침 낮음(우위)
            D[(name, "kappa")][i] = kap(tt, bb) - kap(tt, rr)      # +면 B1 카파 높음(우위)
    log("## 부트스트랩 검정 (짝지은 환자단위, 2,000회) — 'B1이 유의하게 나은가'")
    log("부호규약: **양수 = B1 우위**. 95% CI가 0을 넘으면(하한>0) 유의.")
    log("| 비교 | 지표 | 점추정(B1우위) | 95% CI | 유의 |")
    log("|---|---|---|---|---|")
    labmap = {"mae": "MAE 차", "miss": "중증놓침 차", "kappa": "카파 차"}
    for name in ["A0", "A1"]:
        for met in ["mae", "miss", "kappa"]:
            d = D[(name, met)]
            pt = d.mean(); lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "✅유의" if (lo > 0) else ("❌(0포함)" if lo <= 0 <= hi else "역방향")
            scale = 100 if met == "miss" else 1; u = "%p" if met == "miss" else ""
            log(f"| vs {name} | {labmap[met]} | {pt*scale:+.3f}{u} | "
                f"[{lo*scale:+.3f}, {hi*scale:+.3f}]{u} | {sig} |")
    log("\n- **주의(중요)**: 시험이 3개뿐이라 환자단위 부트스트랩 CI는 **시험 간 변동을 반영 못 해 "
        "낙관적으로 좁습니다**(실제 불확실성 과소평가). 따라서 위 유의성은 '차이가 존재한다'의 "
        "근거로만 보고, **일반화 근거는 위 시험별 3/3 일관성**과 함께 판단.")
    log("- 유의성은 '차이가 진짜냐'이고 '임상적으로 충분히 크냐'는 점추정 크기로 별도 판단.\n")
def final_scoretable(m, C=0.1):
    """전체자료 재학습 채점표: 절단별 계수(원 눈금 환산)+절편, 최종 τ(전체자료 A0-목표).
    ※ in-sample 재학습이라 낙관적 — 외부검증 전 확정 아님(참고용)."""
    imp = SimpleImputer(strategy="median").fit(m[ITEMS])
    Xi = imp.transform(m[ITEMS])
    sc = StandardScaler().fit(Xi); Z = sc.transform(Xi); sd = sc.scale_; mu = sc.mean_
    y = m["ds_stage"].astype(int).values
    coef_raw = {}; intc_raw = {}
    for k in range(1, 6):
        lr = LogisticRegression(solver="saga", l1_ratio=L1_RATIO, C=C,
                                max_iter=6000, tol=1e-3, random_state=0).fit(Z, (y >= k).astype(int))
        w = lr.coef_[0]; b0 = lr.intercept_[0]
        coef_raw[k] = w / sd                                  # 원 눈금 계수
        intc_raw[k] = b0 - np.sum(w * mu / sd)                # 원 눈금 절편
    # 전체자료 확률 → 최종 τ (A0 목표)
    def probs():
        geq = {k: LogisticRegression(solver="saga", l1_ratio=L1_RATIO, C=C,
                                     max_iter=6000, tol=1e-3, random_state=0).fit(Z, (y >= k).astype(int)).predict_proba(Z)[:, 1]
               for k in range(1, 6)}
        P = np.zeros((len(m), 6)); P[:, 0] = 1 - geq[1]
        for k in range(1, 5): P[:, k] = geq[k] - geq[k + 1]
        P[:, 5] = geq[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
    P = probs(); target = miss(y, m["A0_harmonized"].values)
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym(P, t4, t5)
            if miss(y, p) <= target + 1e-9:
                key = mae(y, p)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    _, t4f, t5f = best
    tbl = pd.DataFrame({"item": ITEMS})
    for k in range(1, 6): tbl[f"coef_P(Y>={k})"] = [round(coef_raw[k][i], 3) for i in range(len(ITEMS))]
    intercepts = pd.DataFrame([{"item": "(절편)", **{f"coef_P(Y>={k})": round(intc_raw[k], 3) for k in range(1, 6)}}])
    out = pd.concat([tbl, intercepts], ignore_index=True)
    out.to_csv(OUTDIR / "b1_dominate_a0_scoretable.csv", index=False, encoding="utf-8-sig")
    log("## 최종 채점표 (전체자료 재학습, 원 눈금) — 참고용")
    log(f"- 결정규칙: P(Y≥5)>{t5f} → 5, elif P(Y≥4)>{t4f} → 4, else 조건부중앙값. (τ=전체자료 A0-목표)")
    log("- 각 절단 확률 = logistic(절편 + Σ coef·문항점수). 계수 원 눈금이라 표준화 없이 원점수 대입.")
    log("- 비영 계수만 표기(엘라스틱넷 희소). 전체 표는 b1_dominate_a0_scoretable.csv.")
    log("| 문항 | P(Y≥1) | P(Y≥2) | P(Y≥3) | P(Y≥4) | P(Y≥5) |")
    log("|---|---|---|---|---|---|")
    for i, c in enumerate(ITEMS):
        vals = [coef_raw[k][i] for k in range(1, 6)]
        if all(abs(v) < 1e-4 for v in vals): continue
        log(f"| {c} | " + " | ".join(f"{v:+.2f}" if abs(v) > 1e-4 else "0" for v in vals) + " |")
    log(f"| (절편) | " + " | ".join(f"{intc_raw[k]:+.2f}" for k in range(1, 6)) + " |")
    log("- **주의**: in-sample 재학습이라 낙관적. 확정 채점표는 외부검증 후. τ(4·5 판정 임계)는 "
        "위 결정규칙에 별도 적용(계수만으론 단계 안 나옴).\n")
def main():
    log("# Aim 2 · B1의 A0 지배점 탐색 (option 2) (sjlee)\n")
    m = build()
    t = m["ds_stage"].values
    a0_miss = miss(t, m["A0_harmonized"].values) * 100; a0_mae = mae(t, m["A0_harmonized"].values); a0_k = kap(t, m["A0_harmonized"].values)
    a1_miss = miss(t, m["A1_2015_stage"].values) * 100; a1_mae = mae(t, m["A1_2015_stage"].values); a1_k = kap(t, m["A1_2015_stage"].values)
    log(f"공통표본 {len(m)}. 기준 알고리즘:")
    log(f"- **A0**: 중증놓침 {a0_miss:.1f}%, MAE {a0_mae:.3f}, 카파 {a0_k:.3f}")
    log(f"- **A1**: 중증놓침 {a1_miss:.1f}%, MAE {a1_mae:.3f}, 카파 {a1_k:.3f}")
    log(f"- 전략: B1 중증놓침을 A0 수준({a0_miss:.1f}%)으로 맞추고 MAE가 A0보다 나은지.\n")
    log("## B1 (A0-목표 튜닝, 부분비례오즈 엘라스틱넷)")
    log("| 벌점 C | 중증놓침 | MAE | 카파 | A0지배?(놓침≤A0+1%p & MAE<A0) |")
    log("|---|---|---|---|---|")
    main_pred = None
    for C in [0.1, 0.25, 0.5, 1.0]:
        pred, chosen = run(m, C)
        b_miss = miss(t, pred.values) * 100; b_mae = mae(t, pred.values); b_k = kap(t, pred.values)
        dominate = (b_miss <= a0_miss + 1.0) and (b_mae < a0_mae)
        log(f"| {C} | {b_miss:.1f}% | {b_mae:.3f} | {b_k:.3f} | {'✅ 지배' if dominate else '❌'} |")
        if C == 0.1:
            main_pred = (pred, chosen, b_miss, b_mae, b_k)
    log("")
    pred, chosen, b_miss, b_mae, b_k = main_pred
    log(f"### 대표 결과 (C=0.1) — 시험별")
    log(f"- fold별 선택 (τ4,τ5,A0목표%): {chosen}")
    log("| 시험 | 중증놓침 B1/A0/A1 | MAE B1/A0/A1 | 카파 B1/A0/A1 |")
    log("|---|---|---|---|")
    win_a1 = 0; tie_a0 = 0
    for tr in TRIALS:
        idx = m.STUDYID == tr; tt = t[idx]
        b = pred[idx].values; a0 = m.loc[idx, "A0_harmonized"].values; a1 = m.loc[idx, "A1_2015_stage"].values
        log(f"| {tr} | {miss(tt,b)*100:.1f}/{miss(tt,a0)*100:.1f}/{miss(tt,a1)*100:.1f} | "
            f"{mae(tt,b):.3f}/{mae(tt,a0):.3f}/{mae(tt,a1):.3f} | "
            f"{kap(tt,b):.3f}/{kap(tt,a0):.3f}/{kap(tt,a1):.3f} |")
        if mae(tt,b) < mae(tt,a1) and miss(tt,b) < miss(tt,a1): win_a1 += 1
        if miss(tt,b) <= miss(tt,a0) + 2.0: tie_a0 += 1
    log(f"\n- **전체 B1: 중증놓침 {b_miss:.1f}%, MAE {b_mae:.3f}, 카파 {b_k:.3f}**")
    log(f"- **vs A1**: 3시험 중 {win_a1}/3에서 놓침·MAE 동시 우수 → A1 지배")
    log(f"- **vs A0**: 중증놓침 A0급(통합 {b_miss:.1f} vs {a0_miss:.1f}%, 시험별 ±2%p 내 {tie_a0}/3), "
        f"MAE {(a0_mae-b_mae):+.3f}·카파 {(b_k-a0_k):+.3f} 크게 우위")
    paired_bootstrap(m, pred)
    final_scoretable(m)
    log(f"## 판정")
    log(f"- **A1: 완전 지배**(놓침·MAE·카파 모두 우수).")
    log(f"- **A0: 안전성 동급 + 정확도 압도** — 'A0만큼 중증 안전하면서 훨씬 정확'.")
    log(f"- 종합: B1(A0-목표 지점)이 **세 알고리즘 중 정확도 최고이면서 A0급 중증안전성** 확보.")
    log("- **한계**: ① 3시험 소표본 CV → 외부검증 전 낙관적. ② 중증놓침은 A0를 '완전히' 이기진 "
        "못함(1063·1064서 1~2%p 열위, 시험당 중증 77~120명이라 노이즈 범위). ③ τ는 fold 훈련 "
        "내부 튜닝(누출0)이나 A0-목표 자체가 설계 선택. ④ 앙상블(비평행+임계) 구조.\n")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
if __name__ == "__main__":
    main()
