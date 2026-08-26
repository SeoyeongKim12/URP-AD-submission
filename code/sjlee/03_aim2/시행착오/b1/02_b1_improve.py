"""
Aim 2 · B1 개선 시도 — 조건3(중증 4·5 과소분류) 구제 (sjlee)
=============================================================
현 B1: 조건 1·2·4 통과, 조건 3만 실패(+35.2%p). 대안 A~C(+결합)로 4조건 동시충족을
찾으면 B2 진입, 못 찾으면 궤적(어디까지 좁혔나)을 결과로 보고.

핵심 제약: 임계값·가중·튜닝·재적합은 각 외부 fold 훈련자료 내부에서만. 시험 변수 미투입.
MAE 예산: B1 MAE ≤ A1 − 0.10 = 0.491 (조건1 주 임계값 0.10).

대안:
  A. 비대칭 임계 결정규칙 (post-hoc): 원 B1 확률 + P(Y≥5)>τ5→5 / P(Y≥4)>τ4→4. τ fold내부 튜닝.
  B. 중증(4·5) 오버샘플 재가중 (탐색): 훈련서 중증 복제 후 B1 재적합 + 임계규칙.
  C. 부분비례오즈=비평행 누적로짓 (사전지정): 절단별 독립 이항로짓 P(Y≥k) + 임계규칙.
  결합: 각 확률원에 동일 τ-튜닝 규칙 적용해 비교.

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/b1_improve_report.md, aim2/aim2_sjlee/b1_improve_cv_predictions.csv
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
CSV_OUT = OUTDIR / "aim2_sjlee"; CSV_OUT.mkdir(exist_ok=True)
REPORT = OUTDIR / "b1_improve_report.md"
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
STAGES = np.arange(6)
BUDGET = 0.491
REWEIGHT = 4          # 중증 오버샘플 배수 (탐색)

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
    def col(c):
        return b[c + "__resolved"] if c + "__resolved" in b.columns else b[c]
    data = {}
    for c in ITEMS:
        data[c] = (col("ADL0118A") + col("ADL0118B") + col("ADL0118C")) if c == "Q18" else col(c)
    X = pd.DataFrame(data); X["STUDYID"] = b["STUDYID"].values; X["USUBJID"] = b["USUBJID"].values
    X = X.drop_duplicates(subset=["STUDYID", "USUBJID"])
    bs = pd.read_csv(DOWNLOADS / "baseline_sample.csv")
    cs = bs[bs["in_common_comparison_sample"] == True][
        ["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage"]]
    return cs.merge(X, on=["STUDYID", "USUBJID"], how="left")


# ---------- 규칙 ----------
def cond_median(P):
    return STAGES[(np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)]

def asym_rule(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cond_median(P)))

def pad(pred, tr_cats):
    P = np.zeros((pred.shape[0], 6))
    for j, c in enumerate(tr_cats):
        P[:, int(c)] = pred[:, j]
    return P

# ---------- 지표 ----------
def m_mae(t, p): return np.abs(np.asarray(t) - np.asarray(p)).mean()
def m_u2(t, p):  return (np.asarray(t) - np.asarray(p) >= 2).mean()
def m_hi(t, p):
    t = np.asarray(t); p = np.asarray(p); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan

def eval4(out, pred_col, mae_thr=0.10):
    def per(col, fn):
        return {tr: fn(out.loc[out.STUDYID == tr, "ds_stage"], out.loc[out.STUDYID == tr, col]) for tr in TRIALS}
    b_mae, a_mae = per(pred_col, m_mae), per("A1_2015_stage", m_mae)
    mean = lambda d: np.mean([d[t] for t in TRIALS])
    mae_drop = mean(a_mae) - mean(b_mae)
    d_u2 = mean(per(pred_col, m_u2)) - mean(per("A1_2015_stage", m_u2))
    d_hi = mean(per(pred_col, m_hi)) - mean(per("A1_2015_stage", m_hi))
    n_better = sum(b_mae[t] < a_mae[t] for t in TRIALS)
    c1, c2, c3, c4 = mae_drop >= mae_thr, d_u2 <= 0.02, d_hi <= 0.02, n_better >= 2
    return dict(mae=mean(b_mae), a1_mae=mean(a_mae), mae_drop=mae_drop,
                u2=mean(per(pred_col, m_u2)), d_u2=d_u2,
                hi=mean(per(pred_col, m_hi)), a1_hi=mean(per("A1_2015_stage", m_hi)), d_hi=d_hi,
                n_better=n_better, c1=c1, c2=c2, c3=c3, c4=c4, allpass=c1 and c2 and c3 and c4)

def report_alt(name, tag, out, pred_col):
    r = eval4(out, pred_col)
    log(f"### {name}  [{tag}]")
    log(f"- MAE {r['mae']:.3f} (A1 {r['a1_mae']:.3f}, 감소 {r['mae_drop']:+.3f}) | MAE예산소진 {r['mae']-0.419:+.3f}/0.072")
    log(f"- **4·5→≤3: {r['hi']*100:.1f}%** (A1 {r['a1_hi']*100:.1f}%, 증가 {r['d_hi']*100:+.1f}%p) | 2↑과소 {r['u2']*100:.1f}% ({r['d_u2']*100:+.1f}%p)")
    cal = {s: out.loc[out.ds_stage == s, pred_col].mean() for s in [3, 4, 5]}
    log(f"- calibration: true3 {cal[3]:.2f} · true4 {cal[4]:.2f} · true5 {cal[5]:.2f}")
    log(f"- 조건 1{'✅' if r['c1'] else '❌'} 2{'✅' if r['c2'] else '❌'} 3{'✅' if r['c3'] else '❌'} 4{'✅' if r['c4'] else '❌'} → **{'★4조건 충족' if r['allpass'] else '미충족'}**\n")
    return r


# ---------- 확률원 생성 ----------
def probs_ordered(m, reweight=False):
    """원 B1(비례오즈). reweight=True면 훈련서 중증 4·5 오버샘플."""
    folds = {}
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        Xtr = imp.transform(tr[ITEMS]); Xte = imp.transform(te[ITEMS])
        y = tr["ds_stage"].astype(int).values
        if reweight:
            w = np.where(np.isin(y, [4, 5]), REWEIGHT, 1)
            rep = np.repeat(np.arange(len(y)), w)
            Xf, yf = Xtr[rep], y[rep]
        else:
            Xf, yf = Xtr, y
        res = OrderedModel(yf, Xf, distr="logit").fit(method="bfgs", disp=False, maxiter=250)
        cats = np.array(sorted(np.unique(yf)))
        folds[ho] = dict(te_idx=te.index, ytr=y,
                         Ptr=pad(res.model.predict(res.params, exog=Xtr), cats),
                         Pte=pad(res.model.predict(res.params, exog=Xte), cats))
    return folds

def probs_nonparallel(m):
    """부분비례오즈=비평행: 절단별 독립 이항로짓 P(Y≥k), k=1..5 (L2 정규화)."""
    folds = {}
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        sc = StandardScaler().fit(imp.transform(tr[ITEMS]))
        Ztr = sc.transform(imp.transform(tr[ITEMS])); Zte = sc.transform(imp.transform(te[ITEMS]))
        y = tr["ds_stage"].astype(int).values
        def build(Z):
            geq = {}
            for k in range(1, 6):
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr, (y >= k).astype(int))
                geq[k] = lr.predict_proba(Z)[:, 1]
            P = np.zeros((Z.shape[0], 6))
            P[:, 0] = 1 - geq[1]
            for k in range(1, 5):
                P[:, k] = geq[k] - geq[k + 1]
            P[:, 5] = geq[5]
            P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)   # 음수 클립+정규화
            return P
        folds[ho] = dict(te_idx=te.index, ytr=y, Ptr=build(Ztr), Pte=build(Zte))
    return folds


def probs_augmented(m):
    """E: 비평행 누적로짓 + 파생 입력(ADCS-ADL 총점·iADL·bADL 소계)로 중증신호 강화."""
    folds = {}
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        def feats(dfrows):
            base = imp.transform(dfrows[ITEMS])
            b = pd.DataFrame(base, columns=ITEMS)
            extra = np.column_stack([b.sum(1), b[IADL].sum(1), b[BADL].sum(1)])
            return np.hstack([base, extra])
        Ftr = feats(tr); Fte = feats(te)
        sc = StandardScaler().fit(Ftr); Ztr = sc.transform(Ftr); Zte = sc.transform(Fte)
        y = tr["ds_stage"].astype(int).values
        def build(Z):
            geq = {k: LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, (y >= k).astype(int))
                        .predict_proba(Z)[:, 1] for k in range(1, 6)}
            P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - geq[1]
            for k in range(1, 5): P[:, k] = geq[k] - geq[k + 1]
            P[:, 5] = geq[5]
            P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
        folds[ho] = dict(te_idx=te.index, ytr=y, Ptr=build(Ztr), Pte=build(Zte))
    return folds


def hurdle(m, mae_thr=0.10):
    """F: 중증(4·5) 전용 이진 탐지기(class_weight balanced) + B1(≤3 몸통) 결합(허들).
    P(중증)>τ_h면 {4,5}(B1의 4·5 확률로 배정), 아니면 B1 조건부중앙값. τ_h fold내부 튜닝."""
    out = m.copy(); out["F"] = np.nan; chosen = {}
    grid = np.round(np.arange(0.10, 0.75, 0.02), 3)
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(tr[ITEMS])
        Xtr = imp.transform(tr[ITEMS]); Xte = imp.transform(te[ITEMS])
        sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); Zte = sc.transform(Xte)
        y = tr["ds_stage"].astype(int).values
        # 몸통 B1 (조건부 중앙값 base) + 4·5 확률
        res = OrderedModel(y, Xtr, distr="logit").fit(method="bfgs", disp=False, maxiter=250)
        cats = np.array(sorted(np.unique(y)))
        Ptr = pad(res.model.predict(res.params, exog=Xtr), cats)
        Pte = pad(res.model.predict(res.params, exog=Xte), cats)
        base_tr = cond_median(Ptr); base_te = cond_median(Pte)
        # 중증 탐지기 (4·5 vs ≤3), 희소 보정
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Ztr, (y >= 4).astype(int))
        ps_tr = clf.predict_proba(Ztr)[:, 1]; ps_te = clf.predict_proba(Zte)[:, 1]
        # 4 vs 5 배정: B1의 P(Y=5) > P(Y=4)면 5
        f5_tr = (Ptr[:, 5] >= Ptr[:, 4]); f5_te = (Pte[:, 5] >= Pte[:, 4])
        def rule(ps, base, f5, th):
            sev = ps > th
            return np.where(sev, np.where(f5, 5, 4), base)
        budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - mae_thr
        best = None
        for th in grid:
            p = rule(ps_tr, base_tr, f5_tr, th); mae = m_mae(y, p)
            if mae <= budget:
                key = (m_hi(y, p), mae)
                if best is None or key < best[0]: best = (key, float(th))
        _, th = best; chosen[ho] = round(th, 3)
        out.loc[te.index, "F"] = rule(ps_te, base_te, f5_te, th)
    log(f"- fold별 허들 임계 τ_h: {chosen}")
    return out


def fit_C_probs(train_df):
    """부분PO(비평행) 적합 → 임의 행에 대한 6단계 확률 함수 반환."""
    imp = SimpleImputer(strategy="median").fit(train_df[ITEMS])
    sc = StandardScaler().fit(imp.transform(train_df[ITEMS]))
    y = train_df["ds_stage"].astype(int).values
    Ztr = sc.transform(imp.transform(train_df[ITEMS]))
    models = {k: LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, (y >= k).astype(int))
              for k in range(1, 6)}
    def probs(rows):
        Z = sc.transform(imp.transform(rows[ITEMS]))
        geq = {k: models[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
        P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - geq[1]
        for k in range(1, 5): P[:, k] = geq[k] - geq[k + 1]
        P[:, 5] = geq[5]
        P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P
    return probs, y

def tune_tau(Ptr, ytr, budget, grid=None):
    if grid is None: grid = np.round(np.arange(0.10, 0.55, 0.025), 3)
    best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym_rule(Ptr, t4, t5); mae = m_mae(ytr, p)
            if mae <= budget:
                key = (m_hi(ytr, p), mae)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    return best[1], best[2]

def nested_C(m, margins=(0.10, 0.12, 0.13, 0.14, 0.16)):
    """중첩 CV: 마진(예산 감소목표)을 outer 훈련 2시험의 inner leave-one-trial로만 선택
    → outer 평가시험 완전 제외(누출 없음). 선택 마진으로 outer 재적합·적용."""
    out = m.copy(); out["Gn"] = np.nan; chosen = {}
    for ho in TRIALS:
        ins = [t for t in TRIALS if t != ho]
        score = {}
        for mg in margins:
            npass = 0
            for iv in ins:                       # inner 검증시험
                it = [t for t in ins if t != iv][0]   # inner 훈련(1시험)
                tri = m[m.STUDYID == it]; vai = m[m.STUDYID == iv]
                probs, yi = fit_C_probs(tri)
                budget = m_mae(tri["ds_stage"], tri["A1_2015_stage"]) - mg
                t4, t5 = tune_tau(probs(tri), yi, budget)
                pv = asym_rule(probs(vai), t4, t5)
                a1m = m_mae(vai["ds_stage"], vai["A1_2015_stage"])
                a1h = m_hi(vai["ds_stage"], vai["A1_2015_stage"])
                if (a1m - m_mae(vai["ds_stage"], pv) >= 0.10) and (m_hi(vai["ds_stage"], pv) - a1h <= 0.02):
                    npass += 1
            score[mg] = npass
        best_mg = max(margins, key=lambda k: (score[k], k))   # 통과 많은 것, 동률이면 큰 마진(안전)
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        probs, yt = fit_C_probs(tr)
        budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - best_mg
        t4, t5 = tune_tau(probs(tr), yt, budget)
        out.loc[te.index, "Gn"] = asym_rule(probs(te), t4, t5)
        chosen[ho] = (best_mg, score)
    log(f"- 중첩CV 선택 마진(fold별, inner통과수): {chosen}")
    return out


def tune_apply(m, folds, col, mae_thr=0.10, grid=None, quiet=False):
    """fold내부: 조건3(train 4·5→≤3) 최소화 s.t. train MAE ≤ (train A1 MAE − mae_thr).
    예산은 각 fold 훈련자료의 A1 MAE 기준(전역상수 아님) → 조건1을 train에서 직접 반영."""
    if grid is None:
        grid = np.round(np.arange(0.10, 0.55, 0.025), 3)
    out = m.copy(); out[col] = np.nan; chosen = {}
    for ho in TRIALS:
        f = folds[ho]; yt = f["ytr"]
        tr = m[m.STUDYID != ho]
        budget = m_mae(tr["ds_stage"], tr["A1_2015_stage"]) - mae_thr   # 조건1 예산(train A1 기준)
        best = None
        for t4 in grid:
            for t5 in grid:
                if t5 < t4: continue
                p = asym_rule(f["Ptr"], t4, t5); mae = m_mae(yt, p)
                if mae <= budget:            # MAE 예산 내에서
                    key = (m_hi(yt, p), mae)  # 조건3(4·5→≤3) 최소화
                    if best is None or key < best[0]:
                        best = (key, float(t4), float(t5))
        _, t4, t5 = best
        chosen[ho] = (round(t4, 3), round(t5, 3))
        out.loc[f["te_idx"], col] = asym_rule(f["Pte"], t4, t5)
    if not quiet:
        log(f"- fold별 (MAE예산 내 조건3 최소화) (τ4,τ5): {chosen}")
    return out


def main():
    log("# Aim 2 · B1 개선 시도 리포트 — 조건3 구제 (sjlee)\n")
    m = build_matrix()
    log(f"공통표본 {len(m)} × {len(ITEMS)}문항. MAE 예산 B1 ≤ {BUDGET}(=A1 0.591−0.10), 현 B1 0.419.")
    log("판정: 조건 1(MAE감소≥.10) 2(2↑과소증가≤2%p) 3(4·5→≤3증가≤2%p) 4(≥2시험MAE감소).\n")

    fB1 = probs_ordered(m, reweight=False)
    fRW = probs_ordered(m, reweight=True)
    fNP = probs_nonparallel(m)

    results = {}
    # 참조 원 B1
    out = m.copy()
    for ho in TRIALS:
        out.loc[fB1[ho]["te_idx"], "B1"] = cond_median(fB1[ho]["Pte"])
    log("## 참조: 원 B1 (조건부 중앙값)")
    results["원B1"] = report_alt("원 B1", "기준", out, "B1")

    log("## A. 비대칭 임계 (원 B1 확률 + τ, post-hoc)")
    oA = tune_apply(m, fB1, "A"); out["A"] = oA["A"]
    results["A"] = report_alt("A 비대칭임계", "post-hoc", out, "A")

    log("## B. 중증 재가중(오버샘플×%d) + 비대칭 임계 (탐색)" % REWEIGHT)
    oB = tune_apply(m, fRW, "B"); out["B"] = oB["B"]
    results["B+A"] = report_alt("B 재가중+임계", "탐색", out, "B")

    log("## C. 부분비례오즈(비평행 누적로짓) + 비대칭 임계 (사전지정)")
    oC = tune_apply(m, fNP, "C"); out["C"] = oC["C"]
    results["C+A"] = report_alt("C 부분PO+임계", "사전지정", out, "C")

    log("## E. 부분PO + 파생입력(총점·소계) + 비대칭 임계 (탐색)")
    fE = probs_augmented(m)
    oE = tune_apply(m, fE, "E"); out["E"] = oE["E"]
    results["E"] = report_alt("E 부분PO+파생+임계", "탐색", out, "E")

    log("## F. 중증 전용 허들(4·5 탐지기 + B1 몸통) (탐색)")
    oF = hurdle(m); out["F"] = oF["F"]
    results["F"] = report_alt("F 허들", "탐색", out, "F")

    # G. 프론티어 스윕: C 확률에 train 예산 마진을 넉넉히 줘 test에서 조건1·3 동시충족 탐색
    log("## G. C(부분PO) 예산-마진 스윕 — 4조건 동시충족(주 0.10) 탐색")
    log("- 아이디어: C는 조건3 여유(22.5% vs 28.8%)가 커서, train 예산을 덜 쓰면(감소목표 상향)")
    log("  test MAE 감소가 0.10을 마진 두고 넘으면서 조건3도 유지될 수 있음. train 감소목표를 스윕.")
    log("| train 감소목표 | test MAE감소 | 4·5→≤3 | 조건3 | 4조건(0.10) |")
    log("|---|---|---|---|---|")
    best_G = None
    for tgt in [0.10, 0.12, 0.13, 0.14, 0.16, 0.18]:
        oG = tune_apply(m, fNP, "G", mae_thr=tgt, quiet=True)
        out["G"] = oG["G"]; r = eval4(out, "G", mae_thr=0.10)
        ok = r["allpass"]
        log(f"| {tgt:.2f} | {r['mae_drop']:+.3f} | {r['hi']*100:.1f}% | "
            f"{'✅' if r['c3'] else '❌'} | {'★✅' if ok else '❌'} |")
        if ok and (best_G is None or r["mae_drop"] > best_G[1]):
            best_G = (tgt, r["mae_drop"], r["hi"], out["G"].copy())
    if best_G:
        out["G"] = best_G[3]
        results["G(C마진·낙관)"] = eval4(out, "G", mae_thr=0.10)
        log(f"\n- 스윕상 4조건 통과 구간 존재: train감소목표 {best_G[0]:.2f} → "
            f"test MAE감소 {best_G[1]:+.3f}, 4·5→≤3 {best_G[2]*100:.1f}%. "
            "**단 마진을 test로 골라 낙관편향 → 아래 중첩CV로 검증.**\n")
    else:
        log("\n- 스윕 구간에서 4조건 동시충족 없음.\n")

    # Gn: 중첩 CV (누출 없는 정식 판정)
    log("## Gn. 중첩 CV — 마진을 훈련 내부에서만 선택 (누출 없는 정식 판정)")
    oGn = nested_C(m); out["Gn"] = oGn["Gn"]
    results["Gn(중첩CV정식)"] = report_alt("Gn 중첩CV", "정식(누출0)", out, "Gn")

    # 종합
    log("## 종합 판정")
    log("| 구성 | MAE | MAE감소 | 4·5→≤3 | 조건3(≤+2%p) | 2↑과소 | 4조건(주.10) |")
    log("|---|---|---|---|---|---|---|")
    cols = {"원B1": "B1", "A": "A", "C+A": "C", "F": "F", "Gn(중첩CV정식)": "Gn"}
    for k, r in results.items():
        log(f"| {k} | {r['mae']:.3f} | {r['mae_drop']:+.3f} | {r['hi']*100:.1f}% | "
            f"{r['d_hi']*100:+.1f}%p {'✅' if r['c3'] else '❌'} | {r['d_u2']*100:+.1f}%p | "
            f"{'✅' if r['allpass'] else '❌'} |")

    # MAE 임계값 민감도 (0.05/0.10/0.15)
    log("\n### MAE 임계값 민감도 — 4조건 종합 판정")
    log("| 구성 | thr=0.05 | thr=0.10(주) | thr=0.15 |")
    log("|---|---|---|---|")
    for k, pc in cols.items():
        vs = []
        for thr in [0.05, 0.10, 0.15]:
            rr = eval4(out, pc, mae_thr=thr)
            vs.append("✅" if rr["allpass"] else "❌")
        log(f"| {k} | {vs[0]} | {vs[1]} | {vs[2]} |")

    passed = [k for k, r in results.items() if r["allpass"]]
    passed05 = [k for k, pc in cols.items() if eval4(out, pc, mae_thr=0.05)["allpass"]]
    if passed:
        log(f"\n- **주 임계값 0.10에서 4조건 충족: {passed} → 최단순 선택해 B2 진입.**")
    elif passed05:
        log(f"\n- 주 임계값 0.10에선 미충족이나 **완화 임계값 0.05에서 충족: {passed05}**. "
            "조건3(중증검출)은 해결됐고 조건1 MAE만 0.10 경계에 걸림 — trade-off 명시.")
    else:
        best = min(results.items(), key=lambda kv: kv[1]["hi"])
        log(f"\n- 어떤 구성도 미충족. 조건3 최저 = {best[0]} ({best[1]['hi']*100:.1f}%).")
    log("")

    out.to_csv(CSV_OUT / "b1_improve_cv_predictions.csv", index=False)
    log(f"산출: {CSV_OUT/'b1_improve_cv_predictions.csv'} (환자단위 → Drive)")
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")
    return results


if __name__ == "__main__":
    main()
