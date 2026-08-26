"""
Aim 2 · B2 문항 축약 — 21문항에서 최대한 줄이기 (sjlee)
========================================================
목적: B1(부분비례오즈 엘라스틱넷 + A0-목표 임계) 위에서 문항을 최소로 줄이되
성능(중증놓침·MAE·카파) 유지되는 최소 문항군을 찾는다.

방법:
- 시작 = 21문항(Q6·Q16 쪼갠 상태, 가장 세밀). 줄이며 뭐가 살아남는지가 곧 답.
- 각 외부 fold 훈련자료 내부에서만: 문항 순위(엘라스틱넷 |계수| 합) → 상위 k개 선택
  → 그 k개로 재적합 → A0-목표 τ 튜닝 → 평가시험 예측. (누출0, 시험변수 미투입)
- 판정: 21문항 대비 MAE 증가에 대해 0.05/0.10/0.15 민감도(주 0.10) + 과소평가 증가 ≤2%p.
- 보조: 최소 k vs 21문항 MAE차 쌍체 부트스트랩. 문항 선택 안정성(fold 간 겹침).

입력: ~/Downloads/ { adl_wide.csv, baseline_sample.csv }
산출: aim2/시행착오/b2_item_reduction_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "b2_item_reduction_report.md"
ITEMS = D.ITEMS; TRIALS = D.TRIALS
KS = [21, 18, 15, 12, 10, 8, 6, 5, 4]
LAB = {"ADL0101": "Q1먹기", "ADL0102": "Q2걷기", "ADL0103": "Q3화장실", "ADL0104": "Q4목욕",
       "ADL0105": "Q5몸단장", "ADL0106B": "Q6b옷입기", "ADL0106A": "Q6a옷고르기",
       "ADL0107A": "Q7전화", "ADL0110A": "Q10설거지", "ADL0111A": "Q11식사준비",
       "ADL0112A": "Q12집안일", "ADL0113A": "Q13빨래", "ADL0114A": "Q14가전",
       "ADL0115A": "Q15외출", "ADL0116A": "Q16a쇼핑", "ADL0116B": "Q16b지불",
       "ADL0117A": "Q17금전", "Q18": "Q18혼자있기", "ADL0121A": "Q21글쓰기",
       "ADL0122Q": "Q22취미", "ADL0123L": "Q23가전사용"}
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


def cut_probs(Ztr, Zte, y, C=0.1):
    geq = {k: LogisticRegression(solver="saga", l1_ratio=0.5, C=C, max_iter=6000, tol=1e-3,
                                 random_state=0).fit(Ztr, (y >= k).astype(int)) for k in range(1, 6)}
    def P(Z):
        g = {k: geq[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
        Q = np.zeros((Z.shape[0], 6)); Q[:, 0] = 1 - g[1]
        for k in range(1, 5): Q[:, k] = g[k] - g[k + 1]
        Q[:, 5] = g[5]; Q = np.clip(Q, 1e-9, None); Q /= Q.sum(1, keepdims=True); return Q
    imp_coef = np.sum([np.abs(geq[k].coef_[0]) for k in range(1, 6)], axis=0)  # 문항 중요도(표준화 |계수|합)
    return P(Ztr), P(Zte), imp_coef


def tune_tau_a0(Ptr, ytr, a0_miss_tr):
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = D.asym(Ptr, t4, t5)
            if D.miss(ytr, p) <= a0_miss_tr + 1e-9:
                key = D.mae(ytr, p)
                if best is None or key < best[0]: best = (key, t4, t5)
    if best is None:
        cand = [(D.miss(ytr, D.asym(Ptr, a, b)), D.mae(ytr, D.asym(Ptr, a, b)), a, b)
                for a in grid for b in grid if b >= a]
        _, _, t4, t5 = min(cand); best = (0, t4, t5)
    return best[1], best[2]


def run_k(m, k):
    pred = pd.Series(index=m.index, dtype=float); selected = {}
    for ho in TRIALS:
        tri = m.index[m.STUDYID != ho]; tei = m.index[m.STUDYID == ho]
        imp = SimpleImputer(strategy="median").fit(m.loc[tri, ITEMS])
        Xtr = imp.transform(m.loc[tri, ITEMS])            # numpy (21열)
        Xte = imp.transform(m.loc[tei, ITEMS])
        y = m.loc[tri, "ds_stage"].astype(int).values
        ds_tr = m.loc[tri, "ds_stage"].values
        # 1) 훈련자료서 21문항 적합 → 중요도 순위 → 상위 k (열 인덱스)
        sc0 = StandardScaler().fit(Xtr); Ztr0 = sc0.transform(Xtr)
        _, _, imp_coef = cut_probs(Ztr0, Ztr0[:1], y)
        top_idx = sorted(np.argsort(-imp_coef)[:k].tolist())
        selected[ho] = [ITEMS[i] for i in top_idx]
        # 2) 상위 k열로 재적합 + A0-목표 τ
        sc = StandardScaler().fit(Xtr[:, top_idx])
        Ztr = sc.transform(Xtr[:, top_idx]); Zte = sc.transform(Xte[:, top_idx])
        Ptr, Pte, _ = cut_probs(Ztr, Zte, y)
        a0m = D.miss(ds_tr, m.loc[tri, "A0_harmonized"].values)
        t4, t5 = tune_tau_a0(Ptr, ds_tr, a0m)
        pred.loc[tei] = D.asym(Pte, t4, t5)
    return pred, selected


def main():
    log("# Aim 2 · B2 문항 축약 (21→최소) 리포트 (sjlee)\n")
    m = D.build(); t = m["ds_stage"].values
    a0_mae = D.mae(t, m["A0_harmonized"].values); a1_mae = D.mae(t, m["A1_2015_stage"].values)
    a0_miss = D.miss(t, m["A0_harmonized"].values) * 100; a1_miss = D.miss(t, m["A1_2015_stage"].values) * 100
    log(f"공통표본 {len(m)}. 기준: A1 MAE {a1_mae:.3f}·중증 {a1_miss:.1f}% / A0 MAE {a0_mae:.3f}·중증 {a0_miss:.1f}%\n")

    res = {}; sel = {}
    for k in KS:
        pred, selected = run_k(m, k)
        res[k] = dict(mae=D.mae(t, pred.values), miss=D.miss(t, pred.values) * 100, kap=D.kap(t, pred.values))
        sel[k] = selected

    full = res[21]
    log("## 문항 수별 성능 (leave-one-trial-out CV)")
    log("| 문항수 | MAE | 중증놓침 | 카파 | MAE증가(vs21) | 중증증가(vs21) | A1MAE↓·A0중증급? |")
    log("|---|---|---|---|---|---|---|")
    for k in KS:
        r = res[k]; dmae = r["mae"] - full["mae"]; dmiss = r["miss"] - full["miss"]
        champ = "✅" if (r["mae"] < a1_mae and r["miss"] <= a0_miss + 1.0) else "❌"
        log(f"| {k} | {r['mae']:.3f} | {r['miss']:.1f}% | {r['kap']:.3f} | {dmae:+.3f} | {dmiss:+.1f}%p | {champ} |")
    log("")

    # 허용기준별 최소 문항수 (MAE 증가 ≤ thr AND 과소평가(중증)증가 ≤2%p)
    log("## 허용기준별 최소 문항수 (21문항 대비)")
    log("| MAE 증가 임계 | 최소 문항수(과소평가 증가 ≤2%p 동시) |")
    log("|---|---|")
    for thr in [0.05, 0.10, 0.15]:
        ok = [k for k in KS if (res[k]["mae"] - full["mae"]) <= thr and (res[k]["miss"] - full["miss"]) <= 2.0]
        log(f"| {thr:.2f}{' (주)' if thr==0.10 else ''} | {min(ok) if ok else '없음'} |")
    log("")

    # 문항 선택 안정성 (fold 3개 공통 = consensus)
    log("## 문항 선택 안정성 (3 fold 공통 선택 문항)")
    log("| 문항수 | 3fold 공통(consensus) 문항 |")
    log("|---|---|")
    for k in [12, 10, 8, 6]:
        common = set(sel[k][TRIALS[0]])
        for ho in TRIALS[1:]: common &= set(sel[k][ho])
        names = [LAB[c] for c in ITEMS if c in common]
        log(f"| {k} | ({len(common)}개) {', '.join(names)} |")
    log("")

    # 부트스트랩: 대표 축약(10문항) vs 21문항 MAE차 유의성 (보조)
    log("## 부트스트랩(보조) — 축약 vs 21문항 MAE차 (2,000회 쌍체)")
    rng = np.random.default_rng(20260805); n = len(m)
    p21 = run_k(m, 21)[0].values
    log("| 축약 | MAE차(축약−21) | 95%CI | 유의 악화? |")
    log("|---|---|---|---|")
    for k in [12, 10, 8]:
        pk = run_k(m, k)[0].values
        d = np.empty(2000)
        for i in range(2000):
            ix = rng.integers(0, n, n); tt = t[ix]
            d[i] = D.mae(tt, pk[ix]) - D.mae(tt, p21[ix])
        lo, hi = np.percentile(d, [2.5, 97.5])
        log(f"| {k}문항 | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'유의 악화' if lo > 0 else '동급(0포함)'} |")
    log("")

    log("## 결론")
    ok10 = [k for k in KS if (res[k]["mae"] - full["mae"]) <= 0.10 and (res[k]["miss"] - full["miss"]) <= 2.0]
    log(f"- 주 판정(0.10)에서 **최소 {min(ok10) if ok10 else '?'}문항**까지 축약해도 21문항 대비 허용 내.")
    log("- 위 consensus 문항이 '축약 채점 후보'. 단, fold마다 선택이 조금씩 달라 '독립 단축형'이 아니라 "
        "'약 N문항이면 충분'의 근거임(계획서 문구).")
    log("- 소표본(3시험) 낙관 + 축약해도 A1MAE↓·A0중증급 유지되는지 위 표로 확인.\n")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
