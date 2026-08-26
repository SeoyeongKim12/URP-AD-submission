"""
Aim 2 · B2 중첩CV 랜덤서치 (정직판) (sjlee)
============================================
팀 제안: 각 문항수(5~18)에서 무작위 300조합을 중첩CV로 정직하게 평가해
'몇 문항이면 충분한가'를 낙관 없이 확인. **6a+6b 병합 버전 / 원 21문항(비병합)
버전을 둘 다 돌려서 비교**함.

구조(누출 없음):
- 바깥: leave-one-trial-out(3). 평가시험은 선택에 절대 미사용.
- 안쪽: 남은 2시험 중 leave-one-inner-trial(2). 여기서 300조합 중 최고 선택.
- 선택된 조합을 바깥 훈련(2시험)으로 재학습 → 바깥 평가시험 예측.
→ 각 k의 정직한 성능 = 바깥 예측 취합.

주의: 시험 3개라 안쪽 선택 노이즈 큼 → 조합최적화 이득이 작게/불안정하게 나올 수 있음(정직 결과).
결측 대치는 전역 중앙값 1회(무시 수준 → 누출 미미).

[Claude 수정사항 — 이번 요청]
1) K_RANGE: 5~12 → **5~18**로 확장.
2) N_RANDOM: 200 → **300**으로 증가.
3) **병합/비병합 분기 추가**: ADCS-ADL 6a(옷 고르기)+6b(옷 입기)를 Q6 하나로 합친 버전
   (20문항 풀)과, 원래대로 둘을 분리해서 쓰는 버전(21문항 풀)을 VARIANTS에 따라 각각
   전체 파이프라인(랜덤서치→중첩CV→outer 예측→저장)을 돌림. 파일명은 변형별로
   `_merged`/`_unmerged` 접미사로 구분되고, 마지막에 두 변형을 나란히 비교하는
   요약표(b2_variant_comparison.csv)도 추가로 저장함.
   합산 규칙(Q6=6a+6b)은 이전과 동일 — 다른 결합규칙을 쓰려면 merge_items()만 고치면 됨.

[주의 — 실행시간] k값 14개(5~18) × 300조합 × 3 outer × 2 inner × 5개 절단모형 × 2변형
≈ 이전(8×200×...×1변형) 대비 약 5.25배 연산량. 이전 실행이 수 분이었다면 이번엔
수십 분 단위로 늘어날 수 있음 — 필요하면 VARIANTS를 하나만 남기거나 N_RANDOM을
낮춰서 먼저 확인해보는 걸 권장.

산출(변형별): aim2/시행착오/b2_nested_randomsearch_report_{tag}.md
             b2_performance_by_k_{tag}.csv
             b2_selected_items_by_fold_{tag}.csv
             b2_common_items_by_k_{tag}.csv
             b2_outer_predictions_{tag}.csv
산출(비교):   b2_variant_comparison.csv (merged vs unmerged, k별 성능 나란히)
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
K_RANGE = list(range(5, 19))     # 5..18
N_RANDOM = 300
VARIANTS = ["merged", "unmerged"]   # 6a+6b 병합 / 원 21문항 비병합, 둘 다 실행

LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106A":"Q6a옷고르기","ADL0106B":"Q6b옷입기","Q6":"Q6옷챙겨입기(6a+6b합산)",
 "ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자있기","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}


def merge_items(m: pd.DataFrame):
    """D.ITEMS(21개, ADL0106A=6a·ADL0106B=6b 분리)를 6a+6b 합산 "Q6" 1개로 병합한
    (데이터프레임, 새 ITEMS 리스트)를 반환. D.build()의 원본 문항 정의는 건드리지 않음."""
    m = m.copy()
    m["Q6"] = m["ADL0106A"] + m["ADL0106B"]
    items = [c for c in D.ITEMS if c not in ("ADL0106A", "ADL0106B")]
    items.insert(D.ITEMS.index("ADL0106B"), "Q6")   # 원래 6번 자리에 병합문항 삽입
    return m, items


def get_variant(m0: pd.DataFrame, variant: str):
    """variant='merged'면 6a+6b 병합(20문항), 'unmerged'면 원 21문항 그대로."""
    if variant == "merged":
        return merge_items(m0)
    elif variant == "unmerged":
        return m0.copy(), list(D.ITEMS)
    raise ValueError(variant)


def fit_cuts(Ztr, y):
    return {k: LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, max_iter=1500, tol=1e-3,
            random_state=0).fit(Ztr, (y >= k).astype(int)) for k in range(1, 6)}


def probs(mdl, Z):
    g = {k: mdl[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - g[1]
    for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
    P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P


def tune_tau(Ptr, ds_tr, a0m):
    grid = np.round(np.arange(0.05, 0.55, 0.03), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = D.asym(Ptr, t4, t5)
            if D.miss(ds_tr, p) <= a0m + 1e-9:
                key = D.mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, t4, t5)
    if best is None: return 0.3, 0.5
    return best[1], best[2]


def predict(X, tr, te, cols, y, ds, a0):
    sc = StandardScaler().fit(X[np.ix_(tr, cols)])
    Ztr = sc.transform(X[np.ix_(tr, cols)]); Zte = sc.transform(X[np.ix_(te, cols)])
    mdl = fit_cuts(Ztr, y[tr])
    Ptr = probs(mdl, Ztr); Pte = probs(mdl, Zte)
    t4, t5 = tune_tau(Ptr, ds[tr], D.miss(ds[tr], a0[tr]))
    return D.asym(Pte, t4, t5)


def run_variant(m0: pd.DataFrame, variant: str, TRIALS, rng):
    """한 변형(merged/unmerged) 전체 파이프라인 실행 + 저장. 성능표(perf_df)를 반환해
    나중에 변형 간 비교표를 만드는 데 씀."""
    tag = variant
    _lines = []
    def log(s=""):
        print(f"[{tag}] " + s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

    m, ITEMS = get_variant(m0, variant)
    NI = len(ITEMS)
    log(f"# Aim 2 · B2 중첩CV 랜덤서치 (정직판, {tag}) (sjlee)\n")
    if variant == "merged":
        log(f"문항 병합: ADL0106A(6a)+ADL0106B(6b) → Q6 (21문항 → {NI}문항)\n")
    else:
        log(f"문항 비병합: 원래 6a/6b 분리 그대로 사용 (전체 {NI}문항)\n")

    X = SimpleImputer(strategy="median").fit_transform(m[ITEMS])
    trial = m["STUDYID"].values; y = m["ds_stage"].astype(int).values
    ds = m["ds_stage"].values; a0 = m["A0_harmonized"].values
    a1_mae = D.mae(ds, m["A1_2015_stage"].values); a0_miss = D.miss(ds, a0) * 100
    rows = {t: np.where(trial == t)[0] for t in TRIALS}
    log(f"공통표본 {len(m)}. 기준 A1 MAE {a1_mae:.3f}/A0 중증 {a0_miss:.1f}%. "
        f"k={K_RANGE[0]}~{K_RANGE[-1]}(문항풀 {NI}개), 각 {N_RANDOM}조합, 중첩CV.\n")

    K_here = [k for k in K_RANGE if k < NI]   # 문항풀보다 큰 k는 뽑을 수 없으니 제외
    if len(K_here) < len(K_RANGE):
        log(f"[알림] 문항풀({NI})보다 큰 k는 제외됨: {[k for k in K_RANGE if k >= NI]}\n")

    outer_pred = {k: np.full(len(m), np.nan) for k in K_here}
    sel = {k: {} for k in K_here}
    sel_rows = []
    for ho in TRIALS:
        otr = np.concatenate([rows[t] for t in TRIALS if t != ho]); ote = rows[ho]
        inner = [t for t in TRIALS if t != ho]
        for k in K_here:
            best = (np.inf, None)
            for _ in range(N_RANDOM):
                cols = sorted(rng.choice(NI, k, replace=False).tolist())
                im = []
                for iv in inner:
                    it = [t for t in inner if t != iv][0]
                    pv = predict(X, rows[it], rows[iv], cols, y, ds, a0)
                    im.append(D.mae(ds[rows[iv]], pv))
                mi = np.mean(im)
                if mi < best[0]: best = (mi, cols)
            cols = best[1]; sel[k][ho] = cols
            outer_pred[k][ote] = predict(X, otr, ote, cols, y, ds, a0)
            names = [ITEMS[i] for i in cols]
            sel_rows.append({"variant": tag, "k": k, "outer_fold": ho, "n_items": len(names),
                              "items_code": ",".join(names),
                              "items_label": ",".join(LAB.get(c, c) for c in names)})
        log(f"[진행] outer={ho} 완료")

    log("\n## 중첩CV 랜덤서치 정직 성능 (문항수별)")
    log("| 문항수 | MAE | 중증놓침 | 카파 | A1대비MAE | 둘다이김 |")
    log("|---|---|---|---|---|---|")
    perf_rows = []
    for k in K_here:
        p = outer_pred[k]; mae_k = D.mae(ds, p); miss_k = D.miss(ds, p) * 100; kap_k = D.kap(ds, p)
        champ = (mae_k < a1_mae) and (miss_k <= a0_miss + 1.0)
        log(f"| {k} | {mae_k:.3f} | {miss_k:.1f}% | {kap_k:.3f} | {a1_mae-mae_k:+.3f} | {'✅' if champ else '❌'} |")
        perf_rows.append({"variant": tag, "n_items": k, "mae": round(mae_k, 4), "hi_missed_pct": round(miss_k, 2),
                           "kappa": round(kap_k, 4), "mae_vs_a1": round(a1_mae - mae_k, 4),
                           "a1_mae_ref": round(a1_mae, 4), "a0_miss_pct_ref": round(a0_miss, 2),
                           "champion": bool(champ)})
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUTDIR / f"b2_performance_by_k_{tag}.csv", index=False, encoding="utf-8-sig")

    log("\n## fold별 선택 조합 (안쪽 선택, 참고 — 10/8/6문항만 본문에 표시, 전체는 csv)")
    common_rows = []
    for k in K_here:
        common = set(sel[k][TRIALS[0]])
        for ho in TRIALS[1:]: common &= set(sel[k][ho])
        common_names = [LAB.get(ITEMS[i], ITEMS[i]) for i in sorted(common)]
        common_rows.append({"variant": tag, "n_items": k, "n_common_across_3fold": len(common),
                             "common_items_label": ",".join(common_names)})
        if k in (10, 8, 6):
            log(f"### {k}문항")
            for ho in TRIALS:
                names = [LAB.get(ITEMS[i], ITEMS[i]) for i in sel[k][ho]]
                log(f"- outer={ho}: {', '.join(names)}")
            log(f"- **3fold 공통: {common_names}**")
    pd.DataFrame(sel_rows).to_csv(OUTDIR / f"b2_selected_items_by_fold_{tag}.csv", index=False, encoding="utf-8-sig")
    common_df = pd.DataFrame(common_rows)
    common_df.to_csv(OUTDIR / f"b2_common_items_by_k_{tag}.csv", index=False, encoding="utf-8-sig")

    pred_wide = m[["STUDYID", "USUBJID", "ds_stage", "A1_2015_stage", "A0_harmonized"]].copy()
    for k in K_here:
        pred_wide[f"pred_k{k}"] = outer_pred[k]
    pred_wide.to_csv(OUTDIR / f"b2_outer_predictions_{tag}.csv", index=False, encoding="utf-8-sig")

    log("\n- 정직판이 v2(계수순위/후진제거)보다 낮으면 = 낙관 제거된 진짜 성능. "
        "fold별 선택이 많이 다르면 = 소표본서 조합 최적화 불안정(정직 결론).")
    report_path = OUTDIR / f"b2_nested_randomsearch_report_{tag}.md"
    report_path.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {report_path}")
    print(f">>> 저장: {OUTDIR / f'b2_performance_by_k_{tag}.csv'}")
    print(f">>> 저장: {OUTDIR / f'b2_selected_items_by_fold_{tag}.csv'}")
    print(f">>> 저장: {OUTDIR / f'b2_common_items_by_k_{tag}.csv'}")
    print(f">>> 저장: {OUTDIR / f'b2_outer_predictions_{tag}.csv'}")
    return perf_df


def main():
    m0 = D.build()
    TRIALS = D.TRIALS
    all_perf = []
    for variant in VARIANTS:
        # 변형마다 같은 시드로 시작해서 "같은 난수 계열"을 쓰게 함(둘의 차이가 순수하게
        # 문항풀 차이에서만 나오게 하기 위함 — 완전히 같은 조합이 뽑히진 않지만
        # 동일 시드 재현성은 보장됨).
        rng = np.random.default_rng(20260805)
        perf_df = run_variant(m0, variant, TRIALS, rng)
        all_perf.append(perf_df)
        print()

    # 변형 간 비교표: merged vs unmerged, k별 나란히
    comp = pd.concat(all_perf, ignore_index=True)
    comp_wide = comp.pivot_table(index="n_items", columns="variant",
                                  values=["mae", "hi_missed_pct", "kappa", "champion"])
    comp_wide.columns = [f"{a}_{b}" for a, b in comp_wide.columns]
    comp_wide = comp_wide.reset_index()
    comp_wide.to_csv(OUTDIR / "b2_variant_comparison.csv", index=False, encoding="utf-8-sig")
    print(f">>> 저장: {OUTDIR / 'b2_variant_comparison.csv'} (merged vs unmerged 비교)")


if __name__ == "__main__":
    main()
