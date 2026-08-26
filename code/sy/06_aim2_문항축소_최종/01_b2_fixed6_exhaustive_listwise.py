"""
Aim 2 · B2 고정 6문항 + 나머지 전수조합 탐색 — listwise deletion(완전사례분석)판 (sjlee)
====================================================================================
b2_fixed6_exhaustive_strict.py(fold-내부 중앙값대치 + 0.02τ그리드)에서 결측치 처리
방식만 "중앙값 대치" 대신 "그 조합의 문항 중 하나라도 결측이면 그 사람 제외"로 바꾼
버전. b1_ridge_top3_listwise_test.py로 top3 조합만 먼저 확인했더니 결론(A0·A1 지배)은
안 바뀌었지만 숫자가 꽤 흔들려서, 전체 22,818조합에 대해 제대로 다시 돌려봄.

바뀐 점(b2_fixed6_exhaustive_strict.py 대비):
  - 결측치를 fold 훈련자료 중앙값으로 채우지 않고, "이 조합의 문항 중 하나라도 결측인
    사람"을 훈련/검증 양쪽에서 그냥 제외함(listwise deletion). 조합마다 쓰는 문항이
    다르니 빠지는 사람도 조합마다 다름 — 21문항 전체가 아니라 "그 조합 문항 기준"으로
    결측 여부를 판단하는 게 핵심(문항이 적을수록 결측 걸릴 확률도 낮아짐).
  - fold별로 실제 채점된 인원수(n_scored)를 같이 기록함(결측 많은 조합은 표본이 줄어드는
    걸 확인할 수 있게).
  - A0/A1 기준값(a1_mae_ref, a0_miss_pct_ref)은 조합마다 다시 계산하지 않고 전체
    2,203명 기준 고정값을 그대로 씀 — 조합마다 기준을 바꾸면 조합 간 비교 자체가
    불공정해지므로(어떤 조합은 쉬운 기준과, 어떤 조합은 어려운 기준과 비교하는 셈이
    됨) 전수조사에서는 이렇게 하는 게 맞음. (top3 검증 스크립트는 조합이 3개뿐이라
    참고용으로 부분표본 기준값도 같이 보여준 것뿐, 이번엔 안 함.)

나머지(고정 6문항, 조합수 22,818, EN/L2 penalty, τ그리드 0.02, 병렬화·체크포인트)는
strict판과 동일.

산출: aim2/시행착오/b2_fixed6_exhaustive_listwise_EN.csv
      aim2/시행착오/b2_fixed6_exhaustive_listwise_L2.csv
      aim2/시행착오/b2_fixed6_exhaustive_listwise_best_by_k.csv
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from pathlib import Path
from itertools import combinations
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
FIXED = ["ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0115A", "ADL0116A"]
R_RANGE = range(1, 9)          # 추가 1~8개 → 총 7~14문항
N_JOBS = -1
CHECKPOINT_EVERY = 500

LAB = {"ADL0101": "Q1먹기", "ADL0102": "Q2걷기", "ADL0103": "Q3화장실", "ADL0104": "Q4목욕",
       "ADL0105": "Q5몸단장", "ADL0106A": "Q6a옷고르기", "ADL0106B": "Q6b옷입기",
       "ADL0107A": "Q7전화", "ADL0110A": "Q10설거지", "ADL0111A": "Q11식사준비",
       "ADL0112A": "Q12집안일", "ADL0113A": "Q13빨래", "ADL0114A": "Q14가전",
       "ADL0115A": "Q15외출", "ADL0116A": "Q16a쇼핑", "ADL0116B": "Q16b지불",
       "ADL0117A": "Q17금전", "Q18": "Q18혼자있기", "ADL0121A": "Q21글쓰기",
       "ADL0122Q": "Q22취미", "ADL0123L": "Q23가전사용"}

PENALTY_KW = {
    "EN": dict(penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=1500, tol=1e-3),
    "L2": dict(penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-3),
}


def probs_from_models(mdl, Z):
    g = {k: mdl[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - g[1]
    for k in range(1, 5):
        P[:, k] = g[k] - g[k + 1]
    P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)
    return P


def tune_tau(Ptr, ds_tr, a0m):
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3); best = None   # b1과 동일 간격(0.02)
    for t4 in grid:
        for t5 in grid:
            if t5 < t4:
                continue
            p = D.asym(Ptr, t4, t5)
            if D.miss(ds_tr, p) <= a0m + 1e-9:
                key = D.mae(ds_tr, p)
                if best is None or key < best[0]:
                    best = (key, t4, t5)
    if best is None:
        return 0.3, 0.5
    return best[1], best[2]


def eval_combo(cols_idx, Xraw, rows, TRIALS, y, ds, a0, C, kw):
    """listwise deletion: 이 조합 문항 중 하나라도 결측인 사람은 훈련·검증 양쪽에서 제외.
    결측치대치 단계 자체가 없음."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    pred = np.full(Xraw.shape[0], np.nan)
    for ho in TRIALS:
        tr_idx = np.concatenate([rows[t] for t in TRIALS if t != ho])
        te_idx = rows[ho]
        Xtr_raw = Xraw[np.ix_(tr_idx, cols_idx)]; Xte_raw = Xraw[np.ix_(te_idx, cols_idx)]
        tr_ok = ~np.isnan(Xtr_raw).any(axis=1); te_ok = ~np.isnan(Xte_raw).any(axis=1)
        tr = tr_idx[tr_ok]; te = te_idx[te_ok]
        if len(tr) < 30 or len(te) < 5:   # 극단적으로 표본이 작아지면(사실상 거의 없음) 건너뜀
            continue
        Xtr = Xraw[np.ix_(tr, cols_idx)]; Xte = Xraw[np.ix_(te, cols_idx)]
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
        Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
        ytr = y[tr]
        mdl = {k: LogisticRegression(C=C, random_state=0, **kw).fit(Ztr, (ytr >= k).astype(int))
               for k in range(1, 6)}
        Ptr = probs_from_models(mdl, Ztr); Pte = probs_from_models(mdl, Zte)
        a0m = D.miss(ds[tr], a0[tr])
        t4, t5 = tune_tau(Ptr, ds[tr], a0m)
        pred[te] = D.asym(Pte, t4, t5)
    valid = ~np.isnan(pred)
    n_scored = int(valid.sum())
    if n_scored == 0:
        return np.nan, np.nan, np.nan, 0
    mae_v = D.mae(ds[valid], pred[valid])
    miss_v = D.miss(ds[valid], pred[valid]) * 100
    kap_v = D.kap(ds[valid], pred[valid])
    return mae_v, miss_v, kap_v, n_scored


def run_penalty(tag, kw, all_combos, ITEMS, Xraw, rows, TRIALS, y, ds, a0, a1_mae, a0_miss, C=0.1):
    print(f"\n[{tag}] 조합 {len(all_combos)}개 평가 시작 (n_jobs={N_JOBS})")
    ckpt_path = OUTDIR / f"b2_fixed6_exhaustive_listwise_{tag}.csv"
    all_rows = []
    t_start = time.time()
    n_done = 0
    for chunk_start in range(0, len(all_combos), CHECKPOINT_EVERY):
        chunk = all_combos[chunk_start: chunk_start + CHECKPOINT_EVERY]
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(eval_combo)(cols, Xraw, rows, TRIALS, y, ds, a0, C, kw) for cols in chunk
        )
        for cols, (mae_v, miss_v, kap_v, n_scored) in zip(chunk, results):
            names = [ITEMS[i] for i in cols]
            champ = (not np.isnan(mae_v)) and (mae_v < a1_mae) and (miss_v <= a0_miss + 1.0)
            all_rows.append({
                "penalty": tag, "n_items": len(cols),
                "items_code": ",".join(names),
                "items_label": ",".join(LAB.get(c, c) for c in names),
                "mae": round(mae_v, 4) if not np.isnan(mae_v) else np.nan,
                "hi_missed_pct": round(miss_v, 2) if not np.isnan(miss_v) else np.nan,
                "kappa": round(kap_v, 4) if not np.isnan(kap_v) else np.nan,
                "n_scored": n_scored,
                "mae_vs_a1": round(a1_mae - mae_v, 4) if not np.isnan(mae_v) else np.nan,
                "a1_mae_ref": round(a1_mae, 4), "a0_miss_pct_ref": round(a0_miss, 2),
                "champion": bool(champ),
            })
        n_done += len(chunk)
        elapsed = time.time() - t_start
        rate = n_done / elapsed
        eta = (len(all_combos) - n_done) / rate if rate > 0 else float("nan")
        print(f"[{tag}] {n_done}/{len(all_combos)} 완료 · {elapsed/60:.1f}분 경과 · "
              f"예상잔여 {eta/60:.1f}분 (체크포인트 저장)")
        pd.DataFrame(all_rows).to_csv(ckpt_path, index=False, encoding="utf-8-sig")
    print(f"[{tag}] 완료. 총 {(time.time()-t_start)/60:.1f}분. 저장: {ckpt_path}")
    return pd.DataFrame(all_rows)


def main():
    m = D.build()
    ITEMS = D.ITEMS
    idx = {c: i for i, c in enumerate(ITEMS)}
    FREE = [c for c in ITEMS if c not in FIXED]
    print(f"고정 {len(FIXED)}문항: {[LAB.get(c,c) for c in FIXED]}")
    print(f"탐색풀 {len(FREE)}문항: {[LAB.get(c,c) for c in FREE]}")

    FIXED_IDX = [idx[c] for c in FIXED]
    FREE_IDX = [idx[c] for c in FREE]

    all_combos = []
    for r in R_RANGE:
        for extra in combinations(FREE_IDX, r):
            all_combos.append(tuple(sorted(FIXED_IDX + list(extra))))
    print(f"총 조합수: {len(all_combos)} (문항수 {6+R_RANGE.start}~{6+R_RANGE.stop-1})")

    Xraw = m[ITEMS].to_numpy(dtype=float)   # 결측치대치 없음. NaN 그대로 보관.
    trial = m["STUDYID"].values
    y = m["ds_stage"].astype(int).values
    ds = m["ds_stage"].values
    a0 = m["A0_harmonized"].values
    TRIALS = D.TRIALS
    rows = {t: np.where(trial == t)[0] for t in TRIALS}
    # 기준값은 전체 2,203명(결측 여부 무관) 고정 — 조합 간 공정 비교를 위해 안 바꿈
    a1_mae = D.mae(ds, m["A1_2015_stage"].values)
    a0_miss = D.miss(ds, a0) * 100
    print(f"기준 A1 MAE {a1_mae:.3f} / A0 중증놓침 {a0_miss:.1f}% (전체 {len(m)}명 기준, 고정)\n")

    calib_n = 20
    for tag, kw in PENALTY_KW.items():
        t0 = time.time()
        for cols in all_combos[:calib_n]:
            eval_combo(list(cols), Xraw, rows, TRIALS, y, ds, a0, 0.1, kw)
        per = (time.time() - t0) / calib_n
        est_serial_min = per * len(all_combos) / 60
        print(f"[{tag}] 조합당 약 {per*1000:.0f}ms(단일코어 추정) → 전체 단일코어 예상 "
              f"{est_serial_min:.0f}분 (실제는 n_jobs={N_JOBS} 병렬이라 코어수만큼 단축됨)")
    print()

    en_df = run_penalty("EN", PENALTY_KW["EN"], all_combos, ITEMS, Xraw, rows, TRIALS, y, ds, a0, a1_mae, a0_miss)
    l2_df = run_penalty("L2", PENALTY_KW["L2"], all_combos, ITEMS, Xraw, rows, TRIALS, y, ds, a0, a1_mae, a0_miss)

    both = pd.concat([en_df, l2_df], ignore_index=True)
    valid_both = both.dropna(subset=["mae"])
    best = (valid_both.sort_values(["penalty", "n_items", "mae"])
                       .groupby(["penalty", "n_items"], as_index=False).first())
    best = best.sort_values(["n_items", "penalty"])
    best.to_csv(OUTDIR / "b2_fixed6_exhaustive_listwise_best_by_k.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> 저장: {OUTDIR / 'b2_fixed6_exhaustive_listwise_best_by_k.csv'} (문항수·penalty별 최저MAE 조합)")


if __name__ == "__main__":
    main()
