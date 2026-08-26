"""
Aim 2 · B2 고정 6문항 + 나머지 전수조합 탐색 — b1 방식 엄격판(fold-내부 결측치대치 +
0.02 τ그리드) (sjlee)
====================================================================================
b2_fixed6_exhaustive.py(1차판)와 결과가 다르게 나온 이유를 원본 `b1_dominate_a0.py`
설계와 비교해보니, 1차판은 (1) 21문항 전체 결측치대치를 시험 3개 다 합친 전체표본
기준으로 "한 번만" 하고 시작했고 (2) τ4/τ5 탐색간격이 0.03이었음 — 둘 다 b1의 원래
설계(결측치대치는 매 leave-one-trial-out fold 훈련자료 안에서만 다시 계산, τ간격 0.02)
보다 느슨함. 이번 버전은 그 두 가지를 b1과 동일하게 맞춘 "엄격판"임.

바뀐 점(1차판 b2_fixed6_exhaustive.py 대비):
  1) 결측치대치를 전체표본에서 미리 한 번 하지 않고, cols_idx 원본값(NaN 포함)을 그대로
     들고 있다가 매 fold마다 "그 fold의 훈련자료(2개 시험)"만으로 중앙값을 다시 계산해서
     채움 → 검증용으로 뺀 시험의 정보가 결측치대치 단계에서 전혀 안 섞임(누출 0).
  2) tune_tau() 그리드 간격 0.03 → 0.02로 변경(b1_dominate_a0.py 원본과 동일).
  나머지(고정 6문항, 조합수, EN/L2 penalty 설정, 병렬화·체크포인트 방식)는 1차판과 동일.

속도: 결측치대치를 fold마다 다시 하는 비용은 중앙값 계산 하나 추가되는 정도라 거의 무시
가능. τ그리드가 17→25개 후보(유효 쌍 약 153→325개, ~2.1배)로 늘어난 만큼만 조합당
시간이 조금 늘어남 — 1차판 실측(EN 84~130ms, L2 52~53ms/조합) 대비 1.3~1.5배 예상.

산출: aim2/시행착오/b2_fixed6_exhaustive_strict_EN.csv
      aim2/시행착오/b2_fixed6_exhaustive_strict_L2.csv
      aim2/시행착오/b2_fixed6_exhaustive_strict_best_by_k.csv
(1차판 b2_fixed6_exhaustive_{EN,L2}.csv는 비교용으로 그대로 남겨둠 — 덮어쓰지 않음)
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
    """한 문항조합을 leave-one-trial-out(3)으로 평가. Xraw는 결측치(NaN) 포함 원본값이고,
    각 fold의 훈련자료(검증시험 제외 2개)만으로 중앙값을 다시 계산해 결측치를 채움
    (b1_dominate_a0.py와 동일한 fold-내부 대치, 누출 0)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    pred = np.full(Xraw.shape[0], np.nan)
    for ho in TRIALS:
        tr = np.concatenate([rows[t] for t in TRIALS if t != ho])
        te = rows[ho]
        Xtr_raw = Xraw[np.ix_(tr, cols_idx)]; Xte_raw = Xraw[np.ix_(te, cols_idx)]
        med = np.nanmedian(Xtr_raw, axis=0)               # fold 훈련자료만으로 중앙값 계산
        Xtr = np.where(np.isnan(Xtr_raw), med, Xtr_raw)
        Xte = np.where(np.isnan(Xte_raw), med, Xte_raw)   # 검증자료는 훈련자료의 중앙값으로만 채움
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
        Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
        ytr = y[tr]
        mdl = {k: LogisticRegression(C=C, random_state=0, **kw).fit(Ztr, (ytr >= k).astype(int))
               for k in range(1, 6)}
        Ptr = probs_from_models(mdl, Ztr); Pte = probs_from_models(mdl, Zte)
        a0m = D.miss(ds[tr], a0[tr])
        t4, t5 = tune_tau(Ptr, ds[tr], a0m)
        pred[te] = D.asym(Pte, t4, t5)
    mae_v = D.mae(ds, pred); miss_v = D.miss(ds, pred) * 100; kap_v = D.kap(ds, pred)
    return mae_v, miss_v, kap_v


def run_penalty(tag, kw, all_combos, ITEMS, Xraw, rows, TRIALS, y, ds, a0, a1_mae, a0_miss, C=0.1):
    print(f"\n[{tag}] 조합 {len(all_combos)}개 평가 시작 (n_jobs={N_JOBS})")
    ckpt_path = OUTDIR / f"b2_fixed6_exhaustive_strict_{tag}.csv"
    all_rows = []
    t_start = time.time()
    n_done = 0
    for chunk_start in range(0, len(all_combos), CHECKPOINT_EVERY):
        chunk = all_combos[chunk_start: chunk_start + CHECKPOINT_EVERY]
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(eval_combo)(cols, Xraw, rows, TRIALS, y, ds, a0, C, kw) for cols in chunk
        )
        for cols, (mae_v, miss_v, kap_v) in zip(chunk, results):
            names = [ITEMS[i] for i in cols]
            champ = (mae_v < a1_mae) and (miss_v <= a0_miss + 1.0)
            all_rows.append({
                "penalty": tag, "n_items": len(cols),
                "items_code": ",".join(names),
                "items_label": ",".join(LAB.get(c, c) for c in names),
                "mae": round(mae_v, 4), "hi_missed_pct": round(miss_v, 2),
                "kappa": round(kap_v, 4), "mae_vs_a1": round(a1_mae - mae_v, 4),
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

    # 결측치대치 없이 원본값(NaN 포함) 그대로 들고 있다가 fold별로 대치함(1차판과의 핵심 차이).
    Xraw = m[ITEMS].to_numpy(dtype=float)
    trial = m["STUDYID"].values
    y = m["ds_stage"].astype(int).values
    ds = m["ds_stage"].values
    a0 = m["A0_harmonized"].values
    TRIALS = D.TRIALS
    rows = {t: np.where(trial == t)[0] for t in TRIALS}
    a1_mae = D.mae(ds, m["A1_2015_stage"].values)
    a0_miss = D.miss(ds, a0) * 100
    print(f"기준 A1 MAE {a1_mae:.3f} / A0 중증놓침 {a0_miss:.1f}%\n")

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
    best = (both.sort_values(["penalty", "n_items", "mae"])
                .groupby(["penalty", "n_items"], as_index=False).first())
    best = best.sort_values(["n_items", "penalty"])
    best.to_csv(OUTDIR / "b2_fixed6_exhaustive_strict_best_by_k.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> 저장: {OUTDIR / 'b2_fixed6_exhaustive_strict_best_by_k.csv'} (문항수·penalty별 최저MAE 조합)")


if __name__ == "__main__":
    main()
