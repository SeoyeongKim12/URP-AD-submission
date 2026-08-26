"""
Aim 2 · B2 고정 6문항 + 나머지 전수조합 탐색 (EN vs L2 비교) (sjlee)
================================================================
고정(무조건 포함, 6문항) — 반복적으로 중요하게 뽑혀서 고정하기로 한 문항:
  ADL0103(Q3화장실), ADL0104(Q4목욕), ADL0105(Q5몸단장),
  ADL0106A(Q6a옷고르기), ADL0115A(Q15외출), ADL0116A(Q16a쇼핑)

나머지 15문항 풀(FREE)에서 1~8개를 추가로 뽑아 총문항수 7~14 전 조합을 전수평가.
  sum_{r=1..8} C(15,r) = 22,818 조합 × (EN/L2 2버전) = 45,636회 평가.

이번엔 "고를" 필요가 없음(랜덤서치처럼 안쪽에서 최고조합 선택하는 게 아니라 이미
가능한 조합을 전부 봄) → 바깥 leave-one-trial-out(3)만으로 정직한 성능을 그대로 기록.
τ4/τ5는 각 fold 훈련자료 내부에서만 튜닝(A0-목표, b1_dominate_a0.py와 동일 규칙) →누출 0.

penalty 두 버전:
  - EN : penalty="elasticnet", l1_ratio=0.5, solver="saga"  (기존 스크립트들이 "쓴다고
         적어놨지만" l1_ratio만 넘기고 penalty를 명시 안 해서 실제로는 L2로 돌아갔던
         버그를 여기서는 고쳐서 진짜 elastic-net으로 돌림)
  - L2 : penalty="l2", solver="lbfgs"  (순수 릿지, elastic-net 없이 비교용)

속도: 컬럼 선택별로 StandardScaler를 매번 새로 fit하지 않고, 문항 21개 전체에 대한
fold별 평균/표준편차를 한 번만 계산해둔 뒤 조합마다 필요한 컬럼만 인덱싱해서 표준화함
(표준화는 열별 독립 연산이라 결과는 동일, 반복 fit 비용만 제거). joblib으로 조합 단위
병렬화(n_jobs=-1, 코어 전부 사용). 진행 중 일정 개수마다 체크포인트 CSV를 남겨서 중간에
중단돼도 그때까지 결과는 안 날아가게 함.

산출: aim2/시행착오/b2_fixed6_exhaustive_EN.csv
      aim2/시행착오/b2_fixed6_exhaustive_L2.csv
      aim2/시행착오/b2_fixed6_exhaustive_best_by_k.csv (EN/L2·문항수별 최고 조합 요약)
"""
import os
# sklearn 최신판(1.8+)에서 penalty= 인자가 deprecated돼서 조합마다(수만 번) FutureWarning이
# 찍혀 콘솔이 도배됨(기능상 문제는 없고 그냥 경고 폭탄). joblib이 Windows에서 워커를
# "spawn"으로 새 프로세스로 띄우면 그 프로세스는 완전히 새 인터프리터로 시작되기 때문에,
# 본 파일 안에서 warnings.filterwarnings()로 끄는 건 워커에 안 먹을 수 있음(재현됨).
# → 환경변수로 인터프리터 시작 시점부터 끄면 부모/자식 프로세스 전부에 확실히 적용됨.
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from pathlib import Path
from itertools import combinations
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 부모 프로세스용(이중 안전장치)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

import b1_dominate_a0 as D

OUTDIR = Path(__file__).parent
FIXED = ["ADL0103", "ADL0104", "ADL0105", "ADL0106A", "ADL0115A", "ADL0116A"]
R_RANGE = range(1, 9)          # 추가 1~8개 → 총 7~14문항
N_JOBS = -1                    # 코어 전부 사용. 느리면 정수(예: 4)로 줄이기
CHECKPOINT_EVERY = 500         # 이 개수마다 중간저장 + 진행률 출력 (너무 조용해 보이지 않게 짧게 잡음)

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
    grid = np.round(np.arange(0.05, 0.55, 0.03), 3); best = None
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


def eval_combo(cols_idx, X, rows, TRIALS, y, ds, a0, C, kw):
    """한 문항조합을 leave-one-trial-out(3)으로 평가. cols_idx: X의 열 인덱스 리스트."""
    warnings.filterwarnings("ignore", category=FutureWarning)  # 워커 안에서도 한 번 더(삼중 안전장치)
    pred = np.full(X.shape[0], np.nan)
    for ho in TRIALS:
        tr = np.concatenate([rows[t] for t in TRIALS if t != ho])
        te = rows[ho]
        Xtr = X[np.ix_(tr, cols_idx)]; Xte = X[np.ix_(te, cols_idx)]
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


def run_penalty(tag, kw, all_combos, ITEMS, X, rows, TRIALS, y, ds, a0, a1_mae, a0_miss, C=0.1):
    print(f"\n[{tag}] 조합 {len(all_combos)}개 평가 시작 (n_jobs={N_JOBS})")
    ckpt_path = OUTDIR / f"b2_fixed6_exhaustive_{tag}.csv"
    all_rows = []
    t_start = time.time()
    n_done = 0
    for chunk_start in range(0, len(all_combos), CHECKPOINT_EVERY):
        chunk = all_combos[chunk_start: chunk_start + CHECKPOINT_EVERY]
        # verbose>0이면 joblib이 조합 단위 완료마다 자체적으로 진행 로그를 찍어줘서
        # "화면이 조용해서 멈춘 줄 알고 끄는" 상황을 방지함.
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(eval_combo)(cols, X, rows, TRIALS, y, ds, a0, C, kw) for cols in chunk
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

    X = SimpleImputer(strategy="median").fit_transform(m[ITEMS])
    trial = m["STUDYID"].values
    y = m["ds_stage"].astype(int).values
    ds = m["ds_stage"].values
    a0 = m["A0_harmonized"].values
    TRIALS = D.TRIALS
    rows = {t: np.where(trial == t)[0] for t in TRIALS}
    a1_mae = D.mae(ds, m["A1_2015_stage"].values)
    a0_miss = D.miss(ds, a0) * 100
    print(f"기준 A1 MAE {a1_mae:.3f} / A0 중증놓침 {a0_miss:.1f}%\n")

    # --- 속도 감 잡기: 앞쪽 20조합으로 캘리브레이션 후 두 버전 합산 예상시간 출력 ---
    calib_n = 20
    for tag, kw in PENALTY_KW.items():
        t0 = time.time()
        for cols in all_combos[:calib_n]:
            eval_combo(list(cols), X, rows, TRIALS, y, ds, a0, 0.1, kw)
        per = (time.time() - t0) / calib_n
        est_serial_min = per * len(all_combos) / 60
        print(f"[{tag}] 조합당 약 {per*1000:.0f}ms(단일코어 추정) → 전체 단일코어 예상 "
              f"{est_serial_min:.0f}분 (실제는 n_jobs={N_JOBS} 병렬이라 코어수만큼 단축됨)")
    print()

    en_df = run_penalty("EN", PENALTY_KW["EN"], all_combos, ITEMS, X, rows, TRIALS, y, ds, a0, a1_mae, a0_miss)
    l2_df = run_penalty("L2", PENALTY_KW["L2"], all_combos, ITEMS, X, rows, TRIALS, y, ds, a0, a1_mae, a0_miss)

    both = pd.concat([en_df, l2_df], ignore_index=True)
    best = (both.sort_values(["penalty", "n_items", "mae"])
                .groupby(["penalty", "n_items"], as_index=False).first())
    best = best.sort_values(["n_items", "penalty"])
    best.to_csv(OUTDIR / "b2_fixed6_exhaustive_best_by_k.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> 저장: {OUTDIR / 'b2_fixed6_exhaustive_best_by_k.csv'} (문항수·penalty별 최저MAE 조합)")


if __name__ == "__main__":
    main()