# -*- coding: utf-8 -*-
"""
Aim 2 · B2 문항축약 v9 — 그리디 래퍼(LOCO) + AD-1062 외부검증 (로컬 실행판)
================================================================================
원본(Colab, b2_item_reduction_v9_greedy_wrapper.ipynb)과 로직은 동일하고,
아래 두 가지만 바꿨다.

1. Google Drive 마운트 → 로컬 경로로 교체.
2. §11 외부검증 표본 구축 — 원본은 "AD-1062가 baseline_sample.csv/adl_wide.csv
   안에 STUDYID로 같이 들어있다"고 가정하고 필터링만 했는데, 실제로는
   AD-1062가 그 두 파일에 전혀 없다(3개 시험 AD-1061/1063/1064만 들어있음).
   대신 별도 폴더(dependence_study_csv/extra_validation_ad1062)에
   adl_wide_ad1062.csv(=A0_harmonized·A1_2015_stage 계산까지 끝난 ADL wide)·
   ds_wide_ad1062.csv(=ds_stage 계산까지 끝난 DS wide)·dm_ad1062.csv(인구통계)가
   따로 준비되어 있어, 이 세 파일을 §1과 동일한 로직(병합 → Q18 평균 →
   ds_stage/A0_harmonized 결측 제외)으로 조립해서 m_ext를 새로 만든다.
   ⚠ AD-1062는 방문번호 체계가 달라 baseline이 VISITNUM==1.0이다
   (AD-1061/1063/1064는 VISITNUM==2.0). BASELINE_VISITNUM_EXT로 확인하세요.

나머지(지표 함수·CV·그리디 래퍼·기여도표·시각화 로직)는 원본과 100% 동일하다.

필요 패키지: pip install pandas numpy scikit-learn matplotlib koreanize-matplotlib
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn 1.8+ penalty= 인자 지정 경고 억제

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # noqa: F401  (그래프 한글 깨짐 방지)
except ImportError:
    print("⚠ koreanize-matplotlib이 없습니다. 그래프 한글이 깨지면:")
    print("    pip install koreanize-matplotlib")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

# =================================================================
# 설정 — 로컬 경로
# =================================================================
DATA_DIR = Path(r"C:\Users\USER\Documents\urp-AD\dependence_study_csv")
EXTERNAL_DIR = DATA_DIR / "extra_validation_ad1062"
OUT_DIR = DATA_DIR / "b2v9_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = DATA_DIR / "baseline_sample.csv"
ADL_PATH = DATA_DIR / "adl_wide.csv"

DS_EXT_PATH = EXTERNAL_DIR / "ds_wide_ad1062.csv"
ADL_EXT_PATH = EXTERNAL_DIR / "adl_wide_ad1062.csv"
DM_EXT_PATH = EXTERNAL_DIR / "dm_ad1062.csv"

EXTERNAL_STUDYID = "AD-1062"
BASELINE_VISITNUM_TRAIN = 2.0   # AD-1061/1063/1064 (baseline_sample.csv / adl_wide.csv)
BASELINE_VISITNUM_EXT = 1.0     # AD-1062만 방문번호 체계가 다름 — ds_stage/A0_harmonized 보유율로 확인된 값

ALPHA = 0.05
TRIALS = ["AD-1061", "AD-1063", "AD-1064"]
C_DEFAULT = 0.1          # sjlee Gn 스크립트에서 중첩CV로 선택된 대표 C (원본 v7/v9과 동일)

# ---- 속도/탐색 범위 옵션 (원본과 동일) ----
RUN_L2 = True             # 빠름(lbfgs) — 기본 실행
RUN_ELASTICNET = False    # 느림(saga, l1_ratio=0.5) — 시간 여유 있을 때 True로
ITEM_FLOOR = 8             # 이 문항수까지만 탐색(v7에서 9문항 이하는 성능 붕괴 시작)
SAFETY_MARGIN = 1.0        # 그리디 래퍼 안전조건 여유(%p)
EXT_MISS_MARGIN = 1.0      # 외부검증 판정에 쓰는 A0 + margin(%p) 기준 (원본과 동일)

BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = [
    "ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
    "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
    "ADL0121A", "ADL0122Q", "ADL0123L",
]
ITEMS = BADL + IADL
NI = len(ITEMS)
LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b옷입기","ADL0106A":"Q6a옷고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자있기","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
Q18_SUBITEMS = ["ADL0118A__resolved", "ADL0118B__resolved", "ADL0118C__resolved"]

_lines = []
def log(s=""):
    print(s)
    _lines.append(str(s))

# =================================================================
# 1. 학습표본 구축 — baseline_sample + adl_wide 병합 (AD-1061/1063/1064)
#    원본 §1과 동일 로직
# =================================================================
df_base = pd.read_csv(BASELINE_PATH)
df_adl = pd.read_csv(ADL_PATH)

df_base_use = df_base[df_base["in_aim1_2_sample"] == True].copy()
df_adl_base = df_adl[df_adl["VISITNUM"] == BASELINE_VISITNUM_TRAIN].copy()
m = df_base_use.merge(df_adl_base, on=["STUDYID", "USUBJID"], how="inner", suffixes=("", "_adl"))

missing_sub = [c for c in Q18_SUBITEMS if c not in m.columns]
if missing_sub:
    print("⚠️ Q18 하위문항 중 없는 것:", missing_sub)
else:
    m["Q18"] = m[Q18_SUBITEMS].mean(axis=1)

required_cols = ITEMS + ["ds_stage", "A0_harmonized", "A1_2015_stage"]
missing_cols = [c for c in required_cols if c not in m.columns]
if missing_cols:
    print("⚠️ 다음 컬럼이 없습니다:", missing_cols)
    ITEMS = [c for c in ITEMS if c in m.columns]
    NI = len(ITEMS)
    print("→ 남은 문항수:", NI, ITEMS)

m = m.dropna(subset=["ds_stage"]).reset_index(drop=True)
before_n = len(m)
m = m.dropna(subset=["A0_harmonized"]).reset_index(drop=True)
if len(m) < before_n:
    print(f"⚠️ A0_harmonized 결측 {before_n - len(m)}건 제외 → 남은 표본 {len(m)}명")

log("# Aim 2 · B2 문항축약 v9 — 그리디 래퍼(LOCO) + AD-1062 외부검증 (로컬 실행판)\n")
log(f"학습표본(AD-1061/1063/1064) n={len(m)} (STUDYID별: {m['STUDYID'].value_counts().to_dict()})")

# =================================================================
# 2. 지표 함수 — 원본과 동일
# =================================================================
def mae(t, pred):
    t = np.asarray(t, dtype=float); pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(t - pred)))

def kap(t, pred):
    t = np.asarray(t, dtype=float); pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(t) | np.isnan(pred))
    t = t[mask].astype(int); pred = pred[mask].astype(int)
    if len(t) == 0:
        return float("nan")
    return float(cohen_kappa_score(t, pred, weights="quadratic"))

def miss(t, pred):
    """중증놓침: 실제 4~5단계인데 예측이 4 미만으로 나온 비율."""
    t = np.asarray(t, dtype=float); pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(t) | np.isnan(pred))
    t = t[mask]; pred = pred[mask]
    severe = t >= 4
    if severe.sum() == 0:
        return 0.0
    return float(np.mean(pred[severe] < 4))

def asym(P, t4, t5):
    """비대칭 임계값 분류: 중증(4~5단계)을 덜 놓치는 쪽으로 편향."""
    pred = np.argmax(P, axis=1).astype(float)
    p45 = P[:, 4] + P[:, 5]
    pred = np.where(p45 >= t4, 4.0, pred)
    pred = np.where(P[:, 5] >= t5, 5.0, pred)
    return pred

def metrics_all(t, pred):
    t = np.asarray(t, dtype=float); pred = np.asarray(pred, dtype=float)
    diff = pred - t
    return {
        "MAE": mae(t, pred),
        "중증놓침(%)": miss(t, pred) * 100,
        "kappa": kap(t, pred),
        "과분류(%)": (diff > 0).mean() * 100,
        "과소분류(%)": (diff < 0).mean() * 100,
        "정확일치(%)": (diff == 0).mean() * 100,
        "인접일치±1(%)": (np.abs(diff) <= 1).mean() * 100,
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
    }

# =================================================================
# 3. 기준선(A0/A1) 및 CV 유틸 — 원본과 동일
# =================================================================
a1_mae = mae(m["ds_stage"].values, m["A1_2015_stage"].values)
a1_metrics = metrics_all(m["ds_stage"].values, m["A1_2015_stage"].values)
a0_metrics = metrics_all(m["ds_stage"].values, m["A0_harmonized"].values)
a0_miss = a0_metrics["중증놓침(%)"]
log(f"A1(2015판) 기준: MAE={a1_metrics['MAE']:.3f}, 중증놓침={a1_metrics['중증놓침(%)']:.1f}%, kappa={a1_metrics['kappa']:.3f}")
log(f"A0(2025판 조화) 기준: MAE={a0_metrics['MAE']:.3f}, 중증놓침={a0_metrics['중증놓침(%)']:.1f}%, kappa={a0_metrics['kappa']:.3f}")

class _ConstantProba:
    """희귀 절단(예: Y>=5)이 특정 fold의 훈련셋에서 전부 0(또는 전부 1)인 경우를 대비한 안전장치."""
    def __init__(self, rate):
        self.rate = float(np.clip(rate, 1e-6, 1 - 1e-6))
    def predict_proba(self, Z):
        p1 = np.full(Z.shape[0], self.rate)
        return np.column_stack([1 - p1, p1])

def fit_cuts(Ztr, y, penalty="l2", C=C_DEFAULT):
    if penalty == "elasticnet":
        kwargs = dict(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=C, max_iter=3000, tol=1e-3, random_state=0)
    elif penalty == "l2":
        kwargs = dict(penalty="l2", solver="lbfgs", C=C, max_iter=2000, random_state=0)
    else:
        raise ValueError(f"알 수 없는 penalty: {penalty}")
    models = {}
    for k in range(1, 6):
        yk = (y >= k).astype(int)
        if len(np.unique(yk)) < 2:
            models[k] = _ConstantProba(yk.mean())
        else:
            models[k] = LogisticRegression(**kwargs).fit(Ztr, yk)
    return models

def probs(models, Z):
    g = {k: models[k].predict_proba(Z)[:, 1] for k in range(1, 6)}
    P = np.zeros((Z.shape[0], 6)); P[:, 0] = 1 - g[1]
    for k in range(1, 5): P[:, k] = g[k] - g[k + 1]
    P[:, 5] = g[5]; P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True); return P

def tune_tau(Ptr, ds_tr, a0m):
    grid = np.round(np.arange(0.05, 0.55, 0.02), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym(Ptr, t4, t5)
            if miss(ds_tr, p) <= a0m + 1e-9:
                key = mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, t4, t5)
    if best is None:
        c = [(miss(ds_tr, asym(Ptr, a, b)), mae(ds_tr, asym(Ptr, a, b)), a, b) for a in grid for b in grid if b >= a]
        _, _, t4, t5 = min(c); best = (0, t4, t5)
    return best[1], best[2]

def prep(m):
    folds = {}
    for ho in TRIALS:
        tri = np.array(m.index[m.STUDYID != ho]); tei = np.array(m.index[m.STUDYID == ho])
        imp = SimpleImputer(strategy="median").fit(m.loc[tri, ITEMS])
        Xtr = imp.transform(m.loc[tri, ITEMS]); Xte = imp.transform(m.loc[tei, ITEMS])
        y = m.loc[tri, "ds_stage"].astype(int).values; ds_tr = m.loc[tri, "ds_stage"].values
        a0m = miss(ds_tr, m.loc[tri, "A0_harmonized"].values)
        folds[ho] = dict(tei=tei, Xtr=Xtr, Xte=Xte, y=y, ds_tr=ds_tr, a0m=a0m)
    return folds

def eval_subset(m, folds, keep, penalty="l2", C=C_DEFAULT):
    t = m["ds_stage"].values; pred = np.zeros(len(m))
    for ho in TRIALS:
        f = folds[ho]
        sc = StandardScaler().fit(f["Xtr"][:, keep])
        mdl = fit_cuts(sc.transform(f["Xtr"][:, keep]), f["y"], penalty=penalty, C=C)
        Ptr = probs(mdl, sc.transform(f["Xtr"][:, keep])); Pte = probs(mdl, sc.transform(f["Xte"][:, keep]))
        t4, t5 = tune_tau(Ptr, f["ds_tr"], f["a0m"])
        pred[f["tei"]] = asym(Pte, t4, t5)
    return metrics_all(t, pred)

folds = prep(m)
log("폴드 준비 완료 (leave-one-trial-out ×3)\n")

# =================================================================
# 4. 그리디 후진소거(LOCO wrapper) — 원본과 동일
# =================================================================
def greedy_backward_wrapper(m, folds, penalty="l2", C=C_DEFAULT, item_floor=ITEM_FLOOR,
                             safety_margin=SAFETY_MARGIN, log_every=1):
    miss_limit = a0_miss + safety_margin
    keep = list(range(NI))
    met0 = eval_subset(m, folds, keep, penalty, C)
    history = [{
        "문항수": len(keep), "제거_코드": "-", "제거_라벨": "-", "선택전략": "-",
        "ΔMAE": 0.0, "Δ미스율": 0.0, "Δkappa": 0.0, **met0, "keep": list(keep),
    }]
    step = 0
    while len(keep) > item_floor:
        met_before = history[-1]
        candidates = []
        for j in keep:
            trial = [x for x in keep if x != j]
            met_try = eval_subset(m, folds, trial, penalty, C)
            candidates.append((j, met_try))
        safe = [c for c in candidates if c[1]["중증놓침(%)"] <= miss_limit + 1e-9]
        if safe:
            j_remove, met_try = min(safe, key=lambda c: c[1]["MAE"])
            strategy = "안전조건충족(MAE최소)"
        else:
            j_remove, met_try = min(candidates, key=lambda c: c[1]["중증놓침(%)"])
            strategy = "안전조건전부위반(미스율최소로후퇴)"
        code_ = ITEMS[j_remove]
        keep = [x for x in keep if x != j_remove]
        history.append({
            "문항수": len(keep), "제거_코드": code_, "제거_라벨": LAB[code_], "선택전략": strategy,
            "ΔMAE": met_try["MAE"] - met_before["MAE"],
            "Δ미스율": met_try["중증놓침(%)"] - met_before["중증놓침(%)"],
            "Δkappa": met_try["kappa"] - met_before["kappa"],
            **met_try, "keep": list(keep),
        })
        step += 1
        if step % log_every == 0:
            log(f"  [{penalty}] {len(keep)+1}→{len(keep)}문항: '{LAB[code_]}' 제거 [{strategy}] "
                f"(ΔMAE={history[-1]['ΔMAE']:+.4f}, Δ미스율={history[-1]['Δ미스율']:+.2f}%p, "
                f"미스율={met_try['중증놓침(%)']:.2f}% vs 한도{miss_limit:.2f}%)")
    return history

# =================================================================
# 5. 실행 — L2(빠름, 기본 켜짐) / Elastic-net(선택)
# =================================================================
wrapper_histories = {}
if RUN_L2:
    log("\n### 그리디 래퍼 실행 — penalty=l2")
    wrapper_histories["l2"] = greedy_backward_wrapper(m, folds, penalty="l2")
    df_l2 = pd.DataFrame(wrapper_histories["l2"]).drop(columns=["keep"])
    df_l2.round(4).to_csv(OUT_DIR / "b2v9_wrapper_curve_l2.csv", index=False, encoding="utf-8-sig")
    log(f">>> 저장: {OUT_DIR / 'b2v9_wrapper_curve_l2.csv'}")
else:
    log("\n(RUN_L2=False — L2 그리디 래퍼 건너뜀)")

if RUN_ELASTICNET:
    log("\n### 그리디 래퍼 실행 — penalty=elasticnet (시간이 오래 걸릴 수 있습니다)")
    wrapper_histories["elasticnet"] = greedy_backward_wrapper(m, folds, penalty="elasticnet")
    df_en = pd.DataFrame(wrapper_histories["elasticnet"]).drop(columns=["keep"])
    df_en.round(4).to_csv(OUT_DIR / "b2v9_wrapper_curve_elasticnet.csv", index=False, encoding="utf-8-sig")
    log(f">>> 저장: {OUT_DIR / 'b2v9_wrapper_curve_elasticnet.csv'}")
else:
    log("\n(RUN_ELASTICNET=False — 상단 상수에서 True로 바꾸면 실행됩니다)")

# =================================================================
# 6. 곡선 시각화 + "둘다이김" 요약 — 원본과 동일
# =================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
styles = {"l2": dict(marker="o", linestyle="-", color="tab:green", label="그리디 래퍼 · L2"),
          "elasticnet": dict(marker="s", linestyle="--", color="tab:red", label="그리디 래퍼 · Elastic-net")}
ax = axes[0]
for key, hist in wrapper_histories.items():
    dfk = pd.DataFrame(hist)
    ax.plot(dfk["문항수"], dfk["MAE"], **styles[key])
ax.axhline(a1_mae, color="gray", linestyle=":", label="A1 기준")
ax.invert_xaxis(); ax.set_xlabel("문항수"); ax.set_ylabel("MAE"); ax.set_title("MAE"); ax.legend(fontsize=8)
ax = axes[1]
for key, hist in wrapper_histories.items():
    dfk = pd.DataFrame(hist)
    ax.plot(dfk["문항수"], dfk["중증놓침(%)"], **styles[key])
ax.axhline(a0_miss, color="gray", linestyle=":", label="A0 기준")
ax.invert_xaxis(); ax.set_xlabel("문항수"); ax.set_ylabel("중증놓침(%)"); ax.set_title("중증놓침"); ax.legend(fontsize=8)
ax = axes[2]
for key, hist in wrapper_histories.items():
    dfk = pd.DataFrame(hist)
    ax.plot(dfk["문항수"], dfk["kappa"], **styles[key])
ax.invert_xaxis(); ax.set_xlabel("문항수"); ax.set_ylabel("가중카파"); ax.set_title("가중카파"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "b2v9_wrapper_curve.png", dpi=150)
plt.close(fig)

log("\n## 조합별 '둘다이김'(A1보다 MAE 낮고 A0+1%p 이내 중증놓침) 요약")
for key, hist in wrapper_histories.items():
    dfk = pd.DataFrame(hist)
    ok = dfk[(dfk["MAE"] < a1_mae) & (dfk["중증놓침(%)"] <= a0_miss + 1.0)]
    if len(ok):
        best_row = ok.loc[ok["MAE"].idxmin()]
        log(f"[{key}] 둘다이김 유지 최소 문항수={int(ok['문항수'].min())}, "
            f"구간 내 최저MAE 지점: 문항수={int(best_row['문항수'])}, MAE={best_row['MAE']:.3f}, "
            f"kappa={best_row['kappa']:.3f}, 중증놓침={best_row['중증놓침(%)']:.1f}%")
    else:
        log(f"[{key}] ⚠️ 둘다이김 조건을 만족하는 지점이 없습니다 (item_floor={ITEM_FLOOR}를 낮춰 더 탐색해보세요).")

# =================================================================
# 7. 문항별 성능 기여도표 (LOCO 해석) — 원본과 동일
# =================================================================
def build_contribution_table(hist, penalty_label):
    rows = []
    for i, h in enumerate(hist):
        if h["제거_코드"] == "-":
            continue
        rows.append({
            "패널티": penalty_label,
            "제거순번": i,
            "중요도순위(역순)": len(hist) - i,
            "코드": h["제거_코드"], "라벨": h["제거_라벨"], "선택전략": h.get("선택전략", "-"),
            "제거직전_문항수": hist[i - 1]["문항수"],
            "ΔMAE": h["ΔMAE"], "Δ미스율(%p)": h["Δ미스율"], "Δkappa": h["Δkappa"],
        })
    return pd.DataFrame(rows).sort_values("중요도순위(역순)", ascending=False).reset_index(drop=True)

contribution_tables = {}
for key, hist in wrapper_histories.items():
    ctab = build_contribution_table(hist, key)
    contribution_tables[key] = ctab
    log(f"\n## 문항별 성능 기여도표 (LOCO, {key}) — 아래로 갈수록 덜 중요(먼저 제거)")
    log(ctab.round(4).to_string(index=False))
    ctab.round(4).to_csv(OUT_DIR / f"b2v9_item_contribution_{key}.csv", index=False, encoding="utf-8-sig")

# =================================================================
# 8. 외부검증 — AD-1062 (그리디 래퍼 탐색에 전혀 쓰이지 않은 시험)
#    ⚠ 여기가 원본 노트북과 다른 부분: AD-1062는 baseline_sample.csv/adl_wide.csv에
#    아예 들어있지 않으므로, extra_validation_ad1062 폴더의 세 원자료 파일을
#    §1과 동일 로직으로 직접 병합해서 m_ext를 새로 만든다.
# =================================================================
log(f"\n## 외부검증 원자료 로드: {EXTERNAL_DIR}")
df_ds_ext = pd.read_csv(DS_EXT_PATH)
df_adl_ext_raw = pd.read_csv(ADL_EXT_PATH)
df_dm_ext = pd.read_csv(DM_EXT_PATH)

# ds_stage 보유율로 baseline VISITNUM을 재확인 (하드코딩값이 안 맞으면 경고)
_visit_check = df_ds_ext.groupby("VISITNUM")["ds_stage"].apply(lambda s: s.notna().mean())
log(f"AD-1062 VISITNUM별 ds_stage 관측률: {_visit_check.round(3).to_dict()}")
if BASELINE_VISITNUM_EXT not in _visit_check.index or _visit_check.loc[BASELINE_VISITNUM_EXT] < 0.5:
    log(f"⚠️ BASELINE_VISITNUM_EXT={BASELINE_VISITNUM_EXT}가 baseline이 아닌 것 같습니다 — 위 관측률을 보고 상단 상수를 조정하세요.")

ds_base_ext = df_ds_ext[df_ds_ext["VISITNUM"] == BASELINE_VISITNUM_EXT][["STUDYID", "USUBJID", "ds_stage"]].copy()
adl_base_ext = df_adl_ext_raw[df_adl_ext_raw["VISITNUM"] == BASELINE_VISITNUM_EXT].copy()
dm_use_ext = (df_dm_ext[["STUDYID", "USUBJID", "AGE", "SEX", "ARM"]]
              .drop_duplicates(subset=["STUDYID", "USUBJID"]))

m_ext = ds_base_ext.merge(dm_use_ext, on=["STUDYID", "USUBJID"], how="left")
m_ext = m_ext.merge(adl_base_ext, on=["STUDYID", "USUBJID"], how="inner", suffixes=("", "_adl"))

missing_sub_ext = [c for c in Q18_SUBITEMS if c not in m_ext.columns]
if missing_sub_ext:
    log(f"⚠️ AD-1062 Q18 하위문항 중 없는 것: {missing_sub_ext}")
else:
    m_ext["Q18"] = m_ext[Q18_SUBITEMS].mean(axis=1)

missing_items_ext = [c for c in ITEMS if c not in m_ext.columns]
if missing_items_ext:
    log(f"⚠️ AD-1062에 없는 ITEMS 컬럼: {missing_items_ext} — 외부검증에서 제외됩니다.")

m_ext = m_ext.dropna(subset=["ds_stage"]).reset_index(drop=True)
before_ext = len(m_ext)
m_ext = m_ext.dropna(subset=["A0_harmonized"]).reset_index(drop=True)
if len(m_ext) < before_ext:
    log(f"⚠️ {EXTERNAL_STUDYID} A0_harmonized 결측 {before_ext - len(m_ext)}건 제외 → 남은 표본 {len(m_ext)}명")

log(f"\n## 외부검증 표본: {EXTERNAL_STUDYID} (그리디 래퍼 탐색·학습에 전혀 미사용), n={len(m_ext)}")

def fit_final_and_validate_external(keep_idx, penalty="l2", C=C_DEFAULT, label=""):
    """m(3개 시험) 전체로 최종 모형을 한 번 학습 → m_ext(AD-1062)에 1회 적용."""
    keep_items = [ITEMS[i] for i in keep_idx]
    imp = SimpleImputer(strategy="median").fit(m[keep_items])
    sc = StandardScaler().fit(imp.transform(m[keep_items]))
    Ztr = sc.transform(imp.transform(m[keep_items]))
    y = m["ds_stage"].astype(int).values
    ds_tr = m["ds_stage"].values
    a0m_full = miss(ds_tr, m["A0_harmonized"].values)
    mdl = fit_cuts(Ztr, y, penalty=penalty, C=C)
    Ptr = probs(mdl, Ztr)
    t4, t5 = tune_tau(Ptr, ds_tr, a0m_full)
    Zte_ext = sc.transform(imp.transform(m_ext[keep_items]))
    Pte_ext = probs(mdl, Zte_ext)
    pred_ext = asym(Pte_ext, t4, t5)
    met_ext = metrics_all(m_ext["ds_stage"].values, pred_ext)
    met_a1_ext = metrics_all(m_ext["ds_stage"].values, m_ext["A1_2015_stage"].values)
    passed = (met_ext["MAE"] < met_a1_ext["MAE"]) and (met_ext["중증놓침(%)"] <= a0_miss + EXT_MISS_MARGIN)
    log(f"\n## 외부검증({EXTERNAL_STUDYID}) — {label} (문항수={len(keep_items)})")
    log(f"  선택모형: MAE={met_ext['MAE']:.3f}, 중증놓침={met_ext['중증놓침(%)']:.1f}%, kappa={met_ext['kappa']:.3f}")
    log(f"  A1(2015판) 참고: MAE={met_a1_ext['MAE']:.3f}, 중증놓침={met_a1_ext['중증놓침(%)']:.1f}%, kappa={met_a1_ext['kappa']:.3f}")
    log(f"  판정: {'✅ 외부검증 통과' if passed else '❌ 외부검증 미달'} "
        f"(A1보다 MAE 낮고, 중증놓침 ≤ {a0_miss + EXT_MISS_MARGIN:.2f}% 기준)")
    return dict(keep_items=keep_items, met_ext=met_ext, met_a1_ext=met_a1_ext, passed=passed)

# 원본은 16문항·8문항 두 지점만 검증했지만, 여기서는 그리디 래퍼가 지나온
# 모든 문항수(21→ITEM_FLOOR)에 대해 AD-1062 외부검증을 전부 돌려 곡선으로 남긴다.
hist_l2 = wrapper_histories["l2"]
ext_rows = []
for h in hist_l2:
    res = fit_final_and_validate_external(h["keep"], penalty="l2", label=f"그리디 래퍼 L2 · {h['문항수']}문항")
    ext_rows.append({"문항수": h["문항수"], **res["met_ext"], "통과": res["passed"]})

ext_summary = pd.DataFrame(ext_rows).sort_values("문항수", ascending=False).reset_index(drop=True)
log("\n## 외부검증(AD-1062) 요약 — 문항수별")
log(ext_summary.round(3).to_string(index=False))
ext_summary.round(3).to_csv(EXTERNAL_DIR / "b2v9_external_validation_AD1062.csv", index=False, encoding="utf-8-sig")
log(f"\n>>> 저장: {EXTERNAL_DIR / 'b2v9_external_validation_AD1062.csv'}")

# 내부 CV 곡선 vs 외부검증 곡선 비교 그래프
fig, ax = plt.subplots(figsize=(8, 5))
dfk = pd.DataFrame(hist_l2)
ax.plot(dfk["문항수"], dfk["MAE"], marker="o", color="tab:green", label="내부 CV(leave-one-trial-out) MAE")
ax.plot(ext_summary["문항수"], ext_summary["MAE"], marker="^", linestyle="--", color="tab:purple",
        label="외부검증(AD-1062) MAE")
ax.axhline(a1_mae, color="gray", linestyle=":", label="A1 기준(학습표본)")
ax.invert_xaxis(); ax.set_xlabel("문항수"); ax.set_ylabel("MAE")
ax.set_title("그리디 래퍼(L2): 내부 CV vs AD-1062 외부검증")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "b2v9_external_validation_AD1062_curve.png", dpi=150)
plt.close(fig)
log(f">>> 저장: {OUT_DIR / 'b2v9_external_validation_AD1062_curve.png'}")

# =================================================================
# 9. 리포트 저장
# =================================================================
REPORT_PATH = OUT_DIR / "b2v9_wrapper_report.md"
REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
print(f">>> 저장: {REPORT_PATH}")