"""
Aim2-B3(순서형 랜덤포레스트) 결과 시각화
==========================================
cpad_dependence_aim2_b3_ordinal_rf.py 산출물(3개 csv)을 읽어 4개 그림을 만든다.
    - b3_ordered_forest_metrics.csv      : 시험별 MAE/가중카파/±1단계 정확도/과소평가율
    - b3_ordered_forest_calibration.csv  : 예측단계별 관측평균단계·정확일치율
    - b3_ordered_forest_predictions.csv  : 참가자별 실제(0·1통합 5범주)-예측 쌍

출력(모두 outputs 폴더에 png로 저장):
    fig1_performance_by_trial.png   : 시험별 MAE / 가중카파 / ±1단계 정확도 막대그래프
    fig2_calibration_reliability.png: 시험별 reliability diagram(예측단계 vs 관측평균단계)
    fig3_confusion_matrix.png       : 3개 시험 합산 혼동행렬(정규화, 실제 x 예측)
    fig4_underestimation.png        : 시험별 과소평가 지표 2종 막대그래프
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---- 한글 폰트 설정 (OS별 기본 한글 폰트 자동 탐색) ----
# Windows: 맑은 고딕(Malgun Gothic), macOS: Apple SD Gothic Neo,
# Linux: 시스템에 설치된 Noto Sans CJK 계열. 설치된 폰트 중 먼저 찾은 것을 쓴다.
_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",       # Windows 기본
    "AppleGothic",         # macOS
    "Apple SD Gothic Neo", # macOS
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Noto Sans CJK JP",
]
_installed = {f.name for f in fm.fontManager.ttflist}
_chosen = next((name for name in _KOREAN_FONT_CANDIDATES if name in _installed), None)
if _chosen is None:
    print("[경고] 한글 폰트를 찾지 못해 기본 폰트로 표시함(한글이 깨질 수 있음). "
          "맑은 고딕(Windows) 등 한글 폰트 설치를 확인할 것.")
else:
    plt.rcParams["font.family"] = _chosen
plt.rcParams["axes.unicode_minus"] = False

METRICS_CSV = "b3_ordered_forest_metrics.csv"
CALIB_CSV = "b3_ordered_forest_calibration.csv"
PRED_CSV = "b3_ordered_forest_predictions.csv"

STAGE_LABELS_5CAT = {1: "0·1단계", 2: "2단계", 3: "3단계", 4: "4단계", 5: "5단계"}
TRIAL_COLORS = {"AD-1061": "#4C72B0", "AD-1063": "#DD8452", "AD-1064": "#55A868"}


def fig1_performance_by_trial(metrics: pd.DataFrame, out="fig1_performance_by_trial.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("MAE", "MAE (평균절대오차, 단계)", axes[0], False),
        ("weighted_kappa", "가중 카파(quadratic)", axes[1], False),
        ("acc_within_1stage", "±1단계 정확도", axes[2], True),
    ]
    for col, title, ax, is_pct in panels:
        vals = metrics[col]
        bars = ax.bar(metrics["held_out_STUDYID"], vals,
                       color=[TRIAL_COLORS[s] for s in metrics["held_out_STUDYID"]])
        mean_val = vals.mean()
        ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1)
        ax.text(2.5, mean_val, f" 평균 {mean_val:.3f}", va="bottom", ha="right",
                fontsize=9, color="gray")
        for b, v in zip(bars, vals):
            label = f"{v:.1%}" if is_pct else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"held-out 시험: {title}", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("B3(순서형 랜덤포레스트) leave-one-trial-out 성능 — 시험별", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_calibration_reliability(calib: pd.DataFrame, out="fig2_calibration_reliability.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([1, 5], [1, 5], color="black", linestyle="--", linewidth=1, label="완전 보정(y=x)")
    for study in calib["held_out_STUDYID"].unique():
        sub = calib[(calib["held_out_STUDYID"] == study) & (calib["n"] > 0)]
        ax.plot(sub["predicted_stage"], sub["observed_mean_stage"],
                marker="o", label=study, color=TRIAL_COLORS[study])
        for _, row in sub.iterrows():
            ax.annotate(f"n={int(row['n'])}",
                        (row["predicted_stage"], row["observed_mean_stage"]),
                        textcoords="offset points", xytext=(6, -4), fontsize=8, color="gray")
    ax.set_xlabel("예측단계 (0·1 통합 5범주)")
    ax.set_ylabel("해당 예측단계에서의 관측 평균 실제단계")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([STAGE_LABELS_5CAT[i] for i in [1, 2, 3, 4, 5]])
    ax.set_title("Reliability diagram — 예측단계별 관측 평균 실제단계")
    ax.legend(title="held-out 시험", loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3_confusion_matrix(pred: pd.DataFrame, out="fig3_confusion_matrix.png"):
    classes = [1, 2, 3, 4, 5]
    mat = pd.crosstab(pred["y_true_5cat"], pred["y_pred_5cat"])
    mat = mat.reindex(index=classes, columns=classes, fill_value=0)
    mat_norm = mat.div(mat.sum(axis=1), axis=0)  # 행(실제단계) 기준 정규화

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    im = ax.imshow(mat_norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    labels = [STAGE_LABELS_5CAT[c] for c in classes]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("예측단계")
    ax.set_ylabel("실제단계")
    ax.set_title("3개 시험 합산 혼동행렬 (행 기준 정규화, n=2,208)")
    for i in range(5):
        for j in range(5):
            n_raw = mat.iloc[i, j]
            frac = mat_norm.iloc[i, j]
            if pd.isna(frac):
                continue
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{frac:.0%}\n(n={n_raw})", ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="실제단계 내 비율")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4_underestimation(metrics: pd.DataFrame, out="fig4_underestimation.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    b1 = ax.bar(x - width / 2, metrics["underest_ge2_rate"], width,
                label="2단계 이상 과소평가율", color="#C44E52")
    b2 = ax.bar(x + width / 2, metrics["top2class_underclassified_rate"], width,
                label="실제 4·5단계를 과소분류한 비율(근사)", color="#8172B2")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["held_out_STUDYID"])
    ax.set_ylabel("비율")
    ax.set_ylim(0, 1.0)
    ax.set_title("B3 과소평가 지표 (0·1 통합 5범주 근사치)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig5_permutation_importance(perm_imp: pd.DataFrame, out="fig5_permutation_importance.png", top_n=15):
    sub = perm_imp.sort_values("importance_mae_increase_mean", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y, sub["importance_mae_increase_mean"],
            xerr=sub["importance_mae_increase_std"], color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["item"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("문항 값을 섞었을 때 MAE 증가량 (클수록 중요)")
    ax.set_title("B3 Permutation Importance — 상위 문항")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig6_partial_dependence(pdp: pd.DataFrame, out="fig6_partial_dependence.png"):
    items = pdp["item"].unique()
    fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 4), sharey=True)
    if len(items) == 1:
        axes = [axes]
    for ax, item in zip(axes, items):
        sub = pdp[pdp["item"] == item]
        ax.errorbar(sub["value"], sub["expected_stage_mean"],
                     yerr=1.96 * sub["expected_stage_se"], marker="o", capsize=3,
                     color="#DD8452")
        ax.set_title(item, fontsize=10)
        ax.set_xlabel("문항 점수")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("예측단계 기댓값 E[Y]")
    fig.suptitle("Partial Dependence — permutation importance 상위 문항", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    metrics = pd.read_csv(METRICS_CSV)
    calib = pd.read_csv(CALIB_CSV)
    pred = pd.read_csv(PRED_CSV)

    fig1_performance_by_trial(metrics)
    fig2_calibration_reliability(calib)
    fig3_confusion_matrix(pred)
    fig4_underestimation(metrics)
    print("저장 완료: fig1~fig4 (성능/calibration/혼동행렬/과소평가)")

    try:
        perm_imp = pd.read_csv("b3_permutation_importance.csv")
        fig5_permutation_importance(perm_imp)
        print("저장 완료: fig5_permutation_importance.png")
    except FileNotFoundError:
        print("[스킵] b3_permutation_importance.csv 없음 — cpad_b3_interpretation.py 먼저 실행할 것")

    try:
        pdp = pd.read_csv("b3_partial_dependence_top5.csv")
        fig6_partial_dependence(pdp)
        print("저장 완료: fig6_partial_dependence.png")
    except FileNotFoundError:
        print("[스킵] b3_partial_dependence_top5.csv 없음 — cpad_b3_interpretation.py 먼저 실행할 것")
