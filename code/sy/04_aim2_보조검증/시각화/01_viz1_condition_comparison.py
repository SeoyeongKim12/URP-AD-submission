"""
시각화 1 — B1 vs Gn vs A1 성능 비교 (내부CV + AD-1062 외부검증)
입력: Downloads/b1 결과/B1_03_성능지표.csv, 보조 검증/gn_ad1062_metrics.csv
출력: 시각화/fig1_condition_comparison.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

B1_DIR = Path(r"C:\Users\USER\Downloads\b1 결과")
AUX_DIR = Path(r"C:\Users\USER\Documents\urp-AD\보조 검증")
OUT_DIR = AUX_DIR / "시각화"
OUT_DIR.mkdir(parents=True, exist_ok=True)

perf = pd.read_csv(B1_DIR / "B1_03_성능지표.csv")
perf = perf.set_index(perf.columns[0])
ext = pd.read_csv(AUX_DIR / "gn_ad1062_metrics.csv")

mae_internal = {"B1": float(perf.loc["MAE", "원 B1"]), "Gn": float(perf.loc["MAE", "Gn(최종)"]), "A1": float(perf.loc["MAE", "A1(2015)"])}
hi_internal = {"B1": float(str(perf.loc["실제 4·5→≤3", "원 B1"]).replace("%", "")),
               "Gn": float(str(perf.loc["실제 4·5→≤3", "Gn(최종)"]).replace("%", "")),
               "A1": float(str(perf.loc["실제 4·5→≤3", "A1(2015)"]).replace("%", ""))}

mae_ext = {"Gn": float(ext.loc[ext.model == "Gn", "mae"].iloc[0]), "A1": float(ext.loc[ext.model == "A1", "mae"].iloc[0])}
hi_ext = {"Gn": float(ext.loc[ext.model == "Gn", "hi_missed"].iloc[0]) * 100,
          "A1": float(ext.loc[ext.model == "A1", "hi_missed"].iloc[0]) * 100}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
colors = {"B1": "#888780", "Gn": "#378ADD", "A1": "#D85A30"}

def bar(ax, d, title, ylabel):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    bars = ax.bar(keys, vals, color=[colors[k] for k in keys], width=0.55)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 5 else f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

bar(axes[0, 0], mae_internal, "MAE — 내부 CV (3개 훈련시험)", "MAE (낮을수록 좋음)")
bar(axes[0, 1], hi_internal, "4·5단계→≤3 과소분류율 — 내부 CV", "%  (낮을수록 좋음)")
bar(axes[1, 0], mae_ext, "MAE — AD-1062 독립 외부검증", "MAE (낮을수록 좋음)")
bar(axes[1, 1], hi_ext, "4·5단계→≤3 과소분류율 — AD-1062", "%  (낮을수록 좋음)")

fig.suptitle("B1 / Gn / A1 성능 비교 — 내부 CV vs 독립 외부검증(AD-1062)", fontsize=15, y=1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_condition_comparison.png", dpi=150, bbox_inches="tight")
print(f"저장: {OUT_DIR / 'fig1_condition_comparison.png'}")
