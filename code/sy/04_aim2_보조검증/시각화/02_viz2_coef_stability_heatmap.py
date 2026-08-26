"""
시각화 2 — 절단×문항 계수 안정성 히트맵 (fold 간 일관성)
입력: 보조 검증/gn_coef_stability_summary.csv
출력: 시각화/fig2_coef_stability_heatmap.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

AUX_DIR = Path(r"C:\Users\USER\Documents\urp-AD\보조 검증")
OUT_DIR = AUX_DIR / "시각화"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(AUX_DIR / "gn_coef_stability_summary.csv")
cutpoints = ["P(Y>=1)", "P(Y>=2)", "P(Y>=3)", "P(Y>=4)", "P(Y>=5)"]
items = df["item"].drop_duplicates().tolist()

# 상태 코드: 0=항상 제외(0/3), 1=불안정(1~2/3), 2=일관 포함(3/3, 부호일관)
def status(row):
    if row["n_folds_nonzero"] == 0:
        return 0
    if row["n_folds_nonzero"] == 3 and row["sign_consistent"]:
        return 2
    return 1

df["status"] = df.apply(status, axis=1)
mat = df.pivot(index="item", columns="cutpoint", values="status").reindex(index=items, columns=cutpoints)

fig, ax = plt.subplots(figsize=(7, 8))
cmap = matplotlib.colors.ListedColormap(["#F1EFE8", "#F0997B", "#5DCAA5"])
im = ax.imshow(mat.values, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")

ax.set_xticks(range(len(cutpoints)))
ax.set_xticklabels(cutpoints, fontsize=10)
ax.set_yticks(range(len(items)))
ax.set_yticklabels(items, fontsize=9)

suspect = ["ADL0101", "ADL0110A", "ADL0112A"]
for i, it in enumerate(items):
    if it in suspect:
        ax.get_yticklabels()[i].set_color("#993C1D")
        ax.get_yticklabels()[i].set_fontweight("bold")

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.6)
cbar.ax.set_yticklabels(["항상 제외(0/3)", "불안정(1~2/3)", "일관 포함(3/3)"], fontsize=9)

ax.set_title("절단×문항 계수 안정성 (3-fold leave-one-trial-out)\n빨간 문항명 = 이전에 부호반전 의심됐던 3문항", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_coef_stability_heatmap.png", dpi=150, bbox_inches="tight")
print(f"저장: {OUT_DIR / 'fig2_coef_stability_heatmap.png'}")
