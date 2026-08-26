"""
시각화 3 — fold별 비단조 확률 발생률 vs 음수질량 (이중축)
입력: 보조 검증/gn_monotonic_diagnostic.csv
출력: 시각화/fig3_monotonic_violation.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

AUX_DIR = Path(r"C:\Users\USER\Documents\urp-AD\보조 검증")
OUT_DIR = AUX_DIR / "시각화"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(AUX_DIR / "gn_monotonic_diagnostic.csv")

fig, ax1 = plt.subplots(figsize=(8, 5.5))
x = range(len(df))
bars = ax1.bar(x, df["violation_rate"] * 100, color="#F0997B", width=0.5, label="비단조 발생률(%)")
ax1.axhline(5, color="#993C1D", linestyle="--", linewidth=1.2, label="사전기준 5%")
ax1.set_xticks(list(x))
ax1.set_xticklabels(df["held_out_trial"], fontsize=11)
ax1.set_ylabel("비단조 발생률 (%)", fontsize=11, color="#993C1D")
ax1.set_ylim(0, max(df["violation_rate"] * 100) * 1.3)
for b, v in zip(bars, df["violation_rate"] * 100):
    ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(x, df["neg_mass_per_capita"], color="#185FA5", marker="o", linewidth=2, label="음수질량 인당평균")
ax2.axhline(0.01, color="#0C447C", linestyle="--", linewidth=1.2, label="사전기준 0.01")
ax2.set_ylabel("음수질량 (인당평균)", fontsize=11, color="#185FA5")
ax2.set_ylim(0, 0.012)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, frameon=False)

ax1.set_title("held-out 시험별 비단조 확률 발생률 vs 위반 크기(음수질량)\n발생은 잦지만(막대) 크기는 작음(선) — 대부분 근소한 역전", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_monotonic_violation.png", dpi=150, bbox_inches="tight")
print(f"저장: {OUT_DIR / 'fig3_monotonic_violation.png'}")
