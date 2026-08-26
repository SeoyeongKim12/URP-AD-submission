"""
B2 랜덤서치 결과에서 "상한선(ceiling)에 가까운" 문항조합만 뽑아내는 후처리 스크립트.
전체 파이프라인을 다시 안 돌리고, 이미 저장된 b2_performance_by_k_*.csv /
b2_selected_items_by_fold_*.csv / b2_common_items_by_k_*.csv만 읽어서 계산함.

정의: "상한선"은 merged/unmerged 두 변형·모든 k를 통틀어 가장 낮은(=제일 좋은) MAE로
잡음. 그 값 + MARGIN(기본 0.02단계) 이내인 (variant, k) 조합을 "근접-상한 후보"로
간주하고, 그 k들에서 실제로 fold별 랜덤서치가 뽑은 문항조합 전체를 가져옴.

산출:
  b2_near_ceiling_summary.csv   근접-상한 (variant, k)별 요약(MAE·카파·중증놓침 등)
  b2_near_ceiling_combos.csv    위 (variant, k)에 해당하는 fold별 실제 문항조합 전체
  b2_near_ceiling_item_votes.csv 근접-상한 조합들에서 각 문항이 몇 번 등장했는지 집계
                                 (자주 등장 = 여러 조합에서 반복 채택된 핵심문항 후보)
"""
from pathlib import Path
import pandas as pd

OUTDIR = Path(__file__).parent
VARIANTS = ["merged", "unmerged"]
MARGIN = 0.02   # 상한 MAE + 이 값 이내면 "근접-상한"으로 인정


def main():
    perf_all, sel_all, common_all = [], [], []
    for v in VARIANTS:
        perf_all.append(pd.read_csv(OUTDIR / f"b2_performance_by_k_{v}.csv"))
        sel_all.append(pd.read_csv(OUTDIR / f"b2_selected_items_by_fold_{v}.csv"))
        common_all.append(pd.read_csv(OUTDIR / f"b2_common_items_by_k_{v}.csv"))
    perf = pd.concat(perf_all, ignore_index=True)
    sel = pd.concat(sel_all, ignore_index=True)
    common = pd.concat(common_all, ignore_index=True)

    best_mae = perf["mae"].min()
    threshold = best_mae + MARGIN
    print(f"전체 최저 MAE(상한): {best_mae:.4f}  →  근접-상한 기준: MAE <= {threshold:.4f} (margin={MARGIN})")

    near = perf[perf["mae"] <= threshold].sort_values("mae").reset_index(drop=True)
    print(f"\n근접-상한 (variant, k) 조합 수: {len(near)}")
    print(near[["variant", "n_items", "mae", "kappa", "hi_missed_pct", "champion"]].to_string(index=False))
    near.to_csv(OUTDIR / "b2_near_ceiling_summary.csv", index=False, encoding="utf-8-sig")

    # 근접-상한 (variant, k)에 해당하는 fold별 실제 문항조합 전체를 붙임.
    # b2_selected_items_by_fold_*.csv에는 variant 컬럼이 없어(파일이 이미 변형별로
    # 나뉘어 저장됨) 아래처럼 파일 유래를 태그로 붙여서 안전하게 매칭한다.
    sel_tagged = []
    for v, df in zip(VARIANTS, sel_all):
        df = df.copy(); df["variant"] = v
        sel_tagged.append(df)
    sel_tagged = pd.concat(sel_tagged, ignore_index=True)
    near_key = near[["variant", "n_items", "mae", "kappa", "champion"]].rename(columns={"n_items": "k"})
    combos = sel_tagged.merge(near_key, on=["variant", "k"], how="inner")
    combos = combos.sort_values(["mae", "variant", "k", "outer_fold"])
    combos.to_csv(OUTDIR / "b2_near_ceiling_combos.csv", index=False, encoding="utf-8-sig")
    print(f"\n근접-상한 fold별 실제 문항조합 {len(combos)}행 저장.")

    # 문항 등장빈도 집계 — 여러 근접-상한 조합에서 자주 뽑히는 문항 = 핵심후보
    votes = {}
    total_combos = len(combos)
    for _, row in combos.iterrows():
        for item in str(row["items_label"]).split(","):
            item = item.strip()
            votes[item] = votes.get(item, 0) + 1
    votes_df = pd.DataFrame(sorted(votes.items(), key=lambda x: -x[1]), columns=["item_label", "n_appearances"])
    votes_df["pct_of_near_ceiling_combos"] = (votes_df["n_appearances"] / total_combos * 100).round(1)
    votes_df.to_csv(OUTDIR / "b2_near_ceiling_item_votes.csv", index=False, encoding="utf-8-sig")

    print(f"\n문항별 근접-상한 조합 등장빈도 top15 (총 {total_combos}개 조합 중):")
    print(votes_df.head(15).to_string(index=False))
    print(f"\n>>> 저장: {OUTDIR / 'b2_near_ceiling_summary.csv'}")
    print(f">>> 저장: {OUTDIR / 'b2_near_ceiling_combos.csv'}")
    print(f">>> 저장: {OUTDIR / 'b2_near_ceiling_item_votes.csv'}")


if __name__ == "__main__":
    main()
