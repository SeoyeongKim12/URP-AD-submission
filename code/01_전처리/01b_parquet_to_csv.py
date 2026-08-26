"""
parquet → csv 변환 (01_전처리/01_preprocess_dependence_study.py 출력 → 이후 단계 입력용)
================================================================
01_preprocess_dependence_study.py는 산출물을 parquet로만 저장하는데,
02_verify_independent.py 이후 대부분의 스크립트(02_aim1/, 03_aim2/최종/ 등)는
~/Downloads/ 에서 같은 이름의 csv를 찾습니다. 이 스크립트가 그 변환을 담당합니다.

입력: ~/Downloads/preprocessed/*.parquet (01_preprocess_dependence_study.py의 OUT_DIR)
출력: ~/Downloads/*.csv (다운스트림 스크립트들이 읽는 위치)

주의: adl_wide.parquet, baseline_sample.parquet는 csv뿐 아니라 원본 parquet도
      그대로 필요합니다(02_gn_stability_monotonic_check.py, 03_gn_isotonic_correction.py가
      parquet를 직접 읽음) — 이 스크립트는 parquet를 지우지 않고 csv만 추가로 만듭니다.
"""
from pathlib import Path
import pandas as pd

SRC_DIR = Path.home() / "Downloads" / "preprocessed"   # 01_preprocess_dependence_study.py의 OUT_DIR
DST_DIR = Path.home() / "Downloads"                     # 대부분의 스크립트가 csv를 찾는 위치

# 다운스트림에서 csv로 필요한 6개 (dm_filtered, qs_subset_long은 csv로 안 씀 — 필요시 추가)
FILES = [
    "ds_wide",
    "adl_wide",
    "baseline_sample",
    "mmse_wide",
    "supervision_time",
    "dm_filtered",
]

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(
            f"{SRC_DIR} 없음 — 먼저 01_preprocess_dependence_study.py를 실행해서 "
            "parquet 산출물을 만들어주세요."
        )

    for name in FILES:
        src = SRC_DIR / f"{name}.parquet"
        dst = DST_DIR / f"{name}.csv"
        if not src.exists():
            print(f"[건너뜀] {src} 없음")
            continue
        df = pd.read_parquet(src)
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"[완료] {src.name} → {dst}  ({len(df)}행)")

    print("\n변환 완료. 이제 02_aim1/, 03_aim2/최종/ 등을 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
