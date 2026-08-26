"""
Aim 2 · B1 최종결과 CSV 내보내기 (Drive 업로드용, 한글 안깨짐) (sjlee)
=====================================================================
b1_export_xlsx.sheets()의 6개 표를 각각 CSV로 저장. 인코딩 = utf-8-sig(BOM)
→ Excel에서 한글이 깨지지 않음.

산출: aim2/aim2_sjlee/csv/*.csv  (→ Drive AIM 2/)
"""
from pathlib import Path
from b1_export_xlsx import sheets

OUT = Path(__file__).parent / "aim2_sjlee" / "csv"; OUT.mkdir(parents=True, exist_ok=True)
FNAME = {"요약": "B1_01_요약", "추가가치_4조건": "B1_02_추가가치_4조건",
         "성능지표": "B1_03_성능지표", "B1_계수_pvalue": "B1_04_계수_pvalue",
         "Gn_절단별계수": "B1_05_Gn_절단별계수", "한계_주의": "B1_06_한계_주의"}

if __name__ == "__main__":
    for name, df in sheets().items():
        path = OUT / (FNAME[name] + ".csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")   # BOM → 엑셀서 한글 정상
        print(f">>> {path.name}  ({len(df)}행)")
    print(f">>> 폴더: {OUT}")
