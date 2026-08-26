"""
Aim 2 · ⑤ 구조분석 — PCA · EFA · 군집 (탐색적) (sjlee)
======================================================
위상: 확증(confirmatory)이 아니라 **탐색·기술**. 주결론(A1/DS총점/Gn)을 바꾸지 않음.
목적: 21문항의 잠재구조(bADL/iADL 등)와 환자 하위집단을 탐색해 B2 선택문항 해석을 뒷받침.

- 5-1 PCA: 표준화 문항행렬 주성분 — 설명분산·상위 적재.
- 5-2 EFA: (가능시 polychoric) 탐색적 인자분석 — 평행분석으로 요인수, 회전 적재.
- 5-3 군집: 환자 문항프로파일 k-means — 실루엣으로 k, 군집별 DS단계 프로파일.
모든 표에 '탐색적' 라벨. 구조가 약하면 '뚜렷한 하위구조 없음'도 그대로 보고.

산출: aim2_structure_report.md, aim2_sjlee/structure_pca_loadings.csv,
      structure_factor_loadings.csv, structure_cluster_assign.csv(환자단위→Drive)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import b1_gn_elasticnet as G

warnings.filterwarnings("ignore")
OUTDIR = Path(__file__).parent
CSV = OUTDIR / "aim2_sjlee"; CSV.mkdir(exist_ok=True)
REPORT = OUTDIR / "aim2_structure_report.md"
ITEMS = G.ITEMS
LAB = {"ADL0101":"Q1먹기","ADL0102":"Q2걷기","ADL0103":"Q3화장실","ADL0104":"Q4목욕","ADL0105":"Q5몸단장",
 "ADL0106B":"Q6b입기","ADL0106A":"Q6a고르기","ADL0107A":"Q7전화","ADL0110A":"Q10설거지","ADL0111A":"Q11식사준비",
 "ADL0112A":"Q12집안일","ADL0113A":"Q13빨래","ADL0114A":"Q14가전","ADL0115A":"Q15외출","ADL0116A":"Q16a쇼핑",
 "ADL0116B":"Q16b지불","ADL0117A":"Q17금전","Q18":"Q18혼자","ADL0121A":"Q21글쓰기","ADL0122Q":"Q22취미","ADL0123L":"Q23가전사용"}
BADL = set(["ADL0101","ADL0102","ADL0103","ADL0104","ADL0105","ADL0106B"])
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)


def parallel_analysis(Z, n_iter=50, seed=0):
    """무작위 데이터 고유값과 비교해 유지할 요인/성분 수 판단."""
    rng = np.random.default_rng(seed); n, p = Z.shape
    real = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    rand = np.zeros((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        rand[i] = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))[::-1]
    thr = rand.mean(0)
    keep = int((real > thr).sum())
    return real, thr, max(keep, 1)


def main():
    log("# Aim 2 · ⑤ 구조분석 (PCA·EFA·군집) — 탐색적 (sjlee)\n")
    log("> **위상: 탐색·기술 분석**(확증 아님). 주결론 불변. B2 선택 해석·하위집단 보조.\n")
    m = G.build_matrix().reset_index(drop=True)
    imp = SimpleImputer(strategy="median").fit(m[ITEMS])
    Z = StandardScaler().fit_transform(imp.transform(m[ITEMS]))
    log(f"표본 {len(m)} × 21문항. 표준화 완료. 순서형이라 상관은 피어슨 근사(polychoric 미사용 명시).\n")

    # 5-1 PCA -------------------------------------------------------------------
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
    npc = int((pca.explained_variance_ > 1).sum())     # Kaiser
    log("## 5-1 PCA (탐색)")
    log(f"- 고유값>1 성분수(Kaiser): **{npc}**. 상위 5성분 설명분산: "
        + ", ".join(f"PC{i+1} {evr[i]*100:.1f}%" for i in range(5)) + f" (누적 {cum[4]*100:.1f}%).")
    load = pca.components_[:3].T
    dfL = pd.DataFrame(load, columns=["PC1", "PC2", "PC3"], index=[LAB[i] for i in ITEMS])
    dfL.insert(0, "type", ["bADL" if i in BADL else "iADL" for i in ITEMS])
    dfL.round(3).to_csv(CSV / "structure_pca_loadings.csv", encoding="utf-8-sig")
    log("- PC1 상위 적재(절대값): " + ", ".join(
        f"{dfL.index[j]}({load[j,0]:+.2f})" for j in np.argsort(-np.abs(load[:, 0]))[:6]))
    log(f"- PC1은 전반 손상축으로 해석(대부분 동부호). PC2가 bADL↔iADL 분리를 보이는지 적재표 참조.")
    log(f"- 적재표 CSV: structure_pca_loadings.csv\n")

    # 5-2 EFA -------------------------------------------------------------------
    log("## 5-2 EFA 탐색적 인자분석 (탐색)")
    real, thr, nfac = parallel_analysis(Z)
    log(f"- 평행분석: 유지 요인수 **{nfac}** (실데이터 고유값 > 무작위 평균인 개수).")
    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=max(nfac, 2), rotation="oblimin"); fa.fit(Z)
        L = fa.loadings_; method = "factor_analyzer oblimin"
    except Exception:
        fa = FactorAnalysis(n_components=max(nfac, 2), rotation="varimax").fit(Z)
        L = fa.components_.T; method = "sklearn FactorAnalysis varimax(폴백)"
    ncol = L.shape[1]
    dfF = pd.DataFrame(L, columns=[f"F{i+1}" for i in range(ncol)], index=[LAB[i] for i in ITEMS])
    dfF.insert(0, "type", ["bADL" if i in BADL else "iADL" for i in ITEMS])
    dfF.round(3).to_csv(CSV / "structure_factor_loadings.csv", encoding="utf-8-sig")
    log(f"- 방법: {method}, 요인 {ncol}개.")
    for f in range(min(ncol, 3)):
        top = np.argsort(-np.abs(L[:, f]))[:6]
        log(f"  · F{f+1} 상위: " + ", ".join(f"{LAB[ITEMS[j]]}({L[j,f]:+.2f})" for j in top))
    log("- 요인이 bADL(기초)·iADL(수단)으로 갈리는지 위 적재로 해석. 적재표 CSV: structure_factor_loadings.csv\n")

    # 5-3 군집 ------------------------------------------------------------------
    log("## 5-3 군집분석 (환자 하위집단, 탐색)")
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        sil[k] = silhouette_score(Z, km.labels_)
    kbest = max(sil, key=sil.get)
    log(f"- 실루엣: " + ", ".join(f"k={k}:{sil[k]:.3f}" for k in sil) + f" → 최적 **k={kbest}**.")
    km = KMeans(n_clusters=kbest, n_init=10, random_state=0).fit(Z)
    m2 = m.assign(cluster=km.labels_)
    m2[["STUDYID", "USUBJID", "cluster"]].to_csv(CSV / "structure_cluster_assign.csv", index=False, encoding="utf-8-sig")
    log(f"- 군집별 프로파일(평균 DS단계·인원):")
    log("| 군집 | n | 평균 DS단계 | DS단계 분포 |")
    log("|---|---|---|---|")
    for c in range(kbest):
        g = m2[m2.cluster == c]
        dist = g["ds_stage"].astype(int).value_counts().sort_index().to_dict()
        log(f"| {c} | {len(g)} | {g['ds_stage'].mean():.2f} | {dist} |")
    if max(sil.values()) < 0.15:
        log("\n- **주의**: 실루엣 낮음 → 뚜렷한 하위구조 약함(연속적 중증도 스펙트럼일 가능성). 그대로 보고.")
    log("- 군집배정 CSV: structure_cluster_assign.csv (환자단위→Drive)\n")

    # 5-4 종합 ------------------------------------------------------------------
    log("## 5-4 종합 해석 (탐색적)")
    log(f"- PCA 성분수 {npc} / EFA 요인수 {nfac} / 군집 k {kbest}. 세 분석이 같은 그림(bADL/iADL 또는 "
        "단일 중증도축)을 그리는지 위 적재·프로파일로 판단.")
    log("- B2 선택문항(외출·목욕·화장실·몸단장 등)이 주요 성분/요인을 고르게 대표하면 축약의 구조적 타당성 뒷받침.")
    log("- **탐색적 한계**: 순서형 피어슨 근사, 표본 3시험, 비지도라 판정 아님.\n")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")

if __name__ == "__main__":
    main()
