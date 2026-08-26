"""
CPAD EDA Script
---------------
각 도메인 CSV를 읽어 연구에 필요한 변수 가용성을 빠르게 파악합니다.
실행 전: DATA_DIR을 CPAD CSV 파일들이 있는 폴더 경로로 바꿔주세요.
"""

import os
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────
# 설정: CSV 파일 경로
# ───────────────────────────────────────────
DATA_DIR = "/Users/siyeon/Documents/urp AD/1_fullExportDb-2011-Member-CSV"   # ← 여기를 실제 경로로 변경

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


# ───────────────────────────────────────────
# 0. 도메인 파일 목록 확인
# ───────────────────────────────────────────
def load_domains(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    domains = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].upper()
        try:
            df = pd.read_csv(f, low_memory=False)
            domains[name] = df
        except Exception as e:
            print(f"[로드 실패] {name}: {e}")
    return domains


# ───────────────────────────────────────────
# 1. 전체 인벤토리
# ───────────────────────────────────────────
def inventory(domains):
    print("\n" + "="*60)
    print("1. 도메인별 기본 현황")
    print("="*60)

    rows = []
    for name, df in sorted(domains.items()):
        n_rows = len(df)
        n_subj = df["USUBJID"].nunique() if "USUBJID" in df.columns else None
        n_study = df["STUDYID"].nunique() if "STUDYID" in df.columns else None
        miss_pct = round(df.isnull().mean().mean() * 100, 1)
        rows.append({
            "도메인": name,
            "행 수": n_rows,
            "환자 수": n_subj,
            "시험 수": n_study,
            "전체 결측률(%)": miss_pct
        })

    summary = pd.DataFrame(rows).set_index("도메인")
    print(summary.to_string())


# ───────────────────────────────────────────
# 2. 연구 핵심 도메인 — 환자별 교집합
# ───────────────────────────────────────────
def subject_overlap(domains):
    print("\n" + "="*60)
    print("2. 핵심 도메인 환자 교집합 (AT(N) 구성 가능 여부)")
    print("="*60)

    key_domains = ["SS", "LB", "NV", "QS", "PF", "DM"]
    sets = {}
    for d in key_domains:
        if d in domains and "USUBJID" in domains[d].columns:
            sets[d] = set(domains[d]["USUBJID"].unique())
            print(f"  {d}: {len(sets[d]):,} 명")

    print()
    # SS + LB 동시 보유 → A/T 그룹 구성 가능
    if "SS" in sets and "LB" in sets:
        ss_lb = sets["SS"] & sets["LB"]
        print(f"  SS ∩ LB (아밀로이드+tau 동시):  {len(ss_lb):,} 명")

    if "SS" in sets and "LB" in sets and "NV" in sets:
        ss_lb_nv = sets["SS"] & sets["LB"] & sets["NV"]
        print(f"  SS ∩ LB ∩ NV (AT+영상 모두):    {len(ss_lb_nv):,} 명")

    if "QS" in sets:
        base = sets.get("SS", set()) | sets.get("LB", set())
        if base:
            with_outcome = (sets["SS"] & sets["LB"]) & sets["QS"] if ("SS" in sets and "LB" in sets) else set()
            print(f"  SS ∩ LB ∩ QS (분석 가능 핵심): {len(with_outcome):,} 명")

    if "PF" in sets:
        print(f"  PF (APOE4 보유):                {len(sets['PF']):,} 명")


# ───────────────────────────────────────────
# 3. QS 도메인 — 인지검사 종류 및 시점
# ───────────────────────────────────────────
def qs_summary(domains):
    if "QS" not in domains:
        print("\n[QS 없음]")
        return

    print("\n" + "="*60)
    print("3. QS 도메인 — 인지검사 종류 및 종단 구조")
    print("="*60)

    qs = domains["QS"]
    testcd_col = next((c for c in ["QSTESTCD", "VSTESTCD"] if c in qs.columns), None)
    if testcd_col:
        counts = qs[testcd_col].value_counts()
        print("\n  검사 항목 (상위 20개):")
        print(counts.head(20).to_string())

    # 방문 수 분포
    if "USUBJID" in qs.columns and "VISITNUM" in qs.columns:
        visits_per_subj = qs.groupby("USUBJID")["VISITNUM"].nunique()
        print(f"\n  환자당 방문 수 분포:")
        print(f"    평균: {visits_per_subj.mean():.1f}  "
              f"중앙값: {visits_per_subj.median():.0f}  "
              f"최대: {visits_per_subj.max():.0f}")

    # ADAS-Cog 총점 확인
    if testcd_col:
        adas_keys = [k for k in qs[testcd_col].unique()
                     if isinstance(k, str) and "ADAS" in k.upper() and "TTL" in k.upper()]
        mmse_keys = [k for k in qs[testcd_col].unique()
                     if isinstance(k, str) and "MMSE" in k.upper()]
        print(f"\n  ADAS-Cog 총점 코드: {adas_keys}")
        print(f"  MMSE 코드:         {mmse_keys}")


# ───────────────────────────────────────────
# 4. LB 도메인 — 바이오마커 항목 확인
# ───────────────────────────────────────────
def lb_summary(domains):
    if "LB" not in domains:
        print("\n[LB 없음]")
        return

    print("\n" + "="*60)
    print("4. LB 도메인 — 바이오마커 항목")
    print("="*60)

    lb = domains["LB"]
    testcd_col = "LBTESTCD" if "LBTESTCD" in lb.columns else None
    if testcd_col:
        counts = lb[testcd_col].value_counts()
        print("\n  바이오마커 항목 (상위 30개):")
        print(counts.head(30).to_string())

        # 타우/아밀로이드 관련 항목 필터
        tau_keys = [k for k in lb[testcd_col].unique()
                    if isinstance(k, str) and any(t in k.upper() for t in ["TAU", "PTAU"])]
        ab_keys = [k for k in lb[testcd_col].unique()
                   if isinstance(k, str) and any(t in k.upper() for t in ["AB", "AMYL", "A42", "A40"])]
        print(f"\n  타우 관련: {tau_keys}")
        print(f"  아밀로이드 관련: {ab_keys}")

    # 시험별 바이오마커 커버리지
    if "STUDYID" in lb.columns and testcd_col:
        print("\n  시험별 타우 측정 여부:")
        for study, grp in lb.groupby("STUDYID"):
            items = grp[testcd_col].unique()
            has_tau = any(isinstance(i, str) and "TAU" in i.upper() for i in items)
            if has_tau:
                n = grp["USUBJID"].nunique()
                tau_items = [i for i in items if isinstance(i, str) and "TAU" in i.upper()]
                print(f"    {study}: {n}명  항목={tau_items}")


# ───────────────────────────────────────────
# 5. SS 도메인 — 아밀로이드 양성/음성 분포
# ───────────────────────────────────────────
def ss_summary(domains):
    if "SS" not in domains:
        print("\n[SS 없음]")
        return

    print("\n" + "="*60)
    print("5. SS 도메인 — 아밀로이드 양성/음성 분포")
    print("="*60)

    ss = domains["SS"]
    result_col = next((c for c in ["SSORRES", "SSSTRESC"] if c in ss.columns), None)
    testcd_col = "SSTESTCD" if "SSTESTCD" in ss.columns else None

    if testcd_col:
        print(f"\n  SS 항목: {ss[testcd_col].value_counts().to_dict()}")

    if result_col:
        print(f"\n  아밀로이드 양성/음성 분포:")
        print(ss[result_col].value_counts().to_string())


# ───────────────────────────────────────────
# 6. PF 도메인 — APOE4 분포
# ───────────────────────────────────────────
def pf_summary(domains):
    if "PF" not in domains:
        print("\n[PF 없음]")
        return

    print("\n" + "="*60)
    print("6. PF 도메인 — APOE4 분포")
    print("="*60)

    pf = domains["PF"]
    result_col = next((c for c in ["PFORRES", "PFSTRESC"] if c in pf.columns), None)
    testcd_col = "PFTESTCD" if "PFTESTCD" in pf.columns else None

    if testcd_col:
        print(f"\n  PF 항목: {pf[testcd_col].value_counts().to_dict()}")

    if result_col:
        print(f"\n  유전형 분포:")
        print(pf[result_col].value_counts().to_string())

        # APOE4 보유 여부 (e4/1 또는 e4/4)
        apoe4 = pf[result_col].astype(str).str.contains("4", na=False)
        n_apoe4 = apoe4.sum()
        total = len(pf)
        print(f"\n  APOE4 보유(ε4 포함): {n_apoe4}/{total} ({n_apoe4/total*100:.1f}%)")


# ───────────────────────────────────────────
# 7. MH/CM 도메인 — CVRFs 변수 확인
# ───────────────────────────────────────────
def cvrf_summary(domains):
    print("\n" + "="*60)
    print("7. CVRFs 관련 도메인 (MH, CM, VS)")
    print("="*60)

    cvrf_terms = ["HYPERTENSION", "DIABETES", "HYPERLIPID", "CARDIAC",
                  "CORONARY", "STROKE", "OBESITY", "SMOKING"]

    # MH
    if "MH" in domains:
        mh = domains["MH"]
        term_col = next((c for c in ["MHDECOD", "MHTERM", "MHMODIFY"] if c in mh.columns), None)
        if term_col:
            mh_upper = mh[term_col].astype(str).str.upper()
            print("\n  MH — CVRF 관련 기록 수:")
            for term in cvrf_terms:
                n = mh_upper.str.contains(term, na=False).sum()
                if n > 0:
                    print(f"    {term}: {n}건")

    # CM — 고혈압약, 스타틴
    if "CM" in domains:
        cm = domains["CM"]
        drug_col = next((c for c in ["CMDECOD", "CMTRT", "CMMODIFY"] if c in cm.columns), None)
        if drug_col:
            cm_upper = cm[drug_col].astype(str).str.upper()
            drug_keys = ["STATIN", "ANTIHYPERTENS", "METFORMIN",
                         "INSULIN", "AMLODIPINE", "ATORVASTATIN"]
            print("\n  CM — CVRFs 관련 약물 기록 수:")
            for d in drug_keys:
                n = cm_upper.str.contains(d, na=False).sum()
                if n > 0:
                    print(f"    {d}: {n}건")

    # VS — 혈압
    if "VS" in domains:
        vs = domains["VS"]
        testcd_col = "VSTESTCD" if "VSTESTCD" in vs.columns else None
        if testcd_col:
            bp_items = [k for k in vs[testcd_col].unique()
                        if isinstance(k, str) and any(b in k.upper() for b in ["SYSBP", "DIABP", "PULSE"])]
            print(f"\n  VS — 혈압/맥박 항목: {bp_items}")
            if bp_items:
                n_vs = vs[vs[testcd_col].isin(bp_items)]["USUBJID"].nunique()
                print(f"  혈압 측정 환자 수: {n_vs:,} 명")


# ───────────────────────────────────────────
# 8. 종단 구조 요약 (SV 또는 QS 기반)
# ───────────────────────────────────────────
def longitudinal_summary(domains):
    print("\n" + "="*60)
    print("8. 종단 추적 구조 (SV 기반)")
    print("="*60)

    if "SV" not in domains:
        print("  [SV 없음 — QS로 대체 확인 필요]")
        return

    sv = domains["SV"]
    if "USUBJID" in sv.columns and "VISITDY" in sv.columns:
        sv["VISITDY"] = pd.to_numeric(sv["VISITDY"], errors="coerce")
        follow_up = sv.groupby("USUBJID")["VISITDY"].max()
        print(f"\n  추적 기간(일, VISITDY 최대값):")
        print(f"    평균: {follow_up.mean():.0f}일  "
              f"중앙값: {follow_up.median():.0f}일  "
              f"최대: {follow_up.max():.0f}일")
        # 주 단위
        print(f"    (주 환산: 평균 {follow_up.mean()/7:.0f}주, "
              f"중앙값 {follow_up.median()/7:.0f}주)")

    if "DS" in domains:
        ds = domains["DS"]
        decod_col = "DSDECOD" if "DSDECOD" in ds.columns else None
        if decod_col:
            print(f"\n  DS — 탈락/완료 분포:")
            print(ds[decod_col].value_counts().head(10).to_string())


# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────
if __name__ == "__main__":
    print("CPAD EDA 시작...")
    domains = load_domains(DATA_DIR)
    print(f"로드된 도메인: {sorted(domains.keys())}")

    inventory(domains)
    subject_overlap(domains)
    qs_summary(domains)
    lb_summary(domains)
    ss_summary(domains)
    pf_summary(domains)
    cvrf_summary(domains)
    longitudinal_summary(domains)

    print("\n" + "="*60)
    print("EDA 완료")
    print("="*60)