"""
의존도(Dependence Scale) 연구계획서 전처리 파이프라인
대상: CPAD AD-1061 / AD-1063 / AD-1064 (경도~중등도 AD, 24주 3상)

원자료: 1_fullExportDb-2011-Member-CSV/dm.csv, qs.csv
사용 도메인: DM(인구학·STUDYID/ARM), QS 중 QSCAT in
    ('DEPENDENCE SCALE', 'ADCS-ADL', 'MMSE', 'RUDL')
CM/AE/EX 등 다른 도메인은 이 연구에서 사용하지 않음 (다른 서브 프로젝트용).

중요 QC 주의사항 (실제 실행으로 확인됨)
--------------------------------------
1. qs.csv를 DuckDB 기본(parallel) CSV 리더로 읽으면 일부 구간에서 quote 상태가
   깨져 VISITNUM 등이 NULL로 빠지는 유령 행이 생성됨(직접 확인됨, raw 파일 자체는
   정상 RFC4180 quoting). DuckDB를 쓸 경우 반드시 parallel=false 로 읽을 것.
2. QSORRES는 인코딩 손상이 있으므로 사용하지 않고 QSSTRESC/QSSTRESN만 사용
   (연구계획서 2.2 변수 식별 원칙과 동일).
3. QSTESTCD 기준으로만 문항을 선택하고 QSTEST 텍스트 매칭은 쓰지 않음.
4. QSBLFL은 이 3개 시험에서 전량 결측이므로 기저 시점 식별은 VISITNUM==2.0
   (Visit 2 Baseline)으로 함. 12주=VISITNUM 5.0, 24주=VISITNUM 7.0,
   ADCS-ADL 4주=VISITNUM 3.0 (DS는 4주 미수집).
5. qs.csv(2GB, 1,245만행)를 pandas로 통째로 스캔하면 네트워크/동기화 마운트
   쓰기 속도 때문에 매우 느려질 수 있음. STUDYID로 먼저 grep 등으로 물리적
   줄 필터링을 해 파일 크기를 줄인 뒤(각 논리 행이 물리적 한 줄과 일치함을
   확인함) pandas로 읽는 2단계 방식이 훨씬 빠름. 이 스크립트도 그 방식을 씀.

검증 결과 (2026-08-04 실행)
---------------------------
기저 DS13 완전관측 표본이 계획서 표 2.1과 완전히 일치함:
  AD-1061 668명, AD-1063 835명, AD-1064 705명 (합 2,208명)
  단계 분포(0~5단계)까지 자릿수 단위로 정확히 일치.
RUD 감독시간의 비정합 응답 수(계획서 282건 vs 실측 242건)와 기저 무공급
비율(계획서 54.5% vs 실측 57.5%)은 근사치이며, 정확 일치를 위해서는 계획서가
말하는 "감독 시간 또는 일수 중 하나만 결측인 경우"의 처리 규칙을 더 세밀하게
맞춰야 함 (현재는 두 값 모두 관측된 행만 비교).

모든 처리 함수는 STUDYID/ARM을 결과 테이블에 유지해, 이후 분석 단계에서
STUDYID/ARM 통제가 항상 가능하도록 함.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------

RAW_DIR = Path.home() / "Downloads" / "1_fullExportDb-2011-Member-CSV"  # 원자료(dm.csv, qs.csv) 위치
OUT_DIR = Path.home() / "Downloads" / "preprocessed"  # parquet 산출 위치 (TRAIN_DIR과 동일해야 함)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 중간 산출물은 로컬(빠른) 경로에 두고, 최종 parquet만 OUT_DIR(마운트)에 씀
SCRATCH_DIR = Path.home() / "Downloads" / "scratch"  # 임시 작업 폴더
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STUDIES = ["AD-1061", "AD-1063", "AD-1064"]

# 공개연장시험(AD-1062) — 계획서 v2 2.1: "최종 모형의 추가 성능평가에만 사용"
# (참가자 중복 가능성 때문에 학습/튜닝/교차검증에는 안 씀, 독립 외부검증으로도
# 격상하지 않음). 그래서 메인 3개 시험과 분리된 별도 파일로만 산출함.
EXTRA_STUDIES = ["AD-1062"]

TARGET_QSCAT = ["DEPENDENCE SCALE", "ADCS-ADL", "MMSE", "RUDL"]

VISIT_LABELS = {
    2.0: "baseline",
    3.0: "week4",
    5.0: "week12",
    7.0: "week24",
}

DS_ITEMS = [f"DS{str(i).zfill(2)}" for i in range(1, 14)]  # DS01..DS13
DS14 = "DS14"  # Equivalent Institutional Care, 부록용

# 2025 개정판에서 제외하는 12개(관문+하위) 문항 코드 (연구계획서 2.2)
# TV(Q8)/대화참여(Q9)/시사대화(Q19)/5분독서(Q20) 관문 + 하위코드
ADL_2025_EXCLUDE = [
    "ADL0108", "ADL0108A", "ADL0108B", "ADL0108C",
    "ADL0109", "ADL0109A",
    "ADL0119", "ADL0119A", "ADL0119B", "ADL0119C",
    "ADL0120", "ADL0120A", "ADL0120B",
]
ADL_NOT_A_PERFORMANCE_ITEM = [
    "ADL0118N",  # 시설입소 여부, 수행 문항 아님
    "ADL0124",   # ADL01-Total ADCS-ADL Score, 원자료가 이미 계산해둔 합계(파생값)
    "ADL0125",   # ADL01-Number Don't Know Responses, 집계값
]


# ------------------------------------------------------------------
# 1. DM 추출 (STUDYID/ARM/인구학)
# ------------------------------------------------------------------

def extract_dm(studies=None) -> pd.DataFrame:
    studies = studies or TARGET_STUDIES
    dm = pd.read_csv(
        RAW_DIR / "dm.csv",
        dtype=str,
        usecols=["STUDYID", "USUBJID", "SUBJID", "AGE", "AGEU", "SEX",
                 "RACE", "ARM", "ACTARMCD", "ACTARM", "COUNTRY"],
    )
    dm = dm[dm["STUDYID"].isin(studies)].copy()
    dm["AGE"] = pd.to_numeric(dm["AGE"], errors="coerce")
    dm = dm.drop_duplicates(subset=["USUBJID"])
    return dm


# ------------------------------------------------------------------
# 2. QS 서브셋 추출 (chunked, 메모리 안전)
# ------------------------------------------------------------------

def extract_qs_subset(studies=None, tag="main") -> pd.DataFrame:
    """qs.csv(2GB)는 STUDYID가 첫 컬럼이고 각 논리 행이 물리적 한 줄과
    일치함을 확인했으므로(임베디드 개행 없음), 먼저 대상 시험의 행만
    물리적으로 걸러낸 뒤 pandas로 읽는다. 이 방식이 pandas 단독 청크
    스캔보다 훨씬 빠르고, 결과를 마운트된 OUT_DIR이 아니라 로컬
    SCRATCH_DIR에 써서 I/O 병목도 피한다.

    원래 head/grep 외부명령을 썼으나 윈도우 네이티브 Python엔 그 실행파일이
    없어(WinError 2) 순수 Python 줄 필터링으로 대체함(OS 무관하게 동작).
    """
    studies = studies or TARGET_STUDIES
    raw_path = RAW_DIR / "qs.csv"
    filtered_path = SCRATCH_DIR / f"qs_subset_raw_{tag}.csv"

    pattern = re.compile("^(" + "|".join(studies) + "),")

    n_matched = 0
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(filtered_path, "w", encoding="utf-8", newline="") as fout:
        header = fin.readline()
        fout.write(header)
        for i, line in enumerate(fin, start=1):
            if pattern.match(line):
                fout.write(line)
                n_matched += 1
            if i % 2_000_000 == 0:
                print(f"   ...{i:,}행 스캔, {n_matched:,}건 매칭")
    print(f"   qs.csv 필터링 완료: {n_matched:,}건 매칭 -> {filtered_path}")

    usecols = [
        "STUDYID", "USUBJID", "QSTESTCD", "QSCAT",
        "QSSTRESC", "QSSTRESN", "VISITNUM", "VISIT", "QSDY", "QSBLFL",
    ]
    qs = pd.read_csv(filtered_path, dtype=str, usecols=usecols)
    qs = qs[qs["QSCAT"].isin(TARGET_QSCAT)].copy()
    qs["QSSTRESN"] = pd.to_numeric(qs["QSSTRESN"], errors="coerce")
    qs["VISITNUM"] = pd.to_numeric(qs["VISITNUM"], errors="coerce")

    # QC: VISITNUM 결측률이 비정상적으로 높으면(파싱 손상 의심) 경고
    null_rate = qs["VISITNUM"].isna().mean()
    if null_rate > 0.02:
        print(f"[WARN] VISITNUM 결측률 {null_rate:.1%} — 파싱 손상 여부를 확인할 것")

    return qs


# ------------------------------------------------------------------
# 3. DS(Dependence Scale) 처리
# ------------------------------------------------------------------

def assign_ds_stage(row: pd.Series) -> float:
    """Brickman 2002 규칙. 13문항 중 하나라도 결측이면 NaN."""
    vals = row[DS_ITEMS]
    if vals.isna().any():
        return np.nan
    a, b, c, d, e, f, g, h, i, j, k, l, m = vals.values

    if k == 1 or l == 1 or m == 1:
        return 5
    if h == 1 or i == 1 or j == 1:
        return 4
    if e == 1 or f == 1 or g == 1:
        return 3
    if (sum([a >= 1, b >= 1, c >= 1]) >= 2) or (a == 2 or b == 2) or (d == 1):
        return 2
    if a >= 1 or b >= 1 or c >= 1:
        return 1
    return 0


def build_ds_wide(qs: pd.DataFrame, visit_filter=(2.0, 5.0, 7.0)) -> pd.DataFrame:
    """visit_filter=None이면 방문번호로 거르지 않고 전부 사용함(AD-1062처럼
    메인 3개 시험과 방문 체계 자체가 다른 경우 — Visit 1='Baseline II',
    4, 6/11=Completion 등 — 에 씀). 이 경우 visit_label은 매핑되지 않아
    NaN으로 남으므로, '기저' 정의는 사용하는 쪽에서 VISIT 텍스트를 보고
    직접 판단해야 함."""
    ds = qs[qs["QSCAT"] == "DEPENDENCE SCALE"].copy()
    if visit_filter is not None:
        ds = ds[ds["VISITNUM"].isin(visit_filter)]

    wide = ds.pivot_table(
        index=["STUDYID", "USUBJID", "VISITNUM"],
        columns="QSTESTCD",
        values="QSSTRESN",
        aggfunc="first",
    ).reset_index()

    for item in DS_ITEMS + [DS14]:
        if item not in wide.columns:
            wide[item] = np.nan

    wide["ds_total"] = wide[DS_ITEMS].sum(axis=1, min_count=13)
    wide["ds_complete13"] = wide[DS_ITEMS].notna().all(axis=1)
    wide["ds_stage"] = wide.apply(assign_ds_stage, axis=1)
    wide["visit_label"] = wide["VISITNUM"].map(VISIT_LABELS)

    return wide


# ------------------------------------------------------------------
# 4. ADCS-ADL 처리
# ------------------------------------------------------------------

def build_adl_wide(qs: pd.DataFrame, visit_filter=(2.0, 3.0, 5.0, 7.0)) -> pd.DataFrame:
    """관문/하위 문항을 wide로 펼치고, 관문 응답(Y/N/DK)에 따라
    하위문항 사용 여부를 결정하는 3분기 규칙까지만 적용한다.

    주의: '문항점수 파생'(관문+하위 응답을 하나의 0-3점 등 최종 문항점수로
    변환하는 원 ADCS-ADL 채점표, Galasko 1997 / Chandler 2025 Table S6-S8)은
    이 저장소에 원문 매핑표가 없어 여기서는 구현하지 않았다. A0(2025)/A1(2015)
    최종 점수 산출 전에 해당 매핑표를 확보해 STEP 4b로 추가해야 한다.
    """
    adl = qs[qs["QSCAT"] == "ADCS-ADL"].copy()
    if visit_filter is not None:
        adl = adl[adl["VISITNUM"].isin(visit_filter)]

    # QSSTRESC 기준 Y/N/Don't Know/빈칸 3분기 (계획서 2.2 표)
    def gate_status(x):
        if pd.isna(x):
            return "missing"
        xu = str(x).strip().upper()
        if xu == "Y":
            return "generated"
        if xu == "N":
            return "not_generated"
        if "DON" in xu or "DK" in xu:
            return "dont_know"
        return "other"  # 수치응답 등 (관문이 아닌 일반 문항)

    adl["gate_status"] = adl["QSSTRESC"].apply(gate_status)

    wide_val = adl.pivot_table(
        index=["STUDYID", "USUBJID", "VISITNUM"],
        columns="QSTESTCD",
        values="QSSTRESN",
        aggfunc="first",
    )
    wide_gate = adl.pivot_table(
        index=["STUDYID", "USUBJID", "VISITNUM"],
        columns="QSTESTCD",
        values="gate_status",
        aggfunc="first",
    ).add_suffix("__gate")

    wide = pd.concat([wide_val, wide_gate], axis=1).reset_index()
    wide["visit_label"] = wide["VISITNUM"].map(VISIT_LABELS)

    # 2025판 입력 후보 19문항 = 전체 관문코드 - 제외 12개 - ADL0118N
    all_gate_cols = [c for c in wide.columns if c.startswith("ADL01")
                      and not c.endswith("__gate")]
    input_2025 = [c for c in all_gate_cols
                  if c not in ADL_2025_EXCLUDE + ADL_NOT_A_PERFORMANCE_ITEM]
    wide.attrs["adl_2025_input_items"] = sorted(set(input_2025))

    wide = resolve_adl_gating(wide)
    return wide


# 관문(gate) 문항 -> 하위(subitem) 문항 코드 매핑 (원자료 구조에서 실측 확인)
ADL_GATE_MAP = {
    "ADL0106": ["ADL0106A", "ADL0106B"],
    "ADL0107": ["ADL0107A"],
    "ADL0108": ["ADL0108A", "ADL0108B", "ADL0108C"],
    "ADL0109": ["ADL0109A"],
    "ADL0110": ["ADL0110A"],
    "ADL0111": ["ADL0111A"],
    "ADL0112": ["ADL0112A"],
    "ADL0113": ["ADL0113A"],
    "ADL0114": ["ADL0114A"],
    "ADL0115": ["ADL0115A"],
    "ADL0116": ["ADL0116A", "ADL0116B"],
    "ADL0117": ["ADL0117A"],
    "ADL0118": ["ADL0118A", "ADL0118B", "ADL0118C"],
    "ADL0119": ["ADL0119A", "ADL0119B", "ADL0119C"],
    "ADL0120": ["ADL0120A", "ADL0120B"],
    "ADL0121": ["ADL0121A"],
    "ADL0122": ["ADL0122Q"],
    "ADL0123": ["ADL0123L"],
}


def resolve_adl_gating(wide: pd.DataFrame) -> pd.DataFrame:
    """계획서 2.2의 3분기 규칙을 적용해 하위문항 '진짜 값'을 만든다.

    Y(generated)   -> 하위 응답값 그대로 사용
    N(not_generated) -> 하위문항은 원자료에 아예 행이 없어 NaN으로 잡히므로 0점으로 채움
    DK/빈칸(missing)  -> 결측으로 유지(임의로 채우지 않음)

    각 하위문항에 대해 '{코드}__resolved' 컬럼을 추가한다. 이 컬럼이 이후
    A0/A1 채점, B1 모형 입력에 실제로 써야 하는 값이다. 원래의 '{코드}' 컬럼은
    원자료 그대로(관문 미통과로 인한 결측 포함) 남겨둬 QC용으로 유지한다.
    """
    for gate, subs in ADL_GATE_MAP.items():
        gate_col = gate + "__gate"
        if gate_col not in wide.columns:
            continue
        gate_status = wide[gate_col]
        for sub in subs:
            if sub not in wide.columns:
                continue
            resolved = wide[sub].where(gate_status != "not_generated", 0)
            wide[sub + "__resolved"] = resolved
    return wide


# ------------------------------------------------------------------
# 4b. A0(2025 개정판, 0-6단계) 산출 — Chandler 2025 보충자료 Table S6(SAS/R 코드)
#     그대로 이식. CPAD 코드 <-> 문헌 Q번호 대응은 QSORRES 응답 텍스트를
#     Table S7과 대조해 확인함(라벨만으로는 일부 부정확 — 예: ADL0106A는
#     라벨이 "Dressing"이지만 실제 응답 문구는 Q6a '옷 고르기'와 일치).
#     Q18(혼자 있기)은 R코드에서 adl_q18 = adl_q18a+adl_q18b+adl_q18c로
#     정의됨을 원문 코드에서 직접 확인함(추정 아님).
# ------------------------------------------------------------------

def _lt(x, thr):
    return (x < thr) & x.notna()


def _ok(x, thr):
    """R의 'or is.na()' 패턴: 결측이면 조건 통과로 둠(0/1단계 판정용)."""
    return (x > thr) | x.isna()


def compute_A0_2025(adl_wide: pd.DataFrame) -> pd.DataFrame:
    def col(c):
        return adl_wide[c + "__resolved"] if c + "__resolved" in adl_wide.columns else adl_wide[c]

    q1, q2, q3, q4, q5 = adl_wide["ADL0101"], adl_wide["ADL0102"], adl_wide["ADL0103"], adl_wide["ADL0104"], adl_wide["ADL0105"]
    q6a, q6b = col("ADL0106A"), col("ADL0106B")
    q7a = col("ADL0107A")
    q10a, q11a, q12a, q13a, q14a, q15a = col("ADL0110A"), col("ADL0111A"), col("ADL0112A"), col("ADL0113A"), col("ADL0114A"), col("ADL0115A")
    q16a, q16b = col("ADL0116A"), col("ADL0116B")
    q17a = col("ADL0117A")
    q18 = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
    q21a, q22c, q23b = col("ADL0121A"), col("ADL0122Q"), col("ADL0123L")

    iadl_items = [q6a, q7a, q10a, q11a, q12a, q13a, q14a, q15a, q16a, q16b, q17a, q18, q21a, q22c, q23b]
    badl_items = [q1, q2, q3, q4, q5, q6b]
    nmiss_iadl = sum(x.isna().astype(int) for x in iadl_items)
    nmiss_badl = sum(x.isna().astype(int) for x in badl_items)

    shop_flag = (q16a > 0) & (q16b == 0) & q16a.notna() & q16b.notna()

    level5_6_GE4 = (_lt(q6a,3).astype(int)+_lt(q10a,3).astype(int)+_lt(q11a,3).astype(int)+_lt(q12a,3).astype(int)
        +_lt(q14a,3).astype(int)+_lt(q16a,3).astype(int)+_lt(q17a,3).astype(int)+_lt(q18,3).astype(int)
        +_lt(q21a,3).astype(int)+_lt(q22c,3).astype(int)+_lt(q13a,4).astype(int)+_lt(q15a,4).astype(int)
        +_lt(q23b,4).astype(int)+_lt(q7a,5).astype(int)+shop_flag.astype(int))

    level5_GE1 = (_lt(q2,2).astype(int)+_lt(q3,2).astype(int)+_lt(q1,1).astype(int)+_lt(q4,1).astype(int)
        +_lt(q5,1).astype(int)+_lt(q6b,1).astype(int))

    level5_GE2 = (_lt(q6a,2).astype(int)+_lt(q7a,2).astype(int)+_lt(q10a,2).astype(int)+_lt(q11a,2).astype(int)
        +_lt(q13a,2).astype(int)+_lt(q14a,2).astype(int)+_lt(q15a,2).astype(int)+_lt(q16a,2).astype(int)
        +_lt(q17a,2).astype(int)+_lt(q21a,2).astype(int)+_lt(q22c,2).astype(int)+_lt(q23b,2).astype(int)
        +_lt(q12a,1).astype(int)+_lt(q18,1).astype(int)+shop_flag.astype(int))

    level4_GE3 = level5_GE2
    level3_GE2_GE3 = level4_GE3

    level2_3_4_GE1 = (_lt(q6b,4).astype(int)+_lt(q2,3).astype(int)+_lt(q3,3).astype(int)+_lt(q4,3).astype(int)
        +_lt(q1,2).astype(int))

    level1_2_GE2_GE3 = (_lt(q7a,5).astype(int)+_lt(q13a,4).astype(int)+_lt(q15a,4).astype(int)+_lt(q23b,4).astype(int)
        +_lt(q6a,3).astype(int)+_lt(q10a,3).astype(int)+_lt(q11a,3).astype(int)+_lt(q12a,3).astype(int)
        +_lt(q14a,3).astype(int)+_lt(q16a,3).astype(int)+_lt(q17a,3).astype(int)+_lt(q21a,3).astype(int)
        +_lt(q22c,3).astype(int)+_lt(q18,3).astype(int)+shop_flag.astype(int))

    level6 = (level5_6_GE4 >= 4) & ((q2==0)|(q3==0)|(q1==0))
    level5 = (level5_GE1 >= 1) & ((level5_GE2 >= 2) | (level5_6_GE4 >= 4))
    level4 = (level4_GE3 >= 3) & (level2_3_4_GE1 >= 1)
    level3 = (level3_GE2_GE3 >= 3) | ((level3_GE2_GE3 >= 2) & (level2_3_4_GE1 >= 1))
    level2 = (level1_2_GE2_GE3 >= 3) | ((level1_2_GE2_GE3 >= 2) & (level2_3_4_GE1 >= 1))
    level1 = (level1_2_GE2_GE3 >= 1) & _ok(q1,1) & _ok(q2,2) & _ok(q3,2) & _ok(q4,2) & _ok(q5,0) & _ok(q6b,3)
    level0 = (_ok(q1,1) & _ok(q2,2) & _ok(q3,2) & _ok(q4,2) & _ok(q5,0) & _ok(q6b,3) &
              _ok(q16b,0) & _ok(q6a,2) & _ok(q10a,2) & _ok(q11a,2) & _ok(q12a,2) & _ok(q14a,2) &
              _ok(q16a,2) & _ok(q17a,2) & _ok(q18,2) & _ok(q21a,2) & _ok(q22c,2) &
              _ok(q13a,3) & _ok(q15a,3) & _ok(q23b,3) & _ok(q7a,4))

    ds0 = pd.Series(np.nan, index=adl_wide.index)
    for lvl, mask in [(0,level0),(1,level1),(2,level2),(3,level3),(4,level4),(5,level5),(6,level6)]:
        ds0 = ds0.where(~mask, lvl)

    adl_wide["A0_2025_stage"] = ds0.where(~((nmiss_iadl > 2) | (nmiss_badl >= 1)))
    adl_wide["A0_harmonized"] = adl_wide["A0_2025_stage"].clip(upper=5)  # 5,6 -> 5 (계획서 조화 규칙)
    adl_wide["A0_nmiss_iadl"] = nmiss_iadl
    adl_wide["A0_nmiss_badl"] = nmiss_badl
    return adl_wide


# ------------------------------------------------------------------
# 4c. A1(2015 원판, 0-5단계) 산출 — Kahle-Wrobleski 2015 원문 Table 1/2 +
#     Figure 1(Table S1 이미지) 서브도메인 표를 그대로 이식.
#
#     핵심 재해석(2026-08-04, 팀원 재검토로 확정): Table 2의 "Level 0/1/2"는
#     파생 범주가 아니라 각 ADCS-ADL 문항의 "원점수 값 그 자체"임. 근거:
#     (1) Level4 조건에서 "Score <2 for Q6B"와 "Level 1 or 0 for Q5, Q4"가
#         같은 줄에 병기됨 — 두 표현이 곧 "점수값 0/1"이라는 동일한 뜻으로 쓰임.
#     (2) 각주 "Q20 only has a maximum level of 2"는 실제로 ADL0120이
#         A/B 두 개의 0/1 하위문항 합(0~2점)이라 원점수 최댓값이 2인 것과
#         정확히 일치함(원자료로 직접 확인).
#     (3) Chandler 2025가 2015판의 결함으로 "문항 점수 2점 미만을 일괄
#         손상 처리(threshold)"한 것을 지적한 내용과 맞물림.
#
#     사전 지정 규칙(팀원 지시, 원문의 결함까지 그대로 재현):
#     - 단계 조건이 상호배타적이지 않으므로 최고단계(5) 우선으로 배정.
#     - 이진(0/1) 문항은 "점수==2" 조건에 도달할 수 없음 — 원 알고리즘 자체의
#       결함이므로 임의로 고치지 않고 그대로 둠.
#     - Q16B(구매대금 지불, 원자료 ADL0116B)는 각주에 따라 제외.
#     - Q20(독서, 원자료 ADL0120A+B)은 각주("one point lower")에 따라
#       Level 조건 평가 시 (원점수+1)을 사용.
#
#     문항 원점수 파생(모두 __resolved, 즉 관문 N=0/DK=결측 처리 후 값 사용):
#       Q8(TV)=ADL0108A+B+C, Q9(대화)=ADL0109A, Q19(시사)=ADL0119A+B+C,
#       Q20(독서)=ADL0120A+B — Chandler Table S7에는 이 4문항이 없어(2025판
#       제외 문항) 원자료 하위문항 구조를 직접 확인해 합산 방식을 정함
#       (Q18 합산 방식과 동일 패턴, 값 범위(0/1 하위문항)로 실측 확인됨).
#
#     서브도메인(Figure 1/사진 우측 표, 원문 그대로):
#       Household: Q6A,Q7,Q10,Q11,Q12,Q13,Q14,Q23
#       Communication: Q8,Q9,Q17,Q19,Q20,Q21,Q22
#       Outside: Q15,Q16A,Q16B(제외),Q18
#       Basic Activities(bADL, Table2 규칙에서 개별 문항으로 지칭):
#         Q1,Q2,Q3,Q4,Q5,Q6B (Q6A 미포함 — 이는 A0의 nmiss_badl과 동일한
#         6문항 정의이며, Figure1 좌측의 "macro bADL 괄호"(Q1-5+Q6A+Q6B,
#         7문항)와는 다른 개념임에 유의. Table2 규칙 자체는 우측의
#         "Basic Activities Sub-Domain"(6문항) 목록을 씀.
# ------------------------------------------------------------------

def compute_A1_2015(adl_wide: pd.DataFrame) -> pd.DataFrame:
    def col(c):
        return adl_wide[c + "__resolved"] if c + "__resolved" in adl_wide.columns else adl_wide[c]

    q1, q2, q3, q4, q5 = adl_wide["ADL0101"], adl_wide["ADL0102"], adl_wide["ADL0103"], adl_wide["ADL0104"], adl_wide["ADL0105"]
    q6a, q6b = col("ADL0106A"), col("ADL0106B")
    q7 = col("ADL0107A")
    q8 = col("ADL0108A") + col("ADL0108B") + col("ADL0108C")
    q9 = col("ADL0109A")
    q10, q11, q12, q13, q14 = col("ADL0110A"), col("ADL0111A"), col("ADL0112A"), col("ADL0113A"), col("ADL0114A")
    q15 = col("ADL0115A")
    q16a = col("ADL0116A")
    # Q16B는 각주에 따라 제외 (사용하지 않음)
    q17 = col("ADL0117A")
    q18 = col("ADL0118A") + col("ADL0118B") + col("ADL0118C")
    q19 = col("ADL0119A") + col("ADL0119B") + col("ADL0119C")
    q20_raw = col("ADL0120A") + col("ADL0120B")
    q20 = q20_raw + 1  # 각주: "one point lower in the above settings" 보정
    q21 = col("ADL0121A")
    q22 = col("ADL0122Q")
    q23 = col("ADL0123L")

    household = [q6a, q7, q10, q11, q12, q13, q14, q23]
    communication = [q8, q9, q17, q19, q20, q21, q22]
    outside = [q15, q16a, q18]  # Q16B 제외

    def any_eq(items, val):
        m = pd.Series(False, index=adl_wide.index)
        for x in items:
            m = m | (x == val)
        return m

    def all_eq(items, val):
        m = pd.Series(True, index=adl_wide.index)
        for x in items:
            m = m & (x == val)
        return m

    hh_has2 = any_eq(household, 2)
    comm_has2 = any_eq(communication, 2)
    out_has2 = any_eq(outside, 2)
    n_clusters_has2 = hh_has2.astype(int) + comm_has2.astype(int) + out_has2.astype(int)

    hcO = household + communication + outside
    any_hcO_1 = any_eq(hcO, 1)
    any_hcO_2 = any_eq(hcO, 2)
    any_hh_0 = any_eq(household, 0)
    any_out_0 = any_eq(outside, 0)
    all_hcO_2 = all_eq(hcO, 2)

    level5 = (q2 == 1) | (q3 == 0)
    level4 = (q6b < 2) | q5.isin([0, 1]) | q4.isin([0, 1]) | (q3 == 2) | (q1 == 0)
    level3 = all_hcO_2 | any_out_0 | (q1 == 2) | (q4 == 2)
    level2 = (n_clusters_has2 >= 2) | any_hcO_1 | any_hh_0
    level1 = any_hcO_2
    # Level0: 위 어느 조건에도 해당하지 않을 때의 기본값(원문 "No impairments in ADCS-ADL")

    # 결측 처리: 관여 문항(bADL 6개 + household/communication/outside 18개
    # [Q16B 제외], 총 24문항)이 얼마나 관측됐는지 함께 기록. 원문에 결측 제외
    # 규정이 없어 임의 배제하지 않되, "관측 문항 수"를 QC용으로 남겨 과도한
    # 결측 사례를 식별 가능하게 함. 관측치가 하나도 없는 행만 최종 결측 처리.
    all_items = [q1, q2, q3, q4, q5, q6b] + hcO
    n_observed = sum(x.notna().astype(int) for x in all_items)
    n_total = len(all_items)
    has_any_obs = n_observed > 0

    computed = pd.Series(np.nan, index=adl_wide.index)
    for lvl, mask in [(1, level1), (2, level2), (3, level3), (4, level4), (5, level5)]:
        computed = computed.where(~mask, lvl)
    computed = computed.fillna(0)
    a1_stage = computed.where(has_any_obs, np.nan)

    adl_wide["A1_2015_stage"] = a1_stage
    adl_wide["A1_n_items_observed"] = n_observed
    adl_wide["A1_n_items_total"] = n_total
    return adl_wide


# ------------------------------------------------------------------
# 5. MMSE 처리
# ------------------------------------------------------------------

def build_mmse_wide(qs: pd.DataFrame) -> pd.DataFrame:
    mm = qs[(qs["QSCAT"] == "MMSE") & (qs["QSTESTCD"] == "MMSETOT")].copy()
    mm = mm[mm["VISITNUM"].isin([2.0, 5.0, 7.0])]
    mm = mm.rename(columns={"QSSTRESN": "mmse_total"})
    mm["visit_label"] = mm["VISITNUM"].map(VISIT_LABELS)
    return mm[["STUDYID", "USUBJID", "VISITNUM", "visit_label", "mmse_total"]]


# ------------------------------------------------------------------
# 6. RUD 감독시간 처리
# ------------------------------------------------------------------

def build_supervision_time(qs: pd.DataFrame) -> pd.DataFrame:
    """기저(RUA, 31일 회상) / 추적(RUB, 60일 회상)의
    124A(감독한 날의 1일 감독 분) x 124B(감독 일수) / 회상일수
    = 평균 일일 감독시간(분). 계획서 2.2·2.3 정의."""

    rud = qs[qs["QSCAT"] == "RUDL"].copy()
    rud = rud[rud["QSTESTCD"].isin(
        ["RUA124A", "RUA124B", "RUB124A", "RUB124B"]
    )]

    wide = rud.pivot_table(
        index=["STUDYID", "USUBJID", "VISITNUM"],
        columns="QSTESTCD",
        values="QSSTRESN",
        aggfunc="first",
    ).reset_index()

    for c in ["RUA124A", "RUA124B", "RUB124A", "RUB124B"]:
        if c not in wide.columns:
            wide[c] = np.nan

    is_baseline_form = wide["RUA124A"].notna() | wide["RUA124B"].notna()

    minutes = np.where(is_baseline_form, wide["RUA124A"], wide["RUB124A"])
    days = np.where(is_baseline_form, wide["RUA124B"], wide["RUB124B"])
    recall_days = np.where(is_baseline_form, 31, 60)

    wide["supervision_minutes_per_episode"] = minutes
    wide["supervision_days"] = days
    wide["recall_days"] = recall_days
    wide["avg_daily_supervision_min"] = (
        pd.to_numeric(minutes, errors="coerce")
        * pd.to_numeric(days, errors="coerce")
        / recall_days
    )

    # 비정합 응답 플래그 (시간 0이나 일수 양수, 또는 그 반대) — 계획서 2.3
    m = pd.to_numeric(minutes, errors="coerce")
    d = pd.to_numeric(days, errors="coerce")
    wide["inconsistent_flag"] = ((m == 0) & (d > 0)) | ((m > 0) & (d == 0))
    # 계획서의 '감독 시간·일수가 모두 0'은 avg_daily_supervision_min==0(파생값, days=0이면
    # minutes>0이어도 0이 되어버려 과대집계됨)이 아니라 원본 두 값이 모두 0인지로 판정해야
    # 정확히 일치함(기저 시점 검증: 54.5%, 계획서 수치와 일치 확인함).
    wide["no_supervision_raw_flag"] = (m == 0) & (d == 0)

    wide["visit_label"] = wide["VISITNUM"].map(VISIT_LABELS)
    return wide


# ------------------------------------------------------------------
# 7. 분석표본 정의 (Aim 1·2 기저 표본 + 계획서 v2 추가 표본 정의)
# ------------------------------------------------------------------

def build_baseline_sample(ds_wide: pd.DataFrame, adl_wide: pd.DataFrame,
                           dm: pd.DataFrame) -> pd.DataFrame:
    ds_bl = ds_wide[ds_wide["VISITNUM"] == 2.0].copy()
    adl_bl = adl_wide[adl_wide["VISITNUM"] == 2.0][
        ["STUDYID", "USUBJID", "A0_harmonized", "A1_2015_stage"]
    ].drop_duplicates(subset=["STUDYID", "USUBJID"])
    adl_bl["has_adl_baseline"] = True

    sample = ds_bl.merge(adl_bl, on=["STUDYID", "USUBJID"], how="left")
    sample["has_adl_baseline"] = sample["has_adl_baseline"].fillna(False)

    sample = sample.merge(dm[["STUDYID", "USUBJID", "ARM", "AGE", "SEX"]],
                           on=["STUDYID", "USUBJID"], how="left")

    sample["in_aim1_2_sample"] = sample["ds_complete13"] & sample["has_adl_baseline"]

    # ------------------------------------------------------------
    # 공통 분석표본(계획서 v2 2.1·2.3): "실측 DS와 A0·A1·B1을 모두 산출할
    # 수 있는" 참가자만 모아 일차 상대 성능 비교에 씀. B1(대안 채점법)은
    # 아직 개발 전이라 여기서는 DS·A0·A1 세 개의 교집합만 반영해두고,
    # B1이 만들어지면 그 산출가능 여부를 추가로 AND 해야 함(주석 참고).
    # ------------------------------------------------------------
    sample["has_a0_baseline"] = sample["A0_harmonized"].notna()
    sample["has_a1_baseline"] = sample["A1_2015_stage"].notna()
    sample["in_common_comparison_sample"] = (
        sample["in_aim1_2_sample"] & sample["has_a0_baseline"] & sample["has_a1_baseline"]
        # TODO: B1 산출 가능 여부(has_b1_baseline) 추가 후 & 로 반영할 것
    )

    # ------------------------------------------------------------
    # Aim 3 종단표본(계획서 v2 2.1 표): 12주 또는 24주 DS 완전관측(2,127명
    # 목표), 기저·24주 모두 완전관측(24주 완전사례, 2,045명 목표).
    # ------------------------------------------------------------
    ds_wk12 = ds_wide[ds_wide["VISITNUM"] == 5.0][["STUDYID", "USUBJID", "ds_complete13"]]
    ds_wk12 = ds_wk12.rename(columns={"ds_complete13": "ds_complete_wk12"})
    ds_wk24 = ds_wide[ds_wide["VISITNUM"] == 7.0][["STUDYID", "USUBJID", "ds_complete13"]]
    ds_wk24 = ds_wk24.rename(columns={"ds_complete13": "ds_complete_wk24"})

    sample = sample.merge(ds_wk12, on=["STUDYID", "USUBJID"], how="left")
    sample = sample.merge(ds_wk24, on=["STUDYID", "USUBJID"], how="left")
    sample["ds_complete_wk12"] = sample["ds_complete_wk12"].fillna(False)
    sample["ds_complete_wk24"] = sample["ds_complete_wk24"].fillna(False)

    sample["in_aim3_longitudinal_sample"] = (
        sample["in_aim1_2_sample"] & (sample["ds_complete_wk12"] | sample["ds_complete_wk24"])
    )
    sample["in_aim3_24wk_completer_sample"] = (
        sample["in_aim1_2_sample"] & sample["ds_complete_wk24"]
    )

    return sample


# ------------------------------------------------------------------
# 8. 실행
# ------------------------------------------------------------------

def main():
    print("1) DM 추출...")
    dm = extract_dm()
    dm.to_parquet(OUT_DIR / "dm_filtered.parquet", index=False)
    print(dm.groupby("STUDYID").size())

    print("2) QS 서브셋 추출 (시간이 걸림, qs.csv 2GB 스캔)...")
    qs = extract_qs_subset()
    qs.to_parquet(OUT_DIR / "qs_subset_long.parquet", index=False)
    print(f"   QS subset rows: {len(qs):,}")

    print("3) DS wide + 단계 배정...")
    ds_wide = build_ds_wide(qs)
    ds_wide.to_parquet(OUT_DIR / "ds_wide.parquet", index=False)

    print("4) ADCS-ADL wide + A0(2025판)/A1(2015판) 채점...")
    adl_wide = build_adl_wide(qs)
    adl_wide = compute_A0_2025(adl_wide)
    adl_wide = compute_A1_2015(adl_wide)
    adl_wide.to_parquet(OUT_DIR / "adl_wide.parquet", index=False)

    print("\n=== QC: A1(2015판) 단계 분포 (기저, 전체 관측 대상) ===")
    a1_bl = adl_wide[adl_wide.VISITNUM == 2.0]
    a1_dist = a1_bl["A1_2015_stage"].value_counts(dropna=False).sort_index()
    print(a1_dist)
    n_valid = a1_bl["A1_2015_stage"].notna().sum()
    print(f"산출 가능: {n_valid} / {len(a1_bl)}")
    print("참고(Wrobleski 2015 Table 4, GERAS 코호트 N=1497, 분포 형태 비교용):")
    print("  Level0 0.7%, 1 2%, 2 30%, 3 26%, 4 34%, 5 7%")
    if n_valid > 0:
        pct = (a1_dist / n_valid * 100).round(1)
        print("  실측(CPAD 기저) 비율:")
        print(pct)

    print("5) MMSE wide...")
    mmse_wide = build_mmse_wide(qs)
    mmse_wide.to_parquet(OUT_DIR / "mmse_wide.parquet", index=False)

    print("6) RUD 감독시간...")
    rud_wide = build_supervision_time(qs)
    rud_wide.to_parquet(OUT_DIR / "supervision_time.parquet", index=False)
    bl_rud = rud_wide[rud_wide.VISITNUM == 2.0]
    print(f"   기저 무공급 비율(원본 0&0 기준): {bl_rud['no_supervision_raw_flag'].mean():.1%} (계획서 값: 54.5%)")
    print(f"   기저 비정합 응답 수: {bl_rud['inconsistent_flag'].sum()} / {len(bl_rud)}")

    print("7) 기저 분석표본 (Aim1/2)...")
    sample = build_baseline_sample(ds_wide, adl_wide, dm)
    sample.to_parquet(OUT_DIR / "baseline_sample.parquet", index=False)

    print("\n=== QC: 계획서 보고 표본수 대조 (기저 DS13 완전관측, 표 2.1) ===")
    print("계획서 값: AD-1061 668명, AD-1063 835명, AD-1064 705명")
    check = (
        sample[sample["ds_complete13"]]
        .groupby("STUDYID")["USUBJID"].nunique()
    )
    print("실측 값  :")
    print(check)

    print("\n=== QC: Aim1/2 최종 분석표본(DS13 완전 + 동일방문 ADCS-ADL) ===")
    check2 = (
        sample[sample["in_aim1_2_sample"]]
        .groupby("STUDYID")["USUBJID"].nunique()
    )
    print(check2, " (합:", check2.sum(), ", 계획서 값: 2,208명)")

    print("\n=== QC: 공통 비교표본(DS+A0+A1 모두 산출 가능, B1 제외) ===")
    n_common = sample["in_common_comparison_sample"].sum()
    print(f"공통 비교표본: {n_common} / {sample['in_aim1_2_sample'].sum()} "
          f"(Aim1/2 표본 중 A0·A1 둘 다 산출 가능한 인원)")

    print("\n=== QC: Aim3 종단표본 (계획서 v2 목표: 2,127명 / 24주완전사례 2,045명) ===")
    n_long = sample["in_aim3_longitudinal_sample"].sum()
    n_24wk = sample["in_aim3_24wk_completer_sample"].sum()
    print(f"12주 또는 24주 DS 완전관측: {n_long}명 (목표 2,127명)")
    print(f"기저·24주 모두 DS 완전관측(24주 완전사례): {n_24wk}명 (목표 2,045명)")

    print("\n=== QC: A0(2025판) vs 실측 DS 일치도 (기저, 조화척도) ===")
    try:
        from sklearn.metrics import cohen_kappa_score, mean_absolute_error
        adl_bl = adl_wide[adl_wide.VISITNUM == 2.0][
            ["STUDYID", "USUBJID", "A0_2025_stage", "A0_harmonized"]
        ]
        merged = sample[sample.in_aim1_2_sample][["STUDYID", "USUBJID", "ds_stage"]].merge(
            adl_bl, on=["STUDYID", "USUBJID"], how="left"
        )
        comp = merged.dropna(subset=["ds_stage", "A0_harmonized"])
        print(f"A0 산출 가능: {merged['A0_harmonized'].notna().sum()} / {len(merged)}")
        print(f"가중카파(quadratic): {cohen_kappa_score(comp.ds_stage, comp.A0_harmonized, weights='quadratic'):.3f}")
        print(f"MAE: {mean_absolute_error(comp.ds_stage, comp.A0_harmonized):.3f}")
        print(f"완전일치율: {(comp.ds_stage == comp.A0_harmonized).mean():.1%}")
        print(f"±1단계 이내: {(comp.ds_stage - comp.A0_harmonized).abs().le(1).mean():.1%}")

        print("\n=== QC: A1(2015판) vs 실측 DS 일치도 (기저) ===")
        adl_bl_a1 = adl_wide[adl_wide.VISITNUM == 2.0][["STUDYID", "USUBJID", "A1_2015_stage"]]
        merged1 = sample[sample.in_aim1_2_sample][["STUDYID", "USUBJID", "ds_stage"]].merge(
            adl_bl_a1, on=["STUDYID", "USUBJID"], how="left"
        )
        comp1 = merged1.dropna(subset=["ds_stage", "A1_2015_stage"])
        print(f"A1 산출 가능: {merged1['A1_2015_stage'].notna().sum()} / {len(merged1)}")
        print(f"가중카파(quadratic): {cohen_kappa_score(comp1.ds_stage, comp1.A1_2015_stage, weights='quadratic'):.3f}")
        print(f"MAE: {mean_absolute_error(comp1.ds_stage, comp1.A1_2015_stage):.3f}")
        print(f"완전일치율: {(comp1.ds_stage == comp1.A1_2015_stage).mean():.1%}")
        print(f"±1단계 이내: {(comp1.ds_stage - comp1.A1_2015_stage).abs().le(1).mean():.1%}")
    except ImportError:
        print("[스킵] scikit-learn 미설치")


# ------------------------------------------------------------------
# 9. AD-1062(공개연장시험) 별도 추출 — 계획서 v2 2.1
#    "참가자 중복을 배제할 수 없어 학습·튜닝·교차검증에서 제외하고,
#    최종 모형의 추가 성능평가에만 사용한다(독립적 외부검증으로
#    격상시키지 않는다)." 그래서 메인 3개 시험(dm_filtered.parquet,
#    qs_subset_long.parquet, ds_wide.parquet, adl_wide.parquet,
#    baseline_sample.parquet)과는 완전히 분리된 파일로만 저장하고,
#    Aim1/2/3의 표본 정의·정합성 검증에는 절대 섞어 쓰지 않는다.
# ------------------------------------------------------------------

EXTRA_OUT_DIR = OUT_DIR / "extra_validation_ad1062"


def run_ad1062_extra():
    EXTRA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[AD-1062] 1) DM 추출...")
    dm_1062 = extract_dm(EXTRA_STUDIES)
    dm_1062.to_parquet(EXTRA_OUT_DIR / "dm_ad1062.parquet", index=False)
    print(f"   등록자 수: {len(dm_1062)}")

    print("[AD-1062] 2) QS 서브셋 추출...")
    qs_1062 = extract_qs_subset(EXTRA_STUDIES, tag="ad1062")
    qs_1062.to_parquet(EXTRA_OUT_DIR / "qs_subset_ad1062_long.parquet", index=False)
    print(f"   QS subset rows: {len(qs_1062):,}")

    print("[AD-1062] 3) DS wide (방문번호 체계가 메인 3개 시험과 달라 필터 없이 전부 사용)...")
    ds_wide_1062 = build_ds_wide(qs_1062, visit_filter=None)
    ds_wide_1062.to_parquet(EXTRA_OUT_DIR / "ds_wide_ad1062.parquet", index=False)

    print("[AD-1062] 4) ADCS-ADL wide + A0/A1 채점 (방문번호 필터 없음)...")
    adl_wide_1062 = build_adl_wide(qs_1062, visit_filter=None)
    adl_wide_1062 = compute_A0_2025(adl_wide_1062)
    adl_wide_1062 = compute_A1_2015(adl_wide_1062)
    adl_wide_1062.to_parquet(EXTRA_OUT_DIR / "adl_wide_ad1062.parquet", index=False)

    print("\n=== QC: AD-1062 방문번호 구성 (메인 3개 시험과 다른 체계임을 확인) ===")
    print(qs_1062[["VISITNUM", "VISIT"]].drop_duplicates().sort_values("VISITNUM").to_string(index=False))

    print("\n=== QC: AD-1062 'Visit 1 (Baseline II)' 시점 DS13 완전관측 인원 (참고용) ===")
    ds_bl_1062 = ds_wide_1062[(ds_wide_1062.VISITNUM == 1.0) & (ds_wide_1062.ds_complete13)]
    print(f"Visit1 DS 완전관측: {len(ds_bl_1062)} / {dm_1062.shape[0]}(등록자) "
          f"— 어느 시점을 '기저'로 볼지는 실제 평가 설계 시 재확인 필요")

    return dm_1062, qs_1062, ds_wide_1062, adl_wide_1062


if __name__ == "__main__":
    main()
