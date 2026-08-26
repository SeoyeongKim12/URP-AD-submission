"""
팀원 전처리 산출물 독립 검증 (sjlee)
=====================================
검증 원칙: 팀원 코드를 다시 읽는 게 아니라, 팀원의 파생 컬럼을
원점수에서 '내가 직접' 재계산해 행별로 대조한다. 불일치 = 버그.

이 스크립트는 계획서 §2.2 규칙을 팀원 코드와 무관하게 새로 구현한다.

입력(현재 확보): ds_wide.csv, adl_wide.csv  (환자단위 → git 금지)
산출: aim1_sjlee/verify_report.md (검증 결과 표)

사용법: python verify_independent.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
OUTDIR.mkdir(exist_ok=True)
REPORT = OUTDIR / "verify_report.md"

_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))  # 콘솔 인코딩 안전
    _lines.append(s)

def flush_report():
    REPORT.write_text("\n".join(_lines), encoding="utf-8")


# ============================================================
# C1. 실측 DS 단계 독립 재현 (가장 강력)
# ============================================================
def my_ds_stage(row):
    """계획서 §2.2 Brickman 규칙을 독립 구현. 최고단계 우선.
    DS01~13 중 하나라도 결측이면 NaN.
    데이터 확인: DS01/DS02만 {0,1,2}, DS03~13은 {0,1}."""
    d = {i: row[f"DS{i:02d}"] for i in range(1, 14)}
    if any(pd.isna(v) for v in d.values()):
        return np.nan
    # 5단계: DS11·12·13 중 하나라도 1
    if d[11] == 1 or d[12] == 1 or d[13] == 1:
        return 5
    # 4단계: DS08·09·10 중 하나라도 1
    if d[8] == 1 or d[9] == 1 or d[10] == 1:
        return 4
    # 3단계: DS05·06·07 중 하나라도 1
    if d[5] == 1 or d[6] == 1 or d[7] == 1:
        return 3
    # 2단계: DS01~03 중 값==1이 2개 이상, 또는 DS01/02 중 하나가 2, 또는 DS04==1
    n123_eq1 = sum(1 for i in (1, 2, 3) if d[i] == 1)
    if n123_eq1 >= 2 or d[1] == 2 or d[2] == 2 or d[4] == 1:
        return 2
    # 1단계: 위 미해당 + DS01~03 중 하나라도 1
    if any(d[i] == 1 for i in (1, 2, 3)):
        return 1
    # 0단계: 전 항목 0
    return 0


def check_C1(ds):
    log("## C1. 실측 DS 단계 독립 재현\n")
    ds = ds.copy()
    ds["my_stage"] = ds.apply(my_ds_stage, axis=1)

    both_val = ds["ds_stage"].notna() & ds["my_stage"].notna()
    mism = both_val & (ds["ds_stage"] != ds["my_stage"])
    # 결측 패턴 대조: 한쪽만 NaN인 경우도 불일치
    nan_mismatch = ds["ds_stage"].isna() != ds["my_stage"].isna()

    log(f"- 전체 행: {len(ds)}")
    log(f"- 값 대조 대상(양쪽 값 존재): {both_val.sum()}")
    log(f"- **값 불일치: {int(mism.sum())}건**")
    log(f"- **결측패턴 불일치(한쪽만 NaN): {int(nan_mismatch.sum())}건**")

    ok = (mism.sum() == 0) and (nan_mismatch.sum() == 0)
    log(f"- 판정: {'PASS (불일치 0건)' if ok else 'FAIL — 아래 불일치 행 확인'}\n")

    if mism.sum() > 0:
        cols = [f"DS{i:02d}" for i in range(1, 14)] + ["ds_stage", "my_stage"]
        bad = ds.loc[mism, ["STUDYID", "USUBJID", "VISITNUM"] + cols].head(30)
        log("불일치 상세(최대 30건):\n```")
        log(bad.to_string(index=False))
        log("```\n")
    if nan_mismatch.sum() > 0:
        cols = [f"DS{i:02d}" for i in range(1, 14)] + ["ds_stage", "my_stage", "ds_complete13"]
        bad = ds.loc[nan_mismatch, ["STUDYID", "USUBJID", "VISITNUM"] + cols].head(30)
        log("결측패턴 불일치 상세(최대 30건):\n```")
        log(bad.to_string(index=False))
        log("```\n")
    return ok


# ============================================================
# C2. 게이팅 3분기 규칙 감사
# ============================================================
import re

def parent_gate_of(sub_code):
    """하위문항 코드(ADL0106B, ADL0122Q, ADL0123L)에서 부모 관문 컬럼명 도출.
    끝의 단일 대문자 접미사를 떼어 ADL01xx__gate 를 만든다.
    주의: 게이팅을 지배하는 건 부모 관문질문(ADL0106__gate: generated/
    not_generated/dont_know)이며, 하위문항 자체의 __gate(other/missing)는
    '그 칸이 원자료에 비었는지'만 나타냄 — 게이팅 판정에 쓰면 오탐."""
    base = re.sub(r"[A-Z]$", "", sub_code)   # ADL0106B -> ADL0106
    return base + "__gate"

def check_C2(adl):
    log("## C2. 게이팅 3분기 규칙 감사 (부모 관문 x __resolved)\n")
    log("규칙: 부모관문=not_generated(N) -> resolved 전부 0 | "
        "=dont_know -> resolved 전부 NaN | =generated(Y) -> 원값 유지\n")
    log("(중요: 하위문항 자체의 __gate={other,missing}가 아니라 **부모 관문질문**"
        " __gate={generated,not_generated,dont_know}를 기준으로 판정)\n")

    resolved_cols = [c for c in adl.columns if c.endswith("__resolved")]
    log(f"- __resolved 컬럼 {len(resolved_cols)}개 감사\n")

    violations = []
    for rc in resolved_cols:
        sub = rc.replace("__resolved", "")           # 예: ADL0120A
        raw_col = sub if sub in adl.columns else None
        gate_col = parent_gate_of(sub)
        if gate_col not in adl.columns:
            violations.append((rc, f"부모관문없음({gate_col})", 0, None))
            continue
        g = adl[gate_col].astype("string").str.strip().str.lower()
        r = adl[rc]

        # 하위문항 원값 존재 여부 (직접 응답된 문항이면 관문과 무관하게 원값 보존이 정상)
        raw_missing = adl[raw_col].isna() if raw_col is not None else True
        # N(not_generated) -> 구조적 결측을 0으로 채워야 함. 원값이 실제로 있으면 정상.
        bad_n = (g == "not_generated") & raw_missing & r.notna() & (r != 0)
        # dont_know -> 구조적 결측을 NaN 유지해야 함. 원값이 실제로 있으면(직접응답) 정상.
        bad_dk = (g == "dont_know") & raw_missing & r.notna()
        # generated(Y) -> resolved == 원값 (원값이 있을 때)
        bad_gen = None
        if raw_col is not None:
            bad_gen = (g == "generated") & adl[raw_col].notna() & (r != adl[raw_col])

        if bad_n.sum() > 0:
            violations.append((rc, "N인데 resolved!=0", int(bad_n.sum()),
                               sorted(r[bad_n].dropna().unique().tolist())[:10]))
        if bad_dk.sum() > 0:
            violations.append((rc, "DK인데 resolved!=NaN", int(bad_dk.sum()),
                               sorted(r[bad_dk].dropna().unique().tolist())[:10]))
        if bad_gen is not None and bad_gen.sum() > 0:
            violations.append((rc, "Y인데 resolved!=원값", int(bad_gen.sum()), None))

    log("> 판정노트: ADL0106(옷입기)은 하위 A(Q6a 고르기)만 관문에 종속되고 "
        "B(Q6b 입기)는 직접 응답되는 별도 문항임 — 검증에서 원값이 실제 존재하는 "
        "직접응답은 관문과 무관하게 보존이 정상이므로 위반에서 제외함.\n")

    if not violations:
        log("- 판정: PASS (모든 __resolved가 부모 관문 3분기 규칙 준수)\n")
        return True
    log(f"- **판정: FAIL — 위반 {len(violations)}건**\n")
    log("| resolved컬럼 | 위반유형 | 건수 | 예시값 |")
    log("|---|---|---|---|")
    for rc, kind, cnt, ex in violations:
        log(f"| {rc} | {kind} | {cnt} | {ex} |")
    log("")
    return False


# ============================================================
# C3. 파생 지표 분포·범위 위생  +  C4 일부(A1 편향, Q18)
# ============================================================
def check_C3_C4(adl):
    log("## C3. 파생 지표 범위·분포 위생\n")
    ok = True

    # 범위 확인
    ranges = {"A0_2025_stage": (0, 6), "A0_harmonized": (0, 5), "A1_2015_stage": (0, 5)}
    for col, (lo, hi) in ranges.items():
        v = adl[col].dropna()
        out = ((v < lo) | (v > hi)).sum()
        uniq = sorted(v.unique().tolist())
        status = "OK" if out == 0 else f"범위밖 {out}건"
        log(f"- {col}: 값 {uniq} | 기대 [{lo}..{hi}] | {status} | 결측 {adl[col].isna().sum()}")
        if out != 0:
            ok = False
    log("")

    # 기저(VISITNUM==2.0) A1 분포 대조 (명세서: 0:0.6 1:3.0 2:39.7 3:31.7 4:23.8 5:1.2)
    base = adl[adl["VISITNUM"] == 2.0]
    log("### 기저(VISITNUM=2.0) A1_2015_stage 분포 (명세서 대조)")
    ref = {0: 0.6, 1: 3.0, 2: 39.7, 3: 31.7, 4: 23.8, 5: 1.2}
    dist = base["A1_2015_stage"].value_counts(normalize=True).sort_index() * 100
    log("| 단계 | 내 계산(%) | 명세서(%) |")
    log("|---|---|---|")
    for s in range(6):
        mine = dist.get(float(s), 0.0)
        log(f"| {s} | {mine:.1f} | {ref[s]} |")
    log(f"\n(주: 명세서 분포는 Aim1/2 표본 2,208명 기준. 여기는 baseline_sample 없이 "
        f"VISITNUM=2.0 전체 {len(base)}명 기준이라 근사 대조.)\n")

    # A0 산출가능/harmonized 값 확인
    log(f"- 기저 A0_2025_stage 산출가능(비결측): {base['A0_2025_stage'].notna().sum()} / {len(base)}")
    log(f"- 기저 A1_2015_stage 산출가능(비결측): {base['A1_2015_stage'].notna().sum()} / {len(base)}\n")

    # ---- C4: Q18 게이팅 특수조건 probe ----
    log("## C4. 리스크 지점 probe\n")
    log("### (a) Q18(혼자있기) 잔여 결측 — 명세서 기대 ~29명")
    q18 = ["ADL0118A__resolved", "ADL0118B__resolved", "ADL0118C__resolved"]
    if all(c in adl.columns for c in q18):
        q18_miss = adl[q18].isna().any(axis=1)
        base_q18_miss = adl.loc[adl["VISITNUM"] == 2.0, q18].isna().any(axis=1).sum()
        log(f"- Q18 resolved 하나라도 결측: 전체 {int(q18_miss.sum())}건 / 기저 {int(base_q18_miss)}건")
        log(f"  (명세서: 기저 A0 관점 Q18 잔여결측 29명 — 26명 관문누락 + 3명 통상결측)\n")
    else:
        log("- Q18 resolved 컬럼 없음\n")

    # ---- C4: A1 결측 하향편향 ----
    log("### (b) A1 결측 하향편향 — 관측문항 적은 사람이 저단계로 쏠리나?")
    if "A1_n_items_observed" in adl.columns:
        b = adl[adl["VISITNUM"] == 2.0].dropna(subset=["A1_2015_stage", "A1_n_items_observed"])
        corr = b["A1_2015_stage"].corr(b["A1_n_items_observed"], method="spearman")
        log(f"- 기저 Spearman(A1_stage, n_items_observed) = {corr:.3f}")
        log(f"- n_items_observed 분포: min {b['A1_n_items_observed'].min():.0f} / "
            f"median {b['A1_n_items_observed'].median():.0f} / max {b['A1_n_items_observed'].max():.0f}")
        grp = b.groupby(pd.cut(b["A1_n_items_observed"], bins=[0, 20, 23, 24]))["A1_2015_stage"].mean()
        log("- 관측문항수 구간별 평균 A1 단계:")
        for k, v in grp.items():
            log(f"    {k}: 평균단계 {v:.2f}")
        log("  (양의 상관 없음 = 편향 근거 약함 / 음의 상관 강하면 저단계 쏠림 주의)\n")
    else:
        log("- A1_n_items_observed 컬럼 없음\n")

    return ok


def check_C5(ds, adl):
    """표본 앵커 재현 — baseline_sample.csv 없이 ds_wide×adl_wide 기저 조인으로 근사."""
    log("## C5. 표본 앵커 재현 (baseline_sample 없이 근사)\n")
    ds_b = ds[ds["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "ds_complete13"]]
    adl_b = adl[adl["VISITNUM"] == 2.0][["STUDYID", "USUBJID", "A0_harmonized"]]

    # in_aim1_2_sample = ds_complete13 AND has_adl_baseline
    m = (ds_b[ds_b["ds_complete13"] == True]
         .merge(adl_b[["STUDYID", "USUBJID"]].drop_duplicates(),
                on=["STUDYID", "USUBJID"], how="inner"))
    per = m.groupby("STUDYID")["USUBJID"].nunique().to_dict()
    tot = m["USUBJID"].nunique()
    log(f"- in_aim1_2_sample(근사): {per} | 합 {tot}")
    log(f"  명세서 목표: AD-1061 668 · AD-1063 835 · AD-1064 705 (합 2,208)")
    ok1 = per.get("AD-1061") == 668 and per.get("AD-1063") == 835 and \
          per.get("AD-1064") == 705 and tot == 2208

    # in_common_comparison_sample = + A0 산출가능 (B1 미개발이라 DS+A0+A1 교집합, A1 전원)
    mc = (ds_b[ds_b["ds_complete13"] == True]
          .merge(adl_b[adl_b["A0_harmonized"].notna()][["STUDYID", "USUBJID"]].drop_duplicates(),
                 on=["STUDYID", "USUBJID"], how="inner"))
    tot2 = mc["USUBJID"].nunique()
    log(f"- in_common_comparison_sample(근사): 합 {tot2} | 명세서 목표 2,203")
    ok2 = tot2 == 2203

    log(f"- 판정: {'PASS (앵커 정확 재현)' if ok1 and ok2 else 'CHECK'}\n")
    return ok1 and ok2


def main():
    log("# 팀원 전처리 독립 검증 리포트 (sjlee)")
    log(f"생성 스크립트: verify_independent.py\n")

    ds_path = DOWNLOADS / "ds_wide.csv"
    adl_path = DOWNLOADS / "adl_wide.csv"

    results = {}
    ds = adl = None
    if ds_path.exists():
        ds = pd.read_csv(ds_path)
        results["C1"] = check_C1(ds)
    else:
        log(f"[건너뜀] {ds_path} 없음 — C1 불가\n")

    if adl_path.exists():
        adl = pd.read_csv(adl_path, low_memory=False)
        results["C2"] = check_C2(adl)
        results["C3/C4"] = check_C3_C4(adl)
    else:
        log(f"[건너뜀] {adl_path} 없음 — C2/C3/C4 불가\n")

    if ds is not None and adl is not None:
        results["C5"] = check_C5(ds, adl)

    log("## 종합")
    for k, v in results.items():
        log(f"- {k}: {'PASS' if v else 'CHECK(불일치/편향 존재 — 상세 확인)'}")
    log("\n(C5 표본앵커 · supervision 282건 · D1/D2는 baseline_sample/mmse_wide/"
        "supervision_time 확보 후 진행)")

    flush_report()
    print(f"\n>>> 리포트 저장: {REPORT}")


if __name__ == "__main__":
    main()
