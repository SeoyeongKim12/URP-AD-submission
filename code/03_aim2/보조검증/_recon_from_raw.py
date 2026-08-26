"""
원자료(CPAD qs.csv)에서 DS단계 + ADCS-ADL 21문항 재구성 (sjlee)
================================================================
목적: 전처리 산출물(adl_wide/ds_wide)이 없는 시험(AD-1062 등)의 분석표본을
      **팀원 전처리와 동일한 규칙**으로 원자료에서 직접 구성.

검증완료(3시험 대조, 이 파일 verify_against_wide()):
- DS단계: 668/835/705명 전원 100% 일치.
- ADL __resolved: 18문항 중 17문항 100%, ADL0106B는 DK특례 반영 후 100%.

규칙:
- 기저방문 = VISIT에 'Baseline' 포함(3시험 Visit2/VISITNUM2.0, AD-1062 Visit1/VISITNUM1.0).
- 값 = QSSTRESN. 부모관문 = QSTESTCD 끝 대문자 제거(ADL0116A→ADL0116).
- resolved: 부모관문 QSORRES가 No→0, Don't Know→NaN, Yes→원값.
  · 단 ADL0106B(입기)는 게이팅 함정: No→0, 그 외(DK 포함)→원값.
- DS단계: 계획서 §2.2 Brickman 규칙(최고단계 우선), DS01~13 중 결측 있으면 NaN.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

QS_PATH = Path.home() / "Downloads" / "1_fullExportDb-2011-Member-CSV" / "qs.csv"  # 원자료 위치

BADL = ["ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105", "ADL0106B"]
IADL = ["ADL0106A", "ADL0107A", "ADL0110A", "ADL0111A", "ADL0112A", "ADL0113A",
        "ADL0114A", "ADL0115A", "ADL0116A", "ADL0116B", "ADL0117A", "Q18",
        "ADL0121A", "ADL0122Q", "ADL0123L"]
ITEMS = BADL + IADL
# resolved(게이팅) 적용 대상: 0101~0105(BADL 원값) 제외 전부
RAW_ONLY = {"ADL0101", "ADL0102", "ADL0103", "ADL0104", "ADL0105"}
Q18_PARTS = ["ADL0118A", "ADL0118B", "ADL0118C"]


def _parent(sub):
    return re.sub(r"[A-Z]+$", "", sub) if sub[-1].isalpha() else sub


def _resolve(sub, val, txt):
    """부모관문 3분기 규칙으로 resolved 값 산출."""
    p = _parent(sub)
    raw = val[sub] if sub in val.columns else pd.Series(np.nan, index=val.index)
    if p not in txt.columns:
        return raw
    g = txt[p].astype("string").str.strip().str.lower()
    out = raw.copy()
    out[g.isin(["no"])] = 0
    if sub != "ADL0106B":                      # 0106B는 DK특례(원값 유지)
        out[g.str.contains("know", na=False)] = np.nan
    return out


def ds_stage_row(row):
    d = {i: row.get(f"DS{i:02d}", np.nan) for i in range(1, 14)}
    if any(pd.isna(v) for v in d.values()):
        return np.nan
    if d[11] == 1 or d[12] == 1 or d[13] == 1: return 5
    if d[8] == 1 or d[9] == 1 or d[10] == 1:   return 4
    if d[5] == 1 or d[6] == 1 or d[7] == 1:    return 3
    n = sum(1 for i in (1, 2, 3) if d[i] == 1)
    if n >= 2 or d[1] == 2 or d[2] == 2 or d[4] == 1: return 2
    if any(d[i] == 1 for i in (1, 2, 3)):      return 1
    return 0


def build_from_raw(studyids):
    """지정 시험들의 기저방문 분석표본(21문항 + ds_stage) 반환."""
    # 메모리 절감: 필요한 7개 열만 로드(전체 37열 → OOM 방지).
    cols = ["STUDYID", "USUBJID", "QSCAT", "QSTESTCD", "QSORRES", "QSSTRESN", "VISIT"]
    q = pd.read_csv(QS_PATH, low_memory=False, usecols=cols)
    q = q[q.STUDYID.isin(studyids) & q.VISIT.astype(str).str.contains("Baseline")]

    ds = q[q.QSCAT == "DEPENDENCE SCALE"]
    dsw = ds.pivot_table(index=["STUDYID", "USUBJID"], columns="QSTESTCD",
                         values="QSSTRESN", aggfunc="first")
    ds_stage = dsw.apply(ds_stage_row, axis=1).rename("ds_stage")

    adl = q[q.QSCAT == "ADCS-ADL"]
    val = adl.pivot_table(index=["STUDYID", "USUBJID"], columns="QSTESTCD",
                          values="QSSTRESN", aggfunc="first")
    txt = adl.pivot_table(index=["STUDYID", "USUBJID"], columns="QSTESTCD",
                          values="QSORRES", aggfunc="first")
    feat = pd.DataFrame(index=val.index)
    for it in ITEMS:
        if it == "Q18":
            feat["Q18"] = sum(_resolve(p, val, txt) for p in Q18_PARTS)
        elif it in RAW_ONLY:
            feat[it] = val[it] if it in val.columns else np.nan
        else:
            feat[it] = _resolve(it, val, txt)

    out = feat.join(ds_stage, how="outer").reset_index()
    return out


def verify_against_wide():
    """3시험 재구성이 기존 adl_wide/ds_wide와 일치하는지 검증(자기점검)."""
    DL = Path.home() / "Downloads"
    trials = ["AD-1061", "AD-1063", "AD-1064"]
    rec = build_from_raw(trials)
    dw = pd.read_csv(DL / "ds_wide.csv", low_memory=False)
    dw = dw[dw.VISITNUM == 2.0][["STUDYID", "USUBJID", "ds_stage"]].rename(columns={"ds_stage": "ds_o"})
    m = rec.merge(dw, on=["STUDYID", "USUBJID"], how="inner")
    both = m.dropna(subset=["ds_stage", "ds_o"])
    ds_match = (both.ds_stage == both.ds_o).mean() * 100
    aw = pd.read_csv(DL / "adl_wide.csv", low_memory=False); aw = aw[aw.VISITNUM == 2.0]
    rates = {}
    for it in ITEMS:
        if it == "Q18":
            aw["Q18_o"] = aw[[p + "__resolved" for p in Q18_PARTS]].sum(axis=1, min_count=3)
            oc = "Q18_o"
        else:
            oc = (it + "__resolved") if (it + "__resolved") in aw.columns else it
        tmp = aw[["STUDYID", "USUBJID", oc]].rename(columns={oc: "_o"})
        mm = rec.merge(tmp, on=["STUDYID", "USUBJID"], how="inner")
        a, b = mm[it], mm["_o"]
        rates[it] = (((a.fillna(-999) == b.fillna(-999)) | (a.isna() & b.isna())).mean()) * 100
    return ds_match, rates


if __name__ == "__main__":
    dsm, rates = verify_against_wide()
    print(f"DS단계 3시험 일치율: {dsm:.2f}%")
    print("ADL 문항별 일치율:")
    for it, r in rates.items():
        flag = "OK" if r > 99.9 else "CHECK"
        print(f"  {it}: {r:.2f}% [{flag}]")
