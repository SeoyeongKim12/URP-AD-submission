# -*- coding: utf-8 -*-
"""
1단계: dev/test 분할 재현 + 검증 (sjlee) — test 미접근
  문서 레시피: 공통표본 2,203 → STUDYID×ds_stage 층화 75/25, seed 20260814
  공표값 dev 1,652 / test 551 과 대조. 분할 배정 CSV 저장.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

DL=Path.home()/"Downloads"
OUT=Path(__file__).parent
SEED=20260814

bs=pd.read_csv(DL/"baseline_sample.csv")
com=bs[bs["in_common_comparison_sample"]==True].copy()
com["ds_stage"]=com["ds_stage"].astype(int)
print("공통표본 n =",len(com))
print(com["STUDYID"].value_counts().to_string())

strata=com["STUDYID"].astype(str)+"_"+com["ds_stage"].astype(str)
print("\n층화 셀 수 =",strata.nunique(),"| 최소 셀 크기 =",strata.value_counts().min())

dev_idx,test_idx=train_test_split(com.index,test_size=0.25,stratify=strata,random_state=SEED)
com["split"]=np.where(com.index.isin(test_idx),"test","dev")

ndev=(com.split=="dev").sum(); ntest=(com.split=="test").sum()
print("\n=== 분할 결과 ===")
print(f"dev  = {ndev}  (공표 1,652)   {'✔일치' if ndev==1652 else '�’불일치'}")
print(f"test = {ntest}  (공표 551)     {'✔일치' if ntest==551 else '✗불일치'}")

print("\n=== 시험별 비율 유지 확인 ===")
tab=com.groupby(["STUDYID","split"]).size().unstack(fill_value=0)
tab["dev%"]=(tab["dev"]/tab.sum(1)*100).round(1)
print(tab.to_string())

print("\n=== 단계별 비율 유지 확인 (dev vs test 분포) ===")
d=com[com.split=="dev"]["ds_stage"].value_counts(normalize=True).sort_index()*100
t=com[com.split=="test"]["ds_stage"].value_counts(normalize=True).sort_index()*100
cmp=pd.DataFrame({"dev%":d.round(1),"test%":t.round(1)})
print(cmp.to_string())

# 배정 CSV 저장 (STUDYID,USUBJID,ds_stage,split) — 문서 스키마와 동일
out=com[["STUDYID","USUBJID","ds_stage","split"]].sort_values(["STUDYID","USUBJID"])
out.to_csv(OUT/"b2_dev_test_split_RECON.csv",index=False,encoding="utf-8-sig")
print("\n저장:",OUT/"b2_dev_test_split_RECON.csv")
print("\n※ 이 재현 분할이 1,652/551과 일치하면 공식 CSV와 동일 레시피로 간주.")
print("  단, 최종 확정 전 공식 b2_dev_test_split_assignment.csv와 USUBJID 집합 대조 권장.")
