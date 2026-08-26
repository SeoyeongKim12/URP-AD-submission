# 데이터 명세서 

이 문서는 실행에 필요한 **입력 데이터**(어디서 받아 어디에 두나), ② 이 브랜치에 올라와 있는 **결과 파일**이 각각 무엇인지, ③ **실행법**을 정리한다.

> **데이터 원칙**: 환자단위 파일(행 = 환자, `USUBJID` 포함)은 **git에 올리지 않는다**(민감정보 → 드라이브 전용, `.gitignore` 차단). 이 브랜치에 커밋된 CSV는 **전부 집계·계수·규칙 표**(환자 식별 불가)뿐이다.

---

## 1. 입력 데이터 (실행 전 드라이브에서 받아 `~/Downloads/` 에 둘 것)

스크립트는 아래 파일들을 `~/Downloads/` 에서 읽는다. **모두 환자단위라 git에 없다** — URP 드라이브 `전처리/` 산출물에서 받는다.

| 파일 | 내용 | 핵심 컬럼 | 단위 |
|---|---|---|---|
| `baseline_sample.csv` | 기저 공통 비교표본(분석 대상 정의) | `STUDYID, USUBJID, ds_stage, ds_total, A0_harmonized, A1_2015_stage, in_common_comparison_sample` | 환자 |
| `adl_wide.csv` | ADCS-ADL 21문항(방문별 wide) + 관문상태 | `ADL01xx`(원점수)·`ADL01xx__resolved`(3분기 보정)·`*_gate`, `visit_label` | 환자×방문 |
| `ds_wide.csv` | 실측 DS(방문별) + Brickman 단계 | `DS01~DS14, ds_total, ds_complete13, ds_stage, VISITNUM` | 환자×방문 |
| `mmse_wide.csv` | MMSE 총점(방문별) | `mmse_total, VISITNUM, visit_label` | 환자×방문 |
| `supervision_time.csv` | 감독시간(평균 일일 분) | `avg_daily_supervision_min, inconsistent_flag, VISITNUM` | 환자×방문 |
| CPAD 원자료 `qs.csv` (+`dm.csv`) | 문항 재구성용 원천(보조검증에서만) | `STUDYID, USUBJID, QSTESTCD, QSCAT, QSSTRESC/N, VISITNUM` | 환자×문항×방문 |

> 각 파일의 **컬럼·전처리·검증 상세**는 팀 공용 **전처리 산출물 명세서**(`preprocess_dependence_study.py` 기준)를 따른다. 위 표는 sjlee 코드가 실제 읽는 것만 요약한 것이다.

### 파생 지표 컬럼 (분석의 예측 대상·비교 대상)
| 컬럼 | 정의 | 범위 |
|---|---|---|
| `ds_stage` | 실측 DS 단계 (Brickman 규칙) — **예측 대상** | 0~5 |
| `ds_total` | 실측 DS 총점 (DS01~13 합) | 0~15 |
| `A0_harmonized` | A0(Chandler 2025 조화) 파생 단계 | 0~5 |
| `A1_2015_stage` | A1(Kahle-Wrobleski 2015) 파생 단계 | 0~5 |

---

## 2. 이 브랜치의 결과 파일 (코드 ↔ 결과 매핑)

**전부 집계·계수 표(비환자)라 git에 포함됨.** 환자단위 예측 CSV는 §3 참조(드라이브).

### 01_전처리/ — 팀 전처리 독립 재현검증
| 파일 | 내용 |
|---|---|
| `01_verify_independent.py` (+`_report.md`) | DS단계·게이팅 3분기·A1편향을 원점수에서 재계산해 팀 전처리와 대조 |
| `02_verify_supervision_c4c.py` (+`_report.md`) | 감독시간(C4c) 계산 검증 |
| `검증결과_독립검증_sjlee.md` | 종합 |

### 02_aim1/ — Aim 1 타당성 (코드·리포트, 산출 CSV는 환자단위 → 드라이브)
`01_preliminary` → `02_formal` → `03_sensitivity` → `04_sensitivity_missing` → `05_supervision` → `06_supervision_sens` (+ 각 `_report.md`), 종합 `Aim1_정식결과_sjlee.md`.

### 03_aim2/최종/ — 확정 산출물 (B1→B2)
| 파일 | 내용 |
|---|---|
| `01_b1_gn_elasticnet.py` … `05_b2_test_and_scoretable.py` | Gn 개발 → dev/test 분할 → 파이프라인 검증 → test 1회 + 채점표 생성 |
| `b1_gn_cut_coefficients.csv` | Gn 절단별 계수(원눈금), 21문항 × P(Y≥1..5) |
| `b1_dominate_a0_scoretable.csv` | A0 지배 보강 채점표(절단별 계수) |
| **`b2_scoretable_7item_raw.csv`** | **최종 7문항 절단별 릿지 계수**(원눈금+표준화) |
| **`b2_scoretable_7item_rule.csv`** | **판정규칙** (τ4=0.08, τ5=0.56, 조건부중앙값) |
| `Aim2_결과_sjlee.md`, `b2_holdout_검증_결과.md` | 종합·holdout 결과 |

### 03_aim2/보조검증/
| 파일 | 내용 |
|---|---|
| `_recon_from_raw.py` | 원자료에서 DS단계·21문항 재구성 **공유 모듈**(여러 스크립트가 import → 번호 없음) |
| `01_gn_external_validation_ad1062.py` (+report) | AD-1062 외부검증(비독립 판정) |
| `02_gn_isotonic_correction.py` (+report) | 비단조 PAVA 보정 |
| `gn_final_hyperparams.csv` | Gn 최종 하이퍼파라미터 (C, l1_ratio, τ4, τ5 등) |

### 03_aim2/시행착오/b1/ — B1 탐색(원 비례오즈·개선·벌점견고성·계수/내보내기)
| 파일 | 내용 |
|---|---|
| `01_b1_proportional_odds.py` … `07_b1_export_xlsx.py` (+reports) | 원 B1·개선·벌점·계수·내보내기 |
| `b1_po_coefficients.csv` | 원 비례오즈 계수·OR·SE·z·p (무벌점) |

### 03_aim2/시행착오/b2/ — B2 탐색(구조분석·문항축소·안정성선택 등)
| 파일 | 내용 |
|---|---|
| `01_aim2_structure.py` … `12_b2_graphs.py` (+reports) | 구조분석(FA/PCA)·문항축소·랜덤서치·전수탐색·문항선택·단축형·그래프 |
| `aim2_sjlee/b2_reduced_scoretable.csv` | 축약 채점표(절단×문항 원눈금 계수) |
| `aim2_sjlee/b2_final_shortform_scoretable.csv` | 최종 단축형 채점표 |
| `aim2_sjlee/b2_short12_scoretable.csv`, `…short13_…` | 12·13문항 후보 채점표 |
| `aim2_sjlee/b2_selection_frequency.csv` | 안정성선택 문항 선택빈도(K별·fold별) |
| `aim2_sjlee/structure_factor_loadings.csv` | EFA 요인적재량(F1~F4) |
| `aim2_sjlee/structure_pca_loadings.csv` | PCA 성분적재량(PC1~) |
| `07_fig_b2_overfit_scatter.png` + `b2_overfit_scatter.csv` | 과적합 진단(그림 + k별 in/ex MAE·중증놓침 원자료) |
| `12_fig_lasso_path.png`, `12_fig_perf_vs_k.png` | LASSO 경로·문항수 대비 성능 |

### 03_aim2/aim2_sjlee/ — B1 요약표 (리포트 근거 표)
| 폴더 | 내용 |
|---|---|
| `csv/` | B1/Gn 계열 6표: 요약·추가가치4조건·성능지표·계수pvalue·Gn절단별계수·한계 |
| `csv_a0/` | A0 계열 8표: 요약·성능비교·부트스트랩유의성·채점표계수·한계·축약(문항수별성능·선택지비교) |
| `b1_scoretable_final.csv` | B1 최종 채점표 (item, coef) |

---

## 3. git에 없는 파일 (환자단위 → 드라이브 전용)

아래는 **행 = 환자**라 git에서 제외된다(정상). 드라이브 `AIM 1/` · `AIM 2/` 에 있다.

| 파일 | 내용 |
|---|---|
| `aim1_common_sample_scored.csv` | Aim1 공통표본 채점표(DS·A0·A1) |
| `aim1_supervision_analysis.csv` | Aim1 감독시간+지표 분석표 |
| `aim1_ad1061_ipw_weights.csv` | AD-1061 IPW 가중치 |
| `b1_cv_predictions.csv`, `b1_improve_cv_predictions.csv` | B1 교차검증 예측(환자별) |
| `gn_en_cv_predictions.csv` | Gn 외부예측(환자별) |
| `ad1062_predictions_PATIENT.csv` | AD-1062 환자별 예측 |
| `structure_cluster_assign.csv` | 군집 배정(환자별) |

---

## 4. 실행법

1. §1의 입력 데이터를 드라이브에서 받아 **`~/Downloads/`** 에 둔다.
2. 번호 순으로 실행한다: `00_eda → 01_전처리 → 02_aim1 → 03_aim2`.
3. 예: `PYTHONUTF8=1 python 03_aim2/최종/05_b2_test_and_scoretable.py`
4. 환경: python 3.x · numpy · pandas · scikit-learn · scipy · matplotlib. (Aim3 종단 다절만 R lme4/lmerTest — 팀원 작업, 드라이브 `Aim 3/`.)

> **재현 주의**: dev/test 분할은 공식 `b2_dev_test_split_assignment.csv` 미수령 시 동일 레시피(seed 20260814 · STUDYID×ds_stage 층화 · 75/25)로 재현한 것이다. 최종 확정 시 공식 파일과 `USUBJID` 대조 권장.
