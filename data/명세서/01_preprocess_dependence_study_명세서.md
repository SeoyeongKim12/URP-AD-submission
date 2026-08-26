# 데이터 명세서 — 의존도(Dependence Scale) 연구 전처리 산출물

> **읽기 전 참고**: 아래는 `preprocess_dependence_study.py`(`code/01_전처리/01_preprocess_dependence_study.py`)를 직접 실행했을 때 나오는 **parquet** 산출물 기준 설명이다. 팀 대부분은 이 스크립트를 직접 돌리지 않고, 아래와 동일한 내용을 **csv로 변환해 드라이브에 미리 올려둔 것**(`전처리/전처리 산출물/`)을 받아 `~/Downloads/`에 두고 바로 쓴다 — 컬럼 정의·파생 로직·검증 내역은 parquet든 csv든 완전히 동일하다. 파일 형식만 다르고, 여기 서술된 모든 내용(처리 과정·검증 결과·주의사항)은 그대로 적용된다.

작성 기준: `preprocess_dependence_study.py` 실행 결과 (실행일 2026-08-04)
원천 데이터: CPAD `1_fullExportDb-2011-Member-CSV/dm.csv`, `qs.csv`
대상 시험: AD-1061 · AD-1063 · AD-1064 (경도~중등도 AD, 24주 3상, 연구계획서 2.1)

공통 원칙: 모든 파일은 STUDYID를 보존해 이후 분석에서 STUDYID/ARM 통제가 가능함.
문항 선택은 전부 QSTESTCD 기준이며 QSTEST 텍스트 매칭은 쓰지 않음(동일 코드에
다른 라벨이 붙는 CPAD 특성 때문). 응답값은 QSORRES(인코딩 손상 확인됨) 대신
QSSTRESC/QSSTRESN만 사용함.

---

## 1. dm_filtered.parquet

**내용**: 3개 시험 참가자의 인구학·시험군 정보.

**원천**: `dm.csv` 전체(모든 시험) → STUDYID가 AD-1061/1063/1064인 행만 필터.

**처리 과정**:
1. 필요 컬럼만 선택(STUDYID, USUBJID, SUBJID, AGE, AGEU, SEX, RACE, ARM, ACTARMCD, ACTARM, COUNTRY).
2. STUDYID 필터.
3. USUBJID 기준 중복 제거(원자료에 중복 행이 있을 가능성 대비).

**행 수**: 2,526명 (AD-1061 934 · AD-1063 858 · AD-1064 734, 각 시험 DM상 전체 등록자와 일치 확인함).

**컬럼**: STUDYID, USUBJID, SUBJID, AGE(수치 변환됨), AGEU, SEX, RACE, ARM, ACTARMCD, ACTARM, COUNTRY.

**주의**: 이 표는 "등록자 전체"이며 DS/ADCS-ADL 관측 여부와 무관함. 분석표본을 정의할 때는 이 파일을 다른 파일과 조인해서 조건을 추가로 걸어야 함.

---

## 2. qs_subset_long.parquet

**내용**: QS 도메인 중 이 연구가 쓰는 4개 QSCAT(DEPENDENCE SCALE, ADCS-ADL, MMSE, RUDL)만 남긴 long(세로긴) 포맷 원자료. 나머지 산출물(ds_wide 등)은 전부 이 파일에서 파생됨.

**원천**: `qs.csv` (2GB, 1,245만 행) 전체.

**처리 과정**:
1. `grep`으로 STUDYID가 AD-1061/1063/1064인 물리적 행만 1차로 걸러냄(각 논리 행=물리 행 1줄임을 확인 후 적용, 마운트된 폴더 대신 로컬 스크래치 경로에 써서 속도 확보).
2. pandas로 읽어 QSCAT이 4개 대상 카테고리인 행만 2차 필터.
3. QSSTRESN을 수치형으로, VISITNUM을 수치형으로 변환.
4. QC: VISITNUM 결측률이 2% 넘으면 경고 출력(파싱 손상 탐지용 — DuckDB 기본 병렬 CSV 리더에서 실제로 손상이 발견된 바 있어 넣어둔 안전장치).

**행 수**: 1,045,484행 (ADCS-ADL 473,201 · RUDL 246,810 · MMSE 233,507 · DEPENDENCE SCALE 91,966).

**컬럼**: STUDYID, USUBJID, QSTESTCD, QSCAT, QSSTRESC, QSSTRESN, QSBLFL(이 3개 시험에서 전량 결측 확인됨, 사실상 미사용), VISITNUM, VISIT, QSDY.

**주의**: 행 단위가 "참가자×문항×방문"이라 한 참가자가 여러 행을 가짐. 그대로 분석에 쓰기보다는 이 파일을 기준으로 만든 wide 파일(3~6번)을 쓰는 게 맞음. 환자 단위 원자료 성격이 강해 팀 외부 공유(git 등)는 피할 것.

---

## 3. ds_wide.parquet

**내용**: 실측 Dependence Scale(DS) 결과를 참가자×방문 단위로 펼치고, Brickman 2002 규칙으로 단계까지 계산해둔 표.

**원천**: qs_subset_long.parquet 중 QSCAT='DEPENDENCE SCALE'.

**처리 과정**:
1. VISITNUM이 2.0(기저)·5.0(12주)·7.0(24주)인 행만 사용(DS는 4주 미수집).
2. QSTESTCD(DS01~DS14)를 컬럼으로 피벗(참가자×방문 단위로 wide화).
3. `ds_total` = DS01~DS13 합(13개 모두 있어야 계산, 하나라도 없으면 결측).
4. `ds_complete13` = DS01~DS13이 전부 관측됐는지 여부(True/False).
5. `ds_stage` = Brickman 규칙 적용값(계획서 표 2.2 그대로 구현):
   - 5단계: DS11·12·13 중 하나라도 1
   - 4단계: DS08·09·10 중 하나라도 1
   - 3단계: DS05·06·07 중 하나라도 1
   - 2단계: DS01~03 중 2개 이상 1, 또는 DS01/02 중 하나가 2, 또는 DS04=1
   - 1단계: 위 조건 미해당 + DS01~03 중 하나라도 1
   - 0단계: 전 항목 0
   - 13문항 중 하나라도 결측이면 단계도 결측(NaN).

**행 수**: 6,569행 (참가자×방문 조합 단위).

**컬럼**: STUDYID, USUBJID, VISITNUM, DS01~DS14(원점수), ds_total, ds_complete13, ds_stage, visit_label(baseline/week12/week24).

**검증**: 기저(VISITNUM=2.0) 중 ds_complete13=True인 인원이 AD-1061 668·AD-1063 835·AD-1064 705명으로 연구계획서 표 2.1과 완전히 일치함(단계별 분포까지 일치).

---

## 4. adl_wide.parquet

**내용**: ADCS-ADL 문항을 참가자×방문 단위로 펼친 표. 관문(gate) 문항의 응답상태(Y/N/Don't Know/빈칸)도 별도 컬럼으로 같이 둠.

**원천**: qs_subset_long.parquet 중 QSCAT='ADCS-ADL'.

**처리 과정**:
1. VISITNUM이 2.0(기저)·3.0(4주)·5.0(12주)·7.0(24주)인 행만 사용.
2. 관문 문항(예: ADL0106, ADL0108 등 원 코드)의 QSSTRESC를 3분기 처리해 `gate_status` 생성: Y→generated(하위값 사용), N→not_generated(0점 처리 대상), Don't Know/빈칸→missing. 나머지(관문이 아닌 일반 응답)는 other로 표시.
3. QSTESTCD를 컬럼으로 피벗해 두 세트를 만듦: 응답값 세트(수치)와 `__gate` 접미사가 붙은 관문상태 세트.

**행 수**: 9,683행.

**컬럼**: 총 91개. 문항 응답값 35개 내외(ADL0101~ADL0125 계열), 그에 대응하는 `_gate` 컬럼 다수, visit_label.

**결측 처리(2026-08-04 추가)**: 하위문항 원값 컬럼(예: ADL0120A)은 관문이 N(not_generated)이면 원자료에 해당 행 자체가 없어 결측으로 잡힘 — 이는 진짜 결측이 아니라 "물어보지 않은 것"임. 계획서 2.2의 3분기 규칙(Y→값 사용, N→0점, DK/빈칸→결측 유지)을 적용해 각 하위문항마다 `{코드}__resolved` 컬럼을 추가함. 예: ADL0120A는 원본 결측률 35.8% → resolved 결측률 1.0%로 감소(대부분이 관문 N으로 인한 구조적 결측이었음을 확인). A0/A1 채점, B1 모형 입력에는 원본 컬럼이 아니라 반드시 `__resolved` 컬럼을 써야 함.

**A0(2025판) 채점 완료(2026-08-04)**: 사용자가 업로드한 Chandler 2025 보충자료(mmc1.docx)에서 Table S6(SAS/R 원본 코드)·S7(문항별 응답-점수 매핑)·S8(단계별 판정규칙 서술)·S1(2015판 원 알고리즘 이미지)을 전부 확보함. Table S6의 R 코드를 그대로 파이썬으로 이식해 `adl_wide.parquet`에 `A0_2025_stage`(0~6)·`A0_harmonized`(5·6→5 병합, 0~5) 컬럼을 추가함.

CPAD 코드 ↔ 문헌 Q번호 대응은 QSTEST 라벨이 아니라 **QSORRES 응답 옵션 텍스트를 Table S7과 직접 대조**해 확인함(라벨만으로는 부정확한 경우가 있었음 — 예: ADL0106A는 라벨이 "Dressing Performance"이지만 실제 응답 문구는 Q6a '옷 고르기'와 일치). Q18(혼자 있기)이 ADL0118A+B+C의 합이라는 점은 R 코드 원문(`adl_q18 = adl_q18a + adl_q18b + adl_q18c`)에서 직접 확인함(추정 아님). 상세 대응표는 `urp-AD/adcs_adl_algorithm_source/` 폴더에 원본 자료와 함께 저장해둠.

**검증 결과** (기저, Aim1/2 표본 2,208명 기준): A0 산출 가능 2,203명(99.8%), 실측 DS(조화척도)와 가중카파(quadratic) 0.448, MAE 0.824, 완전일치율 39.4%, ±1단계 이내 82.8%. 참고용 수치이며 최종 Aim1 분석에서는 계획서에 정의된 정식 절차(공통표본, 안전성 지표 등)로 다시 계산해야 함.

**A0 후속 점검 결과(2026-08-04)**:

1. *규칙 공백 케이스*: 결측이 전혀 없는데도 0~6단계 중 어디에도 안 걸리는 사람이 2,515명 중 1명(0.04%) 존재함. Chandler 2025 논문 본문(4.2 한계)은 "Don't Know·결측으로 인한 제외 <2%"만 언급하고 이런 규칙 공백은 다루지 않음 — 원 알고리즘 자체의 미기술 영역으로 판단해 결측으로 유지하기로 함(임의 보간 안 함).
2. *Q18(혼자 있기) 잔여 결측 29명*: 시설입소 등 특수 로직 때문이 아니라 26명은 해당 방문에 관문 문항 자체가 원자료에 없었고(CRF 페이지 누락 추정), 3명은 모름/무응답인 통상적인 결측으로 확인됨. 추가 조치 불필요.
3. *문항 매핑 전수 재검증*: Q10~15·17·21~23 전부 QSORRES 응답문구를 Table S7과 자동대조(단어 자카드 유사도)해 확인함. Q15만 유사도가 낮게 나왔는데 이는 알고리즘이 mile, CPAD가 kilometre 단위를 써서 생긴 표기 차이일 뿐 매핑은 정확함.

5. *하위문항 사용·접미사 불일치·합산 로직 재검증(2026-08-04, 팀원 지적 검토)*: 코드가 관문이 아니라 지정된 하위문항(`ADL0107A`·`ADL0110A`·`ADL0121A`·`ADL0122Q`·`ADL0123L` 등)을 쓰는 게 맞는지, 특히 Q22·Q23은 다른 문항과 달리 하위코드 접미사가 "A"가 아니라 "Q"·"L"이라 별도 확인이 필요하다는 지적이 있었음. 원자료에서 QSTESTCD별 QSTEST 라벨과 QSORRES 응답문구를 직접 뽑아 Table S7과 대조함.

   | 문항 | CPAD 코드 | QSTEST 라벨 | 확인 결과 |
   |---|---|---|---|
   | Q7 전화 | ADL0107A | Highest Telephone Use Performance | 1~5점 응답문구 Table S7과 정확히 일치 |
   | Q10 설거지 | ADL0110A | Clear Dishes Usual Performance | 1~3점 응답문구 일치 |
   | Q21 글쓰기 | ADL0121A | Most Complicated Writings | 1~3점 응답문구 일치 |
   | Q22 취미 | ADL0122Q | Usual Common Pastimes Performance | 접미사 Q지만 라벨·응답문구로 Q22 확정, 1~3점 일치 |
   | Q23 가전기기 | ADL0123L | Use Common Used Appliances | 접미사 L이지만 라벨·응답문구로 Q23 확정, 1~4점 일치 |

   Q18(혼자있기) 합산(`ADL0118A+B+C`)은 R 원본 코드의 `adl_q18 = adl_q18a+adl_q18b+adl_q18c` 정의와 동일함을 재확인했고, Q16 "A>0 AND B=0" 손상 카운트(`shop_flag`)도 Chandler 원 코드의 `(adl_q16a>0 and adl_q16b=0)` 조건이 쓰이는 5개 지점(level5_6_GE4·level5_GE2·level4_GE3·level3_GE2_GE3·level1_2_GE2_GE3) 전부에 빠짐없이 반영돼 있음을 코드 대조로 재확인함. 수정 사항 없음.
4. *체계적 과대평가 경향(A0가 실측 DS보다 평균 +0.41단계 높게 판정, 조화척도 기준)*: 특정 문항 하나의 오류가 아니라 20개 입력문항 전부에서 고르게 나타나는 패턴(전부 음의 상관, p<0.001)이며, 그중 옷 고르기·입기(Q6a·Q6b)가 가장 강함. 초기 인지저하에서 먼저 흔들리는 복합과제(옷차림 결정 등)를 A0가 민감하게 포착하는 반면 실측 DS(신체돌봄 중심 척도)는 상대적으로 둔감하게 반응하는 것으로 해석됨 — 구현 오류가 아니라 두 척도의 개념적 차이로 판단, Aim1 결과로 보고할 사항.

원척도(`A0_2025_stage`, 0~6)와 조화척도(`A0_harmonized`, 0~5) 컬럼을 둘 다 유지함(하나로 합치지 않음).

**참고**: 2025판 입력 후보 문항(TV시청·대화참여·시사대화·5분독서 관문 및 하위코드 제외, 시설입소 여부·합계·Don't Know 집계 제외)은 25개로 확인됨(계획서는 19개로 명시 — 최종 매핑표 적용 시 문항 수 재확인 필요).

**bADL 문항 수 재확인(2026-08-04, 팀원 지적 검토)**: Table S1 이미지 우측 "ADL questions" 도식에서 Q1~Q5와 Q6A·Q6B가 하나의 괄호로 "Basic ADL (bADL)"에 묶여 있어(7문항으로 보일 수 있음) 팀원이 A0의 bADL 정의(6문항)가 틀렸다고 지적함. 이미지를 다시 정밀 확인한 결과, 같은 이미지 안에 **별도의 명시적 텍스트 목록**("Basic Activities Sub-Domain: Q1, Q2, Q3, Q4, Q5, Q6B" / "Household Sub-Domain: Q6A, Q7, Q10, Q11, Q12, Q13, Q14, Q23")이 있고, 이 목록이 Q6A를 Household(iADL 성격)로 명확히 분류함. 즉 이미지 좌측 괄호는 "개념적/거시적 bADL·iADL 구분"이고, 우측 텍스트 목록은 "Table 2 채점 규칙이 실제로 사용하는 서브도메인 구분"으로 서로 다른 두 분류 체계가 한 이미지에 같이 있음. A0(Chandler 2025)의 R 원본 코드(Table S6)가 `nmiss_badl`을 `q1,q2,q3,q4,q5,q6b` 6개로 명시적으로 정의하고 있어(원문 코드 직접 확인, 추정 아님), **A0는 수정 불필요** — 이미 올바르게 6문항 기준으로 산출돼 있었음. 다만 A1(2015 원판)은 Q6A를 "Household" 서브도메인으로 취급함(아래 A1 섹션 참고) — 이 역시 이미지의 우측 텍스트 목록과 일치.

---

## 4b. A1(2015 원판) 채점 — `adl_wide.parquet`에 컬럼 추가

**핵심 재해석(2026-08-04, 팀원 재검토로 확정)**: 이전에는 Kahle-Wrobleski 2015 논문 Table 2의 "Level 0/1/2" 표현을 (A0처럼) 별도로 파생되는 범주로 보고 "원문만으로는 재현 불가"로 결론 냈었음. 팀원이 본문을 재검토해 다음 근거로 "Level 0/1/2가 곧 ADCS-ADL 문항의 원점수 값 자체"라는 걸 확인함:

1. Level4 조건에서 "Score <2 for Q6B"와 "Level 1 or 0 for Q5, Q4"가 같은 줄에 병기됨 — 두 표현이 "점수값 0/1"이라는 동일한 뜻으로 혼용됨.
2. 각주 "Q20 only has a maximum level of 2"가 실제로 CPAD 원자료에서 ADL0120이 A/B 두 개의 0/1 이진 하위문항 합(즉 원점수 최댓값 2)인 것과 정확히 일치함(원자료로 직접 재확인).
3. Chandler 2025가 2015판의 결함으로 지적한 "문항 점수 2점 미만을 일괄 손상 처리"한 방식과 그대로 맞물림.

이 재해석에 따라 A1을 코드로 이식함(`compute_A1_2015()`). 사전 지정한 3가지 규칙(팀원 지시):
- 조건이 상호배타적이지 않아 **최고단계(5) 우선**으로 배정(5→4→3→2→1→0 순으로 검사, 처음 걸리는 단계로 확정).
- **이진(0/1) 문항은 "점수==2" 조건에 절대 도달 못 함** — 이는 원 알고리즘 자체의 결함이나, 임의로 고치지 않고 원문 그대로 재현함(팀원 명시 지시).
- **Q16B(구매대금 지불) 제외**, **Q20(독서)은 원점수+1로 보정**(각주 그대로).

**서브도메인 구성** (Table S1 이미지 우측 텍스트 그대로):
- Household: Q6A, Q7, Q10, Q11, Q12, Q13, Q14, Q23
- Communication: Q8, Q9, Q17, Q19, Q20(+1 보정), Q21, Q22
- Outside: Q15, Q16A, Q18 (Q16B 제외)
- Basic Activities(개별 문항으로 지칭): Q1, Q2, Q3, Q4, Q5, Q6B

**Table 2 규칙 그대로 이식**:
- Level5: Q2==1(보행) 또는 Q3==0(배변)
- Level4: Q6B<2 또는 Q5∈{0,1} 또는 Q4∈{0,1} 또는 Q3==2 또는 Q1==0
- Level3: Household∪Communication∪Outside 전 문항==2 또는 Outside 중 1개라도==0 또는 (Q1==2 또는 Q4==2)
- Level2: 3개 클러스터 중 2개 이상에서 "==2인 문항 존재" 또는 세 클러스터 통틀어 어떤 문항이든==1 또는 Household 중 1개라도==0
- Level1: 세 클러스터 통틀어 어떤 문항이든==2
- Level0: 위 어디에도 안 걸리면 기본값("No impairments")

**문항 원점수 파생(Chandler Table S7에 없는 4문항 — 2025판에서 제외된 TV·대화·시사·독서)**: 원자료 하위문항 구조를 직접 확인해(Q18과 동일한 합산 패턴) 다음과 같이 정의함 — Q8(TV)=ADL0108A+B+C__resolved(0~3), Q9(대화)=ADL0109A__resolved(0~3), Q19(시사)=ADL0119A+B+C__resolved(0~3), Q20(독서)=ADL0120A+B__resolved(0~2, 보정 전).

**결측 처리**: 원문에 결측 제외 규정이 없어 A0처럼 임의로 배제하지 않되, 관여 문항(bADL 6개+클러스터 18개=총 24개) 중 관측된 문항 수를 `A1_n_items_observed` 컬럼에 남김. 관측치가 전혀 없는 경우만 최종 결측(NaN) 처리.

**산출 컬럼**: `A1_2015_stage`(0~5), `A1_n_items_observed`, `A1_n_items_total`(=24).

**검증 결과(2026-08-04 파이프라인 실행)**:
- 산출 가능: 2,208/2,208명(100%, Aim1/2 기저 표본 기준).
- 기저 단계 분포: 0단계 0.6% · 1단계 3.0% · 2단계 39.7% · 3단계 31.7% · 4단계 23.8% · 5단계 1.2%.
- 참고용 대조(Wrobleski 2015 Table 4, GERAS 코호트 N=1,497, 모집단이 달라 정확 일치는 기대하지 않되 분포 형태 비교용): 0단계 0.7% · 1단계 2% · 2단계 30% · 3단계 26% · 4단계 34% · 5단계 7%. 양 끝(0·1·5단계)이 작고 중간(2~4단계)에 몰리는 전체적 형태는 일치하나, CPAD 표본이 2단계 쪽으로(39.7% vs 30%), GERAS는 4단계 쪽으로(34% vs 23.8%) 더 쏠려 있음 — CPAD가 3상 임상시험 등록 기준(경도~중등도) 특성상 상대적으로 덜 심한 쪽에 몰려 있을 가능성, GERAS는 관찰연구라 중등도가 더 많이 포함됐을 가능성 등으로 해석 가능(추가 확인 필요, 알고리즘 이식 오류로 단정할 근거는 없음).
- 실측 DS와의 일치도(기저, Aim1/2 표본): 가중카파(quadratic) 0.498, MAE 0.589, 완전일치율 51.4%, ±1단계 이내 90.9%. A0(가중카파 0.448, MAE 0.824)보다 실측 DS와의 일치도가 더 높게 나옴 — A1이 DS와 같은 0~5 척도라 조화 과정이 없다는 점, 그리고 원 논문 저자들이 애초에 DS와의 정합성을 염두에 두고 설계했을 가능성이 있음.

---

## 5. mmse_wide.parquet

**내용**: MMSE 총점을 참가자×방문 단위로 정리.

**원천**: qs_subset_long.parquet 중 QSCAT='MMSE' & QSTESTCD='MMSETOT'.

**처리 과정**: VISITNUM 2.0/5.0/7.0만 사용, 컬럼명을 mmse_total로 변경.

**행 수**: 4,925행.

**컬럼**: STUDYID, USUBJID, VISITNUM, visit_label, mmse_total.

---

## 6. supervision_time.parquet

**내용**: RUD(자원이용) 설문의 감독시간을 계획서 정의대로 "평균 일일 감독시간(분)"으로 환산한 표.

**원천**: qs_subset_long.parquet 중 QSCAT='RUDL', QSTESTCD가 RUA124A/RUA124B(기저 양식, 31일 회상)/RUB124A/RUB124B(추적 양식, 60일 회상)인 행.

**처리 과정**:
1. 참가자×방문 단위로 피벗.
2. 해당 행이 기저 양식(RUA)인지 추적 양식(RUB)인지 RUA124A/B 존재 여부로 판별.
3. `avg_daily_supervision_min` = 감독한 날의 1일 감독분(124A) × 감독일수(124B) ÷ 회상일수(기저 31, 추적 60).
4. `inconsistent_flag` = 시간이 0인데 일수가 양수, 또는 그 반대인 논리적 비정합 응답 표시(계획서 2.3 나 항목).

**행 수**: 7,241행.

**컬럼**: STUDYID, USUBJID, VISITNUM, RUA124A, RUA124B, RUB124A, RUB124B(원값), supervision_minutes_per_episode, supervision_days, recall_days, avg_daily_supervision_min, inconsistent_flag, visit_label.

**검증 결과(수정됨, 2026-08-04)**: 처음에 "무공급 비율 57.5%"로 계산해 계획서 값(54.5%)과 안 맞는다고 봤는데, 원인은 로직 버그가 아니라 비교 대상을 잘못 잡은 것이었음 — `avg_daily_supervision_min==0`(파생값)으로 재면 days=0일 때 minutes가 양수여도 0으로 계산돼 과대집계됨. 원본 두 값(`supervision_minutes_per_episode`, `supervision_days`)이 **둘 다 0**인 비율로 다시 재면 기저 시점 54.5%로 계획서와 정확히 일치함. `no_supervision_raw_flag` 컬럼을 이 정의로 추가함. 비정합 응답은 기저 시점만 보면 75건/2,518명(2.98%)으로 계획서가 말한 "약 3.1%"와 근접함(계획서의 282건은 표본 범위가 이 스크립트와 달라 절대 건수는 안 맞을 수 있음 — 비율은 일치).

---

## 7. baseline_sample.parquet

**내용**: 연구계획서 Aim 1·2가 쓰는 최종 기저 분석표본. 다른 파일들을 조합해 표본 포함 여부를 판정해둔 표.

**원천**: ds_wide.parquet(기저만) + adl_wide.parquet(기저 존재 여부) + dm_filtered.parquet(ARM/AGE/SEX).

**처리 과정**:
1. ds_wide에서 VISITNUM=2.0(기저)인 행만 추출.
2. adl_wide에서 VISITNUM=2.0에 해당 USUBJID가 하나라도 있으면 `has_adl_baseline`=True.
3. 두 표를 USUBJID로 병합(coalesce), dm_filtered에서 ARM/AGE/SEX 조인.
4. `in_aim1_2_sample` = ds_complete13(DS 13문항 완전관측) AND has_adl_baseline(동일 방문 ADCS-ADL 존재).

**행 수**: 2,271행(기저에 DS 기록이 하나라도 있는 사람 기준. 계획서 2.6의 "기저에 DS 기록이 하나라도 있는 사람" 표와 비교 시 AD-1061 685명과 유사한 수준일 것으로 예상되나 3개 시험 합계 기준이라 별도 시험별 분리 확인 필요).

**컬럼**: STUDYID, USUBJID, VISITNUM, DS01~DS14, ds_total, ds_complete13, ds_stage, visit_label, has_adl_baseline, ARM, AGE, SEX, in_aim1_2_sample.

**검증**: `in_aim1_2_sample`=True인 인원이 AD-1061 668·AD-1063 835·AD-1064 705명(합 2,208명)으로 연구계획서 수치와 정확히 일치함.

**계획서 v2 반영(2026-08-04 추가) — 공통 비교표본·Aim3 종단표본 컬럼**:

- `has_a0_baseline`, `has_a1_baseline`: 기저 시점에 A0(조화척도)·A1이 각각 산출됐는지 여부.
- `in_common_comparison_sample`: v2 2.1·2.3이 요구하는 "실측 DS·A0·A1·B1을 모두 산출할 수 있는" 공통 분석표본. B1(대안 채점법)은 아직 개발 전이라 현재는 DS+A0+A1 교집합만 반영함 — B1이 만들어지면 `has_b1_baseline`을 추가해 AND 조건을 넣어야 함(코드에 TODO로 표시해둠). 현재 값: 2,203명/2,208명(Aim1/2 표본 중 A0가 산출 안 되는 5명만 빠짐, A1은 전원 산출됨).
- `ds_complete_wk12`, `ds_complete_wk24`: 12주·24주 시점 DS13 완전관측 여부.
- `in_aim3_longitudinal_sample`: 기저 Aim1/2 표본이면서 12주 또는 24주 중 하나라도 DS 완전관측 — **2,127명, 계획서 v2 표 2.1의 목표치(2,127명)와 정확히 일치**.
- `in_aim3_24wk_completer_sample`: 기저 Aim1/2 표본이면서 24주까지 DS 완전관측(24주 완전사례) — **2,045명, 계획서 v2 목표치(2,045명)와 정확히 일치**.

---

## 8. extra_validation_ad1062/ (계획서 v2 2.1 — AD-1062 공개연장시험)

**내용**: AD-1062(공개연장시험, open-label extension)의 DM/QS/DS/ADCS-ADL(A0·A1 포함) 자료. **메인 3개 시험(AD-1061/1063/1064)과 완전히 분리된 별도 파일**로만 존재하며, `baseline_sample.parquet` 등 메인 표본 정의에는 전혀 섞이지 않음.

**분리 이유(계획서 v2 원문)**: "공개연장시험 AD-1062는 참가자 중복을 배제할 수 없어 학습·튜닝·교차검증에서 제외하고, 최종 모형의 추가 성능평가에만 사용한다(독립적 외부검증으로 격상시키지 않는다)."

**중요 주의사항 — 방문번호 체계가 메인 3개 시험과 다름**: AD-1062는 DM/QS 추출은 메인과 동일한 방식(grep 사전필터+pandas)을 썼지만, VISITNUM을 메인 시험의 {2.0=기저, 3.0=4주, 5.0=12주, 7.0=24주} 체계로 필터링하면 **행이 전부 사라짐**(실제 확인함). AD-1062의 VISITNUM 구성은 다음과 같이 전혀 다른 체계임(연장시험이라 "2차 기저"부터 시작).

| VISITNUM | VISIT 라벨 |
|---|---|
| 1.0 | Visit 1 (Baseline II) |
| 4.0 | Visit 4 |
| 6.0 | Visit 6 (Completion/Withdrawal) |
| 10.0 | Visit 10 |
| 11.0 | Visit 11 (Completion/Withdrawal) |

그래서 `build_ds_wide`/`build_adl_wide`에 `visit_filter=None`을 넘겨 방문번호 필터 없이 전부 추출함(`visit_label` 컬럼은 매핑 대상이 아니라 전부 NaN으로 남음). **"기저"를 어느 시점(VISITNUM=1.0 "Baseline II"로 볼지, 아니면 다른 기준이 필요한지)으로 정의할지는 아직 확정하지 않았고, 실제 추가 성능평가 설계 시 재확인이 필요함.** 참고로 VISITNUM=1.0(Baseline II) 시점 DS13 완전관측자는 1,400명/1,463명(등록자)임.

**파일 목록**:

| 파일 | 행 수 | 내용 |
|---|---|---|
| `dm_ad1062.parquet/csv` | 1,463 | AD-1062 등록자 DM |
| `qs_subset_ad1062_long.parquet/csv` | 501,953 | AD-1062의 QS 4개 카테고리(DS/ADCS-ADL/MMSE/RUDL), long 포맷, 방문번호 필터 없음 |
| `ds_wide_ad1062.parquet/csv` | 4,368 | AD-1062 DS wide, ds_stage 포함(방문번호 필터 없음) |
| `adl_wide_ad1062.parquet/csv` | 4,290 | AD-1062 ADCS-ADL wide + A0_2025_stage/A0_harmonized + A1_2015_stage(방문번호 필터 없음) |

**컬럼**: 메인 3개 시험의 동명 파일과 동일한 컬럼 구조(단, visit_label만 미매핑으로 전부 결측).

---

## 결측치 처리 원칙 요약 (전체 파이프라인 공통)

결측을 실제로 값으로 채운 곳은 **딱 한 군데**(ADCS-ADL 관문 N 케이스)뿐이고, 나머지는 전부 결측을 결측 그대로 남겨둠. 임의 대치를 최소화하는 게 원칙임.

| 대상 | 결측 발생 상황 | 처리 | 근거 |
|---|---|---|---|
| 실측 DS 13문항 | 13개 중 하나라도 결측 | 전체 결측 처리(대치 안 함) | 계획서 2.2 — 직접 준거를 임의 수정 금지 |
| ADCS-ADL 하위문항 (관문 Y) | 정상 상황인데 값 없음 | 결측 유지 | — |
| ADCS-ADL 하위문항 (관문 N) | 애초에 안 물어봄(원자료에 행 없음) | **0점으로 채움** | 계획서 2.2 3분기 규칙 |
| ADCS-ADL 하위문항 (관문 DK/무응답) | 몰라서 답 안 함 | 결측 유지 | 계획서 2.2 — 0으로 채우면 과소평가 |
| A0(2025판) 점수 | iADL 15개 중 3개↑ 결측 또는 bADL 6개 중 1개↑ 결측 | 그 사람 A0 전체 결측 | Chandler 2025 원 알고리즘(Table S6) 그대로 |
| RUD 감독시간 | 시간/일수 원본 결측 | 대치 안 함, 플래그만 생성(무공급/비정합) | 통계분석 단계(두 부분 모형)에서 처리하도록 원자료 보존 |
| MMSE, DM | — | 원자료 그대로, 별도 처리 없음 | — |

## 전체 파일 간 관계 요약

```
dm.csv ──────────────────┐
                          ├─→ dm_filtered.parquet
qs.csv (2GB, 12.4M행) ────┤
  │ (STUDYID+QSCAT 필터)  │
  └──→ qs_subset_long.parquet
           │
           ├─→ ds_wide.parquet ─────┐
           ├─→ adl_wide.parquet ────┼─→ baseline_sample.parquet
           ├─→ mmse_wide.parquet    │      (+ dm_filtered.parquet)
           └─→ supervision_time.parquet
```

## 남은 작업 (우선순위순)

1. baseline_sample의 2,271명(기저 DS 기록 보유자)을 시험별로 나눠 계획서 표(AD-1061 685명 등)와 대조.
2. A0/A1 모두 산출 완료 — 이제 두 알고리즘을 실측 DS와 함께 Aim1 정식 분석(계획서에 정의된 공통표본·통계모형)에 투입하는 단계로 넘어갈 수 있음.
