# data/ — 데이터 안내 (Data Availability)

**⚠️ 환자 원본 데이터는 이 저장소에 없습니다**

본 연구의 입력 데이터는 CPAD(Critical Path for Alzheimer's Disease) 통합 3상 임상시험 자료(AD-1061·AD-1063·AD-1064)입니다. 이 데이터는 데이터 사용협약(DUA) 아래 제공되어, 환자단위 원본을 공개 저장소에 배포할 수 없습니다.

따라서 이 `data/` 폴더에는 데이터의 설명과 접근 방법만 담습니다.

## 폴더 구성

| 항목 | 내용 |
|---|---|
| `명세서/` | 데이터 파일·컬럼 명세서 (전처리 산출물 + 파생지표 A0/A1/DS 정의) |
| `algorithm_source/` | A0(Chandler 2025)·A1(2015) 채점 알고리즘 원천 자료 (Table S1·S6·S7·S8) |

## 실제 데이터를 받으려면 (팀 내부)

팀 드라이브에 환자단위 데이터가 두 종류로 나뉘어 있습니다 — **원자료(입력)**와 **코드 실행 산출물(출력)** 입니다.

### 1) 원자료 — `전처리 > 전처리 산출물`

`code/`의 모든 분석 스크립트가 입력으로 쓰는 파일들입니다.

```
드라이브/전처리/전처리 산출물/
  baseline_sample.csv, adl_wide.csv, ds_wide.csv, mmse_wide.csv,
  supervision_time.csv, dm_filtered.csv, qs.csv(원자료) 등
```

재현 시: 위 파일들을 받아 `~/Downloads/`에 그대로 두고 `code/`의 스크립트를 순서대로 실행하세요.

### 2) 코드 실행 산출물 — `URP-AD-submission_patient_level/`

`code/02_aim1/`, `code/03_aim2/`의 스크립트를 실행하면 각 스크립트 안에서 환자단위 중간 산출물(개인정보 포함이라 git에 올릴 수 없는 파일)이 자동으로 생성됩니다. 이 파일들은 `.gitignore`로 저장소에서 제외되며, 아래 구조로 드라이브에 별도 보관합니다.

```
드라이브/URP-AD-submission_patient_level/
  02_aim1/
    aim1_common_sample_scored.csv   (가: 공통표본 채점표)
    aim1_supervision_analysis.csv   (나: 감독시간+지표 분석표)
    aim1_ad1061_ipw_weights.csv     (M1-b IPW 가중치표)
  03_aim2/
    gn_en_cv_predictions.csv        (Gn 엘라스틱넷 CV 예측값)
    b2_dev_test_split_RECON.csv     (B2 dev/test 분할 배정표, 환자단위)
```

폴더명(`02_aim1`, `03_aim2`)은 `code/`, `result/`의 폴더 번호와 동일하게 맞췄습니다 — 어떤 스크립트가 어떤 파일을 만들었는지 이름만으로 대조할 수 있게 하기 위함입니다. (참고: 로컬 저장소 안에서는 이 파일들이 `code/02_aim1/aim1_patient_level/`, `code/03_aim2/aim2_patient_level/` 폴더에 그대로 남아있고, `.gitignore`가 이 폴더 전체를 git 추적에서 차단합니다. 드라이브에는 그 안의 내용물만 위 구조로 옮겨 올립니다.)

자세한 컬럼·실행 경로는 `명세서/` 문서를 참고하세요.

## 원칙

행 = 환자(USUBJID 포함)인 파일은 git에 올리지 않습니다. 이 저장소의 `result/` CSV는 전부 집계·계수·규칙 표(환자 식별 불가)입니다.
