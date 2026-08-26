# data/ — 데이터 안내 (Data Availability)

## ⚠️ 환자 원본 데이터는 이 저장소에 없습니다
본 연구의 입력 데이터는 **CPAD(Critical Path for Alzheimer's Disease) 통합 3상 임상시험 자료**(AD-1061·AD-1063·AD-1064)입니다. 이 데이터는 **데이터 사용협약(DUA)** 아래 제공되어, **환자단위 원본을 공개 저장소에 배포할 수 없습니다.**

따라서 이 `data/` 폴더에는 **데이터의 설명과 접근 방법**만 담습니다.

## 폴더 구성
| 항목 | 내용 |
|---|---|
| `명세서/` | 데이터 파일·컬럼 명세서 (전처리 산출물 + 파생지표 A0/A1/DS 정의) |
| `algorithm_source/` | A0(Chandler 2025)·A1(2015) 채점 알고리즘 원천 자료 (Table S1·S6·S7·S8) |

## 실제 데이터를 받으려면 (팀 내부)
환자단위 CSV(`baseline_sample.csv`, `adl_wide.csv`, `ds_wide.csv`, `mmse_wide.csv`, `supervision_time.csv`, 원자료 `qs.csv` 등)는 **URP 팀 드라이브 `전처리 > 전처리 산출물`** 에 있습니다(접근 권한자만).
- 재현 시: 드라이브에서 위 파일들을 받아 **`~/Downloads/`** 에 둔 뒤 `code/` 의 스크립트를 실행하세요.
- 자세한 컬럼·실행 경로는 `명세서/` 문서를 참고하세요.

## 원칙
> 행 = 환자(`USUBJID` 포함)인 파일은 **git에 올리지 않습니다.** 이 저장소의 `result/` CSV는 전부 **집계·계수·규칙 표**(환자 식별 불가)입니다.
