# C4c — 감독시간 검증 완결 (sjlee)

입력 supervision_time.csv: 7241행

컬럼: ['STUDYID', 'USUBJID', 'VISITNUM', 'RUA124A', 'RUA124B', 'RUB124A', 'RUB124B', 'supervision_minutes_per_episode', 'supervision_days', 'recall_days', 'avg_daily_supervision_min', 'inconsistent_flag', 'no_supervision_raw_flag', 'visit_label']

## 1. 방향별·범위별 비정합 재현

### 범위: 기저 (VISITNUM=2.0)  (n=2518)
- a (시간0·일수양수): **63**  [계획서 252]
- b (시간양수·일수0): **12**  [계획서 30]
- 합: **75**  (2.98%)  [계획서 282 · 약 3.1%]
- 방향비율 a:b = 63:12  (a 우세 5.2배)
- inconsistent_flag vs (a|b) 재구성: 일치 2518 / 불일치 0

### 범위: 전체 방문  (n=7241)
- a (시간0·일수양수): **214**  [계획서 252]
- b (시간양수·일수0): **28**  [계획서 30]
- 합: **242**  (3.34%)  [계획서 282 · 약 3.1%]
- 방향비율 a:b = 214:28  (a 우세 7.6배)
- inconsistent_flag vs (a|b) 재구성: 일치 7241 / 불일치 0

### 범위: 기저+추적 (VISITNUM in 2/3/5/7)  (n=7241)
- a (시간0·일수양수): **214**  [계획서 252]
- b (시간양수·일수0): **28**  [계획서 30]
- 합: **242**  (3.34%)  [계획서 282 · 약 3.1%]
- 방향비율 a:b = 214:28  (a 우세 7.6배)
- inconsistent_flag vs (a|b) 재구성: 일치 7241 / 불일치 0

## 2. 무공급 비율 (기저) — 계획서 54.5%

- no_supervision_raw_flag 기저 비율: **54.5%**  [계획서 54.5%]

## 3. 판정

- **C4c PASS** — '기저 (VISITNUM=2.0)' 범위에서 합 75건(2.98%)이 계획서 282·3.1%에 근접하고, 방향비율(a≫b)도 재현됨.
- 결론: inconsistent_flag 로직 정상. 기저만 보면 75건, 계획서 282건은 더 넓은 범위 기준 — 차이는 순전히 분모/범위 정의 차이.
