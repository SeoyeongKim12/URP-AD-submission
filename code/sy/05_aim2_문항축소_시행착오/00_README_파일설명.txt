aim2/시행착오 폴더 파일 설명
================================================================

■ 코드 (2개)

b1_dominate_a0.py
  B1(부분비례오즈 엘라스틱넷)이 A0를 지배하는 지점을 탐색하는 원본 분석 코드.
  b2가 import해서 D.build()·D.ITEMS·D.asym·D.mae·D.miss·D.kap 등을 그대로 씀.
  단독 실행하면 b1_dominate_a0_report.md·b1_dominate_a0_scoretable.csv 생성(현재
  폴더엔 아직 실행 안 해서 없음 — 필요하면 python b1_dominate_a0.py로 실행).

b2_nested_randomsearch.py
  지금 쓰고 있는 최신판. 문항수 5~18개에서 무작위 300조합을 중첩CV로 평가.
  6a+6b 병합(merged)/비병합(unmerged) 두 버전을 한 번에 다 돌려서 아래 CSV·MD들을
  전부 생성함. 이 파일 하나가 밑에 있는 b2_* 결과 파일 전체(변형 comparison 제외 각
  변형당 5개)의 생성 주체.

■ 결과 — 현재판 (b2_nested_randomsearch.py 최신 실행, k=5~18·N=300·2변형)

b2_nested_randomsearch_report_merged.md / _unmerged.md
  사람이 읽는 보고서. 문항수별 성능표 + fold별 선택조합(10/8/6문항만 본문표시) +
  요약 텍스트. merged=6a+6b 합친 버전, unmerged=원래대로 분리된 버전.

b2_performance_by_k_merged.csv / _unmerged.csv
  문항수(n_items)별 성능 요약표. MAE·중증놓침%·카파·A1대비MAE차이·champion(A1·A0
  기준 동시통과 여부). "몇 문항이 적당한가" 볼 때 제일 먼저 볼 파일.

b2_selected_items_by_fold_merged.csv / _unmerged.csv
  (문항수 k) × (바깥fold 3개) 조합마다 실제로 어떤 문항들이 선택됐는지 원자료
  그대로(긴 표, 문항 코드+한글라벨). report.md엔 10/8/6문항만 나오는데 이 csv엔
  k=5~18 전체가 다 있음.

b2_common_items_by_k_merged.csv / _unmerged.csv
  각 k에서 3개 outer fold가 공통으로 고른 문항이 뭔지 요약. "어떤 문항이 항상
  뽑히는가"(화장실·목욕·외출 등 핵심문항 확인용) 볼 때 쓰는 파일.

b2_outer_predictions_merged.csv / _unmerged.csv
  환자단위(STUDYID·USUBJID) wide표. 실제 ds_stage·A1_2015_stage·A0_harmonized +
  k=5~18 각각의 outer 예측값(pred_k5~pred_k18)이 전부 들어있음. 다른 지표를
  새로 계산하고 싶을 때(예: 완전정답률처럼) 원본 재실행 없이 이 파일에서 바로
  뽑아 쓰면 됨 — 실제로 b2_exact_accuracy_*.csv가 이 파일에서 파생됨.

b2_exact_accuracy_merged.csv / _unmerged.csv
  b2_outer_predictions_*.csv에서 파생. 문항수별 완전정답률(exact_acc_pct)·
  ±1단계이내 정답률(within1_acc_pct)·MAE 비교표. "MAE 좋아 보이는 게 착시 아니냐"
  질문에 답하려고 추가로 뽑은 파일.

b2_variant_comparison.csv
  merged vs unmerged를 같은 k끼리 나란히 비교한 표(MAE·중증놓침·카파·champion
  각각 _merged/_unmerged 접미사로 병기). "병합이 나은가 비병합이 나은가"를 k별로
  한눈에 보려면 이 파일.

■ 1차_k5-12_n200_병합버전only/ (구버전, 참고용 — 폴더 분리해둠)

  아직 6a+6b 병합/비병합 분기가 없던 이전 버전 실행 결과(k=5~12, N=200, 병합
  버전만). 지금은 위 최신판이 이 범위를 포함해서 다 덮으니 실질적으로 안 봐도 됨,
  헷갈리지 않게 따로 빼놓기만 함. 지워도 무방.

■ 참고 — 아직 안 만든 것

b1_dominate_a0_report.md / b1_dominate_a0_scoretable.csv
  b1_dominate_a0.py를 단독 실행하면 나오는 산출물(B1 자체의 A0 지배 검증 리포트+
  최종 채점표). 지금은 b2가 import만 하고 있어서 이 폴더에 아직 없음.
