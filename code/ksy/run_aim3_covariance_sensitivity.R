# ============================================================
# Aim3 다절 — 반복측정 혼합모형 + 5-1. 공분산구조 민감도분석 (로컬 R)
#
# 사전 준비: Colab 노트북 "5-0. (로컬 R용) 장기 포맷 데이터 CSV로 내보내기" 셀을
# 실행해서 아래 두 파일을 만든 뒤, 로컬 preprocessed 폴더로 다운로드해두세요.
#   - aim3_long_ds_total.csv
#   - aim3_long_b2_expected.csv
# ============================================================

# ---- 0. 패키지 설치(최초 1회만) & 로드 ----

# R < 4.4.0에서는 %||% 연산자가 base에 없어서 최근 lme4 버전과 충돌할 수 있음
# (에러: 함수 "%||%"를 찾을 수 없습니다) → 없을 때만 직접 정의해서 방지
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

if (!require("lme4", quietly = TRUE))     install.packages("lme4")
if (!require("lmerTest", quietly = TRUE)) install.packages("lmerTest")
library(lme4)
library(lmerTest)

# ---- 1. 데이터 경로 ----
DATA_DIR <- "C:/Users/82109/Downloads/c-path/preprocessed"  # 역슬래시 대신 슬래시 사용 (R 관례, 둘 다 동작함)

ds_path <- file.path(DATA_DIR, "aim3_long_ds_total.csv")
b2_path <- file.path(DATA_DIR, "aim3_long_b2_expected.csv")

if (!file.exists(ds_path) || !file.exists(b2_path)) {
  stop(paste0(
    "CSV를 찾을 수 없습니다. Colab에서 '5-0. 로컬 R용 CSV 내보내기' 셀을 먼저 실행하고,\n",
    "  aim3_long_ds_total.csv / aim3_long_b2_expected.csv 를 아래 경로에 다운로드하세요:\n  ",
    DATA_DIR
  ))
}

long_ds_total <- read.csv(ds_path, stringsAsFactors = FALSE)
long_b2_exp   <- read.csv(b2_path, stringsAsFactors = FALSE)

cat("long_ds_total:", nrow(long_ds_total), "행\n")
cat("long_b2_exp  :", nrow(long_b2_exp), "행\n")

# ---- 2. 형변환 (범주형) ----
prep <- function(df) {
  df$Visit    <- factor(df$Visit, levels = c("V2", "V5", "V7"))
  df$STUDYID  <- factor(df$STUDYID)
  if ("Treatment" %in% colnames(df)) df$Treatment <- factor(df$Treatment)
  df
}
long_ds_total <- prep(long_ds_total)
long_b2_exp   <- prep(long_b2_exp)

build_formula <- function(df) {
  if ("Treatment" %in% colnames(df)) {
    Y_std ~ Visit + STUDYID + Treatment + (1 | USUBJID)
  } else {
    Y_std ~ Visit + STUDYID + (1 | USUBJID)
  }
}

# ---- 3. 주분석: random intercept만 ----
model_ds <- lmer(build_formula(long_ds_total), data = long_ds_total)
model_b2 <- lmer(build_formula(long_b2_exp),   data = long_b2_exp)

cat("\n========== [실측 DS총점] 주분석 (random intercept) ==========\n")
print(summary(model_ds))
cat("\n95% Wald CI:\n")
print(confint(model_ds, method = "Wald"))

cat("\n========== [B2 기대단계] 주분석 (random intercept) ==========\n")
print(summary(model_b2))
cat("\n95% Wald CI:\n")
print(confint(model_b2, method = "Wald"))

# ---- 4. 5-1. 공분산구조 민감도분석: random slope 추가 ----
# ⚠️ 주의: Visit을 범주형(V2/V5/V7)으로 random slope에 넣으면
#   참가자당 임의효과가 3개(절편+V5기울기+V7기울기) 필요한데,
#   2시점만 관측된 참가자가 있어(전체의 약 10%) 임의효과 수가 관측치 수를 넘어서면
#   "number of observations <= number of random effects" 에러로 실행 자체가 안 됩니다.
#   → 연속시간(time_num: 0/12/24)으로 바꿔 참가자당 임의효과를 2개(절편+기울기)로
#     줄이면 식별 가능해집니다.

add_time_num <- function(df) {
  df$time_num <- c(V2 = 0, V5 = 12, V7 = 24)[as.character(df$Visit)]
  df
}
long_ds_total <- add_time_num(long_ds_total)
long_b2_exp   <- add_time_num(long_b2_exp)

build_formula_rs <- function(df) {
  if ("Treatment" %in% colnames(df)) {
    Y_std ~ Visit + STUDYID + Treatment + (1 + time_num | USUBJID)
  } else {
    Y_std ~ Visit + STUDYID + (1 + time_num | USUBJID)
  }
}

model_ds_rs <- lmer(build_formula_rs(long_ds_total), data = long_ds_total,
                    control = lmerControl(optimizer = "bobyqa"))
model_b2_rs <- lmer(build_formula_rs(long_b2_exp), data = long_b2_exp,
                    control = lmerControl(optimizer = "bobyqa"))

cat("\n========== 5-1. AIC 비교: 실측 DS총점 (intercept-only vs +random slope[연속시간]) ==========\n")
print(AIC(model_ds, model_ds_rs))

cat("\n========== 5-1. AIC 비교: B2 기대단계 (intercept-only vs +random slope[연속시간]) ==========\n")
print(AIC(model_b2, model_b2_rs))

cat("\n※ 해석: AIC가 낮을수록 더 나은 모형. random slope 모형의 AIC가 intercept-only보다\n")
cat("  뚜렷이 낮으면 참가자별로 시간에 따른 변화 기울기가 다르다는 뜻(compound symmetry 위반).\n")
cat("  차이가 미미하면 주분석(intercept-only)을 그대로 유지해도 무방.\n")
cat("  (참고: 범주형 Visit random slope는 2시점-전용 참가자 때문에 식별 불가 — 연속시간으로 대체함)\n")

# ---- 5. (선택) 잔차 진단 플롯 ----
# 그래프 창이 뜹니다. 필요 없으면 이 블록은 주석 처리하세요.
plot(model_ds_rs, main = "실측 DS총점 (random slope) 잔차")
plot(model_b2_rs, main = "B2 기대단계 (random slope) 잔차")

# ---- 6. 결과 텍스트 파일로 저장 ----
out_path <- file.path(DATA_DIR, "aim3_다절_결과_R_local.txt")
sink(out_path)
cat("===== [실측 DS총점] 주분석 =====\n")
print(summary(model_ds))
print(confint(model_ds, method = "Wald"))

cat("\n===== [B2 기대단계] 주분석 =====\n")
print(summary(model_b2))
print(confint(model_b2, method = "Wald"))

cat("\n===== 5-1. AIC 비교: 실측 DS총점 =====\n")
print(AIC(model_ds, model_ds_rs))

cat("\n===== 5-1. AIC 비교: B2 기대단계 =====\n")
print(AIC(model_b2, model_b2_rs))
sink()

cat("\n결과 저장 완료:", out_path, "\n")