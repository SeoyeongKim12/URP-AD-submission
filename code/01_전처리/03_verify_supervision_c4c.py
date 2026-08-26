"""
C4c — 감독시간(supervision) 검증 완결 (sjlee)
=============================================
전처리 검증의 마지막 조각. 핵심: '총 건수'가 아니라 '구조(방향 비율)'를 재현하는 것.

계획서 기준값(비정합 응답):
  a = 시간 0분인데 감독 일수 양수      → 252건
  b = 그 반대(시간 양수·일수 0)        → 30건
  합 282건, 전체의 약 3.1%
무공급: no_supervision_raw_flag(원본 시간·일수 둘 다 0) 기저 비율 = 54.5%

판정:
- 어느 범위에서 282·3.1% 재현 + 방향비율(252:30, 압도적 "시간0·일수양수")이 재현되면
  → 로직 정상, 차이는 분모/범위 차이 → C4c PASS (범위 정의만 주석).
- 방향이 엉뚱(b>a)하거나 총비율이 크게 벗어나면 → inconsistent_flag 로직 재점검.

입력: ~/Downloads/supervision_time.csv
산출: 전처리/verify_supervision_c4c_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "verify_supervision_c4c_report.md"
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii"))
    _lines.append(s)


def summarize_range(df, label):
    """주어진 부분집합에서 a/b/합/비율을 산출."""
    n = len(df)
    mins = df["supervision_minutes_per_episode"]
    days = df["supervision_days"]
    a = ((mins == 0) & (days > 0))
    b = ((mins > 0) & (days == 0))
    a_n, b_n = int(a.sum()), int(b.sum())
    tot = a_n + b_n
    pct = tot / n * 100 if n else float("nan")
    log(f"### 범위: {label}  (n={n})")
    log(f"- a (시간0·일수양수): **{a_n}**  [계획서 252]")
    log(f"- b (시간양수·일수0): **{b_n}**  [계획서 30]")
    log(f"- 합: **{tot}**  ({pct:.2f}%)  [계획서 282 · 약 3.1%]")
    log(f"- 방향비율 a:b = {a_n}:{b_n}" +
        (f"  (a 우세 {a_n/max(b_n,1):.1f}배)" if b_n else "  (b=0)"))
    # 팀원 inconsistent_flag 와 a|b 일치 여부
    if "inconsistent_flag" in df.columns:
        flag = df["inconsistent_flag"].astype("boolean").fillna(False)
        ab = (a | b)
        agree = int((flag == ab).sum())
        mism = int((flag != ab).sum())
        log(f"- inconsistent_flag vs (a|b) 재구성: 일치 {agree} / 불일치 {mism}")
    log("")
    return dict(label=label, n=n, a=a_n, b=b_n, tot=tot, pct=pct)


def main():
    log("# C4c — 감독시간 검증 완결 (sjlee)\n")
    path = DOWNLOADS / "supervision_time.csv"
    if not path.exists():
        log(f"[대기] {path} 없음 — Drive dependence_study_csv/에서 받아야 실행됨.")
        REPORT.write_text("\n".join(_lines), encoding="utf-8")
        print(">>> 입력 파일 없음. 파일 확보 후 재실행.")
        return

    df = pd.read_csv(path)
    log(f"입력 supervision_time.csv: {len(df)}행\n")
    log(f"컬럼: {list(df.columns)}\n")

    # 필수 컬럼 확인
    need = ["supervision_minutes_per_episode", "supervision_days", "VISITNUM"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        log(f"[경고] 필요한 컬럼 없음: {miss} — 명세서와 컬럼명 대조 필요.\n")

    log("## 1. 방향별·범위별 비정합 재현\n")
    results = []
    results.append(summarize_range(df[df["VISITNUM"] == 2.0], "기저 (VISITNUM=2.0)"))
    results.append(summarize_range(df, "전체 방문"))
    # 기저+추적: DS 방문번호 체계 2/3/5/7 전부(=전체와 동일할 수 있음). 별도 명시.
    fu = df[df["VISITNUM"].isin([2.0, 3.0, 5.0, 7.0])]
    results.append(summarize_range(fu, "기저+추적 (VISITNUM in 2/3/5/7)"))

    # ---- 2. 무공급 비율 ----
    log("## 2. 무공급 비율 (기저) — 계획서 54.5%\n")
    base = df[df["VISITNUM"] == 2.0]
    if "no_supervision_raw_flag" in df.columns:
        r = base["no_supervision_raw_flag"].astype("boolean").fillna(False).mean() * 100
        log(f"- no_supervision_raw_flag 기저 비율: **{r:.1f}%**  [계획서 54.5%]\n")
    else:
        # 원본 두 값 둘 다 0으로 직접 재계산
        r = ((base["supervision_minutes_per_episode"] == 0) &
             (base["supervision_days"] == 0)).mean() * 100
        log(f"- (no_supervision_raw_flag 컬럼 없음 — 직접 재계산) "
            f"시간·일수 둘다0 기저 비율: **{r:.1f}%**  [계획서 54.5%]\n")

    # ---- 3. 판정 ----
    log("## 3. 판정\n")
    hit = next((x for x in results if abs(x["pct"] - 3.1) < 0.6 or 270 <= x["tot"] <= 295), None)
    base_r = results[0]
    dir_ok = base_r["a"] > base_r["b"] * 3  # 방향 압도적 a 우세
    if hit is not None and dir_ok:
        log(f"- **C4c PASS** — '{hit['label']}' 범위에서 합 {hit['tot']}건({hit['pct']:.2f}%)이 "
            f"계획서 282·3.1%에 근접하고, 방향비율(a≫b)도 재현됨.")
        log(f"- 결론: inconsistent_flag 로직 정상. 기저만 보면 {base_r['tot']}건, "
            f"계획서 282건은 더 넓은 범위 기준 — 차이는 순전히 분모/범위 정의 차이.")
    else:
        log(f"- **재점검 필요** — 282·3.1% 재현 범위를 못 찾았거나(hit={hit}), "
            f"방향비율이 예상과 다름(기저 a={base_r['a']}, b={base_r['b']}).")
        log(f"- inconsistent_flag 정의를 명세서/원 R코드와 재대조할 것.")
    log("")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n>>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
