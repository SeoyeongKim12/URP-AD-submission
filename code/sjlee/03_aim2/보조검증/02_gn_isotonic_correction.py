"""
Aim 2 · Gn 비단조 확률 등위회귀(PAVA) 보정 (sjlee) — 실제 구현판
==================================================================
배경: 부분비례오즈(절단별 독립 로지스틱)는 P(Y>=k)가 k에서 증가(비단조)할 수 있음.
발표에서 'PAVA로 위반 20.9%→0%, 성능 유지'로 나갔으나 실제 실행 코드가 없었음(미착수).
이 스크립트로 **실제로 구현·실행**한다.

방법(정보누출 0):
- leave-one-trial-out 3-fold. 각 fold 훈련 2시험으로 Gn(C=0.1) 적합 → 평가시험 P(Y>=1..5).
- 환자별 PAVA(등위회귀, 감소 강제)로 누적확률을 단조감소로 투영 → 계단확률 재계산.
- τ(비대칭임계)는 훈련 raw에서 도출해 raw·isotonic에 **동일 적용**(공정 비교).
- raw vs isotonic: 비단조 위반율·음수질량 + 4지표(MAE·가중카파·4·5→≤3·2↑과소).

기대(정직 서술):
- isotonic 위반율 0%는 **원리상 보장**(PAVA가 단조 강제) — 발표값과 자동 일치.
- 성능은 위반 크기가 미미하면 거의 불변 예상. **다만 수치는 나온 그대로 기록**.

산출: gn_isotonic_correction_report.md
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "시행착오"))
import b1_gn_elasticnet as G     # build_matrix, fit_en, asym_rule, tune_tau, m_mae, m_hi, m_u2

OUTDIR = Path(__file__).parent
REPORT = OUTDIR / "gn_isotonic_correction_report.md"
TRIALS = G.TRIALS
C = 0.1               # 배포모형 벌점(발표 확정값)
MARGIN = 0.14
_lines = []
def log(s=""):
    print(s.encode("ascii", "replace").decode("ascii")); _lines.append(s)

from sklearn.metrics import cohen_kappa_score
def mae(t, p): return np.abs(np.asarray(t, float) - np.asarray(p, float)).mean()
def miss(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float); hi = t >= 4
    return (p[hi] <= 3).mean() if hi.sum() else np.nan
def u2(t, p): return (np.asarray(t, float) - np.asarray(p, float) >= 2).mean()
def kap(t, p): return cohen_kappa_score(np.asarray(t, int), np.asarray(p, int),
                                        weights="quadratic", labels=list(range(6)))

def asym(P, t4, t5):
    p5 = P[:, 5]; p4 = P[:, 4] + P[:, 5]
    st = np.arange(6)
    cmed = st[(np.cumsum(P, 1) >= 0.5).argmax(1)]
    return np.where(p5 > t5, 5, np.where(p4 > t4, 4, cmed))

def tune_tau_sev(Ptr, ds_tr, a0_target):
    """[중증우선/통일] 제약=중증놓침≤A0목표, 목적=MAE최소."""
    grid = np.round(np.arange(0.10, 0.55, 0.025), 3); best = None
    for t4 in grid:
        for t5 in grid:
            if t5 < t4: continue
            p = asym(Ptr, t4, t5)
            if miss(ds_tr, p) <= a0_target + 1e-9:
                key = mae(ds_tr, p)
                if best is None or key < best[0]: best = (key, float(t4), float(t5))
    if best is None:
        cand = [(miss(ds_tr, asym(Ptr, a, b)), mae(ds_tr, asym(Ptr, a, b)), a, b)
                for a in grid for b in grid if b >= a]
        _, _, a, b = min(cand); return a, b
    return best[1], best[2]

def step_from_G(Gm, clip):
    """누적확률 G(n x5, =P(Y>=1..5)) → 계단확률 P(n x6). clip=True면 음수클립+정규화."""
    n = Gm.shape[0]; P = np.zeros((n, 6))
    P[:, 0] = 1 - Gm[:, 0]
    for k in range(1, 5): P[:, k] = Gm[:, k - 1] - Gm[:, k]
    P[:, 5] = Gm[:, 4]
    if clip:
        P = np.clip(P, 1e-9, None); P /= P.sum(1, keepdims=True)
    return P

def pava_decreasing(Gm):
    """각 환자의 5개 누적확률을 단조감소로 등위회귀 투영."""
    xs = np.arange(5); out = np.empty_like(Gm)
    for i in range(len(Gm)):
        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
        out[i] = ir.fit_transform(xs, Gm[i])
    return out

def viol_and_negmass(Gm):
    """비단조 위반 인원수, 클립된 음수질량 총합."""
    inc = np.any(np.diff(Gm, axis=1) > 1e-9, axis=1)
    P0 = step_from_G(Gm, clip=False)
    negmass = np.clip(-P0, 0, None).sum()
    return int(inc.sum()), float(negmass)


def main():
    log("# Aim 2 · Gn 비단조 등위회귀(PAVA) 보정 (sjlee) — 실제 실행\n")
    log("> 발표 'PAVA 20.9%->0%, 성능 유지'는 미착수였음. 이 리포트가 **실제 실행** 결과.\n")
    m = G.build_matrix().reset_index(drop=True)
    bs = pd.read_csv(Path.home() / "Downloads" / "baseline_sample.csv")
    bs = bs[bs["in_common_comparison_sample"] == True][["STUDYID", "USUBJID", "A0_harmonized"]]
    m = m.merge(bs, on=["STUDYID", "USUBJID"], how="left").reset_index(drop=True)
    ds = m["ds_stage"].values
    pred_raw = np.zeros(len(m)); pred_iso = np.zeros(len(m))
    vr = nr = vi = ni = 0; n_tot = 0
    for ho in TRIALS:
        tr = m[m.STUDYID != ho]; te = m[m.STUDYID == ho]
        probs, geq, ytr, _ = G.fit_en(tr, C)
        a0_target = miss(tr["ds_stage"].values, tr["A0_harmonized"].values)  # 중증우선 통일
        t4, t5 = tune_tau_sev(probs(tr), tr["ds_stage"].values, a0_target)
        g = geq(te); Graw = np.column_stack([g[k] for k in range(1, 6)])
        # raw
        Praw = step_from_G(Graw, clip=True)
        pred_raw[te.index] = G.asym_rule(Praw, t4, t5)
        a, b = viol_and_negmass(Graw); vr += a; nr += b
        # isotonic (PAVA)
        Giso = pava_decreasing(Graw)
        Piso = step_from_G(Giso, clip=False)
        Piso = np.clip(Piso, 1e-12, None); Piso /= Piso.sum(1, keepdims=True)
        pred_iso[te.index] = G.asym_rule(Piso, t4, t5)
        a2, b2 = viol_and_negmass(Giso); vi += a2; ni += b2
        n_tot += len(te)
        log(f"- fold(held-out={ho}): τ4={t4}, τ5={t5}, 평가 {len(te)}명")
    log("")

    log("## 1) 비단조 진단: raw vs isotonic (C=0.1 배포모형)")
    log("| | 위반 인원 | 위반율 | 음수질량 총합 | 인당 음수질량 |")
    log("|---|---|---|---|---|")
    log(f"| raw (보정 전) | {vr}/{n_tot} | {vr/n_tot*100:.1f}% | {nr:.3f} | {nr/n_tot:.4f} |")
    log(f"| isotonic (PAVA 후) | {vi}/{n_tot} | {vi/n_tot*100:.1f}% | {ni:.3f} | {ni/n_tot:.4f} |")
    log(f"\n- 발표 표기값: 위반 20.9% / 음수질량 0.0005. 실측 raw 위반 {vr/n_tot*100:.1f}%.")
    log("- isotonic 위반 0%는 PAVA 단조강제로 **원리상 보장**.\n")

    log("## 2) 성능 유지 확인: raw vs isotonic (4지표)")
    log("| 처리 | MAE | 가중카파 | 4·5→≤3 | 2↑과소 |")
    log("|---|---|---|---|---|")
    for name, p in [("raw (보정 전)", pred_raw), ("isotonic (PAVA 후)", pred_iso)]:
        log(f"| {name} | {mae(ds,p):.3f} | {kap(ds,p):.3f} | {miss(ds,p)*100:.1f}% | {u2(ds,p)*100:.1f}% |")
    dmae = abs(mae(ds, pred_raw) - mae(ds, pred_iso))
    log(f"\n- raw↔isotonic MAE 차 **{dmae:.3f}**. "
        f"{'성능 사실상 동일(비단조 보정이 예측단계를 거의 안 바꿈).' if dmae < 0.01 else '차이 발생 — 위 수치 그대로 해석.'}")
    log("- 결론: 비단조는 등위회귀로 형식상 완전 해소(0%), 성능은 위 표대로 (나온 값 그대로 기록).\n")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f">>> 저장: {REPORT}")


if __name__ == "__main__":
    main()
