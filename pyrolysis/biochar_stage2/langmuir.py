"""
langmuir.py — Langmuir 등온흡착식 비선형 최소제곱 피팅 (선택적 분석 기능).

지시서 §3 (Langmuir fitting):
  사용자가 동일 시료·동일 오염물질·동일 실험조건에서 측정한 여러 (Ce, qe) 쌍을
  입력하면, 비선형 최소제곱법으로 Langmuir 식을 피팅한다.

      qe = (qmax * KL * Ce) / (1 + KL * Ce)

  출력: qmax(mg/g), KL(L/mg), 적합 곡선, 실측 점, 피팅 R² 및 RMSE.
  데이터가 부족하거나 피팅이 식별 불가능하면 결과를 생성하지 않고 사유를 출력한다.

  이 기능은 '입력한 실측 등온흡착 데이터에 대한 피팅'이며, Stage2 머신러닝 예측과는
  별개다. 결과에 그 사실을 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "LangmuirFitResult",
    "LangmuirFitRefusal",
    "langmuir_qe",
    "fit_langmuir",
    "MIN_POINTS",
]

# Langmuir 식 파라미터 2개(qmax, KL)를 식별하려면 최소 서로 다른 Ce 3점 이상 필요.
MIN_POINTS = 3

ANALYSIS_NOTE = (
    "이 결과는 입력한 실측 등온흡착 데이터(Ce, qe)에 대한 Langmuir 피팅이며, "
    "Stage2 머신러닝 예측과는 별개입니다."
)


def langmuir_qe(ce: np.ndarray, qmax: float, kl: float) -> np.ndarray:
    """Langmuir 식: qe = qmax*KL*Ce / (1 + KL*Ce)."""
    ce = np.asarray(ce, dtype=float)
    return (qmax * kl * ce) / (1.0 + kl * ce)


@dataclass
class LangmuirFitResult:
    """Langmuir 피팅 성공 결과."""

    qmax: float
    kl: float
    r2: float
    rmse: float
    n_points: int
    ce: list[float]
    qe: list[float]
    curve_ce: list[float]
    curve_qe: list[float]
    note: str = ANALYSIS_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted": True,
            "qmax_mg_g": self.qmax, "KL_L_mg": self.kl,
            "r2": self.r2, "rmse": self.rmse, "n_points": self.n_points,
            "observed": {"Ce": self.ce, "qe": self.qe},
            "fit_curve": {"Ce": self.curve_ce, "qe": self.curve_qe},
            "note": self.note,
        }


@dataclass
class LangmuirFitRefusal:
    """피팅 거부 (데이터 부족·식별 불가). 결과 수치를 만들지 않는다."""

    reason_code: str
    message_ko: str
    n_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted": False,
            "reason_code": self.reason_code,
            "message_ko": self.message_ko,
            "n_points": self.n_points,
            "note": ANALYSIS_NOTE,
        }


def fit_langmuir(
    ce: list[float] | np.ndarray, qe: list[float] | np.ndarray
) -> LangmuirFitResult | LangmuirFitRefusal:
    """(Ce, qe) 쌍에 Langmuir 식을 비선형 최소제곱으로 피팅한다.

    데이터가 부족하거나 피팅이 식별 불가능하면 LangmuirFitRefusal 을 반환한다
    (수치를 임의로 만들지 않는다).

    거부 사유:
      - INSUFFICIENT_POINTS : 유효한 점이 MIN_POINTS 미만.
      - NON_POSITIVE        : Ce 또는 qe 에 음수가 있음 (물리적으로 불가).
      - DEGENERATE_CE       : 서로 다른 Ce 값이 2개 미만 (곡선 식별 불가).
      - FIT_FAILED          : 최적화가 수렴하지 않음.
      - UNIDENTIFIABLE       : 추정 파라미터가 비물리적이거나 분산이 과도.
    """
    ce = np.asarray(ce, dtype=float)
    qe = np.asarray(qe, dtype=float)

    # 결측·비유한 제거
    finite = np.isfinite(ce) & np.isfinite(qe)
    ce, qe = ce[finite], qe[finite]
    n = int(len(ce))

    if n < MIN_POINTS:
        return LangmuirFitRefusal(
            "INSUFFICIENT_POINTS",
            f"유효한 (Ce, qe) 점이 {n}개로 부족합니다. Langmuir 피팅에는 최소 "
            f"{MIN_POINTS}개의 서로 다른 평형점이 필요합니다.", n)

    if (ce < 0).any() or (qe < 0).any():
        return LangmuirFitRefusal(
            "NON_POSITIVE",
            "Ce 또는 qe 에 음수가 있습니다. 평형 농도·흡착량은 음수가 될 수 없습니다.", n)

    if len(np.unique(ce)) < 2:
        return LangmuirFitRefusal(
            "DEGENERATE_CE",
            "서로 다른 Ce 값이 2개 미만입니다. 등온 곡선을 식별할 수 없습니다.", n)

    try:
        from scipy.optimize import curve_fit
    except Exception:
        return LangmuirFitRefusal(
            "FIT_FAILED", "scipy 를 사용할 수 없어 피팅을 수행할 수 없습니다.", n)

    # 초기값: qmax ~ 관측 qe 최대의 조금 위, KL ~ 1/중앙 Ce
    qmax0 = float(max(qe.max() * 1.1, 1e-6))
    kl0 = float(1.0 / max(np.median(ce), 1e-6))
    try:
        popt, pcov = curve_fit(
            langmuir_qe, ce, qe, p0=[qmax0, kl0],
            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=10000,
        )
    except Exception as exc:
        return LangmuirFitRefusal(
            "FIT_FAILED", f"비선형 최소제곱이 수렴하지 않았습니다: {exc}", n)

    qmax, kl = float(popt[0]), float(popt[1])

    # 식별성 점검: 비물리적 값 또는 공분산이 비유한(정보 부족)이면 거부
    if qmax <= 0 or kl <= 0 or not np.all(np.isfinite(pcov)):
        return LangmuirFitRefusal(
            "UNIDENTIFIABLE",
            "추정 파라미터가 비물리적이거나 공분산을 추정할 수 없습니다 "
            "(식별 불가). 더 넓은 농도 범위의 평형점이 필요합니다.", n)

    pred = langmuir_qe(ce, qmax, kl)
    ss_res = float(np.sum((qe - pred) ** 2))
    ss_tot = float(np.sum((qe - qe.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((qe - pred) ** 2)))

    if not np.isfinite(r2):
        return LangmuirFitRefusal(
            "UNIDENTIFIABLE",
            "qe 값이 모두 같아 R² 를 정의할 수 없습니다 (곡선 식별 불가).", n)

    # 적합 곡선 (표시용)
    ce_grid = np.linspace(0.0, float(ce.max()) * 1.05, 100)
    qe_grid = langmuir_qe(ce_grid, qmax, kl)

    order = np.argsort(ce)
    return LangmuirFitResult(
        qmax=qmax, kl=kl, r2=float(r2), rmse=rmse, n_points=n,
        ce=[float(v) for v in ce[order]],
        qe=[float(v) for v in qe[order]],
        curve_ce=[float(v) for v in ce_grid],
        curve_qe=[float(v) for v in qe_grid],
    )
