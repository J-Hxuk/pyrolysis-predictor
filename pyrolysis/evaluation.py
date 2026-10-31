"""
evaluation.py - 독립 평가 지표 계산.

회귀: MAE, RMSE (원래 단위)
분류: 혼동행렬, 정밀도, 재현율, F1, 클래스별 표본 수

Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

__all__ = [
    "compute_regression_metrics",
    "compute_classification_metrics",
    "ServeStatus",
    "determine_serve_status",
]


def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    unit: str | None = None,
) -> dict[str, Any]:
    """회귀 평가 지표를 계산한다.

    Args:
        y_true: 실측값 배열.
        y_pred: 예측값 배열.
        unit: 지표 단위 (없으면 None).

    Returns:
        {
            "mae": float,
            "rmse": float,
            "unit": str | None,
            "n_samples": int,
            "note": str,
        }
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    return {
        "mae": mae,
        "rmse": rmse,
        "unit": unit,
        "n_samples": n,
        "note": (
            "독립 평가 완료는 높은 정확도를 보장하지 않습니다. "
            "MAE/RMSE는 개별 예측의 신뢰구간이 아닙니다."
        ),
    }


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    target_spec: Any | None = None,
) -> dict[str, Any]:
    """이진 분류 평가 지표를 계산한다.

    Args:
        y_true: 실측 레이블 배열 (0/1).
        y_pred: 예측 레이블 배열 (0/1).
        target_spec: TargetSpec 객체 (label_mapping 참조용, 없어도 됨).

    Returns:
        {
            "precision": float | None,
            "recall": float | None,
            "f1": float | None,
            "confusion_matrix": list[list[int]],
            "class_counts": {"true_0": int, "true_1": int},
            "n_samples": int,
            "threshold_note": str,
        }
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    n_classes = len(classes)

    # 클래스가 하나뿐이면 일부 지표를 계산할 수 없음
    if n_classes < 2:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
            "class_counts": {
                "true_0": int(np.sum(y_true_arr == 0)),
                "true_1": int(np.sum(y_true_arr == 1)),
            },
            "n_samples": len(y_true_arr),
            "threshold_note": (
                f"클래스가 {n_classes}개뿐이어서 정밀도/재현율/F1을 계산할 수 없습니다."
            ),
            "error": f"insufficient_classes: {list(classes)}",
        }

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "class_counts": {
            "true_0": int(np.sum(y_true_arr == 0)),
            "true_1": int(np.sum(y_true_arr == 1)),
        },
        "n_samples": len(y_true_arr),
        "threshold_note": (
            "임계값은 내부 선정 과정에서 확정됩니다. "
            "외부 평가 결과로 임계값을 재조정하지 않습니다."
        ),
    }


# ── 대상별 제공 가능 상태 판정 (작업 14) ─────────────────────────────────────
#
# 설계 6.5 상태:
#   unconfigured        규격 미확정
#   data_insufficient   정답·그룹·클래스 부족
#   trained_unvalidated 학습했으나 독립 평가 미완료
#   evaluated           독립 평가 완료 (배포 선정 가능)
#   active              배포 검사 통과하여 묶음 포함 (배포 단계에서 확정)
#   disabled            제공하지 않도록 정한 대상
#
# 독립 평가 완료는 높은 정확도를 보장하지 않는다. 가상 정확도 목표를 통과 조건으로 넣지 않는다.

_SERVE_STATES = (
    "unconfigured",
    "data_insufficient",
    "trained_unvalidated",
    "evaluated",
    "active",
    "disabled",
)


class ServeStatus:
    """대상 하나의 제공 가능 상태와 사유."""

    def __init__(self, target_id: str, state: str, reason: str = ""):
        if state not in _SERVE_STATES:
            raise ValueError("알 수 없는 제공 상태입니다: " + repr(state))
        self.target_id = target_id
        self.state = state
        self.reason = reason

    @property
    def web_enabled(self) -> bool:
        """웹 활성화 가능 여부. active 만 실제 제공 가능."""
        return self.state == "active"

    def __repr__(self):
        return (
            "ServeStatus(target_id=" + repr(self.target_id)
            + ", state=" + repr(self.state)
            + ", reason=" + repr(self.reason) + ")"
        )


def determine_serve_status(
    target_id: str,
    *,
    configured: bool,
    data_sufficient: bool,
    trained: bool,
    evaluated: bool,
    disabled: bool = False,
    required_classes_present: bool = True,
    reason: str = "",
) -> ServeStatus:
    """플래그로부터 제공 가능 상태를 결정한다.

    우선순위: disabled > unconfigured > data_insufficient >
             (분류 필수 클래스 부족 -> data_insufficient) >
             trained_unvalidated > evaluated.
    active 는 이 함수에서 직접 부여하지 않는다 (배포 검사에서 확정).

    - 평가 미완료 대상, 학습 오차만 있는 대상은 evaluated 가 될 수 없다.
    - 학습 또는 전체 독립 평가에서 필수 클래스가 부족하면 활성화할 수 없다.
    - 성능 임계값은 적용하지 않는다 (평가 완료 여부만 본다).

    Returns:
        ServeStatus (state 는 active 를 제외한 5개 중 하나).
    """
    if disabled:
        return ServeStatus(target_id, "disabled", reason or "제공하지 않도록 설정된 대상입니다.")
    if not configured:
        return ServeStatus(target_id, "unconfigured", reason or "규격이 확정되지 않았습니다.")
    if not data_sufficient:
        return ServeStatus(target_id, "data_insufficient", reason or "필요한 자료가 부족합니다.")
    if not required_classes_present:
        return ServeStatus(
            target_id, "data_insufficient",
            reason or "학습 또는 독립 평가에 필요한 클래스가 부족합니다.",
        )
    if not trained:
        return ServeStatus(target_id, "data_insufficient",
                           reason or "학습이 수행되지 않았습니다.")
    if not evaluated:
        return ServeStatus(target_id, "trained_unvalidated",
                           reason or "학습은 했으나 독립 평가가 완료되지 않았습니다.")
    return ServeStatus(target_id, "evaluated",
                       reason or "독립 평가 완료. 높은 정확도를 보장하지는 않습니다.")
