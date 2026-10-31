"""
models.py - 회귀.분류 모델 후보와 예측 어댑터 등록.

초기 후보:
  회귀: LinearRegression, RandomForestRegressor, XGBRegressor
  분류: DummyClassifier(most_frequent), LogisticRegression,
        RandomForestClassifier, XGBClassifier

Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

__all__ = [
    "get_regression_candidates",
    "get_classification_candidates",
    "apply_label_mapping",
    "invert_label_mapping",
    "LabelMappingError",
    "CandidateStatus",
    "check_classification_candidates",
]


def get_regression_candidates(n_jobs: int = 1) -> dict[str, Any]:
    """회귀 후보 모델 딕셔너리를 반환한다.

    Args:
        n_jobs: 병렬 스레드 수. 기본 1 (무료 서버 고려).

    Returns:
        모델 이름 -> 인스턴스 딕셔너리.
    """
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs,
        ),
        "xgboost": XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs,
            verbosity=0,
        ),
    }


def get_classification_candidates(n_jobs: int = 1) -> dict[str, Any]:
    """분류 후보 모델 딕셔너리를 반환한다.

    Args:
        n_jobs: 병렬 스레드 수. 기본 1.

    Returns:
        모델 이름 -> 인스턴스 딕셔너리.
    """
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=n_jobs,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }


# ── 검출 표지(이진 분류) 라벨 매핑 (작업 12) ─────────────────────────────────
#
# 작용기별 이진 표지를 독립 대상으로 다룬다. 0.1 매핑과 양성의 의미를 기록한다.
# 여러 작용기의 동시 검출은 각 대상을 독립 이진 분류로 두어 표현한다.


class LabelMappingError(Exception):
    """라벨 매핑 적용·역매핑 실패."""


def apply_label_mapping(y, label_mapping: dict):
    """원본 표지(문자열 등)를 내부 0/1 정수로 변환한다.

    - label_mapping 에 없는 값은 임의 치환하지 않고 오류를 발생시킨다.
    - 결측(None/NaN)은 변환하지 않고 그대로 둔다 (호출자가 유효 행만 전달).

    Args:
        y: 원본 표지 시퀀스.
        label_mapping: {원본 표기: 0 또는 1}.

    Returns:
        정수 라벨 리스트.
    """
    out = []
    for v in y:
        if v is None:
            raise LabelMappingError("결측 표지는 매핑할 수 없습니다 (유효 행만 전달하세요).")
        key = str(v)
        if key not in label_mapping:
            raise LabelMappingError(
                "라벨 매핑에 없는 표지입니다: " + repr(v)
                + ". 임의 치환하지 않습니다. 정의된 표지: " + str(sorted(label_mapping))
            )
        out.append(int(label_mapping[key]))
    return out


def invert_label_mapping(y_int, label_mapping: dict):
    """내부 0/1 정수 예측을 원본 표지로 되돌린다.

    Args:
        y_int: 0/1 정수 예측 시퀀스.
        label_mapping: {원본 표기: 0 또는 1}.

    Returns:
        원본 표지 리스트.
    """
    inverse = {int(v): k for k, v in label_mapping.items()}
    out = []
    for v in y_int:
        iv = int(v)
        if iv not in inverse:
            raise LabelMappingError("역매핑할 수 없는 정수 라벨입니다: " + repr(v))
        out.append(inverse[iv])
    return out


# ── 후보 적용 가능 여부와 사유 (작업 12) ─────────────────────────────────────


class CandidateStatus:
    """후보 모델 하나의 적용 가능 여부와 사유.

    실행 실패를 좋은 평가값으로 대체하지 않기 위해, 제외된 후보는
    available=False 와 사유를 남긴다.
    """

    def __init__(self, name: str, available: bool, reason: str = ""):
        self.name = name
        self.available = available
        self.reason = reason

    def __repr__(self):
        return (
            "CandidateStatus(name=" + repr(self.name)
            + ", available=" + repr(self.available)
            + ", reason=" + repr(self.reason) + ")"
        )


def check_classification_candidates(n_positive: int, n_negative: int) -> dict:
    """분류 후보의 적용 가능 여부를 점검한다.

    - 한 클래스라도 표본이 없으면 비기준선 후보를 적용 불가로 표시한다.
    - 다수 클래스 기준선은 한 클래스만 있어도 실행 가능하다.
    - 실행 실패를 좋은 점수로 대체하지 않고 사유를 남긴다.

    Args:
        n_positive: 양성(1) 표본 수.
        n_negative: 음성(0) 표본 수.

    Returns:
        후보 이름 -> CandidateStatus.
    """
    both_present = n_positive > 0 and n_negative > 0
    statuses = {}
    for name in ("majority_baseline", "logistic_regression", "random_forest", "xgboost"):
        if name == "majority_baseline":
            if n_positive + n_negative > 0:
                statuses[name] = CandidateStatus(name, True)
            else:
                statuses[name] = CandidateStatus(name, False, "표본이 없습니다.")
        else:
            if both_present:
                statuses[name] = CandidateStatus(name, True)
            else:
                statuses[name] = CandidateStatus(
                    name, False,
                    "학습에 두 클래스가 모두 필요하지만 한 클래스가 비어 있습니다 "
                    "(양성 " + str(n_positive) + ", 음성 " + str(n_negative) + ").",
                )
    return statuses
