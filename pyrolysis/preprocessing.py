"""
preprocessing.py - 모델별 전처리 Pipeline과 목표 변환 구성.

전처리는 학습 자료에만 fit하며 평가 자료에는 transform만 적용된다.
Pipeline.fit()이 내부적으로 이를 보장한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

__all__ = [
    "build_regression_pipeline",
    "build_classification_pipeline",
    "split_numeric_categorical",
    "TargetTransform",
    "TargetTransformError",
]


def split_numeric_categorical(
    input_specs: list[Any],
) -> tuple[list[str], list[str]]:
    """InputSpec 목록에서 수치형 열 이름과 범주형 열 이름을 분리한다."""
    numeric: list[str] = []
    categorical: list[str] = []
    for spec in input_specs:
        if spec.dtype in ("float", "int"):
            numeric.append(spec.id)
        elif spec.dtype in ("category", "bool"):
            categorical.append(spec.id)
    return numeric, categorical


def _build_preprocessor(input_specs: list[Any]) -> ColumnTransformer:
    """수치형·범주형 열에 맞는 ColumnTransformer를 만든다.

    수치형: 결측 대체(mean) -> StandardScaler
    범주형: 결측 대체(most_frequent) -> OrdinalEncoder
    """
    numeric_cols, categorical_cols = split_numeric_categorical(input_specs)
    transformers: list[tuple] = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_regression_pipeline(input_specs: list[Any], model: Any) -> Pipeline:
    """회귀 Pipeline을 만든다: 전처리 -> 모델.

    Args:
        input_specs: InputSpec 목록.
        model: scikit-learn 호환 회귀 모델 인스턴스.

    Returns:
        fit되지 않은 Pipeline.
    """
    preprocessor = _build_preprocessor(input_specs)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_classification_pipeline(input_specs: list[Any], model: Any) -> Pipeline:
    """분류 Pipeline을 만든다: 전처리 -> 모델.

    Args:
        input_specs: InputSpec 목록.
        model: scikit-learn 호환 분류 모델 인스턴스.

    Returns:
        fit되지 않은 Pipeline.
    """
    preprocessor = _build_preprocessor(input_specs)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ── 명시적 목표 변환과 역변환 (작업 12) ──────────────────────────────────────
#
# 설계 6.3: 목표의 로그 변환 등은 대상별로 명시적으로 설정하고 역변환까지 저장한다.
# 변환을 자동으로 선택하거나 임의의 상수를 더하지 않는다.

import numpy as np


class TargetTransformError(Exception):
    """목표 변환 설정·적용 실패."""


class TargetTransform:
    """대상 하나의 목표 변환과 역변환. 명시적으로 지정한 종류만 적용한다.

    kind:
        "identity" : 변환 없음 (기본). 역변환도 항등.
        "log"      : y -> log(y). 역변환 exp. 양수 목표에만 명시적으로 설정한다.
                     0 이하 값이 있으면 임의 상수를 더하지 않고 오류를 발생시킨다.

    자동으로 변환을 고르지 않는다. 호출자가 kind 를 명시해야 한다.
    """

    _SUPPORTED = ("identity", "log")

    def __init__(self, kind: str = "identity"):
        if kind not in self._SUPPORTED:
            raise TargetTransformError(
                "지원하지 않는 목표 변환입니다: " + repr(kind)
                + ". 지원: " + str(self._SUPPORTED)
            )
        self.kind = kind

    def forward(self, y):
        """학습용 변환. 결측을 채우지 않는다 (NaN 은 NaN 으로 통과)."""
        arr = np.asarray(y, dtype="float64")
        if self.kind == "identity":
            return arr
        if self.kind == "log":
            finite = arr[np.isfinite(arr)]
            if finite.size > 0 and np.any(finite <= 0):
                raise TargetTransformError(
                    "log 변환은 양수 목표에만 적용할 수 있습니다. "
                    "0 이하 값이 있어 임의 상수를 더하지 않고 중단합니다."
                )
            return np.log(arr)
        raise TargetTransformError("알 수 없는 변환: " + repr(self.kind))

    def inverse(self, y):
        """예측값을 원래 단위로 되돌린다."""
        arr = np.asarray(y, dtype="float64")
        if self.kind == "identity":
            return arr
        if self.kind == "log":
            return np.exp(arr)
        raise TargetTransformError("알 수 없는 변환: " + repr(self.kind))
