"""
validation.py — 예측 입력 파일의 값 검증 (작업 9).

행별 자료형·범주·유한성·필수 여부를 검사하고 모든 행의 오류를 수집한다.
비어 있지 않은 잘못된 값을 결측으로 바꾸지 않는다.
학습 범위 이탈(작업 10)과 조성 제약(작업 18)은 여기서 다루지 않는다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pyrolysis.contracts import Issue

__all__ = [
    "validate_candidate_values",
    "check_physical_constraints",
    "check_training_range",
    "validate_candidates",
]


def _is_blank(value: Any) -> bool:
    """결측(빈 셀)으로 간주할 값인지 판정한다. 빈 문자열·공백·None·NaN 만 결측이다."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _coerce_number(value: Any):
    """값을 수치로 변환해 반환한다. 변환 불가면 None 을 반환한다 (대체하지 않음)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def validate_candidate_values(data: pd.DataFrame, input_specs: list) -> list:
    """후보 표의 각 셀을 입력 규격에 따라 검사하고 Issue 목록을 반환한다.

    검사 항목:
    - 필수 입력의 결측 (MISSING_REQUIRED)
    - 수치형(int/float)의 비수치 문자열 (NOT_NUMERIC)
    - 비유한 값 NaN·Inf (NOT_FINITE)
    - int 형의 정수 아님 (NOT_INTEGER)
    - 범주형의 미지원 범주 (UNKNOWN_CATEGORY)

    규칙:
    - 모든 행을 끝까지 검사한다 (첫 오류에서 멈추지 않는다).
    - 비어 있지 않은 잘못된 값을 결측으로 바꾸지 않는다.
    - 범위 이탈은 여기서 다루지 않는다 (작업 10).

    Args:
        data: 후보 표 (열 = 입력 ID). 행 번호는 0-기준 위치.
        input_specs: InputSpec 목록.

    Returns:
        Issue 목록 (severity="error"). 오류가 없으면 빈 목록.
    """
    issues: list = []

    spec_by_id = {s.id: s for s in input_specs}

    for pos in range(len(data)):
        row_no = pos + 1  # 1-기준 (헤더 제외)
        for input_id, spec in spec_by_id.items():
            if input_id not in data.columns:
                # 열 자체가 없으면 파서 단계에서 FILE 수준 오류로 처리한다.
                continue
            value = data.iloc[pos][input_id]
            blank = _is_blank(value)

            if blank:
                if spec.required:
                    issues.append(Issue(
                        code="MISSING_REQUIRED",
                        message_ko="필수 입력이 비어 있습니다.",
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                # 선택 입력의 결측은 작업 10의 저장된 규칙으로 처리한다.
                continue

            if spec.dtype in ("int", "float"):
                num = _coerce_number(value)
                if num is None:
                    issues.append(Issue(
                        code="NOT_NUMERIC",
                        message_ko="수치가 아닌 값입니다: " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                    continue
                if not math.isfinite(float(num)):
                    issues.append(Issue(
                        code="NOT_FINITE",
                        message_ko="유한한 수치가 아닙니다 (NaN 또는 Inf): " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                    continue
                if spec.dtype == "int" and float(num) != int(num):
                    issues.append(Issue(
                        code="NOT_INTEGER",
                        message_ko="정수가 아닙니다: " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                    continue

            elif spec.dtype == "category":
                allowed = {c.value for c in spec.allowed_categories}
                if str(value) not in allowed:
                    issues.append(Issue(
                        code="UNKNOWN_CATEGORY",
                        message_ko="허용되지 않은 범주입니다: " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                    continue

    return issues


# ── 물리적 제약 검사 (작업 10) — 위반은 오류 ────────────────────────────────


def check_physical_constraints(data: pd.DataFrame, input_specs: list) -> list:
    """입력의 물리적 제약(min/max/expression)을 검사하고 오류 Issue 목록을 반환한다.

    물리적 제약 위반은 오류다 (예측 차단). 학습 범위와는 별개다.
    expression 은 지원하는 단순 형식만 평가하며, 평가할 수 없으면 건너뛰고 주의를 남기지 않는다
    (작업 18에서 조합 제약을 확장한다).

    Args:
        data: 후보 표 (열 = 입력 ID).
        input_specs: InputSpec 목록.

    Returns:
        Issue 목록 (severity="error"). code="PHYSICAL_VIOLATION".
    """
    issues: list = []
    spec_by_id = {s.id: s for s in input_specs}

    for pos in range(len(data)):
        row_no = pos + 1
        for input_id, spec in spec_by_id.items():
            if input_id not in data.columns:
                continue
            if not spec.physical_constraints:
                continue
            value = data.iloc[pos][input_id]
            if _is_blank(value):
                continue
            num = _coerce_number(value)
            if num is None:
                # 수치 변환 실패는 값 검증(NOT_NUMERIC)에서 다룬다.
                continue
            num = float(num)
            for pc in spec.physical_constraints:
                if pc.min_value is not None and num < pc.min_value:
                    issues.append(Issue(
                        code="PHYSICAL_VIOLATION",
                        message_ko="물리적 허용 범위 미만입니다 (" + str(pc.min_value) + " 이상): " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
                if pc.max_value is not None and num > pc.max_value:
                    issues.append(Issue(
                        code="PHYSICAL_VIOLATION",
                        message_ko="물리적 허용 범위 초과입니다 (" + str(pc.max_value) + " 이하): " + repr(value),
                        severity="error",
                        row=row_no, col=input_id, input_id=input_id,
                    ))
    return issues


# ── 학습 범위 이탈 검사 (작업 10) — 이탈은 주의 ─────────────────────────────


def check_training_range(data: pd.DataFrame, input_specs: list, training_domain) -> list:
    """대상별 학습 범위 이탈을 주의(warning) Issue 로 계산한다.

    - 수치 입력이 학습 최솟값·최댓값을 벗어나면 TRAINING_RANGE_EXCEEDED 주의.
    - 학습에서 관측하지 못한 (그러나 규격상 허용된) 범주도 TRAINING_RANGE_EXCEEDED 주의.
    - 범위 이탈은 예측을 차단하지 않으며 입력 오류 목록에 넣지 않는다.
    - 대상별 학습 표본이 다르면(training_domain 이 다르면) 판정도 달라진다.

    Args:
        data: 후보 표.
        input_specs: InputSpec 목록 (dtype 확인용).
        training_domain: TrainingDomain. target_id 가 주의 메시지에 포함된다.

    Returns:
        Issue 목록 (severity="warning"). code="TRAINING_RANGE_EXCEEDED".
    """
    issues: list = []
    spec_by_id = {s.id: s for s in input_specs}
    tgt = training_domain.target_id

    for pos in range(len(data)):
        row_no = pos + 1
        # 수치 범위
        for input_id, rng in training_domain.numeric_ranges.items():
            if input_id not in data.columns:
                continue
            value = data.iloc[pos][input_id]
            if _is_blank(value):
                continue
            num = _coerce_number(value)
            if num is None:
                continue
            num = float(num)
            if num < rng.min_value or num > rng.max_value:
                issues.append(Issue(
                    code="TRAINING_RANGE_EXCEEDED",
                    message_ko=(
                        "학습 범위를 벗어났습니다 [" + str(rng.min_value) + ", "
                        + str(rng.max_value) + "]: " + repr(value)
                    ),
                    severity="warning",
                    row=row_no, col=input_id, input_id=input_id, target_id=tgt,
                ))
        # 학습 미관측 범주
        for input_id, seen_values in training_domain.supported_categories.items():
            if input_id not in data.columns:
                continue
            value = data.iloc[pos][input_id]
            if _is_blank(value):
                continue
            if str(value) not in set(seen_values):
                issues.append(Issue(
                    code="TRAINING_RANGE_EXCEEDED",
                    message_ko="학습에서 관측하지 못한 범주입니다: " + repr(value),
                    severity="warning",
                    row=row_no, col=input_id, input_id=input_id, target_id=tgt,
                ))
    return issues


# ── 통합 검증 (단일·파일 입력 공통) ──────────────────────────────────────────


def validate_candidates(data: pd.DataFrame, input_specs: list, training_domains=None) -> dict:
    """단일·파일 입력에 공통으로 적용하는 통합 검증.

    - 값 검증(validate_candidate_values)과 물리적 제약(check_physical_constraints)은 오류다.
    - 학습 범위 이탈(check_training_range)은 대상별 주의로 분리한다.
    - 오류가 있으면 예측하지 않고, 주의만 있으면 예측을 허용한다 (호출자가 판단).

    Args:
        data: 후보 표.
        input_specs: InputSpec 목록.
        training_domains: TrainingDomain 목록 (대상별). None 이면 범위 주의를 계산하지 않는다.

    Returns:
        dict: {
            "errors": list[Issue],            # 값·물리 제약 오류
            "range_warnings": list[Issue],    # 대상별 TRAINING_RANGE_EXCEEDED
        }
    """
    errors: list = []
    errors.extend(validate_candidate_values(data, input_specs))
    errors.extend(check_physical_constraints(data, input_specs))

    range_warnings: list = []
    if training_domains:
        for td in training_domains:
            range_warnings.extend(check_training_range(data, input_specs, td))

    return {"errors": errors, "range_warnings": range_warnings}
