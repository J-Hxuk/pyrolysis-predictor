"""
contracts.py — 설정·입력·출력·배포 묶음의 자료형과 구조 검증.

Pydantic v2 모델로 공통 계약을 정의한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── 입력 변수 규격 ─────────────────────────────────────────────────────────────


class PhysicalConstraint(BaseModel):
    """물리적으로 허용되는 범위·조건식 (학습 범위와 별개)."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., description="제약 조건의 설명")
    min_value: float | None = Field(None, description="물리적 최솟값")
    max_value: float | None = Field(None, description="물리적 최댓값")
    expression: str | None = Field(None, description="조건식 (예: 'x1 + x2 <= 100')")


class AllowedCategory(BaseModel):
    """범주형 입력의 표준 값과 표시 이름."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(..., description="내부 표준 값")
    label_ko: str = Field(..., description="한국어 표시 이름")
    description: str | None = Field(None, description="범주 의미 설명")


class InputSpec(BaseModel):
    """입력 변수 하나의 규격."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="내부 식별자 (영문, 밑줄 허용)")
    label_ko: str = Field(..., description="한국어 이름")
    description: str = Field(..., description="변수 의미 설명")
    role: Literal["feedstock", "preparation", "reaction_test"] = Field(
        ..., description="변수 역할"
    )
    dtype: Literal["int", "float", "category", "bool"] = Field(
        ..., description="허용 자료형"
    )
    unit: str | None = Field(None, description="단위 (무차원이면 null)")
    basis: str | None = Field(None, description="산출 기준")
    required: bool = Field(True, description="필수 여부")
    missing_policy: str | None = Field(
        None,
        description="학습 시 확정한 결측 처리 방법. 필수 변수에는 null",
    )
    allowed_categories: list[AllowedCategory] = Field(
        default_factory=list,
        description="범주형 입력의 허용 값 목록",
    )
    physical_constraints: list[PhysicalConstraint] = Field(
        default_factory=list,
        description="물리적 허용 범위·조건식",
    )
    display_order: int = Field(0, description="표준 양식과 화면의 표시 순서")

    @field_validator("missing_policy")
    @classmethod
    def _missing_policy_only_for_optional(cls, v: str | None) -> str | None:
        # 실제 required/missing_policy 교차 검증은 model_validator 에서 수행
        return v

    @model_validator(mode="after")
    def _required_has_no_missing_policy(self) -> "InputSpec":
        if self.required and self.missing_policy is not None:
            raise ValueError(
                f"입력 '{self.id}': 필수 변수(required=True)에는 "
                "missing_policy를 설정하지 않습니다."
            )
        return self

    @model_validator(mode="after")
    def _category_needs_allowed_list(self) -> "InputSpec":
        if self.dtype == "category" and not self.allowed_categories:
            raise ValueError(
                f"입력 '{self.id}': dtype=category 이면 "
                "allowed_categories 목록이 필요합니다."
            )
        return self


# ── 예측 대상 규격 ─────────────────────────────────────────────────────────────


class OutputConstraint(BaseModel):
    """출력 제약 (비음수·비율·유한성 등)."""

    model_config = ConfigDict(extra="forbid")

    non_negative: bool = Field(False, description="비음수 제약")
    finite: bool = Field(True, description="유한값 제약")
    min_value: float | None = Field(None, description="출력 최솟값")
    max_value: float | None = Field(None, description="출력 최댓값")
    tolerance: float | None = Field(
        None, description="수치 허용 오차 (저장 전후 일치 판정용)"
    )
    description: str | None = Field(None, description="제약 설명")


class TargetSpec(BaseModel):
    """예측 대상 하나의 규격."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="내부 식별자")
    label_ko: str = Field(..., description="한국어 이름")
    family: Literal["gas", "liquid", "functional_group", "pore", "performance"] = Field(
        ..., description="기체·액체·작용기·기공·성능 범주"
    )
    task: Literal["regression", "binary_classification"] = Field(
        ..., description="회귀 또는 이진 분류"
    )
    definition: str = Field(..., description="수치·분류의 의미")
    unit: str | None = Field(None, description="단위 (무차원이면 null)")
    basis: str | None = Field(None, description="조성의 분모 등 산출 기준")
    measurement: str | None = Field(
        None, description="분석 방법, 전처리·피크 분리·산출 기준"
    )
    input_ids: list[str] = Field(
        default_factory=list,
        description="이 모델이 실제로 사용하는 입력 ID 목록",
    )
    label_mapping: dict[str, int] | None = Field(
        None, description="분류 정답의 원본 표기와 내부 0·1 대응"
    )
    positive_label: str | None = Field(
        None, description="양성 클래스의 원본 표기"
    )
    detection_rule: str | None = Field(
        None, description="검출 표지 정의 (수치에서 변환 시 근거·기준)"
    )
    reaction_context: str | None = Field(
        None, description="촉매 반응과 지표 산출 조건 (다른 대상에는 null)"
    )
    output_constraints: OutputConstraint = Field(
        default_factory=OutputConstraint,
        description="출력 제약",
    )
    display_precision: int = Field(
        4, description="화면 표시 소수점 자릿수. 계산값은 바꾸지 않음"
    )

    @model_validator(mode="after")
    def _classification_needs_label_mapping(self) -> "TargetSpec":
        if self.task == "binary_classification":
            if self.label_mapping is None:
                raise ValueError(
                    f"대상 '{self.id}': binary_classification 에는 "
                    "label_mapping이 필요합니다."
                )
            if self.positive_label is None:
                raise ValueError(
                    f"대상 '{self.id}': binary_classification 에는 "
                    "positive_label이 필요합니다."
                )
        return self


# ── 문제 설정 전체 ─────────────────────────────────────────────────────────────




class InputSchema(BaseModel):
    """Input schema: union of active target inputs (Task 6)."""

    model_config = ConfigDict(extra="forbid")
    schema_id: str = Field(..., description="12-char hash of input contract fields")
    input_specs: list[InputSpec] = Field(..., description="Active InputSpec list sorted by id")

    @property
    def input_ids(self) -> list:
        return [s.id for s in self.input_specs]

    @classmethod
    def build(cls, specs: list) -> "InputSchema":
        ids = [s.id for s in specs]
        seen = set()
        dups = []
        for id_ in ids:
            if id_ in seen: dups.append(id_)
            seen.add(id_)
        if dups:
            raise ValueError("Duplicate input IDs: " + str(dups))
        ss = sorted(specs, key=lambda s: s.id)
        return cls(schema_id=cls._compute_sid(ss), input_specs=ss)

    @staticmethod
    def _compute_sid(ss: list) -> str:
        fp = []
        for s in ss:
            fp.append({"id": s.id, "dtype": s.dtype, "unit": s.unit,
                       "basis": s.basis, "required": s.required,
                       "categories": sorted([c.value for c in s.allowed_categories])})
        raw = json.dumps(fp, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]




class NumericRange(BaseModel):
    """대상별 학습 자료에서 관측한 수치 입력의 최소·최대 (학습 범위)."""

    model_config = ConfigDict(extra="forbid")
    min_value: float = Field(..., description="학습 자료 관측 최솟값")
    max_value: float = Field(..., description="학습 자료 관측 최댓값")


class TrainingDomain(BaseModel):
    """대상 하나의 실제 학습 입력 범위와 지원 범주 (training_domain.json 근거).

    학습 범위는 예측을 차단하지 않는 주의(warning) 계산에만 쓰인다.
    대상별 학습 표본이 다르면 범위도 달라진다.
    물리적 제약(InputSpec.physical_constraints)과는 별개다.
    """

    model_config = ConfigDict(extra="forbid")
    target_id: str = Field(..., description="이 범위가 속한 대상 ID")
    numeric_ranges: dict[str, NumericRange] = Field(
        default_factory=dict, description="입력 ID -> 수치 학습 범위"
    )
    supported_categories: dict[str, list[str]] = Field(
        default_factory=dict, description="입력 ID -> 학습에서 관측한 범주 값 목록"
    )


class ProblemSpec(BaseModel):
    """문제 설정 전체. YAML에서 파싱하여 사용한다."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., description="설정 파일 스키마 버전")
    problem_id: str | None = Field(None, description="문제 식별자")
    status: Literal["draft", "configured"] = Field(
        ..., description="draft=구조 검사만, configured=학습 준비 완료"
    )
    inputs: list[InputSpec] = Field(default_factory=list)
    targets: list[TargetSpec] = Field(default_factory=list)
    composition_groups: list[Any] = Field(
        default_factory=list,
        description="조성 그룹 규격 (데이터 확보 후 정의)",
    )
    # problem.template.yaml 의 추가 필드를 허용 (extra는 ignore)
    physical_constraints: list[Any] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _schema_version_must_be_1(self) -> "ProblemSpec":
        if self.schema_version != 1:
            raise ValueError(
                f"지원하는 schema_version은 1입니다. 현재: {self.schema_version}"
            )
        return self


    @model_validator(mode="after")
    def _no_duplicate_input_ids(self) -> "ProblemSpec":
        ids = [i.id for i in self.inputs]
        seen = set()
        dups = []
        for id_ in ids:
            if id_ in seen: dups.append(id_)
            seen.add(id_)
        if dups: raise ValueError("Duplicate input IDs: " + str(dups))
        return self

    @model_validator(mode="after")
    def _no_duplicate_target_ids(self) -> "ProblemSpec":
        ids = [tgt.id for tgt in self.targets]
        seen = set()
        dups = []
        for id_ in ids:
            if id_ in seen: dups.append(id_)
            seen.add(id_)
        if dups: raise ValueError("Duplicate target IDs: " + str(dups))
        return self

    @model_validator(mode="after")
    def _target_inputs_exist_when_configured(self) -> "ProblemSpec":
        if self.status != "configured":
            return self
        input_set = set(s.id for s in self.inputs)
        for tgt in self.targets:
            miss = [id_ for id_ in tgt.input_ids if id_ not in input_set]
            if miss:
                raise ValueError("Target " + repr(tgt.id) + " refs unknown inputs: " + str(miss))
        return self

    def is_prediction_ready(self) -> bool:
        """실제 웹 예측을 실행할 수 있는 상태인지 반환한다.

        예측 준비 상태의 최소 조건:
        - status == 'configured'
        - inputs 가 하나 이상
        - targets 가 하나 이상
        """
        return (
            self.status == "configured"
            and len(self.inputs) > 0
            and len(self.targets) > 0
        )

    def is_training_ready(self) -> bool:
        """학습을 시작할 수 있는 상태인지 반환한다."""
        return self.status == "configured" and len(self.inputs) > 0 and len(self.targets) > 0


# ── 예측 파이프라인 자료형 ──────────────────────────────────────────────────────


class Issue(BaseModel):
    """검증 오류 또는 주의 정보 하나."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="오류 코드 (예: INPUT_INVALID)")
    message_ko: str = Field(..., description="한국어 설명")
    severity: Literal["error", "warning"] = Field(
        ..., description="오류=예측 중단, 주의=예측 계속"
    )
    row: int | None = Field(None, description="원본 파일 행 번호 (1-indexed)")
    col: str | None = Field(None, description="열 이름")
    input_id: str | None = Field(None, description="관련 입력 변수 ID")
    target_id: str | None = Field(None, description="관련 예측 대상 ID")


class CandidateTable(BaseModel):
    """예측 요청의 후보 표. 단일 예측도 한 행으로 표현한다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema_id: str = Field(..., description="입력 규격 식별자")
    # 원본 행 번호 목록 (1-indexed, 파일 헤더 제외 기준)
    source_rows: list[int] = Field(..., description="원본 파일 행 번호 목록")
    # 후보 ID 목록. 없으면 source_rows 기반 자동 부여
    candidate_ids: list[str] = Field(..., description="후보 식별자 목록")
    # 정규화된 입력 표 (열=입력 ID, 행=후보)
    data: pd.DataFrame = Field(..., description="정규화된 입력 표")

    @model_validator(mode="after")
    def _lengths_must_match(self) -> "CandidateTable":
        n = len(self.source_rows)
        if len(self.candidate_ids) != n:
            raise ValueError("source_rows와 candidate_ids의 길이가 다릅니다.")
        if len(self.data) != n:
            raise ValueError("data의 행 수가 source_rows 길이와 다릅니다.")
        return self


class TargetPrediction(BaseModel):
    """대상 하나의 예측 결과."""

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(..., description="예측 대상 ID")
    # 회귀: float, 분류: 표지 문자열
    values: list[float | str] = Field(
        ..., description="원래 단위의 예측값 또는 분류 표지"
    )
    unit: str | None = Field(None, description="단위")
    measurement_ref: str | None = Field(None, description="측정 기준 참조")
    warnings: list[Issue] = Field(
        default_factory=list,
        description="이 대상에 관련된 주의 정보",
    )


class PredictionBatch(BaseModel):
    """예측 요청 전체의 결과."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="요청 식별자")
    bundle_id: str = Field(..., description="배포 묶음 ID")
    bundle_hash: str = Field(..., description="manifest.json SHA-256 해시")
    input_snapshot_hash: str = Field(..., description="입력 표 SHA-256 해시")
    # 대상 ID -> TargetPrediction
    predictions: dict[str, TargetPrediction] = Field(
        default_factory=dict, description="대상별 결과"
    )
    range_warnings: list[Issue] = Field(
        default_factory=list,
        description="학습 범위 이탈 주의 정보",
    )
    completed: bool = Field(
        False,
        description="전체 대상·행이 성공했을 때만 True",
    )
    error: Issue | None = Field(None, description="전체 요청 실패 원인")




class RankedCandidate(BaseModel):
    """순위표의 후보 한 줄."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="후보 식별자")
    source_row: int = Field(..., description="원본 행 번호")
    value: float = Field(..., description="원래 정밀도의 예측값")
    sort_value: float = Field(..., description="정렬에 쓴 값 (target은 절대 차이)")
    rank: int = Field(..., description="공동 순위 (1,1,3)")
    range_exceeded: bool = Field(False, description="학습 범위 이탈 여부 표시")


class RankingResult(BaseModel):
    """후보 순위 결과. 모델 재실행 없이 기준만 바꿔 재정렬할 수 있다."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="원본 요청 ID")
    target_id: str = Field(..., description="순위 대상 ID")
    unit: str | None = Field(None, description="대상 단위")
    direction: Literal["maximize", "minimize", "target"] = Field(..., description="순위 기준")
    target_value: float | None = Field(None, description="target 기준의 목표값")
    ranked: list[RankedCandidate] = Field(default_factory=list, description="순위 매긴 후보")


# ── PredictorAdapter 프로토콜 ──────────────────────────────────────────────────


@runtime_checkable
class PredictorAdapter(Protocol):
    """입력 표를 받아 지정한 대상의 결과를 반환하는 공통 계약.

    초기 구현은 대상별 sklearn Pipeline 어댑터를 사용한다.
    공동 출력·연계 구조가 필요해지면 같은 계약의 어댑터를 추가한다.
    """

    @property
    def target_id(self) -> str:
        """이 어댑터가 담당하는 예측 대상 ID."""
        ...

    def predict(self, candidates: CandidateTable) -> TargetPrediction:
        """후보 표를 받아 대상 하나의 예측 결과를 반환한다."""
        ...


# ── 공개 목록 ──────────────────────────────────────────────────────────────────

__all__ = [
    "PhysicalConstraint",
    "AllowedCategory",
    "InputSpec",
    "OutputConstraint",
    "TargetSpec",
    "ProblemSpec",
    "Issue",
    "CandidateTable",
    "TargetPrediction",
    "PredictionBatch",
    "PredictorAdapter",
]
