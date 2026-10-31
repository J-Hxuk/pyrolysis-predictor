"""
schema.py — Stage2 흡착 데이터 입력 스키마 정의.

지시서 §3 이 명시한 '향후 수용할' 입력 항목을 코드로 고정한다. 모든 필드가 이미
준비됐다는 뜻은 아니며, 데이터셋 연결 시 검증 모듈(validate_stage2_frame)이
적격성을 판정한다.

필드 분류:
  - identity     : material_id, original_experiment_doi, provenance_group
  - feedstock    : feedstock 및 제조조건
  - property     : measured_BET_m2_g, measured_TPV_cm3_g  (Stage1 결합 지점)
  - condition    : pollutant, initial_concentration_mg_L, solution_pH,
                   adsorbent_dose_g_L, adsorption_temperature_C, contact_time_min,
                   equilibrium_concentration_mg_L
  - target       : measured_qe_mg_g  (평형 검증된 실측 qe)
  - status       : equilibrium_verified, measurement_status, qa_status
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Stage2Field",
    "STAGE2_FIELDS",
    "STAGE2_FIELD_IDS",
    "IDENTITY_FIELDS",
    "PROPERTY_FIELDS",
    "CONDITION_FIELDS",
    "TARGET_FIELD",
    "STATUS_FIELDS",
    "VERIFIED_EQUILIBRIUM_VALUES",
    "APPROVED_MEASUREMENT_STATUS",
    "QE_VALUE_KIND",
]


FieldKind = Literal["identity", "feedstock", "property", "condition", "target", "status"]


@dataclass(frozen=True)
class Stage2Field:
    """Stage2 입력 필드 하나의 규격."""

    id: str
    label_ko: str
    kind: FieldKind
    dtype: Literal["str", "float", "bool", "category"]
    unit: str | None
    required_for_training: bool
    description: str


# 지시서 §3 의 입력 항목. required_for_training 은 '학습 대상 승인'에 필요한지 여부.
STAGE2_FIELDS: list[Stage2Field] = [
    Stage2Field("material_id", "시료 ID", "identity", "str", None, True,
                "Stage1 바이오차 시료 식별자. Stage1 결합의 키."),
    Stage2Field("original_experiment_doi", "원실험 DOI", "identity", "str", None, False,
                "원 실험 논문 DOI. 수집 논문 DOI 와 구분한다."),
    Stage2Field("provenance_group", "출처 그룹", "identity", "str", None, True,
                "GroupKFold 분할 단위. 출처 누설 방지."),
    Stage2Field("feedstock", "원료", "feedstock", "str", None, False,
                "바이오차 원료 및 제조조건 요약."),
    Stage2Field("measured_BET_m2_g", "측정 BET", "property", "float", "m2/g", False,
                "실측 BET 비표면적 (측정값). Stage1 예측 BET 와 구분한다."),
    Stage2Field("measured_TPV_cm3_g", "측정 총공극부피", "property", "float", "cm3/g", False,
                "실측 총공극부피 (측정값). Stage1 예측 TPV 와 구분한다."),
    Stage2Field("pollutant", "오염물질", "condition", "str", None, True,
                "흡착 대상 오염물질."),
    Stage2Field("initial_concentration_mg_L", "초기 농도", "condition", "float", "mg/L", True,
                "흡착 실험 초기 농도 C0."),
    Stage2Field("solution_pH", "용액 pH", "condition", "float", None, False,
                "흡착 용액 pH."),
    Stage2Field("adsorbent_dose_g_L", "흡착제 투입량", "condition", "float", "g/L", True,
                "흡착제 농도(dose)."),
    Stage2Field("adsorption_temperature_C", "흡착 온도", "condition", "float", "°C", False,
                "흡착 실험 온도."),
    Stage2Field("contact_time_min", "접촉시간", "condition", "float", "min", False,
                "흡착 접촉시간."),
    Stage2Field("equilibrium_concentration_mg_L", "평형 농도", "condition", "float", "mg/L", False,
                "평형 농도 Ce. 평형 검증의 근거."),
    Stage2Field("measured_qe_mg_g", "실측 평형흡착량", "target", "float", "mg/g", True,
                "평형 검증된 실측 qe. 미검증 reported capacity 와 구분한다."),
    Stage2Field("equilibrium_verified", "평형 검증 여부", "status", "bool", None, True,
                "평형 도달이 검증됐는지. 미검증이면 학습 대상 자동 승인 금지."),
    Stage2Field("measurement_status", "측정 상태", "status", "category", None, True,
                "measured / langmuir_qmax / model_predicted / reported_unverified 중 하나."),
    Stage2Field("qa_status", "QA 상태", "status", "category", None, False,
                "품질검토 상태."),
]

STAGE2_FIELD_IDS: list[str] = [f.id for f in STAGE2_FIELDS]

IDENTITY_FIELDS = [f.id for f in STAGE2_FIELDS if f.kind == "identity"]
PROPERTY_FIELDS = [f.id for f in STAGE2_FIELDS if f.kind == "property"]
CONDITION_FIELDS = [f.id for f in STAGE2_FIELDS if f.kind == "condition"]
STATUS_FIELDS = [f.id for f in STAGE2_FIELDS if f.kind == "status"]
TARGET_FIELD = "measured_qe_mg_g"

# 평형 검증으로 인정하는 equilibrium_verified 값
VERIFIED_EQUILIBRIUM_VALUES = {True, "true", "True", "TRUE", "yes", "1", 1}

# 학습 대상으로 승인 가능한 measurement_status (실측만). 나머지는 승인 금지.
APPROVED_MEASUREMENT_STATUS = {"measured"}

# qe 값의 종류 구분 (혼동 금지):
#   measured            : 평형 검증된 실측 qe (학습 후보)
#   langmuir_qmax       : Langmuir 피팅 최대 흡착량 (실측 qe 아님)
#   model_predicted     : 모델 예측값 (실측 아님)
#   reported_unverified : 논문 보고 미검증 capacity (실측 qe 로 승격 금지)
QE_VALUE_KIND = {
    "measured": "평형 검증된 실측 qe",
    "langmuir_qmax": "Langmuir 피팅 qmax (실측 등온 데이터 피팅 결과, 단일 qe 아님)",
    "model_predicted": "모델 예측값 (실측 아님)",
    "reported_unverified": "논문 보고 미검증 capacity (실측 qe 로 승격 금지)",
}


def field_rows() -> list[dict]:
    """표시·문서용으로 필드 메타데이터를 반환한다."""
    return [
        {
            "id": f.id, "label_ko": f.label_ko, "kind": f.kind, "dtype": f.dtype,
            "unit": f.unit, "required_for_training": f.required_for_training,
            "description": f.description,
        }
        for f in STAGE2_FIELDS
    ]
