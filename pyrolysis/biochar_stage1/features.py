"""
features.py — Stage1 BET 입력변수 계약.

eval37/INPUT_CONTRACT_KO.md 의 정의를 코드로 고정한다.

입력 10개 (순서 고정):
  1. C_pct              원료 원소분석 C, 질량%, dry ash-free basis
  2. H_pct              원료 원소분석 H, 질량%, dry ash-free basis
  3. N_pct              원료 원소분석 N, 질량%, dry ash-free basis
  4. O_pct              원료 원소분석 O, 질량%, dry ash-free basis (차감법)
  5. VM_pct             원료 공업분석 휘발분, 질량%, dry basis
  6. Ash_pct            원료 공업분석 회분, 질량%, dry basis
  7. FC_pct             원료 공업분석 고정탄소, 질량%, dry basis
  8. temperature_C      열분해 온도, °C
  9. time_min           유지시간, min
 10. heating_rate_C_min 승온속도, °C/min

출력:
  value -> BET 비표면적, m²/g

중요:
  - 원료 조성과 생성 바이오차 조성을 혼동하지 않는다.
  - C/H/N/O 합을 임의로 100 으로 정규화하지 않는다.
  - 공업분석 VM/Ash/FC 와 원소분석 C/H/N/O 의 basis 가 다르다.
  - 적용 범위: Leng2022 기반 비활성화(unmodified) 바이오매스 열분해.
  - 활성화제·함침·세척·반응기·분위기 등은 입력에 없다 (고려한 것처럼 표시 금지).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 입력 순서는 모델 학습·예측·저장에서 반드시 동일하게 유지한다.
BET_INPUT_ORDER: list[str] = [
    "C_pct",
    "H_pct",
    "N_pct",
    "O_pct",
    "VM_pct",
    "Ash_pct",
    "FC_pct",
    "temperature_C",
    "time_min",
    "heating_rate_C_min",
]

BET_TARGET_COL = "value"  # eval37 base.csv 의 BET 값 열
BET_TARGET_UNIT = "m2/g"
BET_GROUP_COL = "provenance_group"  # 출처 그룹 (GroupKFold 분할 단위)


@dataclass(frozen=True)
class InputVariable:
    """입력 변수 하나의 메타데이터 (단위·basis·설명)."""

    id: str
    label_ko: str
    unit: str
    basis: str
    group: str  # "elemental" | "proximate" | "pyrolysis"
    description: str


# 각 입력의 단위·분석 basis 를 명시한다. basis 는 원료(feedstock) 기준이다.
BET_INPUT_VARIABLES: dict[str, InputVariable] = {
    "C_pct": InputVariable(
        "C_pct", "탄소 C", "wt%", "dry ash-free (daf)", "elemental",
        "원료 원소분석 탄소 함량. 생성 바이오차가 아닌 원료 기준."),
    "H_pct": InputVariable(
        "H_pct", "수소 H", "wt%", "dry ash-free (daf)", "elemental",
        "원료 원소분석 수소 함량."),
    "N_pct": InputVariable(
        "N_pct", "질소 N", "wt%", "dry ash-free (daf)", "elemental",
        "원료 원소분석 질소 함량."),
    "O_pct": InputVariable(
        "O_pct", "산소 O", "wt%", "dry ash-free (daf)", "elemental",
        "원료 원소분석 산소 함량. 논문에서 차감법으로 산출."),
    "VM_pct": InputVariable(
        "VM_pct", "휘발분 VM", "wt%", "dry basis", "proximate",
        "원료 공업분석 휘발분."),
    "Ash_pct": InputVariable(
        "Ash_pct", "회분 Ash", "wt%", "dry basis", "proximate",
        "원료 공업분석 회분."),
    "FC_pct": InputVariable(
        "FC_pct", "고정탄소 FC", "wt%", "dry basis", "proximate",
        "원료 공업분석 고정탄소."),
    "temperature_C": InputVariable(
        "temperature_C", "열분해 온도", "°C", "—", "pyrolysis",
        "열분해 최고 온도(HTT)."),
    "time_min": InputVariable(
        "time_min", "유지시간", "min", "—", "pyrolysis",
        "최고 온도 유지시간."),
    "heating_rate_C_min": InputVariable(
        "heating_rate_C_min", "승온속도", "°C/min", "—", "pyrolysis",
        "승온속도."),
}


@dataclass(frozen=True)
class TargetVariable:
    """출력 변수 메타데이터."""

    id: str
    label_ko: str
    unit: str
    description: str


BET_TARGET = TargetVariable(
    id="BET_m2_g",
    label_ko="BET 비표면적",
    unit=BET_TARGET_UNIT,
    description="원료 조성·열분해 조건으로 예측하는 바이오차 BET 비표면적.",
)

# 적용 범위 선언 (지시서 §2). 모델에 없는 인자를 고려한 것처럼 표시하지 않는다.
APPLICABILITY_SCOPE = {
    "domain": "Leng2022 기반 비활성화(unmodified) 바이오매스 열분해",
    "elemental_basis": "dry ash-free (daf), 원료 기준",
    "proximate_basis": "dry basis, 원료 기준",
    "not_modeled": [
        "활성화제", "함침제", "세척", "반응기 종류", "분위기", "입도", "가스 유량",
    ],
    "note": (
        "입력에 없는 인자(활성화·함침·세척·반응기·분위기 등)의 영향은 검증하지 않았다. "
        "활성화 공정 등으로 확장하려면 별도 입력과 평가가 필요하다."
    ),
}


def input_variable_rows() -> list[dict[str, Any]]:
    """표시·문서용으로 입력 변수 메타데이터를 순서대로 반환한다."""
    rows = []
    for vid in BET_INPUT_ORDER:
        v = BET_INPUT_VARIABLES[vid]
        rows.append({
            "id": v.id, "label_ko": v.label_ko, "unit": v.unit,
            "basis": v.basis, "group": v.group, "description": v.description,
        })
    return rows
