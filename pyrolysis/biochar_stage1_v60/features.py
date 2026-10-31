"""
features.py — Stage 1 v60 입력 계약과 모델 사용 변수 명세.

v60 는 **제조 전 원료 정보 + 제조·활성화 조건만으로** BET·TPV 를 예측하는 연구용
프로토타입이다(REPORT_KO §1). 입력은 제조조건 중심의 JSON 이며, 예측기(`predict_stage1_v60.py`)
내부에서 Eng59 파생 특징으로 확장된다.

검증된 사실 (실제 모델 객체 점검, _v60_probe/_v60_cols 결과):
  - 모델 파이프라인의 ColumnTransformer 는 `features` 메타데이터에 명시된 56개 열
    (수치 40 + 범주 16)만 사용하고, 나머지(원료 조성 등)는 remainder='drop' 으로 버린다.
  - 따라서 **원료 원소·공업분석 조성(feedstock_C/H/O/N/ash/VM/FC/moisture)은 전처리에서
    drop 되어 실제 추정기에 전달되지 않는다.** master CSV·pickle 의 feature_names_in_ 에는
    전체 149열이 보이지만, 이는 fit 시 전체 DataFrame 을 넘겼기 때문이며 실제 사용 변수가 아니다.
  - 즉 REPORT 의 "원소조성을 입력변수로 사용하지 않는다"는 전처리 drop 으로 실현된다.

이 모듈은 사람이 확인·표시하기 위한 명세이며, 예측 자체는 패키지 예측기에 위임한다.
"""

from __future__ import annotations

__all__ = [
    "V60_PRIMARY_INPUT_FIELDS",
    "V60_TARGETS",
    "V60_MODEL_USED_NUMERIC",
    "V60_MODEL_USED_CATEGORICAL",
    "V60_COMPOSITION_FIELDS_DROPPED",
    "V60_APPLICABILITY",
]

# 사용자가 JSON 으로 제공하는 1차 제조조건 입력(예측기가 받는 주요 키).
# example_input.json 기준. 모든 키가 필수는 아니며 미제공 시 결측으로 처리된다.
V60_PRIMARY_INPUT_FIELDS = [
    {"id": "feedstock", "label_ko": "원료", "unit": None, "kind": "category"},
    {"id": "feedstock_group", "label_ko": "원료군", "unit": None, "kind": "category"},
    {"id": "pyrolysis_temperature_C", "label_ko": "열분해 온도", "unit": "°C", "kind": "float"},
    {"id": "pyrolysis_hold_min", "label_ko": "열분해 유지시간", "unit": "min", "kind": "float"},
    {"id": "pyrolysis_atmosphere", "label_ko": "열분해 분위기", "unit": None, "kind": "category"},
    {"id": "pyrolysis_N2_flow_mL_min", "label_ko": "N2 유량", "unit": "mL/min", "kind": "float"},
    {"id": "heating_rate_C_min", "label_ko": "승온속도", "unit": "°C/min", "kind": "float"},
    {"id": "activation_method", "label_ko": "활성화 방법", "unit": None, "kind": "category"},
    {"id": "activation_agent", "label_ko": "활성화제", "unit": None, "kind": "category"},
    {"id": "activation_ratio_value", "label_ko": "활성화 비율", "unit": None, "kind": "float"},
    {"id": "activation_ratio_basis", "label_ko": "활성화 비율 기준", "unit": None, "kind": "category"},
    {"id": "activation_temperature_C", "label_ko": "활성화 온도", "unit": "°C", "kind": "float"},
    {"id": "activation_time_min", "label_ko": "활성화 시간", "unit": "min", "kind": "float"},
    {"id": "process_stage", "label_ko": "공정 단계", "unit": None, "kind": "category"},
]

V60_TARGETS = [
    {"id": "BET", "label_ko": "BET 비표면적", "unit": "m2/g"},
    {"id": "TPV", "label_ko": "총 기공부피", "unit": "cm3/g"},
]

# 모델 파이프라인이 실제로 사용하는 변수(features 메타데이터 기준, 점검으로 확인).
V60_MODEL_USED_NUMERIC = [
    "pyrolysis_temperature_C", "pyrolysis_hold_min", "heating_rate_C_min",
    "pyrolysis_N2_flow_mL_min", "activation_ratio_value", "activation_temperature_C",
    "activation_time_min", "activation_gas_flow_mL_min",
    "activation_carrier_gas_flow_mL_min", "activation_steam_water_rate_mL_min",
    "activation_steam_fraction_pct", "activation_CO2_fraction_pct",
    "component_1_fraction", "component_2_fraction",
    "activation_agent_concentration_pct_wv", "activation_solution_to_solid_ratio",
    "pretreatment_temperature_C", "pretreatment_time_min",
    "post_treatment_time_min", "post_treatment_temperature_C",
    # eng_* 파생 특징
    "eng_is_activated", "eng_is_chemical", "eng_is_physical", "eng_is_steam",
    "eng_is_co2", "eng_is_zncl2", "eng_is_koh", "eng_effective_activation_temp_C",
    "eng_effective_activation_time_min", "eng_pyro_severity", "eng_activation_severity",
    "eng_temp_gap", "eng_ratio_log1p", "eng_temp_ratio_interaction",
    "eng_physical_temp_time", "eng_chemical_temp_ratio", "eng_has_blend",
    "eng_has_pretreatment", "eng_has_postwash", "eng_steam_co2_balance",
]
V60_MODEL_USED_CATEGORICAL = [
    "feedstock_group", "feedstock", "activation_method", "activation_agent",
    "activation_agent_2", "activation_ratio_basis", "pyrolysis_atmosphere",
    "mixture_component_1", "mixture_component_2", "process_stage", "v59_route_family",
    "pretreatment_agent", "post_treatment_agent", "particle_size_mesh",
    "eng_activation_timing", "eng_agent_family",
]

# 모델에 '전달되지만 drop' 되는 원료 조성 변수 (실제 추정기 미사용, 검증됨).
V60_COMPOSITION_FIELDS_DROPPED = [
    "feedstock_C_pct_original", "feedstock_H_pct_original", "feedstock_O_pct_original",
    "feedstock_N_pct_original", "feedstock_S_pct_original", "feedstock_ash_pct_original",
    "feedstock_VM_pct_original", "feedstock_FC_pct_original",
    "feedstock_moisture_pct_original", "feedstock_elemental_basis",
]

V60_APPLICABILITY = {
    "domain": "제조 전 원료 정보 + 제조·활성화 조건 기반 바이오차/활성탄 BET·TPV 예측 (연구용 프로토타입)",
    "not_used_inputs": "제조 후 BET·TPV·원소조성·pH·회분 등은 입력으로 사용하지 않음(조성 변수는 전처리 drop).",
    "support_grades": {
        "HIGH": "동일 route 에서 비교적 지지가 있는 조건",
        "MODERATE": "일부 조건이 학습범위 밖이거나 자료가 제한적",
        "LOW": "외삽 가능성이 높아 수치를 강하게 신뢰하면 안 됨",
    },
    "research_interval_note": "research_interval_approx 는 잔차 기반 연구용 범위이며 산업용 신뢰구간이 아님.",
    "status": "research_prototype_complete_not_production_validated",
    "external_generalization_warning": (
        "잠금 외부 challenge(3 신규 DOI, 23행)에서 BET R²=-0.891, TPV R²=-3.331 로 음수였다. "
        "post-lock 전체자료 LODO(R² BET 0.380 / TPV 0.222)는 challenge 데이터가 학습에 포함된 "
        "진단값이며 독립 외부검증 성능이 아니다. 실험 측정을 대체하는 확정값으로 쓰면 안 된다."
    ),
}
