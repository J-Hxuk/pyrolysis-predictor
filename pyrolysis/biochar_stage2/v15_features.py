"""
v15_features.py — Stage2 모델 A/B/C/D 입력 계약 (11_Feature_Contract_A_B_C_D.csv 근거).

지시·계약 요지:
  - ID·분할·provenance·타깃·유도값·검증 플래그는 어떤 모델에도 입력하지 않는다.
  - A = 흡착조건만.  B = A + 원료명/공정.  C = B + 실측 BET/TPV.  D = B + Stage1 예측(비활성).
  - feedstock 명칭은 범주로만 쓰고 이름에서 조성을 유도하지 않는다.
  - raw 원료 조성은 선택(OPTIONAL_AFTER_BASIS_AUDIT)이며 필수 입력으로 강제하지 않는다.
  - 열분해 후 바이오차 원소조성(BIOCHAR_POST_*)을 원료 조성으로 대체하지 않는다.
"""

from __future__ import annotations

__all__ = [
    "TARGET_COLUMN",
    "TARGET_UNIT",
    "GROUP_DOI",
    "GROUP_FEEDSTOCK",
    "MASS_BALANCE_CAP_COL",
    "C0_COL",
    "DOSE_COL",
    "ADSORPTION_INPUTS",
    "MANUFACTURING_INPUTS",
    "RAW_COMPOSITION_OPTIONAL",
    "MEASURED_PROPERTY_INPUTS",
    "STAGE1_PREDICTED_INPUTS",
    "FORBIDDEN_FIELDS",
    "model_inputs",
    "MODEL_DEFS",
]

# ── 타깃·분할·물리제약 열 ─────────────────────────────────────────────────────
TARGET_COLUMN = "target_value"          # Core 97: qe_reported_mg_g 값
TARGET_UNIT = "mg/g"
GROUP_DOI = "doi"                        # DOI 그룹 분할 단위 (== doi_group)
GROUP_FEEDSTOCK = "feedstock_label_normalized"  # 원료 외삽 검증 단위
MASS_BALANCE_CAP_COL = "mass_balance_cap_mg_g"
C0_COL = "C0_mg_L"
DOSE_COL = "dose_g_L"

# ── 모델별 허용 입력 ─────────────────────────────────────────────────────────
# A: 흡착조건만 (오염물질·C0·pH·투입량·온도·접촉시간)
ADSORPTION_INPUTS = [
    "pollutant",
    "C0_mg_L",
    "initial_pH",
    "dose_g_L",
    "adsorption_T_C",
    "contact_time_min",
]

# B 추가: 원료명/분류 + 공정 변수 (조성은 포함하지 않음 — 별도 선택 후보)
MANUFACTURING_INPUTS = [
    "feedstock_label_normalized",
    "manufacturing_route",
    "process_step_count",
    "pyrolysis_temperature_C",
    "pyrolysis_hold_min",
    "heating_rate_C_min",
    "pyrolysis_atmosphere",
    "activation_or_modification",
    "secondary_temperature_C",
    "secondary_hold_min",
]

# B/C/D 선택 후보(기본 필수 아님; basis 감사 후에만, 결측 매우 높음)
RAW_COMPOSITION_OPTIONAL = [
    "raw_C_wt_pct", "raw_H_wt_pct", "raw_O_wt_pct", "raw_N_wt_pct", "raw_S_wt_pct",
    "raw_ash_wt_pct", "raw_VM_wt_pct", "raw_FC_wt_pct", "raw_moisture_wt_pct",
    "raw_cellulose_pct", "raw_hemicellulose_pct", "raw_lignin_pct",
]

# C 추가: 실측 물성 (사용자 제조조건만으로는 모름 → 배포 추천 모델로 쓰지 않음)
MEASURED_PROPERTY_INPUTS = ["BET_measured_m2_g", "TPV_measured_cm3_g"]

# D 추가(비활성): Stage1 v60 예측 물성. 이번 작업에서 실제 전달 비활성화.
STAGE1_PREDICTED_INPUTS = ["BET_predicted_m2_g", "TPV_predicted_cm3_g"]

# ── 어떤 모델에도 입력 금지 (ID·분할·provenance·타깃·유도·검증·char 조성) ──────
FORBIDDEN_FIELDS = {
    # SPLIT_OR_PROVENANCE
    "doi_group", "observation_id", "material_uid", "sample_group",
    "experiment_series_id", "source_url", "target_origin", "target_qualification",
    "doi", "feedstock_group_for_novelty_check", "model_use_tier", "stage1_v60_eligible",
    "route_warning", "warnings", "feedstock_raw",  # feedstock_raw 는 검색 단서(자연어), 모델은 normalized 범주 사용
    # LEAKAGE_OR_TARGET
    "target_value", "target_name", "target_unit", "mass_balance_cap_mg_g",
    "physical_cap_pass", "reported_removal_pct", "qe_derived_from_removal_mg_g",
    "qmax_mg_g", "BET_POST_C", "O_POST",
    # 활성화제 원문 라벨(역할 정규화 전): 모델 확정 입력 아님 (REVIEW_REQUIRED)
    "reported_activation_agent", "reported_impregnation_agent",
    "raw_composition_basis", "raw_composition_evidence",
}


def model_inputs(model: str, *, include_raw_composition: bool = False) -> list[str]:
    """모델(A/B/C)의 허용 입력 열 목록을 반환한다.

    include_raw_composition=True 이면 B/C 에 raw 조성 선택 후보를 추가한다
    (기본 False — 결측률이 매우 높아 필수 입력으로 강제하지 않음).
    D 는 Stage1 연계 비활성으로 여기서 학습 입력을 제공하지 않는다.
    """
    model = model.upper()
    if model == "A":
        cols = list(ADSORPTION_INPUTS)
    elif model == "B":
        cols = list(ADSORPTION_INPUTS) + list(MANUFACTURING_INPUTS)
        if include_raw_composition:
            cols += list(RAW_COMPOSITION_OPTIONAL)
    elif model == "C":
        cols = list(ADSORPTION_INPUTS) + list(MANUFACTURING_INPUTS) + list(MEASURED_PROPERTY_INPUTS)
        if include_raw_composition:
            cols += list(RAW_COMPOSITION_OPTIONAL)
    else:
        raise ValueError(
            f"지원하지 않는 모델: {model!r}. A/B/C 만 학습한다 (D 는 Stage1 연계 비활성)."
        )
    # 금지 열이 섞이지 않았는지 방어적으로 제거
    cols = [c for c in cols if c not in FORBIDDEN_FIELDS]
    return cols


# 모델 정의 메타(표시·검증용)
MODEL_DEFS = {
    "A": {"name": "흡착조건 Baseline", "desc": "흡착조건만으로 qe 예측"},
    "B": {"name": "제조조건 직접 예측(핵심)", "desc": "A + 원료명/공정 → qe. 사용자 배포 핵심 모델"},
    "C": {"name": "실측 물성 추가(참고)", "desc": "B + 실측 BET/TPV. 물성 정보 기여 평가용(배포 추천 아님)"},
    "D": {"name": "Stage1 예측 물성(비활성)", "desc": "B + Stage1 예측 BET/TPV. 이번 작업 비활성"},
}
