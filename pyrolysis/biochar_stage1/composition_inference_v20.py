"""Experimental, portable Stage1 BET predictor (v20).

Nonactivated biomass, single-stage pyrolysis ONLY. Input: raw precursor
composition + pyrolysis conditions. Neither TPV nor adsorption qe is predicted.
The bundled forests are exported as tree arrays to avoid cross-version pickle
loading. Training/provenance and grouped validation must be kept distinct.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

PROCESS = ("temperature_C", "time_min", "heating_rate_C_min")
ELEMENTAL = ("C_pct", "H_pct", "N_pct", "O_pct")
PROXIMATE = ("VM_pct", "Ash_pct", "FC_pct")

FEATURES = {
    "proximate": list(PROCESS + PROXIMATE),
    "full": list(PROCESS + ELEMENTAL + PROXIMATE),
}
RAW_MAP = {
    "VM_pct": "raw_VM_wt_pct", "Ash_pct": "raw_ash_wt_pct", "FC_pct": "raw_FC_wt_pct",
    "C_pct": "raw_C_wt_pct", "H_pct": "raw_H_wt_pct",
    "N_pct": "raw_N_wt_pct", "O_pct": "raw_O_wt_pct",
}
PROCESS_MAP = {
    "temperature_C": "pyrolysis_temperature_C",
    "time_min": "pyrolysis_hold_min",
    "heating_rate_C_min": "heating_rate_C_min",
}
MODEL_FILENAME = "experimental_composition_bet_v20.json"


def default_model_path(root: Path) -> Path:
    return Path(root) / "deployment" / "experimental_stage1_bet" / MODEL_FILENAME


def load_portable_model(path: Path) -> dict:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_version") != 1 or obj.get("research_status") != "EXPLORATORY_NOT_VALIDATED_FOR_DEPLOYMENT":
        raise ValueError("Unrecognized experimental BET model metadata")
    for name, spec in obj["models"].items():
        if name not in FEATURES or spec["features"] != FEATURES[name]:
            raise ValueError("Invalid feature contract")
        if not spec["trees"]:
            raise ValueError("Empty forest")
    return obj


def _convert_numeric(value, key):
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"필수 입력 누락: {key}")
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"숫자가 아닌 값: {key}") from None
    if not math.isfinite(number):
        raise ValueError(f"유효하지 않은 수: {key}")
    return number


def assess_inputs(
    model: Mapping,
    process: Mapping,
    composition: Mapping,
    *,
    elemental_basis: str | None,
    proximate_basis: str | None,
    process_type: str,
    activated: bool,
) -> dict:
    """Check strict applicability before predicting. Does not guess missing chemistry.

    Unknown/unseen feedstock name is allowed because it is NOT a fitted feature;
    this does not establish true external-feedstock generalization.
    """
    errors: list[str] = []
    if process_type != "SINGLE_STAGE_PYROLYSIS" or activated:
        errors.append("비활성화 원료의 단일 단계 열분해에만 적용할 수 있습니다.")
    if proximate_basis != "dry":
        errors.append("Ash/VM/FC는 반드시 원료의 dry basis 실측값이어야 합니다.")
    data = {}
    for k, v in PROCESS_MAP.items():
        try:
            data[k] = _convert_numeric(process.get(v), k)
        except ValueError as exc:
            errors.append(str(exc))
    for k in PROXIMATE:
        try:
            data[k] = _convert_numeric(composition.get(RAW_MAP[k]), k)
        except ValueError as exc:
            errors.append(str(exc))
    full_present = all(composition.get(RAW_MAP[k]) not in (None, "") for k in ELEMENTAL)
    variant = "full" if full_present else "proximate"
    if full_present:
        if elemental_basis not in ("dry ash-free (daf)", "daf"):
            errors.append("C/H/N/O는 원료의 dry ash-free (daf) 기준이어야 합니다.")
        for k in ELEMENTAL:
            try:
                data[k] = _convert_numeric(composition.get(RAW_MAP[k]), k)
            except ValueError as exc:
                errors.append(str(exc))
    else:
        supplied = [k for k in ELEMENTAL if composition.get(RAW_MAP[k]) not in (None, "")]
        if supplied:
            errors.append("원소분석은 C/H/N/O 4개가 모두 있을 때만 반영됩니다. 빈 항목을 추정해 채우지 않습니다.")

    if not errors:
        for k, v in data.items():
            if v < 0 or (k in ELEMENTAL + PROXIMATE and v > 100):
                errors.append(f"{k}: 음수 또는 100 wt% 초과 값은 허용하지 않습니다.")
        prox_sum = sum(data[k] for k in PROXIMATE)
        if abs(prox_sum - 100) > 2.0:
            errors.append(f"원료 dry 공업분석의 VM + Ash + FC 합계는 약 100%여야 합니다 (현재 {prox_sum:.2f}%).")
        if variant == "full":
            elem_sum = sum(data[k] for k in ELEMENTAL)
            if not (95 <= elem_sum <= 101):
                errors.append(f"원료 daf 기준 C+H+N+O 합계({elem_sum:.2f}%)를 점검해 주세요 (S 등 미포함 가능).")

    # A marginal-range check is NOT proof that the joint input combination is
    # represented in training.  Retain strict in-range prediction by default.
    # A separately opted-in, clearly labelled provisional estimate can be
    # shown for *composition-only* OOD, provided processing inputs are in-range.
    # No process-temperature extrapolation is enabled by this relaxation.
    outside = []
    outside_keys = []
    strong_outside_keys = []
    if not errors:
        limits = model["models"][variant]["feature_range"]
        for key in FEATURES[variant]:
            low, high = limits[key]
            if data[key] < low or data[key] > high:
                outside_keys.append(key)
                outside.append(f"{key}: {data[key]:g} (학습범위 {low:g}–{high:g})")
                span = high-low
                if span > 0 and (low-data[key] if data[key]<low else data[key]-high)/span >= 0.2:
                    strong_outside_keys.append(key)
    in_range = not errors and not outside
    composition_only_ood = bool(outside_keys) and all(key not in PROCESS for key in outside_keys)
    return {
        "variant": variant, "values": data, "errors": errors,
        "outside_training_range": outside,
        "outside_feature_names": outside_keys,
        "strongly_outside_feature_names": strong_outside_keys,
        "inside_training_range": in_range,
        "can_explore_composition_ood": not errors and composition_only_ood,
        "predictable": in_range,  # backwards-compatible: normal prediction is strictly in-range
        "used_features": FEATURES[variant],
    }


def forest_predict_one(spec: Mapping, values: Mapping) -> float:
    """Exact sklearn RandomForestRegressor tree-average for one numeric row.

    Exported tree indices follow sklearn's <= threshold branch convention.
    """
    x = [float(values[k]) for k in spec["features"]]
    pred = 0.0
    for tree in spec["trees"]:
        node = 0
        left = tree["left"]
        while left[node] != -1:
            feat = tree["feature"][node]
            node = left[node] if x[feat] <= tree["threshold"][node] else tree["right"][node]
        pred += tree["value"][node]
    return pred / len(spec["trees"])


def estimate_bet(model: Mapping, check: Mapping, *, allow_composition_ood: bool = False) -> float:
    """Predict BET; require explicit opt-in for composition-only OOD.

    Random forests produce an output outside feature limits but do NOT
    extrapolate physical trends there. Such values are provisional only.
    Invalid units, unsupported manufacturing routes or out-of-range process
    conditions remain blocked, even when allow_composition_ood=True.
    """
    normal = bool(check.get("predictable", False))
    provisional = allow_composition_ood and bool(check.get("can_explore_composition_ood", False))
    if not (normal or provisional):
        raise ValueError("입력 오류 또는 적용 불가 조건입니다. 범위 밖 조성은 별도 동의 후 참고값만 제공합니다.")
    return forest_predict_one(model["models"][check["variant"]], check["values"])
