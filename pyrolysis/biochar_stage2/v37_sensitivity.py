"""Pure helpers for on-change sensitivity exploration of frozen Stage 2 models.

No Streamlit dependency. Calls persisted model input contract, never changes
training, never treats mass-balance clipping as a trained prediction.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2.input_support import predict_from_contract

# Canonical sliders only; no inferred raw biomass chemistry.
VARIABLES = {
    "initial_pH": ("초기 pH", 0.0, 14.0, 0.1, 5.0),
    "C0_mg_L": ("초기농도 C₀ (mg/L)", 1.0, 500.0, 1.0, 100.0),
    "dose_g_L": ("흡착제 투입량 (g/L)", 0.1, 10.0, 0.1, 2.0),
    "contact_time_min": ("접촉시간 (min)", 0.0, 2880.0, 10.0, 1440.0),
    "adsorption_T_C": ("흡착온도 (°C)", 0.0, 70.0, 1.0, 25.0),
}


def eval_one(model_obj: Mapping, inputs: Mapping) -> dict:
    c0, dose = float(inputs["C0_mg_L"]), float(inputs["dose_g_L"])
    if not all(math.isfinite(v) and v > 0 for v in (c0, dose)):
        raise ValueError("C0, dose는 양의 유한한 값이어야 합니다")
    raw, clipped, cap = predict_from_contract(model_obj, inputs)
    if not all(math.isfinite(x) for x in (raw, clipped, cap)):
        raise ValueError("비유한 예측 또는 물질수지 상한")
    return {
        "qe_raw": raw, "qe_clipped": clipped, "cap": cap,
        "violates": bool(raw < 0 or raw > cap),
        "removal_raw_pct": 100.0 * raw / cap,
        "removal_clipped_pct": 100.0 * clipped / cap,
    }


def sensitivity_frame(models: Mapping, base: Mapping, *, variable: str,
                      values, show_model_b: bool = False) -> pd.DataFrame:
    """Evaluate one variable sweep. Do not silently impute fields or clip raw values."""
    if variable not in VARIABLES:
        raise ValueError(f"지원하지 않는 분석 변수: {variable}")
    labels = ["A"] + (["B"] if show_model_b else [])
    labels = [label for label in labels if label in models]
    rows = []
    for value in values:
        val = float(value)
        if not math.isfinite(val):
            raise ValueError("가변 입력이 유한하지 않습니다")
        row = dict(base)
        row[variable] = val
        for label in labels:
            pred = eval_one(models[label], row)
            rows.append({"x": val, "model": label, **pred})
    return pd.DataFrame(rows)


def make_sweep_values(variable: str, low: float, high: float, current: float,
                      n: int = 31) -> list[float]:
    if variable not in VARIABLES or not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("유효하지 않은 민감도 범위")
    if low >= high or n < 2 or n > 101:
        raise ValueError("민감도 상·하한 또는 점 개수 오류")
    return sorted(set(float(v) for v in np.r_[np.linspace(low, high, n), current]
                      if low <= v <= high))
