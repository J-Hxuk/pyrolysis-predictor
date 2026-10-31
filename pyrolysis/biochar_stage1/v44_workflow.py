"""v44 presentation workflow: an independent BET-v20 sweep and explicit condition handoff.

This module never fits models, forwards BET into Stage2, or claims the plotted
temperature response is a validated manufacturing optimum.
"""
from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from pyrolysis.biochar_stage1.composition_inference_v20 import estimate_bet


def build_bet_record(feedstock: str, process: Mapping, check: Mapping,
                     bet_m2_g: float, *, allow_composition_ood: bool = False) -> dict:
    """Capture a validated single-stage process and the *separate* v20 BET result."""
    if not isinstance(feedstock, str) or not feedstock.strip():
        raise ValueError("원료명이 필요합니다")
    if not (check.get("predictable") or
            (allow_composition_ood and check.get("can_explore_composition_ood"))):
        raise ValueError("예측 승인되지 않은 입력입니다")
    val = float(bet_m2_g)
    if not math.isfinite(val) or val < 0:
        raise ValueError("유효하지 않은 BET 예측값")
    record = {"feedstock_note_only": feedstock.strip(),
              "model": "Stage1 experimental BET v20",
              "model_variant": check["variant"],
              "predicted_BET_m2_g": val,
              "input_scope": ("COMPOSITION_OOD_PROVISIONAL" if not check.get("predictable")
                              else "MARGINAL_RANGE_EXPLORATORY"),
              "manufacturing_route": "SINGLE_STAGE_PYROLYSIS",
              "activated": False,
              "pyrolysis_atmosphere": None}
    for key in ("pyrolysis_temperature_C", "pyrolysis_hold_min", "heating_rate_C_min"):
        n = float(process[key])
        if not math.isfinite(n) or n <= 0:
            raise ValueError(f"잘못된 제조조건: {key}")
        record[key] = n
    return record


def handoff_to_stage2(record: Mapping) -> dict:
    """Copy only actual process inputs; never smuggle estimated BET into Stage2."""
    if record.get("manufacturing_route") != "SINGLE_STAGE_PYROLYSIS" or record.get("activated"):
        raise ValueError("지원하지 않는 제조경로")
    name = str(record.get("feedstock_note_only", "")).strip()
    if not name:
        raise ValueError("원료명 누락")
    values = {"s2_bio": name, "s2_enable_B": True,
              "s2_route": "SINGLE_STAGE_PYROLYSIS", "s2_steps": 1,
              "s2_hr_unknown": False}
    for source, dest in (("pyrolysis_temperature_C", "s2_pt"),
                         ("pyrolysis_hold_min", "s2_hold"),
                         ("heating_rate_C_min", "s2_hr")):
        value = float(record[source])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"잘못된 제조조건: {source}")
        values[dest] = value
    assert not any("bet" in k.lower() for k in values)
    return values


def bet_temperature_sweep(model: Mapping, check: Mapping, *,
                          n_points: int = 13, allow_composition_ood: bool = False) -> pd.DataFrame:
    """Plot the fixed-composition v20 response across its *training* temperature range."""
    if not (check.get("predictable") or
            (allow_composition_ood and check.get("can_explore_composition_ood"))):
        raise ValueError("허용되지 않은 입력 상태")
    if not 2 <= n_points <= 41:
        raise ValueError("그래프 표본 수는 2~41")
    low, high = model["models"][check["variant"]]["feature_range"]["temperature_C"]
    low, high = float(low), float(high)
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError("학습 온도 범위가 올바르지 않습니다")
    rows = []
    for i in range(n_points):
        temp = low + (high - low) * i / (n_points - 1)
        row_check = dict(check)
        row_check["values"] = {**check["values"], "temperature_C": temp}
        value = float(estimate_bet(model, row_check,
                                   allow_composition_ood=allow_composition_ood))
        if not math.isfinite(value) or value < 0:
            raise ValueError("모델이 유효하지 않은 BET 값을 반환했습니다")
        rows.append({"열분해온도 (°C)": round(temp, 3),
                     "예측 BET (m²/g)": round(value, 6),
                     "model_variant": check["variant"],
                     "input_scope": ("COMPOSITION_OOD_PROVISIONAL"
                                     if not check.get("predictable") else "MARGINAL_RANGE_EXPLORATORY")})
    return pd.DataFrame(rows)


def bet_result_csv(record: Mapping, sweep: pd.DataFrame) -> bytes:
    """Single UTF-8 BOM CSV with process-input metadata, no invented qe/TPV."""
    df = sweep.copy()
    for k in ("feedstock_note_only", "pyrolysis_hold_min", "heating_rate_C_min",
              "model", "predicted_BET_m2_g"):
        df[k] = record[k]
    return df.to_csv(index=False).encode("utf-8-sig")
