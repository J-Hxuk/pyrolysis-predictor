"""v48 exploratory Stage 1 BET/TPV *research model* inference.

All numeric/categorical values are copied from the chosen v46 material record.
The only editable numerical variable is pyrolysis temperature (in that feedstock
and processing-route observational range). Never infer an unreported input from
feedstock name or use predicted properties as Stage 2 inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math
import numpy as np
import pandas as pd

from scripts.build_v46_phase8 import S1_NUM, S1_CAT, features

DATA_REL = "data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv"
MODEL_FILES = {
    "BET": "V46_Stage1_BET_m2_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
    "TPV": "V46_Stage1_TPV_cm3_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
}
TARGETS = {"BET": "BET_m2_g_training_candidate", "TPV": "TPV_cm3_g_training_candidate"}
UNITS = {"BET": "m²/g", "TPV": "cm³/g"}


def load_research_data(root: Path) -> pd.DataFrame:
    path = Path(root) / DATA_REL
    if not path.is_file():
        raise FileNotFoundError(f"연구용 시료 데이터 누락: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "material_id" not in frame or frame.material_id.isna().any() or not frame.material_id.is_unique:
        raise ValueError("시료 ID가 유일하지 않습니다. 임의로 중복 제거하지 않습니다.")
    if not set(S1_NUM + S1_CAT).issubset(frame.columns):
        raise ValueError("v46 학습 입력 열이 누락되었습니다.")
    if not set(TARGETS.values()).issubset(frame.columns):
        raise ValueError("BET/TPV 측정 열이 누락되었습니다.")
    return frame


def specimen_row(data: pd.DataFrame, specimen_id: str) -> pd.Series:
    found = data.loc[data.material_id.astype(str).eq(str(specimen_id))]
    if len(found) != 1:
        raise ValueError("유효한 단일 시료를 선택해야 합니다.")
    return found.iloc[0]


def source_temperature_range(data: pd.DataFrame, row: Mapping) -> tuple[float, float] | None:
    """Only show temperature response when same feedstock+process has >=2 temperatures."""
    same = data.loc[(data["feedstock"] == row["feedstock"]) &
                    (data["activation_method"].fillna("UNKNOWN") ==
                     ("UNKNOWN" if pd.isna(row["activation_method"]) else row["activation_method"]))]
    temps = pd.to_numeric(same["pyrolysis_temperature_C"], errors="coerce")
    temps = temps[np.isfinite(temps) & (temps > 0)].unique()
    if len(temps) < 2:
        return None
    return float(min(temps)), float(max(temps))


def material_input(row: Mapping, *, temperature_C: float | None = None) -> pd.DataFrame:
    """Match the deployed v46 research model's exact 15-feature contract."""
    values = {key: row.get(key, np.nan) for key in S1_NUM + S1_CAT}
    if temperature_C is not None:
        temp = float(temperature_C)
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("열분해온도는 0보다 큰 유한한 값이어야 합니다.")
        values["pyrolysis_temperature_C"] = temp
    original_temp = pd.to_numeric(values["pyrolysis_temperature_C"], errors="coerce")
    if not np.isfinite(original_temp) or original_temp <= 0:
        raise ValueError("제조온도가 원문에서 확인되지 않은 시료입니다.")
    return features(pd.DataFrame([values]), S1_NUM + S1_CAT, S1_NUM, S1_CAT)


def predict_properties(models: Mapping, row: Mapping, *, temperature_C: float | None = None) -> dict:
    if not all(target in models for target in ("BET", "TPV")):
        raise ValueError("BET/TPV 연구용 두 모델이 모두 필요합니다.")
    x = material_input(row, temperature_C=temperature_C)
    output = {}
    for target in ("BET", "TPV"):
        value = float(models[target].predict(x)[0])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{target}: 유효하지 않은 모델 출력")
        output[target] = value
    return output


def response_curve(models: Mapping, data: pd.DataFrame, row: Mapping,
                   *, n_points: int = 11) -> pd.DataFrame:
    bounds = source_temperature_range(data, row)
    if bounds is None:
        return pd.DataFrame(columns=["열분해온도 (°C)", "예측 BET (m²/g)", "예측 TPV (cm³/g)"])
    if not 2 <= n_points <= 41:
        raise ValueError("곡선 점 개수는 2~41개여야 합니다.")
    out = []
    for t in np.linspace(*bounds, num=n_points):
        y = predict_properties(models, row, temperature_C=float(t))
        out.append({"열분해온도 (°C)": round(float(t), 3),
                    "예측 BET (m²/g)": round(y["BET"], 6),
                    "예측 TPV (cm³/g)": round(y["TPV"], 7)})
    return pd.DataFrame(out)


def measurement_note(row: Mapping) -> dict:
    """Distinguish measured values from surrogate output and imputed missing inputs."""
    missing = [k for k in S1_NUM + S1_CAT if pd.isna(row.get(k, np.nan))]
    return {
        "sample_id": str(row["material_id"]),
        "doi": str(row.get("doi", "")),
        "missing_training_inputs": missing,
        "measured_BET": pd.to_numeric(row.get(TARGETS["BET"]), errors="coerce"),
        "measured_TPV": pd.to_numeric(row.get(TARGETS["TPV"]), errors="coerce"),
        "source_ratio_not_in_model": bool(
            pd.notna(row.get("source_K2CO3_to_char_mass_ratio", np.nan)) or
            pd.notna(row.get("source_FeCl3_to_char_mass_percent", np.nan))
        ),
    }


def export_comparison(row: Mapping, predicted: Mapping, notes: Mapping,
                      *, pyrolysis_temperature_C: float) -> bytes:
    vals = {
        "source_material_id": notes["sample_id"],
        "doi": notes["doi"],
        "feedstock": row.get("feedstock"),
        "pyrolysis_temperature_C_selected": pyrolysis_temperature_C,
        "BET_predicted_m2_g_V46_RESEARCH": float(predicted["BET"]),
        "TPV_predicted_cm3_g_V46_RESEARCH": float(predicted["TPV"]),
        "BET_measured_m2_g_original_sample": notes["measured_BET"],
        "TPV_measured_cm3_g_original_sample": notes["measured_TPV"],
        "measured_values_pertain_to_original_temperature_only": True,
        "training_missing_inputs": ";".join(notes["missing_training_inputs"]),
        "source_activation_ratios_unused_in_model": notes["source_ratio_not_in_model"],
        "model_status": "RESEARCH_ONLY_NOT_ACTIVE",
        "stage2_qe_uses_predicted_BET_TPV": False,
    }
    return pd.DataFrame([vals]).to_csv(index=False).encode("utf-8-sig")
