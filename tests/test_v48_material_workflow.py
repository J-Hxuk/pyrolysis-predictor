"""v48 read-only BET/TPV exploratory material view tests."""
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from pyrolysis.biochar_stage1.v48_research_bridge import (
    DATA_REL, MODEL_FILES, TARGETS, S1_CAT, S1_NUM,
    load_research_data, specimen_row, source_temperature_range, material_input,
    predict_properties, response_curve, measurement_note, export_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load_research_data(ROOT)


@pytest.fixture(scope="module")
def models():
    return {k: joblib.load(ROOT / "deployment/research_v46_NOT_ACTIVE" / path)
            for k, path in MODEL_FILES.items()}


def test_v48_source_is_inherited_verbatim(data):
    assert len(data) == 417
    assert data.material_id.is_unique
    assert sum(data[TARGETS["BET"]].notna()) == 361
    assert sum(data[TARGETS["TPV"]].notna()) == 231


def test_v48_contract_exact_columns_and_reported_nulls(data):
    row = data.iloc[0]
    inputs = material_input(row)
    assert list(inputs.columns) == S1_NUM + S1_CAT
    assert np.isnan(inputs["pyrolysis_hold_min"].iloc[0])
    assert inputs["feedstock"].iloc[0] == row.feedstock


def test_v48_no_unreported_manufacturing_fabrication(data):
    src = data.loc[data.material_id.str.startswith('V46_')]
    if src.empty:
        src = data[data.source_stage.eq('V46_PHASE8_SOURCE_SPECIFIC_RESEARCH_ONLY')]
    assert len(src) == 13
    row = src.iloc[0]
    inputs = material_input(row)
    assert np.isnan(inputs["activation_agent_feedstock_mass_ratio"].iloc[0])
    note = measurement_note(row)
    assert note["source_ratio_not_in_model"] is True or any(
        pd.notna(src[c]).any() for c in ("source_K2CO3_to_char_mass_ratio", "source_FeCl3_to_char_mass_percent")
    )


def test_v48_selection_must_match_unique_material(data):
    with pytest.raises(ValueError):
        specimen_row(data, "does-not-exist")
    assert specimen_row(data, data.material_id.iloc[0]).material_id == data.material_id.iloc[0]


def test_v48_temperature_range_is_matched_to_feedstock_and_processing_route(data):
    row = data.iloc[0]
    assert source_temperature_range(data, row) == (450.0, 650.0)
    other = data.loc[data.source_stage.eq('V46_PHASE8_SOURCE_SPECIFIC_RESEARCH_ONLY')].iloc[0]
    check = source_temperature_range(data, other)
    assert check is None or check[0] < check[1]


def test_v48_bad_temperature_refused(data):
    row = data.iloc[0]
    for bad in (-1, 0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            material_input(row, temperature_C=bad)


def test_v48_saved_models_produce_finite_nonnegative_predictions(data, models):
    for row in (data.iloc[0], data.iloc[5], data.iloc[-1]):
        if pd.isna(row.pyrolysis_temperature_C):
            continue
        predictions = predict_properties(models, row)
        assert set(predictions) == {"BET", "TPV"}
        assert all(math.isfinite(v) and v >= 0 for v in predictions.values())


def test_v48_temperature_curve_separate_units(data, models):
    row = data.iloc[0]
    curve = response_curve(models, data, row, n_points=6)
    assert len(curve) == 6
    assert list(curve.columns) == ["열분해온도 (°C)", "예측 BET (m²/g)", "예측 TPV (cm³/g)"]
    assert curve.iloc[0, 0] == 450
    assert curve.iloc[-1, 0] == 650
    assert curve.iloc[:, 1:].ge(0).all().all()
    with pytest.raises(ValueError):
        response_curve(models, data, row, n_points=1)


def test_v48_report_separates_measured_prediction_and_qe(data, models):
    row = data.iloc[0]
    note = measurement_note(row)
    value = predict_properties(models, row)
    export = export_comparison(row, value, note, pyrolysis_temperature_C=450)
    assert export.startswith(b'\xef\xbb\xbf')
    content = export.decode('utf-8-sig')
    assert 'RESEARCH_ONLY_NOT_ACTIVE' in content
    assert 'stage2_qe_uses_predicted_BET_TPV' in content
    assert 'False' in content
    assert row.material_id in content


def test_v48_model_loading_errors_are_not_silent(data):
    with pytest.raises(ValueError):
        predict_properties({"BET": object()}, data.iloc[0])


def test_v48_v47_active_registry_files_unchanged():
    assert (ROOT / "deployment/stage2_models/active.json").is_file()
    assert (ROOT / "deployment/stage1_models/active.json").is_file()
    assert (ROOT / "V47_MANIFEST.json").is_file()


def test_v48_stage2_handoff_is_only_manufacturing_input():
    from pyrolysis.biochar_stage1.v44_workflow import handoff_to_stage2
    row = {
        "feedstock_note_only": "rice husk",
        "manufacturing_route": "SINGLE_STAGE_PYROLYSIS",
        "activated": False,
        "pyrolysis_temperature_C": 500,
        "pyrolysis_hold_min": 60,
        "heating_rate_C_min": 10,
        "predicted_BET_m2_g": 1000,
        "predicted_TPV_cm3_g": .4,
    }
    out = handoff_to_stage2(row)
    assert not any("bet" in key.lower() or "tpv" in key.lower() for key in out)


def test_v48_checksum_allowlist_only_research_models(tmp_path):
    from scripts.verify_v48 import _locally_rebuilt_v46_hashes
    from pathlib import Path
    import json
    directory = tmp_path / "deployment" / "research_v46_NOT_ACTIVE"
    directory.mkdir(parents=True)
    allowed = [
        "V46_Stage1_BET_m2_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
        "V46_Stage1_TPV_cm3_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
        "V46_Stage2_target_value_A_RESEARCH_NOT_ACTIVE.joblib",
        "V46_Stage2_target_value_B_RESEARCH_NOT_ACTIVE.joblib",
    ]
    report = {"active_model_changed": False, "data_changed": False,
              "models": [{"file": n, "sha256": "a" * 64} for n in allowed]}
    (directory / "V46_LOCAL_REBUILD_MANIFEST.json").write_text(json.dumps(report))
    found = _locally_rebuilt_v46_hashes(tmp_path)
    assert len(found) == 4
    report["active_model_changed"] = True
    (directory / "V46_LOCAL_REBUILD_MANIFEST.json").write_text(json.dumps(report))
    assert _locally_rebuilt_v46_hashes(tmp_path) == {}
