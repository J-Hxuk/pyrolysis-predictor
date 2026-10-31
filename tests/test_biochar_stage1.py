"""
test_biochar_stage1.py — Stage1 BET 플랫폼 자동 검증.

완료 기준(지시서 §7) 대응:
  2. 실제 입력값으로 BET 예측이 수행된다.
  3. 모델 저장 후 재로드해도 동일 입력에 동일한 예측값.
  4. Stage1 입력 스키마와 단위가 유지된다.
  5. 교차검증에서 출처 그룹 누설이 발생하지 않는다.
  6. Stage2 데이터가 없어도 Stage1 은 독립 동작.
 10. 예측값·성능 수치를 임의로 생성하지 않는다 (재현 경로만 사용).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyrolysis.biochar_stage1 import data_validation, evaluation, inference
from pyrolysis.biochar_stage1.data_loader import load_stage1_dataset
from pyrolysis.biochar_stage1.extract import extract_required_files
from pyrolysis.biochar_stage1.features import (
    BET_INPUT_ORDER,
    BET_INPUT_VARIABLES,
    BET_TARGET,
)
from pyrolysis.biochar_stage1.model_registry import Stage1Registry
from pyrolysis.biochar_stage1.training import save_bet_model, train_bet_model

_REPO = Path(__file__).resolve().parents[1]

SAMPLE = {
    "C_pct": 48.9, "H_pct": 6.0, "N_pct": 0.5, "O_pct": 44.4,
    "VM_pct": 78.26, "Ash_pct": 1.13, "FC_pct": 20.61,
    "temperature_C": 600, "time_min": 10, "heating_rate_C_min": 30.0,
}


@pytest.fixture(scope="module")
def dataset():
    extract_required_files(_REPO)
    return load_stage1_dataset(_REPO)


@pytest.fixture(scope="module")
def trained(dataset):
    # CV 생략해 빠르게 학습 (determinism·schema 검증용)
    return train_bet_model(dataset, base=_REPO, run_cv=False)


def test_input_schema_and_units_preserved():
    """기준 4: 10개 입력 순서·단위·basis 가 고정이다."""
    assert BET_INPUT_ORDER == [
        "C_pct", "H_pct", "N_pct", "O_pct", "VM_pct", "Ash_pct", "FC_pct",
        "temperature_C", "time_min", "heating_rate_C_min",
    ]
    assert BET_INPUT_VARIABLES["C_pct"].basis.startswith("dry ash-free")
    assert BET_INPUT_VARIABLES["VM_pct"].basis == "dry basis"
    assert BET_TARGET.unit == "m2/g"


def test_dataset_loaded_from_real_eval37(dataset):
    """실제 eval37 데이터(182 base / 99 extra)를 읽는다."""
    assert len(dataset.base) == 182
    assert len(dataset.extra) == 99
    assert len(dataset.dataset_version_hash) == 16


def test_prediction_runs_with_real_input(trained):
    """기준 2: 실제 입력값으로 BET 예측이 수행된다."""
    handle = inference.BetModelHandle(trained.pipeline, trained.metadata, _REPO)
    pred = inference.predict_bet(handle, SAMPLE)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_save_reload_determinism(trained, tmp_path):
    """기준 3: 저장 후 재로드해도 동일 입력에 동일 예측."""
    vdir = save_bet_model(trained, tmp_path)
    handle = inference.load_bet_model(vdir)
    p_mem = inference.predict_bet(
        inference.BetModelHandle(trained.pipeline, trained.metadata, vdir), SAMPLE)
    p_reload = inference.predict_bet(handle, SAMPLE)
    assert float(p_mem[0]) == float(p_reload[0])


def test_metadata_has_required_fields(trained):
    """메타데이터에 데이터셋 해시·입력·단위·하이퍼파라미터·범위 등이 기록된다."""
    m = trained.metadata
    assert m["model_type"] == "RandomForestRegressor"
    assert m["hyperparameters"]["n_estimators"] == 300
    assert m["hyperparameters"]["min_samples_leaf"] == 2
    assert m["hyperparameters"]["random_state"] == 42
    assert m["input_order"] == BET_INPUT_ORDER
    assert "dataset_version_hash" in m["training_data"]
    assert m["training_numeric_ranges"]["temperature_C"]["min"] >= 0
    assert "applicability" in m


def test_groupkfold_no_source_leakage(dataset):
    """기준 5: GroupKFold 에서 출처 그룹 누설이 없다."""
    res = evaluation.evaluate_groupkfold(dataset.base, dataset.extra, model_name="RF")
    assert res.leakage_checks_passed is True
    assert res.n_groups == 24
    # 지표는 유한하며 임의 생성이 아니라 재현 경로에서 계산된다
    assert np.isfinite(res.mean_r2)
    assert np.isfinite(res.mean_mae)


def test_reproduction_compared_to_reference(dataset):
    """기준 10: eval37 보고 수치와 비교하되 임의 수치를 만들지 않는다."""
    res = evaluation.evaluate_groupkfold(dataset.base, dataset.extra, model_name="RF")
    cmp = evaluation.compare_with_reference(res, _REPO)
    assert cmp["reference_found"] is True
    # 차이가 보고된다 (환경 차이). 극단적으로 벗어나지 않는지 느슨한 상한만 확인.
    assert abs(cmp["delta"]["mae"]) < 20
    assert abs(cmp["delta"]["r2"]) < 0.3


def test_input_sanity_rejects_bad_composition(trained):
    """비정상 조성(질량%>100)은 오류로 잡는다."""
    bad = {**SAMPLE, "C_pct": 150.0}
    verdict = inference.check_input_sanity(bad)
    assert verdict.ok is False
    assert any(e["input_id"] == "C_pct" for e in verdict.errors)


def test_domain_extrapolation_warning(trained, tmp_path):
    """학습 범위 이탈이 TRAINING_RANGE_EXCEEDED 주의로 표시된다."""
    vdir = save_bet_model(trained, tmp_path)
    handle = inference.load_bet_model(vdir)
    far = {**SAMPLE, "temperature_C": 5000}
    verdict = inference.check_applicability_domain(far, handle)
    assert any(w.get("code") == "TRAINING_RANGE_EXCEEDED" for w in verdict.warnings)


def test_within_range_not_guaranteed_note(trained, tmp_path):
    """범위 안이어도 조합 검증을 보장하지 않는다는 주의를 남긴다."""
    vdir = save_bet_model(trained, tmp_path)
    handle = inference.load_bet_model(vdir)
    verdict = inference.check_applicability_domain(SAMPLE, handle)
    assert any(w.get("code") == "WITHIN_RANGE_NOT_GUARANTEED" for w in verdict.warnings)


def test_stage1_independent_of_stage2(dataset):
    """기준 6: Stage2 없이도 Stage1 학습·예측이 독립 동작한다."""
    # Stage2 모듈을 전혀 쓰지 않고 Stage1 전 경로가 동작하는지 확인
    model = train_bet_model(dataset, base=_REPO, run_cv=False)
    handle = inference.BetModelHandle(model.pipeline, model.metadata, _REPO)
    assert np.isfinite(inference.predict_bet(handle, SAMPLE)[0])


def test_composition_not_renormalized(dataset):
    """조성 합을 100 으로 재조정하지 않는다 (점검만 한다)."""
    audit = data_validation.audit_dataset(dataset.base, dataset.extra)
    chno = audit["base"]["composition_sanity"]["elemental_CHNO_sum"]
    # 원자료 합은 97.7~100 범위 (재조정 안 함)
    assert chno["sum_min"] < 100.0
    assert "재조정" in audit["base"]["composition_sanity"]["note"] or \
           "100" in audit["base"]["composition_sanity"]["note"]
