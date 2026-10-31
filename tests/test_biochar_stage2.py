"""
test_biochar_stage2.py — Stage2 인터페이스·Langmuir 자동 검증.

완료 기준(지시서 §7) 대응:
  7. Stage2 에 미검증 데이터가 입력되면 학습이 차단된다.
  + Langmuir fitting 성공·거부, qe 종류 구분, Stage1 결합 분리.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrolysis.biochar_stage2 import adapter, langmuir
from pyrolysis.biochar_stage2.adapter import TrainingBlockedError
from pyrolysis.biochar_stage2.schema import (
    QE_VALUE_KIND,
    STAGE2_FIELD_IDS,
    TARGET_FIELD,
)


def _frame():
    return pd.DataFrame([
        # 적격: measured + 평형 검증 + qe
        {"material_id": "M1", "provenance_group": "g1", "pollutant": "Pb",
         "initial_concentration_mg_L": 50, "adsorbent_dose_g_L": 1.0,
         "measured_qe_mg_g": 30.0, "equilibrium_verified": True,
         "measurement_status": "measured"},
        # 부적격: measured 지만 평형 미검증
        {"material_id": "M2", "provenance_group": "g1", "pollutant": "Pb",
         "initial_concentration_mg_L": 50, "adsorbent_dose_g_L": 1.0,
         "measured_qe_mg_g": 40.0, "equilibrium_verified": False,
         "measurement_status": "measured"},
        # 미검증 reported
        {"material_id": "M3", "provenance_group": "g2", "pollutant": "Cd",
         "initial_concentration_mg_L": 20, "adsorbent_dose_g_L": 2.0,
         "measured_qe_mg_g": 10.0, "equilibrium_verified": True,
         "measurement_status": "reported_unverified"},
    ])


def test_schema_contains_required_fields():
    for fid in [
        "material_id", "original_experiment_doi", "measured_BET_m2_g",
        "measured_TPV_cm3_g", "pollutant", "initial_concentration_mg_L",
        "solution_pH", "adsorbent_dose_g_L", "adsorption_temperature_C",
        "contact_time_min", "equilibrium_concentration_mg_L", "measured_qe_mg_g",
        "equilibrium_verified", "measurement_status", "qa_status", "provenance_group",
    ]:
        assert fid in STAGE2_FIELD_IDS
    assert TARGET_FIELD == "measured_qe_mg_g"


def test_qe_value_kinds_distinguished():
    """실측 / Langmuir qmax / 모델 예측 / 미검증 reported 를 구분한다."""
    assert set(QE_VALUE_KIND) == {
        "measured", "langmuir_qmax", "model_predicted", "reported_unverified",
    }


def test_validate_counts_eligible():
    res = adapter.validate_stage2_frame(_frame())
    assert res.n_rows == 3
    assert res.n_training_eligible == 1  # M1 만
    assert res.n_unverified_capacity == 1
    assert any(i["code"] == "MEASURED_BUT_EQUILIBRIUM_UNVERIFIED" for i in res.issues)


def test_training_blocked_when_no_verified_rows():
    """기준 7: 미검증 데이터만 있으면 학습이 차단된다."""
    df = _frame()
    df = df[df.measurement_status != "measured"]  # 적격 행 제거
    with pytest.raises(TrainingBlockedError):
        adapter.approve_training_rows(df)


def test_training_blocked_when_all_unverified_even_if_measured():
    """measured 표기여도 평형 미검증이면 승인되지 않는다."""
    df = _frame()
    df.loc[df.material_id == "M1", "equilibrium_verified"] = False
    with pytest.raises(TrainingBlockedError):
        adapter.approve_training_rows(df)


def test_approve_returns_only_eligible():
    approved = adapter.approve_training_rows(_frame())
    assert len(approved) == 1
    assert approved.iloc[0]["material_id"] == "M1"


def test_stage1_link_keeps_measured_and_predicted_separate():
    link = adapter.Stage1LinkInput(
        bet_m2_g=398.3, tpv_cm3_g=0.1,
        bet_source="stage1_predicted", tpv_source="measured",
        bet_uncertainty=90.0)
    row = adapter.build_stage2_input_row(
        {"pollutant": "Pb", "initial_concentration_mg_L": 50}, link)
    # 예측 BET 는 measured_BET 로 들어가지 않는다
    assert row["measured_BET_m2_g"] is None
    # 측정 TPV 는 들어간다
    assert row["measured_TPV_cm3_g"] == 0.1
    assert row["_stage1_link"]["bet_source"] == "stage1_predicted"


# ── Langmuir ─────────────────────────────────────────────────────────────────


def test_langmuir_fit_recovers_parameters():
    ce = np.array([1, 2, 5, 10, 20, 40, 80], dtype=float)
    qe = langmuir.langmuir_qe(ce, 120.0, 0.08)
    result = langmuir.fit_langmuir(ce, qe)
    d = result.to_dict()
    assert d["fitted"] is True
    assert d["qmax_mg_g"] == pytest.approx(120.0, rel=0.05)
    assert d["KL_L_mg"] == pytest.approx(0.08, rel=0.1)
    assert d["r2"] > 0.99
    assert "Stage2 머신러닝 예측과는 별개" in d["note"]


def test_langmuir_refuses_insufficient_points():
    r = langmuir.fit_langmuir([1, 2], [3, 4])
    d = r.to_dict()
    assert d["fitted"] is False
    assert d["reason_code"] == "INSUFFICIENT_POINTS"


def test_langmuir_refuses_degenerate_ce():
    r = langmuir.fit_langmuir([5, 5, 5, 5], [1, 2, 3, 4])
    assert r.to_dict()["reason_code"] == "DEGENERATE_CE"


def test_langmuir_refuses_non_positive():
    r = langmuir.fit_langmuir([1, 2, 3], [-1, 2, 3])
    assert r.to_dict()["reason_code"] == "NON_POSITIVE"
