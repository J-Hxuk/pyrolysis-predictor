"""Regression tests for v141 UI state and composition opt-in.

No training data, predictions, or model hyperparameters are changed by this file.
"""
from pathlib import Path

import pytest

from pyrolysis.biochar_ui.v141_ui_state import (
    comparison_delta, input_signature, safe_export_stem,
)
from pyrolysis.biochar_stage2 import v112_single_engine as engine


ROOT = Path(__file__).resolve().parents[1]


def test_state_fingerprint_stable_and_sensitive_to_every_input():
    base = {
        "feedstock": "rice husk",
        "process": {"temperature": 500, "atmosphere": "N2"},
        "adsorption": {"pollutant": "Phenol", "pH": 5.5},
        "composition": {"VM": ""},
        "sweep": [300, 700],
        "step": 50,
        "allow_composition_ood": False,
    }
    assert input_signature(base) == input_signature(dict(reversed(list(base.items()))))
    variants = [
        ("feedstock", "wheat straw"),
        ("process", {"temperature": 500, "atmosphere": "Ar"}),
        ("adsorption", {"pollutant": "Phenol", "pH": 6.5}),
        ("composition", {"VM": "70"}),
        ("sweep", [400, 800]),
        ("step", 25),
        ("allow_composition_ood", True),
    ]
    for key, value in variants:
        changed = {**base, key: value}
        assert input_signature(base) != input_signature(changed), key


def test_export_filename_has_no_path_injection():
    name = safe_export_stem("../../rice husk", "Phenol/Cu(II)", 525)
    assert name.startswith("biochar_v141_")
    assert "/" not in name and "\\" not in name
    assert ".." not in name
    assert "Phenol" in name


def test_ab_delta_is_signed_and_does_not_round_predictions():
    a = {"BET_m2_g": 100, "TPV_cm3_g": 0.25, "reported_capacity_mg_g": 52.511212}
    b = {"BET_m2_g": 110, "TPV_cm3_g": 0.24, "reported_capacity_mg_g": 52.514111}
    delta = comparison_delta(a, b)
    assert delta["BET_m2_g"] == 10
    assert delta["TPV_cm3_g"] == pytest.approx(-0.01)
    assert delta["reported_capacity_mg_g"] == pytest.approx(0.002899)


class _FakePredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        assert len(frame) == 1
        return [self.value]


@pytest.fixture
def fake_models():
    return {"s1_reference": {}, "s1_BET": _FakePredictor(100.0),
            "s1_TPV": _FakePredictor(0.3), "composition_bet": object()}


def _predict(fake_models, allow=False):
    return engine.predict_properties(
        fake_models, ROOT, feedstock="rice husk", temperature=500,
        hold=120, heat=10, atmosphere="N2", activated=False,
        composition={"raw_ash_wt_pct": "70"},
        activation_method="none", allow_composition_ood=allow,
    )


def test_composition_ood_requires_explicit_consent(monkeypatch, fake_models):
    check = {"predictable": False, "can_explore_composition_ood": True,
             "errors": [], "outside_training_range": ["Ash outside"]}
    monkeypatch.setattr(engine, "assess_inputs", lambda *a, **kw: check)
    monkeypatch.setattr(engine, "estimate_bet", lambda *a, **kw: 210.0)
    denied = _predict(fake_models, allow=False)
    accepted = _predict(fake_models, allow=True)
    assert denied["BET_m2_g"] == 100.0 and denied["tpv_path"] == accepted["tpv_path"]
    assert denied["composition_status"] == "FALLBACK"
    assert "허용" in "; ".join(denied["warnings"])
    assert accepted["BET_m2_g"] == 210.0
    assert accepted["composition_status"] == "APPLIED"


def test_in_range_composition_retains_previous_behavior(monkeypatch, fake_models):
    check = {"predictable": True, "can_explore_composition_ood": False,
             "errors": [], "outside_training_range": []}
    monkeypatch.setattr(engine, "assess_inputs", lambda *a, **kw: check)
    monkeypatch.setattr(engine, "estimate_bet", lambda *a, **kw: 155.0)
    assert _predict(fake_models, allow=False)["BET_m2_g"] == 155.0
    assert _predict(fake_models, allow=True)["BET_m2_g"] == 155.0


def test_unsupported_composition_never_overrides_hgb(monkeypatch, fake_models):
    check = {"predictable": False, "can_explore_composition_ood": False,
             "errors": ["unsupported route"], "outside_training_range": []}
    monkeypatch.setattr(engine, "assess_inputs", lambda *a, **kw: check)
    monkeypatch.setattr(engine, "estimate_bet", lambda *a, **kw: pytest.fail("should not call"))
    assert _predict(fake_models, allow=True)["BET_m2_g"] == 100.0


def test_unified_ui_has_stale_guard_and_no_form_submit():
    source = (ROOT / "pyrolysis/biochar_ui/v112_unified_page.py").read_text(encoding="utf-8")
    assert "payload.get('input_signature') != current_signature" in source
    assert "st.form_submit_button(" not in source
    assert "allow_composition_ood=allow_comp_ood" in source
    assert "v141_comparison" in source
