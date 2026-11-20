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


def test_v143_ab_comparison_renders_three_separate_original_unit_charts(monkeypatch):
    from contextlib import nullcontext
    from pyrolysis.biochar_ui import v112_unified_page as ui

    charts = []
    captions = []
    monkeypatch.setattr(ui.st, "columns", lambda n: [nullcontext() for _ in range(n)])
    monkeypatch.setattr(ui.st, "markdown", lambda *_a, **_kw: None)
    monkeypatch.setattr(ui.st, "caption", lambda text: captions.append(text))
    monkeypatch.setattr(ui.st, "altair_chart", lambda chart, **_kw: charts.append(chart))
    a = {"BET_m2_g": 40.0, "TPV_cm3_g": 0.12, "reported_capacity_mg_g": 52.500001}
    b = {"BET_m2_g": 80.0, "TPV_cm3_g": 0.16, "reported_capacity_mg_g": 52.500003}
    ui._render_ab_comparison_charts(a, b)
    assert len(charts) == 3
    expected = [(40.0, 80.0), (0.12, 0.16), (52.500001, 52.500003)]
    for chart, pair in zip(charts, expected):
        assert tuple(chart.data["예측값"]) == pair
    assert any("+0.000002" in note for note in captions)


def test_v143_removed_feature_status_expander_and_replaced_ab_table_with_chart():
    source = (ROOT / "pyrolysis/biochar_ui/v112_unified_page.py").read_text(encoding="utf-8")
    assert "입력변수 활용 현황 · 모델별 차이" not in source
    assert "st.dataframe(ab_table" not in source
    assert "_render_ab_comparison_charts(r, br)" in source
    assert "scale=alt.Scale(domain=[0, axis_max])" in source
    assert "A/B 비교 CSV" in source


def test_v144_activation_labels_and_tab_spacing_are_ui_only():
    """Keep the trained activation code unchanged while simplifying the display name."""
    source = (ROOT / "pyrolysis/biochar_ui/v112_unified_page.py").read_text(encoding="utf-8")
    assert "'화학적 활성화':'chemical_impregnation'" in source
    assert "화학적 활성화/함침" not in source
    assert "'없음','화학적 활성화','물리적 활성화','수증기 활성화'" in source
    assert source.count('v144-tab-bottom-space') == 3
    assert 'div[data-baseweb=' in source and 'tab-panel' in source
    assert "padding-bottom:4.5rem" in source


def test_v145_air_is_unavailable_in_both_user_atmosphere_selectors():
    """Air is unseen in Stage 1; keep backend guard, not UI choice."""
    from pyrolysis.biochar_ui import v112_unified_page as ui

    assert "air" not in ui.ATMOSPHERE_OPTIONS
    assert ui.ATMOSPHERE_OPTIONS == ("N2", "inert_N2", "oxygen_limited", "Ar", "NR")
    source = (ROOT / "pyrolysis/biochar_ui/v112_unified_page.py").read_text(encoding="utf-8")
    assert "st.selectbox('열분해 분위기',ATMOSPHERE_OPTIONS)" in source
    assert "atmospheres=ATMOSPHERE_OPTIONS" in source
    backend_source = (ROOT / "pyrolysis/biochar_stage2/v112_single_engine.py").read_text(encoding="utf-8")
    assert "'air'" in backend_source
    assert "air는 현재 Stage 1" in backend_source


def test_sidebar_footer_displays_active_model_versions_and_author():
    """The sidebar credits are presentation-only and must stay vertically ordered."""
    source = (ROOT / "pyrolysis/biochar_ui/stage2_pages.py").read_text(encoding="utf-8")
    assert "biochar-sidebar-footer" in source
    assert "bottom:1.25rem" in source
    assert source.index("Model version:") < source.index("Stage1 v134") < source.index("Stage2 v139")
    assert "Made by 한양대학교 자원환경공학과 한재혁" in source
    assert "unsafe_allow_html=True" in source


def test_v146_sweep_and_rank_have_csv_downloads_without_on_screen_tables():
    source = (ROOT / "pyrolysis/biochar_ui/v112_unified_page.py").read_text(encoding="utf-8")
    sweep = source.split("    with tab_sweep:", 1)[1].split("    with tab_ab:", 1)[0]

    # Full sweep DataFrame is exported, including temperature coverage diagnostics.
    assert "df=attach_coverage(df, models, stage2_bounds)" in source
    assert "온도 비교 결과 CSV" in sweep
    assert "data=df.to_csv(index=False).encode('utf-8-sig')" in sweep
    assert "_temperature_comparison.csv" in sweep

    # Ranking retains raw scores, rank, and provenance fields in the CSV.
    assert "rank_export=rank_table.copy()" in sweep
    assert "rank_export['rank_model']" in sweep
    assert "rank_export['rank_training_snapshot_sha256']" in sweep
    assert "제조조건 순위 CSV" in sweep
    assert "rank_export.to_csv(index=False).encode('utf-8-sig')" in sweep

    # Neither of the dense result tables nor the grey explanatory captions render.
    assert "st.dataframe(" not in sweep
    assert "st.caption(" not in sweep
    assert "st.markdown('### 제조조건 순위 추천')" in sweep
    assert "Manufacturing RF')" not in sweep

    # Visual prediction and safety messages remain in the screen.
    assert "_render_temperature_curve(df)" in sweep
    assert "st.warning(" in sweep
    assert "a.metric('예측 흡착량 최대 후보'" in sweep
    assert "b.metric('RF 1위 후보 (탐색)'" in sweep
