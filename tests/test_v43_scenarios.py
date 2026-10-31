"""Small functional tests for v43's user-facing saving/export workflow."""
import math
import pandas as pd
import pytest
from pyrolysis.biochar_stage2.v43_scenarios import (
    MAX_SCENARIOS, build_snapshot, append_snapshot, comparison_frame,
    printable_report, comparison_csv,
)


def inputs(**updates):
    row = {"feedstock_label_normalized": "rice husk", "pollutant": "Cd(II)",
           "C0_mg_L": 100, "initial_pH": 5, "dose_g_L": 2,
           "contact_time_min": 120, "adsorption_T_C": 25,
           "pyrolysis_temperature_C": float('nan')}
    row.update(updates)
    return row


def result(qe=25.0, cap=50.0, raw=None):
    x = qe if raw is None else raw
    return {"qe_raw": x, "qe_clipped": qe, "cap": cap,
            "violates": x != qe}


def test_capture_raw_and_bounded_preserved_and_missing_cleaned():
    s = build_snapshot("조건 1", inputs(), {"A": result(qe=25, raw=60)},
                       scope_status={"A": "LIMITED_EVIDENCE"})
    assert s["predictions"]["A"]["qe_raw"] == 60
    assert s["predictions"]["A"]["qe_clipped"] == 25
    assert s["predictions"]["A"]["removal_bounded_pct"] == 50
    assert s["inputs"]["pyrolysis_temperature_C"] is None
    assert s["predictions"]["A"]["scope"] == "LIMITED_EVIDENCE"


def test_no_fabricated_model_b_and_csv_utf8_bom():
    cases = [build_snapshot("비교", inputs(), {"A": result()})]
    df = comparison_frame(cases)
    assert len(df) == 1 and pd.isna(df.loc[0, "B 원시 qe (mg/g)"])
    assert comparison_csv(cases).startswith(b'\xef\xbb\xbf')


def test_multiple_scenarios_not_falsely_ranked():
    s1 = build_snapshot("Pb", inputs(pollutant="Pb(II)"), {"A": result()})
    s2 = build_snapshot("Cd", inputs(), {"A": result()})
    html = printable_report([s1, s2])
    assert "서로 다른 오염물질" in html
    assert "실험 결과가 아닙니다" in html


def test_xss_escaped_in_offline_report():
    s = build_snapshot("<script>alert(1)</script>", inputs(), {"A": result()})
    html = printable_report([s])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<meta charset='utf-8'>" in html


def test_batch_cap_and_order_preserved():
    history = []
    for n in range(MAX_SCENARIOS + 2):
        history = append_snapshot(history, build_snapshot(f"case{n}", inputs(), {"A": result()}))
    assert len(history) == MAX_SCENARIOS
    assert history[0]['name'] == 'case2'
    assert history[-1]['name'] == f'case{MAX_SCENARIOS+1}'


def test_bad_label_and_nonfinite_results_rejected():
    with pytest.raises(ValueError):
        build_snapshot(" ", inputs(), {"A": result()})
    with pytest.raises(ValueError):
        build_snapshot("bad", inputs(), {"A": result(qe=math.nan)})
    with pytest.raises(ValueError):
        printable_report([])


def test_offline_report_includes_sweep():
    s = build_snapshot("A", inputs(), {"A": result()})
    sensitivity = pd.DataFrame([{"x": 4., "model": "A", "qe_raw": 25.,
                                 "qe_clipped": 25., "cap": 50., "violates": False}])
    report = printable_report([s], sensitivity=sensitivity, sweep_label="초기 pH")
    assert '변수별 모델 응답' in report
    assert '초기 pH' in report
    assert '물질수지' in report
    assert '<svg ' in report
    assert '<polyline ' in report


def test_duplicate_scenario_names_get_unique_chart_labels():
    first = build_snapshot("조건 1", inputs(), {"A": result()})
    history = append_snapshot([first], first)
    assert [x["name"] for x in history] == ["조건 1", "조건 1 (2)"]
    assert comparison_frame(history)["시나리오"].is_unique
