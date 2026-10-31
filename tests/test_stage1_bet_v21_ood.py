"""v21 regression: explicit research-only composition OOD; fixed analysis basis."""
from pathlib import Path
import pytest
from pyrolysis.biochar_stage1.composition_inference_v20 import (
    load_portable_model, default_model_path, assess_inputs, estimate_bet,
)
from pyrolysis.biochar_stage1.data_loader import load_bet_base

ROOT = Path(__file__).resolve().parents[1]


def evaluate(process, comp, pb="dry", eb="dry ash-free (daf)", route="SINGLE_STAGE_PYROLYSIS"):
    model=load_portable_model(default_model_path(ROOT))
    check=assess_inputs(model, process, comp, elemental_basis=eb,
                        proximate_basis=pb, process_type=route, activated=False)
    return model,check


def example():
    row=load_bet_base(ROOT).iloc[0]
    process={"pyrolysis_temperature_C":row["temperature_C"],
             "pyrolysis_hold_min":row["time_min"],
             "heating_rate_C_min":row["heating_rate_C_min"]}
    composition={"raw_VM_wt_pct":row["VM_pct"],"raw_ash_wt_pct":row["Ash_pct"],
                 "raw_FC_wt_pct":row["FC_pct"]}
    return process,composition


def test_vm20_fc60_dry_basis_strict_reject_opt_in_reference():
    p,c=example()
    c.update(raw_VM_wt_pct=20.0,raw_ash_wt_pct=20.0,raw_FC_wt_pct=60.0)
    m,check=evaluate(p,c)
    assert not check["errors"] and not check["predictable"]
    assert check["can_explore_composition_ood"]
    assert set(check["outside_feature_names"]) == {"VM_pct", "FC_pct"}
    assert set(check["strongly_outside_feature_names"]) == {"VM_pct", "FC_pct"}
    with pytest.raises(ValueError): estimate_bet(m,check)
    assert estimate_bet(m,check, allow_composition_ood=True)>=0


def test_no_process_extrapolation_even_with_opt_in():
    p,c=example()
    p["pyrolysis_temperature_C"]=10000.0
    m,check=evaluate(p,c)
    assert not check["can_explore_composition_ood"]
    with pytest.raises(ValueError): estimate_bet(m,check,allow_composition_ood=True)


def test_unsupported_basis_and_invalid_composition_not_overridable():
    p,c=example()
    c.update(raw_VM_wt_pct=20,raw_ash_wt_pct=20,raw_FC_wt_pct=60)
    m,ch=evaluate(p,c,pb="as received")
    assert ch["errors"]
    assert not ch["can_explore_composition_ood"]
    with pytest.raises(ValueError): estimate_bet(m,ch,allow_composition_ood=True)
    c["raw_ash_wt_pct"]=10
    m,ch=evaluate(p,c)
    assert ch["errors"]
    with pytest.raises(ValueError): estimate_bet(m,ch,allow_composition_ood=True)


def test_same_in_range_prediction_and_existing_validation():
    p,c=example()
    m,ch=evaluate(p,c)
    assert ch["predictable"] and not ch["can_explore_composition_ood"]
    assert estimate_bet(m,ch)==estimate_bet(m,ch,allow_composition_ood=True)


def test_no_as_received_options_in_streamlit_ui():
    stage1=(ROOT/"pyrolysis/biochar_ui/stage1_bet_v20_page.py").read_text(encoding="utf-8")
    stage2=(ROOT/"pyrolysis/biochar_ui/stage2_pages.py").read_text(encoding="utf-8")
    assert '"as received"' not in stage1
    assert '"as received"' not in stage2
