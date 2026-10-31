"""v26 contract tests: no source editing; bounded model, real output provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from pyrolysis.biochar_stage2.v26_screening_policy import assess_scope
from pyrolysis.biochar_stage2.v15_loader import load_core_qe
from scripts.evaluate_v26_physics_aware import safe_f, bounded_predictions

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/v26_physics_aware'


def test_safe_f_invariant_and_failure():
    d=pd.DataFrame({'C0_mg_L':[100],'dose_g_L':[2],'target_value':[25]})
    f,cap=safe_f(d)
    assert f[0]==.5 and cap[0]==50.
    with pytest.raises(ValueError):
        safe_f(d.assign(target_value=[51]))
    with pytest.raises(ValueError):
        safe_f(d.assign(dose_g_L=[0]))


def test_holdout_fraction_is_always_bounded():
    from scripts.compare_phase8_v23 import stage2_targets
    _,df=stage2_targets()
    train=df.iloc[:90].copy(); test=df.iloc[90:].copy()
    result=bounded_predictions(train,test)
    _,cap=safe_f(test)
    assert (result['fraction_logit_ridge']>=0).all()
    assert (result['fraction_logit_ridge']<=cap+1e-10).all()
    assert np.isfinite(result['fraction_logit_ridge']).all()


def test_stage1_external_is_heldout_both_targets():
    df=pd.read_csv(REPORT/'stage1_external33_log_vs_raw.csv')
    from scripts.compare_phase8_v23 import stage1_targets
    _,tr=stage1_targets()
    assert len(df)==66
    assert set(df.doi).isdisjoint(tr.doi)
    assert set(df.target)=={'BET','TPV'}
    assert df.log_rf.notna().all()
    assert df.raw_rf_v25.notna().all()


def test_stage1_group_cv_unique_by_material_and_target():
    x=pd.read_csv(REPORT/'stage1_lodo_log_vs_raw.csv')
    assert not x.duplicated(['target','material_id']).any()
    assert (x.log_rf>=0).all()
    assert set(x.target)=={'BET','TPV'}


def test_stage2_external_ood_and_mass_balance_visible():
    ex=pd.read_csv(REPORT/'stage2_external51_bounded_fraction.csv')
    assert len(ex)==51 and ex.doi.nunique()==1
    assert (ex.dose_g_L==100).all()
    assert (ex.fraction_logit_ridge<=ex.cap+1e-8).all()
    assert (ex.fraction_logit_ridge>=0).all()
    assert ex.f_predicted.mean()>.99  # warns that model saturated, not predictive success


def test_lodo_excludes_external():
    old=pd.read_csv(REPORT/'stage2_lodo_bounded_fraction.csv')
    ext=pd.read_csv(REPORT/'stage2_external51_bounded_fraction.csv')
    assert len(old)==98 and old.doi.nunique()==26
    assert set(old.doi).isdisjoint(set(ext.doi))


def test_report_does_not_auto_upgrade():
    r=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert 'no deployment' in r['scope']
    assert any('v24 data remains excluded' in s for s in r['interpretation'])
    assert r['stage2']['lodo_core98']['methods']['fraction_logit_ridge']['physical_violation_pct']==0


def test_screen_gate_novel_feedstock_or_unsupported_dose():
    core=load_core_qe(ROOT).df
    ads={'C0_mg_L':100,'dose_g_L':2,'initial_pH':5,'adsorption_T_C':25,'contact_time_min':1440}
    s=assess_scope(core,None,'Pb(II)',ads)
    assert not s.eligible_for_exploratory_ranking
    s=assess_scope(core,'peanut shell','Pb(II)',{**ads,'dose_g_L':100})
    assert not s.eligible_for_exploratory_ranking
    assert 'outside_training_range:dose_g_L' in s.reasons


def test_scope_pass_only_enables_exploratory_not_verified():
    core=load_core_qe(ROOT).df
    ads={'C0_mg_L':100,'dose_g_L':2,'initial_pH':5,'adsorption_T_C':25,'contact_time_min':1440}
    res=assess_scope(core,'peanut shell','Pb(II)',ads)
    assert res.unique_candidates == 2
    assert '탐색적' in res.disclaimer
    assert not res.reasons or all(r.startswith('outside_training_range:') for r in res.reasons)
