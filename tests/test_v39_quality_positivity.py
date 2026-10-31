"""Tests for v39. Run in a repo with v37 & v38 applied."""
import numpy as np
import pandas as pd
import pytest
from scripts.build_v39_quality_positivity import (
    stage1_factory,stage2_cohorts,doi_bootstrap_delta,coverage,
    historical_new_doi_diagnostic,S1_NUM,S1_CAT
)
from scripts.build_v37_full_refit import input_frame


def test_stage1_positive_model_never_predicts_negative(tmp_path):
    toy=pd.DataFrame({**{col:[1.,2.,3.,4.,5.] for col in S1_NUM},
                      **{col:['x','y','x','y','x'] for col in S1_CAT}})
    model=stage1_factory('Poisson_HGB');model.fit(input_frame(toy,S1_NUM,S1_CAT),np.array([0.1,1.2,3.0,6.0,0.2]))
    p=model.predict(input_frame(toy,S1_NUM,S1_CAT));assert np.all(p>0)


def test_no_silent_unknown_quality_tier():
    d=pd.DataFrame({'model_use_tier':['UNKNOWN'], 'doi':['a']})
    with pytest.raises(ValueError, match='quality tier'):stage2_cohorts(d)


def test_stage2_doi_leakage_rejected():
    d=pd.DataFrame({'model_use_tier':['CORE_EXPLORATORY','V34_RESEARCH_CONDITIONAL_NOT_CERTIFIED'],
                    'doi':['same','same']})
    with pytest.raises(ValueError,match='overlap'):stage2_cohorts(d)


def test_doi_bootstrap_reproducible_and_zero_delta():
    grp=['a','a','b','c'];abs0=[1.,2.,5.,1.]
    assert doi_bootstrap_delta(grp,abs0,abs0)==(0.,0.)


def test_unavailable_variant_rejected():
    with pytest.raises(ValueError):stage1_factory('MAGIC')


def test_staging_coverage_keeps_doi_count():
    s1=pd.DataFrame({'doi':['A','A','B'], 'BET_m2_g_training_candidate':[1.,2.,3.],
                     'TPV_cm3_g_training_candidate':[.1,.2,.3],
                     'pyrolysis_temperature_C':[300.,400.,500.]})
    for col in ('pyrolysis_hold_min','heating_rate_C_min','feedstock_C_pct_original',
                'feedstock_VM_pct_original','feedstock_FC_pct_original'):
        s1[col]=np.nan
    s2=pd.DataFrame({'doi':['C'], 'model_use_tier':['CORE_EXPLORATORY']})
    for col in ('initial_pH','contact_time_min','adsorption_T_C',
                'pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min',
                'raw_VM_wt_pct','raw_FC_wt_pct'):
        s2[col]=np.nan
    cov=coverage(s1,s2)
    x=cov[(cov.target=='BET')&(cov.variable=='pyrolysis_temperature_C')].iloc[0]
    assert x.n_doi_reporting==2 and x.reported==3


def test_real_staging_contracts_and_status_files():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    a=root/'reports/v38_new_dois/Stage1_Master_351_RESEARCH_STAGING.csv'
    b=root/'reports/v38_new_dois/Stage2_reported_uptake_125_RESEARCH_CONDITIONAL.csv'
    if not (a.exists() and b.exists()):pytest.skip('v38 not present')
    s1=pd.read_csv(a);s2=pd.read_csv(b)
    assert len(s1)==351 and len(s2)==125
    core,v34,v38=stage2_cohorts(s2)
    assert (len(core),len(v34),len(v38))==(98,15,12)
    assert s2.target_name.eq('qe_reported_mg_g').all()
    assert (s2.target_value <= s2.C0_mg_L/s2.dose_g_L+1e-5).all()


def test_report_raw_predictions_no_hidden_clipping():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    p=root/'reports/v39_quality_and_positivity/stage1_oof_predictions.csv'
    if not p.exists():pytest.skip('v39 report not generated')
    r=pd.read_csv(p)
    assert (r[r.variant=='Ridge'].pred_raw<0).any()
    assert (r[r.variant=='Poisson_HGB'].pred_raw>=0).all()
    assert (r[r.variant=='RF_log1p'].pred_raw>=0).all()


def test_new_doi_diagnostic_preserves_holdout():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    old=root/'data/imports/v34_temperature/Stage1_Master_333_RESEARCH_STAGING_NOT_TRAINED.csv'
    new=root/'data/imports/v38_new_dois/new_materials_18_SOURCE_STAGING.csv'
    if not (old.exists() and new.exists()):pytest.skip('v34/v38 not present')
    s1=pd.read_csv(old);s2=pd.read_csv(new)
    assert not (set(s1.doi)&set(s2.doi))
