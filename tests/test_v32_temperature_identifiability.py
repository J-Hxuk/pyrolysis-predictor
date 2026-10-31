"""v32 independently executable guards and research result checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_v32_temperature_identifiability import (
    cleaned_inputs,train_predict,temperature_design_audit,
    feature_ablation_lodo,response_curves,run,ROOT
)
from scripts.compare_phase8_v23 import stage2_targets
from scripts.evaluate_v31_rice_temperature_external import load_external_unmodified
from pyrolysis.biochar_stage2.v15_features import model_inputs


def test_temperature_missing_is_not_zero():
    test=pd.DataFrame([{'pyrolysis_temperature_C':'', 'feedstock_label_normalized':'rice husk'}])
    x=cleaned_inputs(test,['pyrolysis_temperature_C','feedstock_label_normalized'])
    assert np.isnan(x.pyrolysis_temperature_C.iloc[0])
    assert x.feedstock_label_normalized.iloc[0]=='rice husk'


def test_design_audit_expected_cohort():
    _,core=stage2_targets()
    bydoi,a=temperature_design_audit(core)
    assert (len(cored:=core)==98 and core.doi.nunique()==26)
    assert a['reported_temperature_observations']==79
    assert a['unreported_temperature_observations']==19
    assert a['doi_with_more_than_one_reported_temperature']==0
    assert a['strict_temperature_contrast_groups']==0
    assert len(bydoi)==26


def test_design_detects_added_temperature_contrast():
    _,core=stage2_targets(); sampled=core.iloc[[0]].copy()
    sampled['pyrolysis_temperature_C']=997
    sampled['observation_id']='TEST_V32_TEMPERATURE_CONTRAST'
    sampled['material_uid']='TEST_V32_MATERIAL'
    changed=pd.concat([core,sampled],ignore_index=True)
    _,a=temperature_design_audit(changed)
    assert a['doi_with_more_than_one_reported_temperature']==1


def test_feature_contract_distinguishes_temperature():
    full=model_inputs('B')
    without=[x for x in full if x!='pyrolysis_temperature_C']
    assert len(full)==len(without)+1
    assert set(without).issubset(set(full))
    assert 'target_value' not in full and 'doi' not in full


def test_unrecognized_algorithm_blocked():
    with pytest.raises(ValueError,match='Only declared'):
        train_predict(pd.DataFrame(),pd.DataFrame(),'unsupported',['pyrolysis_temperature_C'])


def test_external_paper_not_in_training():
    _,core=stage2_targets()
    e=load_external_unmodified()
    assert e.doi.nunique()==1 and e.doi.iloc[0] not in set(core.doi)
    assert len(e)==6 and e.training_approved.astype(str).str.lower().eq('false').all()
    assert e.groupby('pollutant').pyrolysis_temp_C.nunique().eq(3).all()


def test_curve_guard_rejects_source_doi_in_training():
    _,core=stage2_targets();e=load_external_unmodified()
    bad=core.copy();bad.loc[bad.index[0],'doi']=e.doi.iloc[0]
    with pytest.raises(ValueError,match='leaked'):
        response_curves(bad,e)


def test_curve_counterfactual_only_temperature_changes():
    _,core=stage2_targets();e=load_external_unmodified()
    curves,qa,meta=response_curves(core,e)
    assert len(curves)==2*2*9
    assert curves.groupby(['pollutant','algorithm']).pyrolysis_temperature_C.nunique().eq(9).all()
    assert not qa.curve_has_interior_strict_local_maximum.any()
    assert qa.observed_best_C.eq(400).all()
    assert qa.predicted_best_C_300_400_500.eq(500).all()
    assert meta['evaluation_points']==[300,400,500]


def test_ablation_same_rows_and_folds():
    _,core=stage2_targets()
    long,pair,stats=feature_ablation_lodo(core)
    assert len(long)==98*2*2
    assert len(pair)==98*2
    assert long.groupby(['algorithm','setting']).observation_id.nunique().eq(98).all()
    assert pair[['doi','observation_id','algorithm']].duplicated().sum()==0
    assert all(np.isfinite(pair[k]).all() for k in ['with_temperature','no_temperature'])
    assert set(stats)=={'Ridge','XGBoost'}


def test_existing_saved_results_are_not_training_modified(tmp_path):
    result=run(out=tmp_path)
    assert result['released_model'] is False
    assert result['ranking_training_feasible'] is False
    assert result['app_changed'] is False
    assert result['design']['strict_temperature_contrast_groups']==0
    assert (tmp_path/'summary.json').exists()
    assert (tmp_path/'ablation_paired_oof.csv').exists()
    assert len(json.loads((tmp_path/'summary.json').read_text(encoding='utf8'))['rice_curve_diagnostics'])==4
