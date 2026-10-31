"""v31 safety gates, source contract and reproducible external DOI challenge."""
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_v31_rice_temperature_external import (
    HELDOUT_DOI, ROOT, UNMODIFIED_PROCESS, load_external_unmodified,
    model_input_rows, validate_holdout, score_one, run,
)
from scripts.compare_phase8_v23 import stage2_targets
from pyrolysis.biochar_stage2.v15_features import model_inputs


def source():
    return load_external_unmodified()


def test_original_v30_cohort_six_nontrained_rows():
    s=source()
    assert len(s)==6 and s.pollutant.nunique()==2
    assert not s.training_approved.any()
    assert s.stage1_material_id.nunique()==3
    assert (s.digitization_tolerance_mg_g==0.5).all()
    assert (s.qe_reported_approx_mg_g <= s.C0_mg_L/s.dose_g_L).all()


def test_only_temperature_varies_within_each_pollutant():
    x=model_input_rows(source())
    for pollutant,grp in x.groupby('pollutant'):
        assert sorted(grp.pyrolysis_temperature_C.tolist())==[300,400,500]
        for col in x.columns:
            if col != 'pyrolysis_temperature_C':
                assert grp[col].nunique(dropna=False)==1,(pollutant,col)


def test_exactly_model_b_allowed_inputs_and_no_target():
    x=model_input_rows(source())
    assert set(x.columns)==set(model_inputs('B'))
    assert not {'qe_reported_approx_mg_g','target_value','fitted_table_qe2_DO_NOT_USE',
                'stage1_BET_measured_m2_g','stage1_TPV_measured_cm3_g'} & set(x.columns)


def test_source_methods_known_unmodified_not_unreported():
    assert UNMODIFIED_PROCESS['activation_or_modification']=='none'
    assert UNMODIFIED_PROCESS['process_step_count']==1
    assert UNMODIFIED_PROCESS['heating_rate_C_min']==10
    assert UNMODIFIED_PROCESS['pyrolysis_hold_min']==120


def test_training_doi_is_external_and_missing_process_combination():
    _,train=stage2_targets()
    audit=validate_holdout(train,source(),model_input_rows(source()))
    assert audit['train_rows']==98
    assert audit['train_dois']==26
    assert not audit['external_doi_seen_in_stage2_training']
    assert audit['rice_husk_train_rows']==13
    assert audit['rice_husk_single_stage_train_rows']==0
    assert not audit['explicit_unmodified_agent_category_seen_in_training']


def test_reject_doi_leakage():
    _,train=stage2_targets()
    corrupt=train.copy()
    corrupt.loc[corrupt.index[0],'doi']=HELDOUT_DOI
    with pytest.raises(ValueError,match='LEAKAGE'):
        validate_holdout(corrupt,source(),model_input_rows(source()))


def test_reject_modified_or_noncomparable_data(tmp_path):
    path=ROOT/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv'
    d=pd.read_csv(path)
    d.loc[d.sample_id.eq('BC300'),'training_approved']=True
    corrupt=tmp_path/'changed.csv';d.to_csv(corrupt,index=False)
    with pytest.raises(ValueError,match='NOT_TRAINED'):
        load_external_unmodified(corrupt)


def test_reject_absent_temperature_stratum(tmp_path):
    path=ROOT/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv'
    d=pd.read_csv(path)
    d=d.loc[~((d.sample_id=='BC300')&(d.pollutant=='Cd(II)'))].copy()
    corrupt=tmp_path/'missing.csv';d.to_csv(corrupt,index=False)
    with pytest.raises(ValueError):
        load_external_unmodified(corrupt)


def test_reject_distinct_adsorption_conditions():
    s=source()
    s.loc[(s.pollutant=='Cd(II)')&(s.pyrolysis_temp_C==500),'initial_pH']=6
    # Double check input contract also rejects within-group differences.
    with pytest.raises(ValueError,match='temperature-only'):
        model_input_rows(s)


def test_pairwise_expected_for_monotone_temperature_predictions():
    y=np.array([2.7,7.2,4.2])
    s=score_one(y,np.array([-6.3,-5.7,-5.0]))
    assert s['n_pairs']==3
    assert s['pairwise_tie_half']==pytest.approx(2/3)
    assert s['top1_expected']==0
    assert s['regret_mg_g']==pytest.approx(3)
    assert s['observed_best_has_digitization_margin']


def test_tie_baseline_not_falsely_declared_failure():
    y=np.array([2.7,7.2,4.2])
    s=score_one(y,np.array([10,10,10]))
    assert s['pairwise_tie_half']==0.5
    assert s['top1_expected']==pytest.approx(1/3)
    assert s['prediction_ties']==3


def test_run_reports_no_clipped_score_and_no_train_rewrite(tmp_path):
    target=tmp_path/'v31'
    metrics=run(out=target)
    assert metrics['external_doi_seen_in_stage2_training'] is False
    assert metrics['train_model_uses_graph_data'] is False
    assert metrics['no_clipping_used_for_model_scoring']
    assert metrics['models']['B_Ridge']['top1_expected_mean']==0
    assert metrics['models']['B_Ridge']['raw_constraint_violations']==3
    assert metrics['models']['B_XGBoost_refitCore97_sensitivity']['top1_expected_mean']==0
    assert metrics['models']['B_Ridge_refitCore97_sensitivity']['raw_constraint_violations']==3
    assert metrics['models']['A_XGBoost']['pairwise_macro']==0.5
    assert (target/'source_backed_model_inputs_NO_TARGET.csv').exists()
    x=pd.read_csv(target/'source_backed_model_inputs_NO_TARGET.csv')
    assert 'qe_reported_approx_mg_g' not in x
