"""Research-only v29 integrity: source-backed agent recovery + temperature readiness."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.compare_phase8_v23 import stage2_targets
from scripts.evaluate_v29_process_feature_recovery import (
    ROOT,OUT,extract_reagent_signature,material_feature_table,compare_identifiability,
)
from scripts.audit_v29_temperature_readiness import audit_temperature, audit_external_poultry_temperature


def test_agent_from_explicit_method_not_label():
    row=pd.Series({'sample_name':'SOMETHING',
                   'source_method_process_text':'FeCl3.6H2O + KH2PO4 in 100mL',
                   'source_pretreatment_activation_text':'','reported_activation_agent':''})
    assert extract_reagent_signature(row)==('FECL3|KH2PO4','SOURCE_REAGENT_EXPLICIT')


def test_unverified_phosphate_control_not_fabricated():
    r=pd.Series({'sample_name':'P@SBC','source_method_process_text':'P-only labeledcontrol;exactP@SBC synthesis not separatelyreported;NR',
                 'source_pretreatment_activation_text':'','reported_activation_agent':''})
    assert extract_reagent_signature(r)==('P_ONLY_REPORTED_LABEL_NOT_VERIFIED','LABEL_ONLY_METHOD_NR')
    r2=r.copy();r2['source_method_process_text']='noimpregnation; dry and pyrolyze'
    assert extract_reagent_signature(r2)==('NO_REAGENT_REPORTED','SOURCE_CONTROL_EXPLICIT')


def test_avoid_water_wash_hydration_as_steam_activation():
    r=pd.Series({'source_method_process_text':'washwater;FeCl3.6H2O and FeSO4.7H2O',
                 'source_pretreatment_activation_text':'','reported_activation_agent':''})
    a,e=extract_reagent_signature(r)
    assert a=='FECL3|FESO4'
    assert 'STEAM' not in a


def test_new_phase8_material_not_in_old_master_is_nr():
    _,core=stage2_targets()
    ev=material_feature_table(core)
    missing=ev.loc[ev.material_uid.eq('V15_ALMOND_AS-RAW-SP')]
    assert len(missing)==1
    assert missing.iloc[0].v29_source_tier=='UNREPORTED'
    assert missing.iloc[0].v29_reagent_signature=='REAGENT_NOT_REPORTED'


def test_all_six_collisions_resolved_for_30_evaluable_pairs():
    _,core=stage2_targets()
    evidence=material_feature_table(core)
    pairs=compare_identifiability(core,evidence)
    assert len(pairs)==30
    assert pairs.old_inputs_identical.sum()==6
    assert pairs.new_inputs_identical.sum()==0
    assert pairs.loc[pairs.old_inputs_identical,'doi'].nunique()==3


def test_saved_lodo_baseline_exactly_reproduces_v27_in_scope():
    oof=pd.read_csv(OUT/'lodo_oof_baseline_vs_enriched.csv')
    old=pd.read_csv(ROOT/'reports/v27_recommendation_audit/stage2_heldout_doi_predictions.csv')
    joined=old.merge(oof,on='observation_id',validate='one_to_one')
    assert len(joined)==33
    assert np.allclose(joined.pred_b_raw,joined.pred_baseline,rtol=0,atol=1e-8)
    assert len(oof)==98 and oof.doi.nunique()==26


def test_report_no_unearned_model_improvement():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    assert s['enriched']['mae']>s['baseline']['mae']
    assert s['enriched']['raw_violations']>=s['baseline']['raw_violations']
    assert not s['pairwise_feature_identifiability']['new_collisions']


def test_temperature_series_are_fitted_qmax_not_qe():
    j,g,s=audit_temperature()
    assert len(j)==32 and j.material_id.notna().all()
    assert s['temp_contrast_qmax_series']==8
    assert s['temp_contrast_qmax_dois']==3
    assert s['temp_contrast_qmax_sample_rows']==26
    assert not s['qe_temperature_ranking_validated']
    assert not g.qe_ranking_ready.any()
    assert set(g.target_type)=={'LANGMUIR_QMAX_FITTED_NOT_REPORTED_QE'}


def test_no_manufacturing_model_deployment_changes_in_patch():
    # This is a contract test of the patch inventory (packaging checked separately).
    assert not (OUT/'active.json').exists()
    for p in [OUT/'lodo_oof_baseline_vs_enriched.csv',OUT/'pairwise_feature_resolution.csv']:
        assert p.is_file()


def test_external_poultry_repeated_temperature_groups_are_one_doi_not_replicated_dois():
    groups,summary=audit_external_poultry_temperature()
    assert summary['records']==51
    assert summary['groups']==17 and summary['groups_with_3_temperatures']==17
    assert summary['independent_dois']==1
    assert summary['all_100_g_L']
    assert 0 < summary['max_within_group_qe_gap_mg_g'] < 0.1
    assert not summary['approved_for_temperature_model_training']
