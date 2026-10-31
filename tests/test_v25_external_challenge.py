"""Fast v25 evidence contract tests; no model training on staging rows."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/imports/v24_source_audit'
REPORT=ROOT/'reports/v25_v24_external_challenge'


def test_new_stage1_all_six_doi_disjoint_from_phase8():
    new=pd.read_csv(SOURCE/'stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    old=pd.read_csv(ROOT/'data/imports/Biochar_Phase8_Integrated_Data_Package/data/stage1/Stage1_Master_264_UNCHANGED.csv')
    assert len(new)==33 and new.doi.nunique()==6
    assert set(new.doi).isdisjoint(old.doi)
    assert len(new[['doi','sample_name']].drop_duplicates())==33


def test_staging_model_targets_unapproved_and_zero_released():
    master=pd.read_csv(SOURCE/'Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv',low_memory=False)
    added=master[master.material_id.astype(str).str.startswith('S1ADD24_')]
    assert len(added)==33
    assert added.BET_m2_g_training_candidate.isna().all()
    assert added.TPV_cm3_g_training_candidate.isna().all()
    assert (added.new_data_release_gate=='NOT_APPROVED_FOR_TRAINING').all()
    staged=pd.read_csv(SOURCE/'Stage2_Core_167_RESEARCH_STAGING_NOT_TRAINED.csv')
    s2=staged[staged.observation_id.astype(str).str.startswith('S2ADD24_')]
    assert len(s2)==69
    assert s2.target_qualification.astype(str).str.contains('NOT_CERTIFIED').all()


def test_quarantine_and_pomelo_missing_feature():
    new=pd.read_csv(SOURCE/'stage2_new72_source_reported_QE_STAGING.csv')
    assert len(new)==72
    assert len(new[new.physical_cap_pass==False])==3
    assert len(new[new.initial_pH.isna()])==18
    assert set(new[new.initial_pH.isna()].doi)=={'10.3390/w13060868'}


def test_stage1_route_mapping_in_report():
    rows=pd.read_csv(REPORT/'stage1_external33_predictions.csv')
    assert len(rows)==66
    for target in ['BET','TPV']:
        one=rows[rows.target.eq(target)]
        assert len(one)==33
        assert (one.is_single_stage==True).sum()==17
        assert one.doi.nunique()==6
        assert one.pred.notna().all()


def test_heldout_poultry_and_original_inputs_not_promoted():
    data=pd.read_csv(REPORT/'stage2_poultry51_stress_predictions.csv')
    assert len(data)==102 and data.doi.nunique()==1
    assert set(data.model)=={'A','B'}
    assert data.external_challenge_only.all()
    assert (data.dose_g_L==100).all()
    assert (data.actual_mg_g <= data.physical_cap_mg_g+1e-10).all()
    assert data.raw_violation.all()


def test_clipping_can_mimic_physical_baseline():
    data=pd.read_csv(REPORT/'stage2_poultry51_stress_predictions.csv')
    a=data[data.model.eq('A')]
    assert (a.pred_raw_mg_g>a.physical_cap_mg_g).all()
    assert (a.pred_physically_clipped_mg_g==a.physical_cap_mg_g).all()
    b=data[data.model.eq('B')]
    assert (b.pred_raw_mg_g <0).all()
    assert (b.pred_physically_clipped_mg_g==0).all()


def test_summary_no_training_claim():
    x=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert x['not_training_on_v24'] and x['not_model_upgrade']
    assert x['stage2']['input_exclusions']['training_approved_new_rows']==0
    assert x['stage2']['input_exclusions']['training_dose_max']==10.
