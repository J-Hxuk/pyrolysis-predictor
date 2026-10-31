"""Nonmutating Phase8 checks. Verify exact ID lineage and cohort metadata."""
from pathlib import Path
import csv
import pytest
from pyrolysis.biochar_stage2.phase8_evidence import audit_phase8,build_stage2_core98_candidate,read_csv,phase8_path
from scripts.build_phase8_v22_reports import build

ROOT = Path(__file__).resolve().parents[1]

def test_integrated_counts_and_new_ids():
    d=audit_phase8(ROOT)
    assert (d['stage1']['previous'],d['stage1']['integrated'],d['stage1']['added'])==(256,264,8)
    assert (d['stage2']['previous'],d['stage2']['integrated'],d['stage2']['added'])==(97,98,1)
    assert d['stage2']['strict_new']==1
    assert len({r['doi'] for r in d['stage1_new']})==2


def test_new_materials_are_real_observations_not_duplicate_rows():
    d=audit_phase8(ROOT)
    assert len({x['material_id'] for x in d['stage1_new']})==8
    assert len({x['adsorption_id'] for x in d['stage2_new']})==1
    assert d['stage2_new'][0]['source_target_status']=='USE_DIRECT_EQUILIBRIUM'
    assert d['stage2_new'][0]['adsorption_id']=='V15_ADS_ALMOND_SP_C500'


def test_optional_98_row_research_cohort_preserves_original_97():
    before=read_csv(ROOT/'data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/03_Stage2_QE_Core_97.csv')
    after=build_stage2_core98_candidate(ROOT)
    assert before==after[:97]
    new=after[-1]
    assert set(new)==set(before[0])
    assert new['feedstock_label_normalized']=='almond shell'
    assert new['target_value']=='103.0'
    assert new['manufacturing_route']=='SINGLE_STAGE_PYROLYSIS'
    assert new['pyrolysis_temperature_C']=='600'
    assert new['target_qualification']=='USE_DIRECT_EQUILIBRIUM'
    assert new['raw_C_wt_pct']=='' and new['raw_VM_wt_pct']==''
    assert float(new['target_value'])<=float(new['C0_mg_L'])/float(new['dose_g_L'])


def test_composition_coverage_do_not_overclaim_generalization():
    d=audit_phase8(ROOT)
    cov=d['composition_coverage']
    assert cov['BET']['fields']['proximate_VM_Ash_FC_complete']=={'n':17,'doi':1}
    assert cov['TPV']['fields']['proximate_VM_Ash_FC_complete']=={'n':0,'doi':0}
    assert cov['BET']['fields']['CHON_complete']['doi']==3
    assert cov['TPV']['fields']['CHON_complete']['doi']==2


def test_snapshot_contains_no_deployed_model_replacement():
    d=audit_phase8(ROOT)
    assert not d['model_deployment_changed'] and not d['model_retrained']
    assert d['recipes']['adapter_allowed']==162


def test_export_in_temp_does_not_mutate_models(tmp_path):
    # Read model source from current project; reports are generated only in a separate clone.
    import shutil
    # Use minimal existing sources from their immutable snapshots.
    copy=tmp_path/'project'
    (copy/'data/imports').mkdir(parents=True)
    for name in ['Biochar_Phase8_Integrated_Data_Package',
                 'Biochar_Stage1_v60','Stage1_Stage2_Model_Ready_v15']:
        original=ROOT/'data/imports'/name
        # Keep test inexpensive: symlink to readonly source if supported, else copytree.
        try:
            (copy/'data/imports'/name).symlink_to(original,target_is_directory=True)
        except OSError:
            shutil.copytree(original,copy/'data/imports'/name)
    result=build(copy)
    assert result['stage1']['added']==8
    assert len(read_csv(copy/'reports/phase8_v22/stage2_98_RESEARCH_CANDIDATE_NOT_TRAINED.csv'))==98
    assert not (copy/'deployment').exists()
