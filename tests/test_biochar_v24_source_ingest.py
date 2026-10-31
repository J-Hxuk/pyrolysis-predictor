"""Source staging checks; does not train models or mutate older sources."""
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'imports'/'v24_source_audit'

def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def test_source_table_counts():
    a,b,c=read('stage1_new33_BET_TPV_SOURCE_STAGING.csv'),read('stage2_new72_source_reported_QE_STAGING.csv'),read('all_qmax_32_FITTED_NOT_QE.csv')
    assert (len(a),len(b),len(c))==(33,72,32)
    assert len({(r['doi'],r['sample_name']) for r in a})==33

def test_physical_constraints_and_quarantine():
    obs=read('stage2_new72_source_reported_QE_STAGING.csv')
    ok=[r for r in obs if r['physical_cap_pass']=='True']
    no=[r for r in obs if r['physical_cap_pass']=='False']
    assert len(ok)==69 and len(no)==3
    assert len(read('stage2_new3_PHYSICAL_FAIL_QUARANTINE.csv'))==3
    assert all(r['pollutant']=='Cd(II)' and int(r['C0_mg_L'])==100 and r['doi']=='10.3390/ma14216566' for r in no)
    assert all(r['stage2_training_eligibility']!='MODEL_APPROVED' for r in obs)

def test_pomelo_confirmed_unit_and_rounding():
    po=[r for r in read('stage2_new72_source_reported_QE_STAGING.csv') if r['doi']=='10.3390/w13060868']
    assert len(po)==18
    assert all(r['unit_assumption'].startswith('umol/g_') for r in po)
    match=next(r for r in po if r['pollutant']=='Cu(II)' and r['sample_name']=='BC400')
    assert abs(float(match['qe_mg_g'])-6.2097)<1e-3
    pb=next(r for r in po if r['pollutant']=='Pb(II)' and r['sample_name']=='MBC600')
    assert 'ROUNDING_TOLERANCE_CAP_EXCESS' in pb['quality_flag']

def test_staging_does_not_activate_models():
    s1=read('Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv')
    s2=read('Stage2_Core_167_RESEARCH_STAGING_NOT_TRAINED.csv')
    assert len(s1)==297 and len(s2)==167
    staged=[r for r in s1 if r['material_id'].startswith('S1ADD24_')]
    assert len(staged)==33
    assert all(not r['BET_m2_g_training_candidate'] and not r['TPV_cm3_g_training_candidate'] for r in staged)
    staged2=[r for r in s2 if r['observation_id'].startswith('S2ADD24_')]
    assert len(staged2)==69
    assert all(r['model_use_tier'] in {'STAGING_ONLY_REVIEW_UNIT_CONFIRMATION','HOLD_FOR_MECHANISM_REVIEW'} for r in staged2)

def test_hydrothermal_not_mislabeled_as_dry_pyrolysis():
    s1=read('stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    h=[r for r in s1 if r['doi']=='10.1016/j.heliyon.2023.e14930']
    assert len(h)==6
    assert all(not r['pyrolysis_temp_C'] and r['HTC_temp_C']=='180' for r in h)

def test_source_paper_ratios_and_TPV_unit():
    s1=read('stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    c=next(r for r in s1 if r['sample_name']=='C1P1500')
    assert c['pyrolysis_temp_C']=='500'
    assert abs(float(c['TPV_cm3_g'])-0.0558)<1e-9
    assert float(c['mix_component_2_weight_fraction'])==.5
