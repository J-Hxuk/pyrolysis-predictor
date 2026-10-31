import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_v40_new9 import source_materials,new_qe,source_catalog,separate_targets,references,DOIS

def test_nine_papers_and_dedup():
    s=source_catalog()
    assert len(s)==9 and s.doi.nunique()==9
    assert int(s.category.eq('duplicate DOI').sum())==2

def test_staged_materials_and_targets():
    d=source_materials()
    assert len(d)==30 and d.material_id.is_unique
    assert d.BET_m2_g_training_candidate.notna().sum()==23
    assert d.TPV_cm3_g_training_candidate.notna().sum()==8
    assert (pd.to_numeric(d.BET_m2_g_training_candidate,errors='coerce').dropna()>=0).all()

def test_same_doi_raw_composition_only():
    d=source_materials()
    j=d[d.doi==DOIS['jiang']]
    assert len(j)==18 and j.feedstock_VM_pct_original.notna().all()
    assert d[d.doi==DOIS['chicken']].feedstock_C_pct_original.isna().all()
    assert references().join_policy.str.contains('NO_CROSS_DOI').sum()==2

def test_qe_mass_balance_and_source_kind():
    d=new_qe()
    assert len(d)==3 and d.target_name.eq('qe_reported_mg_g').all()
    assert d.target_value.le(d.C0_mg_L/d.dose_g_L).all()
    assert d.initial_pH.isna().all()
    assert d.model_use_tier.str.contains('CONDITIONAL').all()

def test_problematic_qe_not_training():
    d=separate_targets()
    assert len(d)==9
    assert not d.training_use.eq('QMAX_FOR_QE').any()
    assert (d[d.doi==DOIS['rice']].value_mg_g>5).all()

def test_no_missing_bet_fabrication():
    d=source_materials()
    assert d[(d.doi==DOIS['jiang']) & (d.original_sample_name.isin(['Jiang_BC_CD_300','Jiang_BC_CD_400']))].BET_m2_g_training_candidate.isna().all()
    assert d[(d.doi==DOIS['coffee']) & d.original_sample_name.str.endswith('NoSteam')].BET_m2_g_training_candidate.isna().all()

def test_total_vs_micropore():
    d=source_materials(); t=d[d.TPV_cm3_g_training_candidate.notna()]
    assert len(t)==8 and t.v40_pore_type.eq('TOTAL').all()

from scripts.v40_applicability import training_envelope,check_input
from scripts.audit_project_hygiene_v40 import scan

def test_applicability_not_guaranteed_when_within_ranges():
    e=training_envelope(pd.DataFrame([{'doi':'a','t':300,'feed':'x'},{'doi':'b','t':500,'feed':'y'}]),['t'],['feed'])
    v=check_input({'t':400,'feed':'x'},e)
    assert v['status']=='MARGINAL_RANGE_ONLY_NOT_CERTIFIED'
    out=check_input({'t':700,'feed':'new'},e,essential=['pH'])
    assert {a['level'] for a in out['alerts']}=={'marginal_out_of_range','unseen_category','missing'}

def test_hygiene_scanner_read_only(tmp_path):
    (tmp_path/'Biochar_v99_fake_Patch.zip').write_bytes(b'hello')
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__'/'a.pyc').write_bytes(b'a')
    initial=sorted(str(x) for x in tmp_path.rglob('*'))
    report=scan(tmp_path)
    assert len(report['historical_patch_archives'])==1 and report['cache_count']==1
    assert initial==sorted(str(x) for x in tmp_path.rglob('*'))
