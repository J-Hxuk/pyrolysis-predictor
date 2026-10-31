"""Synthetic-only contract tests. Never interpret their scores as real biochar model quality."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v53_stage2_patch as patch


def test_source_evidence_and_overlap():
    x=patch.source_audit()
    assert x['strict_qe_label_verified']==8
    assert x['dose_1_20_review_non_strict']==16
    assert x['external_holdout_approved']==0
    assert x['physical_material_batch_link_confirmed'] is False


def test_denormalized_doi_metal_and_source24():
    ref=patch.load_ref()
    assert len(ref)==24
    assert ref['__key'].nunique()==24
    assert patch.doi('https://doi.org/10.3390/en13133498')==patch.ZHAO
    assert patch.metal('Cu(II)')=='cu'
    assert patch.metal('Zn(II)')=='zn'


def construct(tmp_path, corrupt=False):
    ref=patch.load_ref().copy()
    rng=np.random.default_rng(7)
    core=[]
    for i in range(140):
        doi='10.0000/synthetic_'+str(i//10)
        metal=['Cu(II)','Zn(II)'][i%2]
        dose=[2.,4.,8.][i%3]
        c0=float(80+i%4)
        qe=float(min(c0/dose*.4,10)+rng.normal(0,.15))
        core.append({'doi':doi,'pollutant':metal,'dose_g_L':dose,'pyrolysis_temp_C':300+(i%4)*100,
           'observed_adsorption_mg_g':qe,'C0_mg_L':c0,'initial_pH':5.,'adsorption_temperature_C':25.,
           'contact_time_min':1440.,'feedstock_original':'synthetic_'+str(i//10)})
    base=pd.DataFrame(core)
    cand=ref.rename(columns={'pyrolysis_temp_C':'pyrolysis_temp_C'}).copy()
    cand['feedstock_original']='apple tree branches'
    cand=cand[[*base.columns]].copy()
    if corrupt:cand.loc[cand.index[0],'observed_adsorption_mg_g']+=1
    v52=pd.concat([base,cand],ignore_index=True)
    p1=tmp_path/'baseline140_SYNTHETIC.csv';p2=tmp_path/'v52_164_SYNTHETIC.csv'
    base.to_csv(p1,index=False);v52.to_csv(p2,index=False)
    conf=json.loads((ROOT/'configs'/'feature_mapping_TEMPLATE.json').read_text(encoding='utf-8'))
    conf['estimators']=['Mean']
    pc=tmp_path/'feature_config.json';pc.write_text(json.dumps(conf),encoding='utf-8')
    return p1,p2,pc


def test_audit_and_evaluate_only_on_140_baseline(tmp_path):
    p1,p2,pc=construct(tmp_path)
    cmd=[sys.executable,str(ROOT/'scripts'/'v53_stage2_patch.py'), 'evaluate',
       '--baseline',str(p1),'--v52',str(p2),'--config',str(pc),
       '--output',str(tmp_path/'evaluation'),'--confirm-research-only']
    x=subprocess.run(cmd,capture_output=True,text=True)
    assert x.returncode==0,x.stderr
    s=json.loads((tmp_path/'evaluation'/'04_comparison.json').read_text(encoding='utf-8'))
    assert s['v52_overlap']['ready_for_three_scenarios']
    assert s['results']['A_Mean']['140']['n']==140
    assert s['results']['B_Mean']['140_plus_strict8']['n']==140
    folds=pd.read_csv(tmp_path/'evaluation'/'03_fold_lineage.csv')
    assert set(folds['n_train_added'])=={0,8,24}
    assert s['deployment']=='NOT_PROMOTED_EXISTING_DEPLOYMENT_UNCHANGED'


def test_source_conflict_blocks_training(tmp_path):
    p1,p2,pc=construct(tmp_path,corrupt=True)
    cfg=json.loads(pc.read_text())
    report,_=patch.report_cohort(pd.read_csv(p2),cfg)
    assert not report['ready_for_three_scenarios']
    assert len(report['zhao_qe_conflicts'])==1


def test_forbidden_leakage_features(tmp_path):
    p1,p2,pc=construct(tmp_path)
    cfg=json.loads(pc.read_text()); cfg['models']['A']['numeric'].append('observed_adsorption_mg_g')
    with pytest.raises(patch.AuditError):patch.assert_config(cfg,pd.read_csv(p1),'synthetic')


def test_mass_balance_warning_is_not_silently_deleted():
    d=patch.metrics([5,6],[100,-1],[60,60],[5,5])
    assert d['prediction_physical_violation_rate']==1
    assert d['mae_after_physical_clipping_sensitivity_only']<d['mae']
