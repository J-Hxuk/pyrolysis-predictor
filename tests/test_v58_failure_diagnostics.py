from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from scripts.run_v58_failure_diagnostics import derive,build
from scripts.build_v46_phase8 import features

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/v58_failure_diagnostics'
DEP=ROOT/'deployment/research_v58_NOT_ACTIVE'

def test_report_has_matched_rows():
    x=pd.read_csv(REP/'derived_cap_feature_fixed140_oof.csv')
    old=pd.read_csv(ROOT/'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv')
    assert len(x)==len(old)==140
    assert x.observation_id.tolist()==old.observation_id.tolist()
    assert old.doi.nunique()==32

def test_derived_feature_is_not_target_leakage():
    x=pd.DataFrame({'C0_mg_L':[100,50],'dose_g_L':[2,5],'target_value':[10,25]})
    a=derive(x,'cap_log');b=derive(x.assign(target_value=[500,900]),'cap_log')
    assert np.allclose(a.physical_cap_mg_g,[50,10])
    assert np.allclose(a.physical_cap_mg_g,b.physical_cap_mg_g)
    assert np.allclose(a.log_physical_cap,b.log_physical_cap)

def test_saved_oof_metrics_recomputed():
    p=pd.read_csv(REP/'derived_cap_feature_fixed140_oof.csv')
    report=pd.read_csv(REP/'derived_cap_feature_fixed140_metrics.csv')
    for r in report.itertuples(index=False):
        if r.feature_set=='original_archived': continue
        key=f'{r.model}__{r.scenario}__{r.feature_set}'
        actual=mean_absolute_error(p.target_value,p[key])
        assert np.isclose(actual,r.mae_mg_g,atol=1e-7)

def test_rank_top_ties_cannot_be_top1():
    rank=pd.read_csv(REP/'chen16_rank_feature_comparison.csv')
    subset=rank[(rank.model=='B')&(rank.feature_set.isin(['cap','cap_log']))]
    assert len(subset)==2
    assert (subset.top_tie_count==12).all()
    assert subset.strict_top1_regret.isna().all()
    assert (subset.top1_unresolved_due_to_tie.astype(str)=='True').all()

def test_source_out_no_training_overlap():
    add=pd.read_csv(ROOT/'data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv')
    old=pd.read_csv(ROOT/'data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv')
    assert len(add)==36 and add.doi.nunique()==2
    assert not set(add.doi)&set(old.doi)
    assert add.groupby('doi').size().sort_values().tolist()==[7,29]

def test_research_models_reload():
    manifest=json.loads((DEP/'manifest_v58.json').read_text(encoding='utf-8'))
    assert len(manifest)==4
    old=pd.read_csv(ROOT/'data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv')
    for rec in manifest:
        mod=joblib.load(ROOT/rec['model_path'])
        data=derive(old.iloc[:3],rec['feature_set'])
        v=mod.predict(features(data,rec['features'],rec['numeric'],rec['categorical']))
        assert len(v)==3 and np.isfinite(v).all()
        assert not rec['active']
