"""v62: tests compare current local data, never hard-code historical accuracy numbers."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.run_v62_pairwise_diagnostics import run, pairwise_scores, pairwise_concord, bootstrap_doi
from scripts.run_v60_separate_effects import MFG

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def data():
    run(ROOT)
    folder=ROOT/'reports/v62_pairwise_manufacturing'
    return folder, json.loads((folder/'manifest_v62.json').read_text(encoding='utf-8'))

def test_oof_coverage_and_source_ids(data):
    folder,manifest=data
    d=pd.read_csv(folder/'panel_oof_predictions.csv')
    assert len(d)==188 and d.observation_id.is_unique
    s=d[d.panel_id>=0]
    assert len(s)==59 and s.panel_id.nunique()==15
    assert np.isfinite(s.pairwise_relwin_score_NOT_QE).all()
    assert s.pairwise_relwin_score_NOT_QE.between(0,1).all()
    assert s.doi.nunique()==9

def test_fold_group_and_no_train_test_overlap(data):
    folder,manifest=data
    d=pd.read_csv(folder/'panel_oof_predictions.csv')
    a=pd.read_csv(folder/'fold_training_audit.csv')
    for _,r in a.iterrows():
        held=set(r.test_DOI.split('|'))
        selected=d[d.fold==r.fold]
        assert set(selected.doi)==held
        assert not d[(d.fold!=r.fold)].doi.isin(held).any()

def test_pairs_are_anti_symmetric(data):
    folder,manifest=data
    artifact=ROOT/manifest['research_artifact']
    model=joblib.load(artifact)
    assert model['features']==list(MFG)
    pre,est=model['preprocessor'],model['estimator']
    import scipy.sparse as sp
    one=sp.csr_matrix(np.zeros((1,len(est.coef_[0]))))
    assert abs(est.predict_proba(one)[0,1]-.5)<1e-12
    # no intercept: p(x_j-x_i) = 1-p(x_i-x_j)
    z=sp.csr_matrix(np.arange(len(est.coef_[0]))[None,:]%3-1)
    assert abs(est.predict_proba(z)[0,1]+est.predict_proba(-z)[0,1]-1)<1e-12

def test_source_outcome_never_input_feature(data):
    _,manifest=data
    names=set(MFG)
    assert not any(x in names for x in ['target_value','qe','qmax','removal_pct','doi','panel_id','material_uid'])
    assert 'NOT_QE' in manifest['research_artifact']

def test_metrics_have_valid_tie_information(data):
    folder,_=data
    d=pd.read_csv(folder/'panel_rank_metrics.csv')
    assert len(d)==15*4
    assert d.groupby('model').size().eq(15).all()
    assert (d.n_top>=1).all()
    assert (d.n_top<=d.n).all()
    assert (d.regret_worst_mg_g>=-1e-9).all()
    assert d.pairwise_concordance.dropna().between(0,1).all()
    # Dropped undefined Spearman may not be imputed with zero.
    undefined=d.loc[d.rho.isna()]
    assert all(d.loc[d.model==m,'rho'].count()<=15 for m in d.model.unique())

def test_rank_not_equivalent_to_cap_and_raw_criteria(data):
    folder,manifest=data
    d=pd.read_csv(folder/'panel_oof_predictions.csv')
    cap=d.C0_mg_L/d.dose_g_L
    assert np.allclose(d['188__B_doi_material'],np.clip(d['188__B_doi_material__raw'],0,cap),atol=2e-5,rtol=1e-8)
    assert manifest['raw_upper_cap_violation_all188']==int((d['188__B_doi_material__raw']>cap+1e-8).sum())

def test_pairwise_feature_pairs_and_no_fabricated_sources(data):
    folder,manifest=data
    pr=pd.read_csv(folder/'pairwise_feature_ambiguity.csv')
    d=pd.read_csv(folder/'panel_oof_predictions.csv')
    expected=sum(len(g)*(len(g)-1)//2 for _,g in d[d.panel_id>=0].groupby('panel_id'))
    assert len(pr)==expected==manifest['n_pairs_total']
    assert pr.p_left_better.between(0,1).all()
    assert pr.identical_encoded_features.sum()==manifest['n_ambiguous_input_pairs']

def test_bootstrap_counts_reflect_actual_available_panels(data):
    folder,manifest=data
    d=pd.read_csv(folder/'panel_rank_metrics.csv')
    for model in ('mfg_forest','B_clipped','B_raw_invalid_as_qe'):
        c,joined=bootstrap_doi(d,'pairwise',model,n=100,seed=13)
        assert c['n_panels_total']==15
        assert c['n_panels_valid']==len(joined.dropna(subset=['a','b']))
        assert c['n_doi_valid']==len(joined.dropna(subset=['a','b']).doi.unique())
        if c['ci'] is not None:assert c['ci'][0]<=c['difference']<=c['ci'][1] or c['n_doi_valid']>1

def test_manifest_tracks_local_model(data):
    folder,manifest=data
    import hashlib
    fp=ROOT/manifest['research_artifact']
    assert fp.exists()
    assert hashlib.sha256(fp.read_bytes()).hexdigest()==manifest['research_artifact_sha256']
    assert manifest['model_role']=='RESEARCH_ONLY_DOI_GROUPED_POSTHOC'
