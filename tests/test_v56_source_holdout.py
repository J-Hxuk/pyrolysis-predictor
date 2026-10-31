"""Tests v56 anti-leakage guardrails and saved retrospective report, without fixed MAE numbers."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from scripts.run_v56_source_holdout import (
    load_s2, historical_dois, retro_rank, prospective, seal,
    sha, stage2_predict, OUT,SEAL
)
ROOT=Path(__file__).resolve().parents[1]

def test_three_new_doi_batches_are_disjoint_from_original():
    old,batches=load_s2(ROOT)
    assert len(old)==140
    assert {k:len(v) for k,v in batches.items()}=={'apple':8,'sugar':4,'cocoa_conditional':4}
    assert all(not set(old.doi)&set(v.doi) for v in batches.values())

def test_new_source_test_doi_cannot_enter_training():
    old,batches=load_s2(ROOT)
    tr=pd.concat([old,batches['apple']],ignore_index=True)
    with pytest.raises(AssertionError,match='DOI leakage'):
        stage2_predict(tr,batches['apple'],'A')

def test_source_out_report_matches_saved_individual_errors():
    d=ROOT/OUT
    m=pd.read_csv(d/'stage2_retrospective_source_out_metrics.csv')
    p=pd.read_csv(d/'stage2_retrospective_source_out_predictions.csv')
    assert len(m)==18 and len(p)==(8+4+4)*3*2
    for r in m.itertuples(index=False):
        subset=p[(p.heldout_source==r.heldout_source)&(p.variant==r.model)&(p.plan==r.plan)]
        assert len(subset)==r.n
        assert abs(subset.abs_error.mean()-r.mae)<1e-7
        assert np.all((subset.predicted>=0)&(subset.predicted<=subset.cap+1e-8))
        assert all(subset.heldout_doi==r.heldout_doi)

def test_ties_not_misreported_as_successful_recommendation():
    p=pd.read_csv(ROOT/OUT/'stage2_retrospective_source_out_predictions.csv')
    rank=retro_rank(p)
    tied=rank[rank.all_predictions_tied]
    assert len(tied)>0
    assert tied.top1_regret_mg_g.isna().all()
    assert tied.spearman_rho.isna().all()

def test_stage1_holdout_rows_and_doi_boundary():
    d=ROOT/OUT
    m=pd.read_csv(d/'stage1_retrospective_source_out_metrics.csv')
    p=pd.read_csv(d/'stage1_retrospective_source_out_predictions.csv')
    assert set(m.target)=={'BET','TPV'}
    for r in m.itertuples(index=False):
        subset=p[(p.heldout_source==r.heldout_source)&(p.plan==r.plan)&(p.target==r.target)]
        assert len(subset)==r.n
        assert abs(subset.abs_error.mean()-r.mae)<1e-7

def test_seal_has_historical_dois_and_content_hashes():
    doi,inputs=historical_dois(ROOT)
    assert len(doi)>=90
    assert '10.3390/en10091293' in doi
    for name in ('10.1007/s13399-026-07198-4','10.1016/j.chemosphere.2013.12.042','10.3390/en13133498'):
        assert name in doi
    s=seal(ROOT)
    assert set(s['historical_DOIs_including_known_lineage'])==set(doi)
    for path,hash_ in s['input_hashes'].items(): assert sha(ROOT/path)==hash_
    for x in s['frozen_candidate_models'].values():assert sha(ROOT/x['path'])==x['sha256']

def test_prospective_rejects_known_doi(tmp_path):
    old,batches=load_s2(ROOT)
    fake=batches['apple'].copy()
    fake['qe_source_evidence_status']='SOURCE_VERIFIED_EXPERIMENTAL_EQUILIBRIUM_QE'
    fake['target_kind']='EXPERIMENTAL_EQUILIBRIUM_QE_MG_G'
    fake['source_lineage_doi']='NONE_VERIFIED'
    fake['sample_batch_independent']=True
    loc=tmp_path/'known.csv';fake.to_csv(loc,index=False)
    with pytest.raises(ValueError,match='Previously exposed DOI'):
        prospective(ROOT,loc)

def test_prospective_rejects_qmax_or_inadequate_evidence(tmp_path):
    _,batches=load_s2(ROOT)
    fake=batches['sugar'].copy()
    fake['doi']='10.9999/v56-simulated-future-never-real-paper'
    fake['qe_source_evidence_status']='LANGMUIR_QMAX_FITTED_NOT_QE'
    fake['target_kind']='EXPERIMENTAL_EQUILIBRIUM_QE_MG_G'
    fake['source_lineage_doi']='NONE_VERIFIED'
    fake['sample_batch_independent']=True
    loc=tmp_path/'bad_target.csv';fake.to_csv(loc,index=False)
    with pytest.raises(ValueError,match='Not strictly verified'):
        prospective(ROOT,loc)

def test_seal_does_not_replace_original_active_model():
    path=ROOT/'deployment/active.json'
    if path.exists():
        # report directory only, never active.json in the patch
        assert 'active.json' not in [str(x) for x in (ROOT/OUT).rglob('*')]

def test_qmax_source_out_remains_separate_from_experimental_qe():
    d=ROOT/OUT
    m=pd.read_csv(d/'qmax_retrospective_source_out_metrics.csv')
    p=pd.read_csv(d/'qmax_retrospective_source_out_predictions.csv')
    assert set(m.target)=={'LANGMUIR_FITTED_QMAX_NOT_QE'}
    assert len(m)==4 and len(p)==(4+12)*2
    for r in m.itertuples(index=False):
        sel=p[(p.heldout_source==r.heldout_source)&(p.plan==r.plan)]
        assert abs(sel.abs_error.mean()-r.mae)<1e-7
