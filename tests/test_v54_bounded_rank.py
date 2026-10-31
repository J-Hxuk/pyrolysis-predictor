"""Tests for v54 research-only bounded ML and manuscript-validated rank benchmark."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.run_v54_bounded_rank import (verify_and_load,cap_of,check_target,
    fit_model,infer,rank_metrics,chen_features,CHEN,ROOT_REPORT,ROOT_DEPLOY,digest)

ROOT=Path(__file__).resolve().parents[1]

def test_data_lineage_and_approval_are_kept_separate():
    base,apple8,apple16,apple24,matches,tea=verify_and_load(ROOT)
    assert (len(base),base.doi.nunique(),len(apple8),len(apple16),len(apple24))==(140,32,8,16,24)
    assert (len(matches),len(tea))==(8,5)
    assert not(set(base.doi)&set(apple24.doi))
    assert set(apple8.target_origin)=={'V52_ZHAO2020_TABLE3_QE_EXP'}

def test_bounded_predictor_is_physically_bounded_without_posthoc_clipping():
    base,*_=verify_and_load(ROOT)
    tr=base.iloc[:75];te=base.iloc[75:85]
    for method in ('logit_fraction','logistic_fraction'):
        model,cols,nums,cats=fit_model(tr,'A',method)
        pred,raw,frac=infer(model,cols,nums,cats,te,method)
        cap=cap_of(te)
        assert np.isfinite(pred).all()
        assert (pred>=0).all() and (pred<=cap+1e-8).all()
        assert (frac>=0).all() and (frac<=1).all()

def test_mass_balance_and_invalid_c0_dose_guard():
    d=pd.DataFrame({'C0_mg_L':[20.0],'dose_g_L':[2.0],'target_value':[11.0]})
    with pytest.raises(ValueError):check_target(d)
    d['target_value']=9.0
    d['dose_g_L']=0.0
    with pytest.raises(ValueError):cap_of(d)

def test_rank_ties_are_not_turned_into_artificial_top1():
    s=rank_metrics(['a','b','c'],[1.0,3.0,2.0],[0.5,0.5,0.5])
    assert s['all_prediction_tie']
    assert s['predicted_top1_id'] is None
    assert s['top1_regret'] is None

def test_chen_input_pH_is_not_invented():
    c=pd.read_csv(ROOT/CHEN)
    x=chen_features(c)
    assert len(x)==16
    assert (x.pollutant=='Cd(II)').all()
    assert x.initial_pH.isna().all()
    assert c.initial_pH.isna().all()
    scenario=chen_features(c,6.0)
    assert np.all(scenario.initial_pH==6.0)
    assert c.initial_pH.isna().all()

def test_v54_report_same_original_v53_direct_baseline_and_reload():
    # Scientific comparability is defined by the SAME OOF predictions and
    # report/model provenance, not a hard-coded historical MAE from sklearn 1.8.
    from scripts.audit_v54_local_consistency import verify_cross_report
    scores53, scores54, summary53, summary54 = verify_cross_report(ROOT)
    assert len(scores53) == 6
    assert len(scores54) == 18
    assert len(summary54['models']) == 8
    base,*_=verify_and_load(ROOT)
    for record in summary54['models']:
        path=ROOT/record['model_path']
        m=joblib.load(path)
        from scripts.build_v46_phase8 import s2_model
        _,cols,nums,cats=s2_model(record['variant'])
        pp,raw,f=infer(m,cols,nums,cats,base.head(2),record['method'])
        assert np.isfinite(pp).all() and (pp<=cap_of(base.head(2))).all()
