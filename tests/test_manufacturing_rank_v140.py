from pathlib import Path
import numpy as np
import pandas as pd
from pyrolysis.biochar_stage2.manufacturing_rank_v140 import load_ranker,fit_ranker,features,rank_sweep

ROOT=Path(__file__).resolve().parents[1]


def test_archived_doi_folds_and_fig7():
    bundle=load_ranker(ROOT); d=bundle['data']; pred=np.full(len(d),np.nan)
    for f in sorted(d.fold.unique()):
        mask=d.fold.eq(f)
        assert not set(d.loc[mask,'doi']) & set(d.loc[~mask,'doi'])
        pred[mask]=fit_ranker(d.loc[~mask]).predict(features(d.loc[mask]))
    assert np.isfinite(pred).all()
    hits=[];regrets=[]
    for _,g in d.assign(score=pred).loc[lambda x:x.pollutant.ne('Zn(II)')].groupby('panel_id'):
        y=g.target_value.to_numpy(); s=g.score.to_numpy()
        top=np.isclose(s,s.max(),atol=1e-9,rtol=0)
        hits.append(int(top.sum()==1 and np.isclose(y[top][0],y.max(),atol=1e-9,rtol=0)))
        regrets.append(y.max()-y[top].min())
    assert len(hits)==12 and sum(hits)==5
    np.testing.assert_allclose(np.mean(regrets),2.7808333333333306,atol=1e-9)


def test_rank_sweep_gating_and_ties():
    b=load_ranker(ROOT)
    process={'activation_method':'none','pyrolysis_atmosphere':'oxygen_limited','pyrolysis_hold_min':120,'heating_rate_C_min':10}
    rank,notes=rank_sweep(b,'maize straw',process,'Cd(II)',[200,300,400,500,600,1000])
    assert len(rank)==4 and rank['RF 순위'].min()==1 and notes
    assert rank_sweep(b,'maize straw',{**process,'activation_method':'physical_activation'},'Cd(II)',[300,400,500])[0] is None
    assert rank_sweep(b,'maize straw',process,'Phenol',[300,400,500])[0] is None
    assert rank_sweep(b,'maize straw',process,'Cd(II)',[300,400])[0] is None
