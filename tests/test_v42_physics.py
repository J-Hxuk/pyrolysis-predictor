"""Self contained v42 tests; no v15/v41 large source archive required."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pyrolysis.biochar_stage2.v42_physics import physical_cap, clip_raw_predictions, BoundedUptakeRegressor
from scripts.evaluate_v42_physics import predict_oof, doi_bootstrap


def toy():
    xs=[]
    for j in range(5):
        for i in range(4):
            c0=25.+j*2+i*5;dose=1+i*.25
            xs.append(dict(observation_id=f'{j}-{i}',doi=f'10.9999/example{j}',target_name='qe_reported_mg_g',
              target_value=(c0/dose)*(.2+.1*i),C0_mg_L=c0,dose_g_L=dose,
              pollutant='Cd(II)',feedstock_label_normalized='rice husk',manufacturing_route='single_stage',
              pyrolysis_atmosphere='inert_N2',activation_or_modification='none',
              initial_pH=5.,adsorption_T_C=25.,contact_time_min=120,
              pyrolysis_temperature_C=300+i*100,pyrolysis_hold_min=60.,heating_rate_C_min=10.,
              model_use_tier='CORE_EXPLORATORY',target_origin='toy',))
    return pd.DataFrame(xs)


def test_cap_formula_and_units():
    x=pd.DataFrame({'C0_mg_L':[100,20],'dose_g_L':[2,4]})
    assert np.allclose(physical_cap(x),[50,5])

@pytest.mark.parametrize('dose',[-1,0,np.nan,np.inf])
def test_cap_rejects_invalid_dose(dose):
    with pytest.raises(ValueError):physical_cap(pd.DataFrame({'C0_mg_L':[10.], 'dose_g_L':[dose]}))

def test_clip_is_explicit_postprocessing():
    x=pd.DataFrame({'C0_mg_L':[10,10],'dose_g_L':[2,2]})
    assert np.allclose(clip_raw_predictions([-8,50],x),[0,5])

@pytest.mark.parametrize('kind',['logit_ridge','fraction_forest'])
@pytest.mark.parametrize('model',['A','B'])
def test_fraction_model_bounds_and_serialization(kind,model,tmp_path):
    import joblib
    d=toy();clf=BoundedUptakeRegressor(model,kind).fit(d,d.target_value)
    test=d.copy();test.loc[0,'C0_mg_L']=10000
    pp=clf.predict(test);cap=physical_cap(test)
    assert np.isfinite(pp).all() and (pp>=0).all() and (pp<=cap+1e-9).all()
    path=tmp_path/'clf.joblib';joblib.dump(clf,path)
    assert np.allclose(pp,joblib.load(path).predict(test))

def test_physics_training_rejects_bad_target():
    d=toy();d.loc[0,'target_value']=d.loc[0,'C0_mg_L']/d.loc[0,'dose_g_L']*1.1
    with pytest.raises(ValueError):BoundedUptakeRegressor().fit(d,d.target_value)

def test_oof_fold_grouping_and_metrics():
    d=toy();dd, preds,metrics, perdoi=predict_oof(d)
    assert len(preds)==len(d)*2*6
    assert dd.doi.nunique()==5
    assert preds.groupby('doi')['fold'].nunique().max()==1
    assert metrics.query("method in ['Logit_Ridge_fraction','RF_fraction']").raw_mass_balance_violations.eq(0).all()
    assert perdoi.doi.nunique()==5
    ci=doi_bootstrap(preds,'A','RF_fraction',n_boot=50)
    assert len(ci['bootstrap_CI95_mg_g'])==2
