"""Focused research contract checks (no edits to deployment or raw dataset)."""
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.evaluate_v50_model_utility import DATA, model, S2_A, evaluate_global, evaluate_properties, physical_clip
from scripts.train_v50_bounded_candidate import predict_for_cu

ROOT=Path(__file__).resolve().parents[1]

def load():
    d=pd.read_csv(ROOT/DATA,low_memory=False)
    assert len(d)==140 and d.doi.nunique()==32
    return d

def test_source_dataset_is_unchanged_and_unique():
    d=load()
    assert d.observation_id.is_unique
    assert (d.target_value>=0).all()
    assert (d.target_value<=d.C0_mg_L/d.dose_g_L+1e-6).all()

def test_A_features_only_adsorption_variables():
    assert set(S2_A)=={'pollutant','C0_mg_L','initial_pH','dose_g_L','adsorption_T_C','contact_time_min'}
    assert 'doi' not in model(S2_A).named_steps['pre'].get_params()['transformers'][0][2]

def test_clip_is_physical_without_modifying_observations():
    d=load().iloc[:5]
    p=np.array([-1,1e6,5.1,2,3])
    clipped=physical_clip(p,d)
    assert np.isfinite(clipped).all()
    assert ((clipped>=0)&(clipped<=d.C0_mg_L.to_numpy()/d.dose_g_L.to_numpy())).all()
    assert np.array_equal(p,np.array([-1,1e6,5.1,2,3]))

def test_bounded_candidate_rejects_wrong_pollutant():
    d=load()
    rows=d.loc[d.pollutant=='Pb(II)'].head(1)
    from pytest import raises
    with raises(ValueError,match=r'Cu\(II\)'):
        predict_for_cu({},rows)

def test_v50_existing_results_have_common_folds_and_no_silent_promotion():
    d=load();p=ROOT/'reports/v50_model_utility/doi_oof_predictions.csv'
    assert p.exists(), 'Run: python -m scripts.evaluate_v50_model_utility --root .'
    o=pd.read_csv(p)
    assert len(o)==len(d) and o.observation_id.is_unique
    assert set(o.observation_id)==set(d.observation_id)
    assert np.isfinite(o[['A_raw','B_raw','A_bounded_fraction','B_bounded_fraction']].to_numpy(float)).all()
    assert (o.A_bounded_fraction>=0).all() and (o.A_bounded_fraction<=o.physical_cap).all()
