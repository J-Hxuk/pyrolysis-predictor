"""Experimental BET inference: source-faithful, restricted, independent of sklearn pickle."""
from pathlib import Path
import math
import numpy as np
import pytest
from pyrolysis.biochar_stage1.composition_inference_v20 import (
    load_portable_model, default_model_path, assess_inputs, estimate_bet,
    forest_predict_one, FEATURES,
)
from pyrolysis.biochar_stage1.comp_ablation_v18 import make_model
from pyrolysis.biochar_stage1.data_loader import load_bet_base

ROOT=Path(__file__).resolve().parents[1]


def inputs(row):
    process={"pyrolysis_temperature_C":row["temperature_C"],
             "pyrolysis_hold_min":row["time_min"],
             "heating_rate_C_min":row["heating_rate_C_min"]}
    comp={"raw_VM_wt_pct":row["VM_pct"],"raw_ash_wt_pct":row["Ash_pct"],"raw_FC_wt_pct":row["FC_pct"],
          "raw_C_wt_pct":row["C_pct"],"raw_H_wt_pct":row["H_pct"],
          "raw_N_wt_pct":row["N_pct"],"raw_O_wt_pct":row["O_pct"]}
    return process,comp


def check(model,process,comp):
    return assess_inputs(model,process,comp,elemental_basis="dry ash-free (daf)",proximate_basis="dry",
                         process_type="SINGLE_STAGE_PYROLYSIS",activated=False)


def test_full_and_proximate_and_no_imputation():
    m=load_portable_model(default_model_path(ROOT))
    r=load_bet_base(ROOT).iloc[0]
    p,c=inputs(r)
    all_features=check(m,p,c)
    assert all_features['predictable'] and all_features['variant']=='full'
    assert estimate_bet(m,all_features)>=0
    partial=c.copy()
    partial['raw_C_wt_pct']=None
    result=check(m,p,partial)
    assert not result['predictable'] and result['errors']
    no_elem={k:v for k,v in c.items() if k not in ("raw_C_wt_pct","raw_H_wt_pct","raw_N_wt_pct","raw_O_wt_pct")}
    reduced=check(m,p,no_elem)
    assert reduced['predictable'] and reduced['variant']=='proximate'
    assert estimate_bet(m,reduced)>=0


def test_scope_guard_and_basis_and_range():
    m=load_portable_model(default_model_path(ROOT))
    p,c=inputs(load_bet_base(ROOT).iloc[0])
    assert not assess_inputs(m,p,c,elemental_basis='dry ash-free (daf)',proximate_basis='dry',
                              process_type='OTHER',activated=False)['predictable']
    assert not assess_inputs(m,p,c,elemental_basis='dry ash-free (daf)',proximate_basis='dry',
                              process_type='SINGLE_STAGE_PYROLYSIS',activated=True)['predictable']
    assert not assess_inputs(m,p,c,elemental_basis='dry',proximate_basis='dry',
                              process_type='SINGLE_STAGE_PYROLYSIS',activated=False)['predictable']
    outside=dict(p,pyrolysis_temperature_C=10000)
    v=check(m,outside,c)
    assert not v['predictable'] and v['outside_training_range']
    with pytest.raises(ValueError): estimate_bet(m,v)


def test_tree_export_matches_sklearn_and_no_feedstock_feature():
    m=load_portable_model(default_model_path(ROOT))
    df=load_bet_base(ROOT)
    for name in FEATURES:
        cols=FEATURES[name]
        pipe=make_model('RandomForest')
        pipe.fit(df[cols],df['value'].to_numpy(float))
        wanted=pipe.predict(df[cols].head(20))
        got=np.array([forest_predict_one(m['models'][name],r.to_dict()) for _,r in df[cols].head(20).iterrows()])
        assert np.allclose(wanted,got,atol=1e-8,rtol=1e-8)
        assert not any('feedstock' in x.lower() for x in cols)


def test_no_outcome_leak_or_tpv():
    m=load_portable_model(default_model_path(ROOT))
    assert m['target']=='BET_m2_g'
    assert m['train_rows']==182 and m['train_doi_groups']==24
    assert all('value' not in x['features'] and 'qe' not in x['features'] for x in m['models'].values())
