from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_v46_phase8 import load_inputs,reconcile,S1_NUM,S1_CAT,S2_A,S2_B,S2_CATS,features


def test_phase8_source_reconciliation_counts():
    s1,s2,qa,stage=load_inputs(ROOT)
    x,y,extra_y,extra_x=reconcile(s1,s2,qa,stage)
    assert (len(x),len(y),len(extra_x),len(extra_y))==(417,140,13,2)
    assert y.observation_id.is_unique and x.material_id.is_unique
    assert len(y[y.phase8_source_verified_v2])==45
    assert y.doi.nunique()==s2.doi.nunique()==32


def test_verified_observations_not_repeated():
    s1,s2,qa,stage=load_inputs(ROOT)
    _,y,ex,_=reconcile(s1,s2,qa,stage)
    assert len(set(s2.observation_id)&set(qa.adsorption_id))==43
    assert set(ex.observation_id)=={'UPG_NI_MSBC_KINETIC_18H','UPG2_OAK_CU_PH3'}
    assert len(y)-len(s2)==2


def test_source_measured_values_unchanged():
    s1,s2,qa,stage=load_inputs(ROOT)
    x,y,ex,ex1=reconcile(s1,s2,qa,stage)
    assert np.allclose(y.iloc[:len(s2)].target_value.to_numpy(float),s2.target_value.to_numpy(float))
    assert np.allclose(x.iloc[:len(s1)].BET_m2_g_training_candidate.fillna(-999),s1.BET_m2_g_training_candidate.fillna(-999))
    assert np.isclose(ex[ex.observation_id=='UPG2_OAK_CU_PH3'].target_value.iloc[0],17.11)
    assert np.isclose(ex[ex.observation_id=='UPG_NI_MSBC_KINETIC_18H'].target_value.iloc[0],24.55)


def test_mass_balance_and_missing_pH():
    s1,s2,qa,stage=load_inputs(ROOT)
    _,y,ex,_=reconcile(s1,s2,qa,stage)
    assert (y.target_value.astype(float)<=y.C0_mg_L.astype(float)/y.dose_g_L.astype(float)+1e-6).all()
    assert pd.isna(ex[ex.observation_id=='UPG2_OAK_CU_PH3'].contact_time_min.iloc[0])
    assert ex[ex.observation_id=='UPG2_OAK_CU_PH3'].initial_pH.iloc[0]==3


def test_no_misuse_of_activation_ratio_basis():
    s1,s2,qa,stage=load_inputs(ROOT)
    x,_,_,extra=reconcile(s1,s2,qa,stage)
    assert extra.activation_agent_feedstock_mass_ratio.isna().all()
    assert 'source_K2CO3_to_char_mass_ratio' in extra
    assert not set(extra.doi)&set(s1.doi)
    assert (extra.release_status=='RESEARCH_SENSITIVITY_ONLY_NOT_AUTO_APPROVED').all()


def test_fold_group_isolation():
    from sklearn.model_selection import GroupKFold
    s1,s2,qa,stage=load_inputs(ROOT)
    _,y,ex,_=reconcile(s1,s2,qa,stage)
    for train,test in GroupKFold(5).split(s2,groups=s2.doi):
        held=set(s2.iloc[test].doi)
        added=ex[~ex.doi.isin(held)]
        assert not set(added.doi)&held
        assert not set(s2.iloc[train].doi)&held


def test_saved_research_models_predict_finite():
    s1,s2,qa,stage=load_inputs(ROOT)
    x,y,_,_=reconcile(s1,s2,qa,stage)
    for st,target,kind,df in [('Stage1','BET_m2_g_training_candidate','Poisson',x),
                              ('Stage1','TPV_cm3_g_training_candidate','Poisson',x),
                              ('Stage2','target_value','A',y),('Stage2','target_value','B',y)]:
        path=ROOT/'deployment/research_v46_NOT_ACTIVE'/f'V46_{st}_{target}_{kind}_RESEARCH_NOT_ACTIVE.joblib'
        model=joblib.load(path)
        if st=='Stage1': cols=S1_NUM+S1_CAT;nums=S1_NUM;cats=S1_CAT
        else:
            cols=S2_A if kind=='A' else S2_B
            cats=[c for c in cols if c in S2_CATS];nums=[c for c in cols if c not in cats]
        p=model.predict(features(df.iloc[:2],cols,nums,cats))
        assert np.isfinite(p).all()
        if st=='Stage1':assert (p>=0).all()
