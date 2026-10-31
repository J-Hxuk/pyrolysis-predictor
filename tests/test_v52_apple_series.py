import numpy as np
from scripts.build_v52_apple_series import DOI,candidate_rows,stage1_rows,load_and_build
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_24_distinct_source_anchored_uptake_rows():
    data=candidate_rows()
    assert len(data)==24
    assert set(data.pollutant)=={'Cu(II)','Zn(II)'}
    assert int((data.source_table_qe_type=='qe_exp').sum())==8
    assert int((data.source_table_qe_type=='reported_1440min_qt_equilibrium_per_paper').sum())==16
    assert (data.target_value<=data.C0_mg_L/data.dose_g_L).all()
    assert data.observation_id.is_unique
    assert data.doi.nunique()==1
    assert not data[['pollutant','pyrolysis_temperature_C','dose_g_L']].duplicated().any()

def test_two_four_temperature_panels():
    data=candidate_rows()
    for (metal,dose),subset in data.groupby(['pollutant','dose_g_L']):
        assert sorted(subset.pyrolysis_temperature_C.tolist())==[300,400,500,600]
        assert float(subset.loc[subset.target_value.idxmax(),'pyrolysis_temperature_C'])==500
        assert subset.contact_time_min.eq(1440).all()

def test_raw_composition_not_fabricated():
    data=candidate_rows()
    for col in ['raw_C_wt_pct','raw_H_wt_pct','raw_O_wt_pct']:
        assert col not in data.columns or data[col].isna().all()
    assert data.pyrolysis_hold_min.isna().all()

def test_stage1_four_bet_no_tpv():
    data=stage1_rows()
    assert len(data)==4
    assert data.BET_m2_g_original_source.tolist()==[2.39,7,37.25,108.59]
    assert 'TPV_cm3_g_training_candidate' not in data.columns

def test_integrated_counts_and_original_ids():
    o1,o2,e1,e2,a1,a2=load_and_build(ROOT)
    assert (len(o1),len(o2),len(a1),len(a2))==(417,140,421,164)
    assert a2.doi.nunique()==33 and a1.doi.nunique()==59
    assert set(o2.observation_id).issubset(set(a2.observation_id))
    assert set(a2.loc[a2.doi==DOI,'target_qualification'])=={'TABLE3_EXPERIMENTAL_EQUILIBRIUM_QE_EXP','REPORTED_1440MIN_UPTAKE_SOURCE_TEXT'}
