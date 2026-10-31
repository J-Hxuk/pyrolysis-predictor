"""Method checks for matched DOI GroupCV ablation (not a model quality guarantee)."""
import numpy as np
import pandas as pd
import pytest
from pyrolysis.biochar_stage1.comp_ablation_v18 import (
    PROCESS, ELEMENTAL, PROXIMATE, FEATURE_SETS, validate_data,
    evaluate, doi_cluster_bootstrap, v60_composition_coverage, tpv_data_readiness,
)
from pyrolysis.biochar_stage1.data_loader import load_bet_base


def toy_data():
    rows=[]
    for k in range(5):
        for j in range(4):
            row={name:1.0+j+0.2*k for name in PROCESS+ELEMENTAL+PROXIMATE}
            row.update({'value': 5+j*2 +k, 'provenance_group':f'doi_{k}'})
            rows.append(row)
    return pd.DataFrame(rows)


def test_nested_feature_sets():
    assert set(FEATURE_SETS['process_only']) < set(FEATURE_SETS['process_all_composition'])
    assert len(FEATURE_SETS['process_all_composition'])==10


def test_matched_eval():
    df=toy_data()
    pred, metric=evaluate(df,algorithms=('Ridge',))
    assert len(pred)==len(df)*len(FEATURE_SETS)
    assert pred.groupby('feature_set').row_index.nunique().eq(len(df)).all()
    assert pred.groupby('feature_set').doi.nunique().eq(5).all()
    boot=doi_cluster_bootstrap(pred,repeats=100)
    assert len(boot)==3
    assert boot.ci95_lo.notna().all()


def test_data_validation_rejects_missing():
    df=toy_data();df.loc[0,'Ash_pct']=np.nan
    with pytest.raises(ValueError,match='fully observed'):
        validate_data(df)


def test_real_data_contract():
    import pathlib
    root=pathlib.Path(__file__).resolve().parents[1]
    frame=load_bet_base(root)
    info=validate_data(frame)
    assert info['rows']==182 and info['doi_groups']==24
    cov,meta=v60_composition_coverage(root)
    assert meta['rows']==256 and cov.shape[0]==9


def test_tpv_not_automatically_approved():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    d=tpv_data_readiness(root)
    assert d['status']=='TPV_COMPOSITION_ABLATION_NOT_CERTIFIED'
    assert d['complete_from_unverified_secondary_source']==d['complete_composition_and_temp_time']
