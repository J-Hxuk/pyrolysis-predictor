"""v23 DOI-paired fixed benchmark checks; no model deployment changes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import compare_phase8_v23 as m
from pyrolysis.biochar_stage2 import v15_features as fc


def test_input_frame_cannot_leak_target_or_doi():
    d=pd.DataFrame({'doi':['a','b'], 'target_value':[1,2],
                    'pyrolysis_temperature_C':[400,500], 'feedstock_group':['A','B']})
    x=m.input_frame(d,m.S1_NUMERIC,m.S1_CATEGORICAL)
    assert 'doi' not in x and 'target_value' not in x
    assert x.shape[0] == 2
    assert x.pyrolysis_hold_min.isna().all()


def test_incremental_rows_and_ids():
    old,new=m.stage1_targets()
    assert (len(old),len(new))==(256,264)
    assert len(set(new.material_id)-set(old.material_id))==8
    assert len(set(new.doi)-set(old.doi))==2
    a,b=m.stage2_targets()
    assert (len(a),len(b))==(97,98)
    assert len(set(b.observation_id)-set(a.observation_id))==1
    assert set(b[fc.TARGET_COLUMN].astype(float).dropna().index)==set(range(98))


def test_cluster_bootstrap_paired_and_direction():
    frame=pd.DataFrame({'doi':['paper1','paper1','paper2'],
                        'err_original':[1.,1.,1.], 'err_expanded':[2.,2.,3.]})
    res=m.cluster_delta_ci(frame,seed=4,reps=150)
    assert res['doi_count']==2
    assert res['delta_mae']==pytest.approx(4/3)
    assert res['ci95'][0]>0


def test_single_row_r2_not_claimed():
    one=pd.DataFrame({'actual':[100.],'pred':[40.]})
    assert m.observed_metrics(one)['r2'] is None
    assert m.observed_metrics(one)['mae']==60


def test_train_and_test_doi_disjoint_for_simple_fit():
    old,_=m.stage2_targets()
    doi=old.iloc[0].doi
    test=old.loc[old.doi.eq(doi)]
    train=old.loc[~old.doi.eq(doi)]
    assert set(train.doi).isdisjoint(set(test.doi))
    assert set(fc.model_inputs('A')).isdisjoint(fc.FORBIDDEN_FIELDS)
    assert set(fc.model_inputs('B')).isdisjoint(fc.FORBIDDEN_FIELDS)
