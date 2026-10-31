import numpy as np
import pandas as pd
import pytest
from scripts.audit_v41_paired_stage1 import paired_oof


def data():
    rows=[]
    for doi, name, actual, ridge, poisson in [
        ('d1','a',5,20,10),('d1','b',10,20,10),('d2','c',40,55,41)]:
        for model, pred in [('Ridge',ridge), ('Poisson_Process',poisson)]:
            rows.append(dict(stage='Stage1',target='BET_m2_g_training_candidate',
                             model=model,doi=doi,record_id=name,actual=actual,pred=pred))
    return pd.DataFrame(rows)


def test_paired_metric_and_cluster_ci():
    r=paired_oof(data(),'BET_m2_g_training_candidate','Poisson_Process',bootstrap=80)
    assert r['n']==3 and r['n_doi']==2
    assert r['MAE_challenger']<r['MAE_baseline']
    assert len(r['bootstrap_DOI_95CI'])==2


def test_reject_mismatched_ids_and_actuals():
    frame=data()
    bad=frame.copy()
    bad.loc[bad.model=='Poisson_Process','record_id']=['x','y','z']
    with pytest.raises(ValueError,match='서로 다른 DOI'):
        paired_oof(bad,'BET_m2_g_training_candidate','Poisson_Process',bootstrap=5)
    mismatch=frame.copy()
    mismatch.loc[mismatch.model=='Poisson_Process','actual'] += 1
    with pytest.raises(ValueError,match='실측값'):
        paired_oof(mismatch,'BET_m2_g_training_candidate','Poisson_Process',bootstrap=5)


def test_reject_duplicate_record_ids():
    frame=data()
    with pytest.raises(ValueError,match='중복'):
        paired_oof(pd.concat([frame,frame.iloc[[0]]]),'BET_m2_g_training_candidate','Poisson_Process',bootstrap=5)
