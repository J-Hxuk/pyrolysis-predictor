import numpy as np
import pandas as pd
from pyrolysis.biochar_stage1.unified_hgb_v134 import make_stage1_model,prepare_inputs,S1_FEATURES
from pyrolysis.biochar_stage2.v112_single_engine import normalize_activation_details

def test_stage1_factory_and_optional_inputs():
    x=prepare_inputs(pd.DataFrame([{'feedstock':'rice husk','pyrolysis_atmosphere':'N2','activation_method':'none','pyrolysis_temperature_C':400+i*10} for i in range(25)]))
    assert len(S1_FEATURES)==10 and x.activation_agent.eq('UNKNOWN').all()
    m=make_stage1_model().fit(x,np.arange(1,26))
    assert m.named_steps['model'].loss=='poisson'
    assert np.isfinite(m.predict(x)).all()
    assert (m.predict(x)>=0).all()

def test_activation_details_not_invented():
    a=normalize_activation_details('none',{'activation_agent':'KOH','activation_time_min':5})
    assert a['activation_agent']=='none' and np.isnan(a['activation_time_min'])
    b=normalize_activation_details('chemical_impregnation',{'activation_agent':'KOH','activation_time_min':'60'})
    assert b['activation_agent']=='KOH' and b['activation_time_min']==60
    assert np.isnan(b['activation_temperature_C'])


def test_air_warning_reflects_stage1_source():
    from pathlib import Path
    from pyrolysis.biochar_stage2.v112_single_engine import (
        _stage1_reference, assess_stage1_input, resolve_stage1_feedstock,
    )

    source = Path(__file__).resolve().parents[1] / (
        'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
    )
    data = pd.read_csv(source, low_memory=False)
    # Neither BET nor TPV training observations use a literal air category.
    assert not data['pyrolysis_atmosphere'].fillna('').eq('air').any()
    data[S1_FEATURES] = prepare_inputs(data)
    reference = _stage1_reference(
        data, 'BET_m2_g_training_candidate', 'TPV_cm3_g_training_candidate',
    )
    models = {'s1_reference': reference}
    process = {
        'pyrolysis_atmosphere': 'air',
        'activation_method': 'none',
        'activation_agent': 'none',
        'pyrolysis_temperature_C': 500,
        'pyrolysis_hold_min': 120,
        'heating_rate_C_min': 10,
    }
    verdict = assess_stage1_input(models, feedstock='rice husk', process=process)
    assert any('air' in w and 'BET·TPV 학습 원장' in w for w in verdict['warnings'])
    assert any('OneHotEncoder' in w for w in verdict['warnings'])
    import pytest
    with pytest.raises(ValueError, match='원료명'):
        resolve_stage1_feedstock(models, '   ')
