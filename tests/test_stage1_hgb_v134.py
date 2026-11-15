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
