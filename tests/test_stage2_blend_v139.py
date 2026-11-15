from pathlib import Path
import numpy as np
import pandas as pd
from pyrolysis.biochar_stage2.stage2_blend_v139 import load_blend, FEATURES, SNAPSHOT, MODEL_VERSION

def test_blend_full_fit_prediction_bounds():
    root=Path(__file__).resolve().parents[1]
    d=pd.read_csv(root/SNAPSHOT,low_memory=False)
    b=load_blend(root)
    assert b['selected_candidate']==MODEL_VERSION and len(b['supported_labels'])==26
    assert b['target_transform']=='fraction'
    p=b['pipeline'].predict(d[FEATURES])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    m=b['pipeline']
    np.testing.assert_allclose(p,.5*np.clip(m.knn.predict(d[FEATURES]),0,1)+.5*np.clip(m.et.predict(d[FEATURES]),0,1))
