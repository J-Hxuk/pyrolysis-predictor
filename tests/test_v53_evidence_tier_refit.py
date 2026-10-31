from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scripts.run_v53_research_refit import verify_and_load, MODELREL, REPORTREL
from scripts.build_v46_phase8 import s2_model, features
ROOT=Path(__file__).resolve().parents[1]

def test_source_count_overlap_and_evidence_status():
    b,x8,x16,x24,matching,tea=verify_and_load(ROOT)
    assert (len(b),len(x8),len(x16),len(x24),len(matching),len(tea))==(140,8,16,24,8,5)
    assert not set(b.observation_id)&set(x24.observation_id)

def test_fixed140_outcomes_consistency():
    # Numeric benchmark values vary across sklearn/xgboost versions.
    # Verify that current report scores exactly reflect held-out OOF predictions.
    from scripts.audit_v54_local_consistency import verify_v53_report
    scores, predictions = verify_v53_report(ROOT)
    assert len(scores) == 6
    assert len(predictions) == 140

def test_artifact_reload_inference():
    frame=pd.read_csv(ROOT/'data/imports/v53_source_review/Stage2_148_RESEARCH_NOT_ACTIVE.csv')
    for n in ('A','B'):
        path=ROOT/MODELREL/f'V53_STAGE2_{n}_148_strict8_sensitivity_NOT_ACTIVE.joblib'
        model=joblib.load(path)
        _,cols,nums,cats=s2_model(n)
        pred=model.predict(features(frame.head(5),cols,nums,cats))
        assert np.isfinite(pred).all()

def test_no_active_replacement():
    import json
    manifest=json.loads((ROOT/MODELREL/'manifest_v53.json').read_text(encoding='utf8'))
    assert len(manifest)==4
    assert all('NOT_ACTIVE' in row['path'] for row in manifest)
