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
    z=pd.read_csv(ROOT/REPORTREL/'v53_fixed140_metrics.csv')
    a=z.query('model=="A" and scenario=="base140"').iloc[0]
    b=z.query('model=="A" and scenario=="plus24"').iloc[0]
    assert abs(a.mae_mg_g-17.246810957570563)<1e-5
    assert abs(b.mae_mg_g-16.760127406300303)<1e-5
    assert z.physical_violations_after_clip.sum()==0

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
