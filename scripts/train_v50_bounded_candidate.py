"""Train the *global* v50 bounded-uptake A model locally. Research-only.

Candidate was evaluated on Cu(II) as a subgroup, but was trained on all 140 rows
inside each cross-validation fold. Preserve that distinction on full-data fit.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scripts.evaluate_v50_model_utility import DATA, S2_A, digest, features, model

OUT='deployment/research_v50_NOT_ACTIVE'

def train(root:Path):
    root=Path(root)
    df=pd.read_csv(root/DATA,low_memory=False)
    if len(df)!=140 or df.doi.nunique()!=32 or not df.observation_id.is_unique:
        raise ValueError('Unexpected research dataset; refusing to train')
    cap=df.C0_mg_L.to_numpy(float)/df.dose_g_L.to_numpy(float)
    y=df.target_value.to_numpy(float)
    if not (np.isfinite(cap).all() and np.isfinite(y).all() and (cap>0).all() and (y>=0).all() and (y<=cap+1e-6).all()):
        raise ValueError('Invalid training targets or mass-balance limits')
    f=np.clip(y/cap,1e-4,1-1e-4)
    x=features(df,S2_A,[v for v in S2_A if v!='pollutant'],['pollutant'])
    m=model(S2_A)
    m.fit(x,logit(f))
    trial=cap[:10]*expit(np.asarray(m.predict(x.iloc[:10]),float))
    if not (np.isfinite(trial).all() and (trial>0).all() and (trial<cap[:10]).all()):
        raise ValueError('Invalid predicted uptake; no model saved')
    out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    path=out/'v50_globalA_boundedFraction_CU_SCOPE_RESEARCH_NOT_ACTIVE.joblib'
    bundle={
        'pipeline':m,'input_columns':S2_A,'target_transform':'logit(qe/(C0/dose))',
        'training_records':len(df),'training_dois':int(df.doi.nunique()),
        'source_sha256':digest(root/DATA),'evaluation_scope':'Cu(II) exploratory only',
        'deployment_approved':False,'warning':'Cross-validation CI includes zero; NOT an improved production model.'
    }
    temp=path.with_suffix('.joblib.tmp')
    try:
        joblib.dump(bundle,temp,compress=3)
        loaded=joblib.load(temp)
        pred=expit(loaded['pipeline'].predict(x.iloc[:10]))*cap[:10]
        if not np.allclose(pred,trial,rtol=1e-7):raise ValueError('model serialization mismatch')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    manifest={k:v for k,v in bundle.items() if k!='pipeline'}
    manifest['model_file']=path.name
    (out/'MODEL_INFO.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def predict_for_cu(bundle, rows:pd.DataFrame):
    if not set(rows.pollutant.dropna().unique()).issubset({'Cu(II)'}) or rows.pollutant.isna().any():
        raise ValueError('Experimental candidate is limited to Cu(II)')
    cap=np.asarray(rows.C0_mg_L,float)/np.asarray(rows.dose_g_L,float)
    if not np.isfinite(cap).all() or (cap<=0).any():raise ValueError('Invalid C0/dose')
    cols=bundle['input_columns']
    x=features(rows,cols,[v for v in cols if v!='pollutant'],['pollutant'])
    return cap*expit(bundle['pipeline'].predict(x))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.',type=Path)
    print(json.dumps(train(p.parse_args().root),ensure_ascii=False,indent=2))
