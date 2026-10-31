"""Training-domain screening for v40 research outputs, not calibrated uncertainty.
Missing or out-of-range values are warnings, not autonomous imputations or fake data.
"""
from __future__ import annotations
import numpy as np,pandas as pd

def training_envelope(frame,nums,cats, group_col='doi'):
    env={'n_rows':len(frame),'n_doi':int(frame[group_col].nunique()),'numerical':{},'categorical':{}}
    for c in nums:
        arr=pd.to_numeric(frame[c],errors='coerce').dropna().to_numpy(float)
        env['numerical'][c]={'reported':int(len(arr)),'min':float(arr.min()) if len(arr) else None,
             'max':float(arr.max()) if len(arr) else None}
    for c in cats:
        arr=frame[c].dropna().astype(str)
        env['categorical'][c]={'reported':int(len(arr)),'known':sorted(arr.unique().tolist())}
    return env

def check_input(row,env,essential=()):
    alerts=[]
    for col in essential:
        if pd.isna(row.get(col)):
            alerts.append(dict(level='missing',field=col,message='essential input not reported'))
    for col,info in env['numerical'].items():
        val=row.get(col)
        if pd.isna(val):continue
        try: val=float(val)
        except (TypeError,ValueError):
            alerts.append(dict(level='invalid',field=col,message='not numeric'));continue
        if info['min'] is not None and (val<info['min'] or val>info['max']):
            alerts.append(dict(level='marginal_out_of_range',field=col,message=f'{val:g} outside {info["min"]:g}–{info["max"]:g}'))
    for col,info in env['categorical'].items():
        val=row.get(col)
        if pd.isna(val):continue
        if str(val) not in info['known']:
            alerts.append(dict(level='unseen_category',field=col,message='unseen in training source categories'))
    return dict(status='REVIEW_REQUIRED' if alerts else 'MARGINAL_RANGE_ONLY_NOT_CERTIFIED',alerts=alerts,
      caveat='Passing each feature range does NOT prove joint feature support or prediction reliability.',
      training_doi=env['n_doi'])
