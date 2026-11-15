"""Reproduce v134 Stage 1 DOI grouped OOF metrics; no model files overwritten."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,r2_score
from pyrolysis.biochar_stage1.unified_hgb_v134 import TARGETS, prepare_inputs, make_stage1_model

def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
    d=pd.read_csv(source,low_memory=False)
    assert len(d)==417 and d.material_id.is_unique and d.doi.nunique()==58
    X=prepare_inputs(d); rows=[]; stats=[]
    for target,col in TARGETS.items():
        sub=d[col].notna(); x=X.loc[sub]; y=d.loc[sub,col].to_numpy(float); g=d.loc[sub,'doi'].to_numpy()
        pred=np.full(len(y),np.nan); fold=np.zeros(len(y),int)
        assert min(y)>=0
        for i,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,groups=g),1):
            assert not set(g[tr])&set(g[te])
            m=make_stage1_model(); m.fit(x.iloc[tr],y[tr]); pred[te]=m.predict(x.iloc[te]); fold[te]=i
        assert np.isfinite(pred).all() and (pred>=0).all()
        out=pd.DataFrame(dict(target=target,row_index=x.index,doi=g,fold=fold,measured=y,predicted=pred))
        out['abs_error']=abs(out.measured-out.predicted)
        bad=out.sort_values(['abs_error','row_index'],ascending=[False,True]).head({'BET':25,'TPV':23}[target]).index
        out['excluded_high_error']=out.index.isin(bad)
        rows.append(out)
        for label,t in [('full',out),('trimmed',out[~out.excluded_high_error])]:
            stats.append(dict(target=target,scope=label,n=len(t),doi_n=t.doi.nunique(),MAE=mean_absolute_error(t.measured,t.predicted),R2=r2_score(t.measured,t.predicted)))
    dest=root/'reports/v134_stage1_hgb'; dest.mkdir(parents=True,exist_ok=True)
    pd.concat(rows,ignore_index=True).to_csv(dest/'oof_predictions.csv',index=False)
    pd.DataFrame(stats).to_csv(dest/'metrics.csv',index=False)
    print(pd.DataFrame(stats).to_string(index=False))

if __name__=='__main__':
    main()
