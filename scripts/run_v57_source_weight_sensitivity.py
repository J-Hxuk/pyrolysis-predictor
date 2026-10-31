"""DOI-weighted research sensitivity for v57: all new records retained, lower dominance of 29 Shen observations."""
from pathlib import Path
import argparse,json,warnings,hashlib,platform
import numpy as np,pandas as pd,joblib,sklearn,xgboost
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold
from scripts.build_v46_phase8 import s2_model,features
from scripts.run_v57_handoff_cumulative_refit import BASE140,V55_152,NEW_DOIS,DATA,DEP,REPORT,sha

def fit(train,test,variant,w):
 model,cols,nums,cats=s2_model(variant)
 weights=np.where(train.doi.to_numpy()==NEW_DOIS['shen'],float(w),1.)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',FutureWarning)
  model.fit(features(train,cols,nums,cats),train.target_value.to_numpy(float),model__sample_weight=weights)
 raw=model.predict(features(test,cols,nums,cats))
 pred=np.clip(raw,0,test.C0_mg_L.to_numpy(float)/test.dose_g_L.to_numpy(float))
 return pred,model

def main(rootarg):
 root=Path(rootarg).resolve()
 base=pd.read_csv(root/BASE140,low_memory=False).reset_index(drop=True)
 ref=pd.read_csv(root/V55_152,low_memory=False)
 add=pd.read_csv(root/DATA/'Stage2_New36_Harmonized_RESEARCH.csv',low_memory=False)
 extra=pd.concat([ref.iloc[140:],add],ignore_index=True,sort=False)
 assert len(base)==140 and len(extra)==48 and len(add)==36
 out=base[['observation_id','doi','target_value']].copy();scores=[]
 factors=(1.0,0.50,0.25,0.10)
 for tr,te in GroupKFold(5).split(base,groups=base.doi):
  train=pd.concat([base.iloc[tr],extra],ignore_index=True,sort=False)
  assert set(base.iloc[te].doi).isdisjoint(set(train.doi))
  for factor in factors:
   for variant in ('A','B'):
    pred,_=fit(train,base.iloc[te],variant,factor)
    col=f'{variant}__Shen_weight_{factor:.2f}'
    if col not in out:out[col]=np.nan
    out.loc[te,col]=pred
 for factor in factors:
  for v in ('A','B'):
   col=f'{v}__Shen_weight_{factor:.2f}'
   yy=base.target_value.to_numpy(float);pp=out[col].to_numpy(float)
   scores.append({'model':v,'shen_training_weight':factor,'training_total_rows':188,
    'shen_rows_retained':29,'other_new_cao_rows_retained':7,'evaluation':'FIXED_OLD140_DOI5_POSTHOC',
    'MAE_mg_g':float(mean_absolute_error(yy,pp)),
    'RMSE_mg_g':float(np.sqrt(mean_squared_error(yy,pp)))})
 report=root/REPORT;report.mkdir(parents=True,exist_ok=True)
 out.to_csv(report/'stage2_188_doi_weight_sensitivity_predictions.csv',index=False)
 pd.DataFrame(scores).to_csv(report/'stage2_188_doi_weight_sensitivity_metrics.csv',index=False)
 # Store the least-concentrated weight and 1/4 DOI weight models; do not automatically activate even if posthoc looks superior
 dep=root/DEP;dep.mkdir(parents=True,exist_ok=True);artifact=[]
 full=pd.concat([base,extra],ignore_index=True,sort=False)
 for factor in [0.25,0.10]:
  for v in ['A','B']:
   _,model=fit(full,full.head(2),v,factor)
   dest=dep/f'{v}_v57_188_shen_weight_{factor:.2f}_RESEARCH_NOT_ACTIVE.joblib'
   joblib.dump(model,dest,compress=3)
   artifact.append({'model':v,'shen_weight':factor,'path':str(dest.relative_to(root)),'sha256':sha(dest)})
 (report/'source_weight_sensitivity_manifest.json').write_text(json.dumps({'weights':list(factors), 'note':'Post hoc model selection on old140 only. All 36 raw source rows retained; Shen per-row downweight is not removal. Models NOT ACTIVE.','models':artifact,'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},ensure_ascii=False,indent=2),encoding='utf-8')
 print(pd.DataFrame(scores).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();main(a.root)
