"""v57: full source preservation, cumulative 36-row experimental uptake study.
Research-only patch for v47: never changes active models or earlier input tables.
Execute: python -m scripts.run_v57_handoff_cumulative_refit --root .
"""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn, xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scripts.build_v46_phase8 import s2_model, s1_model, features

NEW_DOIS={'shen':'10.1007/s11356-017-8847-2','cao':'10.1016/j.scitotenv.2019.07.102'}
NEW_CSV='data/imports/v57_handoff_sources/Kiro_approved_experimental_qe_rows_SOURCE_ORIGINAL.csv'
BASE140='data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
V55_152='data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv'
V55_156='data/imports/v55_cumulative_sources/Stage2_156_v55_156_plus_cocoa_conditional_RESEARCH_NOT_ACTIVE.csv'
V55_168='data/imports/v55_cumulative_sources/Stage2_168_v55_168_all24_plus_sugar_sensitivity_RESEARCH_NOT_ACTIVE.csv'
V55_172='data/imports/v55_cumulative_sources/Stage2_172_v55_172_all24_plus_cocoa_sensitivity_RESEARCH_NOT_ACTIVE.csv'
S1_446='data/imports/v55_cumulative_sources/Stage1_446_RESEARCH_NOT_ACTIVE.csv'
REPORT='reports/v57_handoff_cumulative_refit'
DATA='data/imports/v57_handoff_sources'
DEP='deployment/research_v57_NOT_ACTIVE'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sanitize_json(obj):
 if isinstance(obj,dict):return {str(k):sanitize_json(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [sanitize_json(v) for v in obj]
 if isinstance(obj,(float,np.floating)):
  return float(obj) if np.isfinite(obj) else None
 if isinstance(obj,(int,np.integer)):return int(obj)
 if isinstance(obj,np.bool_):return bool(obj)
 return obj

def score(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 return {'n':int(len(y)),'mae_mg_g':float(mean_absolute_error(y,p)),
 'rmse_mg_g':float(np.sqrt(mean_squared_error(y,p))),
 'r2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None}
def pipefit(train,test,variant):
 pipe,cols,nums,cats=s2_model(variant)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',FutureWarning)
  pipe.fit(features(train,cols,nums,cats),train.target_value.to_numpy(float))
  raw=np.asarray(pipe.predict(features(test,cols,nums,cats)),float)
 cap=test.C0_mg_L.to_numpy(float)/test.dose_g_L.to_numpy(float)
 pred=np.clip(raw,0,cap)
 return pred,raw,pipe

def ingest(root):
 ddir=root/DATA; ddir.mkdir(parents=True,exist_ok=True)
 orig=ddir/'Kiro_approved_experimental_qe_rows_SOURCE_ORIGINAL.csv'
 raw=pd.read_csv(orig,low_memory=False)
 assert len(raw)==36 and raw.source_doi.nunique()==2, 'Unexpected incoming batch; stop before training'
 assert set(raw.source_doi)==set(NEW_DOIS.values()),'Source DOI not whitelisted'
 assert (raw.qe_type=='experimental').all()
 assert all(raw.source_value_type.str.contains('experimental marker|direct reported',case=False))
 assert (raw.C0_mg_L>0).all() and (raw.dose_g_L>0).all() and (raw.qe_mg_g>=0).all()
 assert (raw.qe_mg_g<=raw.C0_mg_L/raw.dose_g_L+1e-6).all(), 'Physical bound violation'
 assert raw[['source_doi','material_id','pollutant','C0_mg_L','dose_g_L','qe_mg_g']].notna().all().all()
 assert not raw.duplicated(subset=['source_doi','material_id','pollutant','C0_mg_L','dose_g_L']).any()
 base=pd.read_csv(root/BASE140,low_memory=False)
 # Keep entire row lineage unmodified; no made-up TPV/hold_time/atmosphere.
 rows=[]
 for i,r in enumerate(raw.itertuples(index=False)):
  row={k:np.nan for k in base.columns}
  doi=r.source_doi
  feed={'Wheat straw pellets':'wheat straw pellets','Rice husk':'rice husk','Wheat straw':'wheat straw'}.get(r.feedstock,r.feedstock.lower())
  doi_name='Shen2017' if doi==NEW_DOIS['shen'] else 'Cao2019'
  row.update(observation_id=f'V57_{doi_name}_{r.material_id}_{r.pollutant.split("(")[0]}_{i:03}',
    doi=doi,doi_group=doi,material_uid=f'{doi}::{r.material_id}',sample_group=f'{doi}::{r.material_id}',
    experiment_series_id=f'{doi}::{r.material_id}::{r.pollutant}',
    pollutant=r.pollutant,feedstock_label_normalized=feed,feedstock_raw=r.feedstock,
    manufacturing_route='SINGLE_STAGE_PYROLYSIS',
    process_step_count=1,pyrolysis_temperature_C=r.pyrolysis_temp_C,
    pyrolysis_hold_min=r.hold_time_min,heating_rate_C_min=r.heating_rate_C_min,
    pyrolysis_atmosphere=r.atmosphere,
    C0_mg_L=r.C0_mg_L,dose_g_L=r.dose_g_L,
    initial_pH=r.initial_pH,adsorption_T_C=r.adsorption_temp_C,contact_time_min=r.contact_time_min,
    BET_measured_m2_g=r.BET_m2_g,TPV_measured_cm3_g=r.TPV_cm3_g,
    target_value=r.qe_mg_g,target_unit='mg/g',target_name='experimental_adsorption_uptake_mg_g',
    target_origin='FIGURE6_EQUILIBRIUM_MARKER_DIGITIZED' if doi==NEW_DOIS['shen'] else 'TABLE2_EXPERIMENTAL_QTOTAL_48H',
    target_qualification='EQUILIBRIUM_GRAPH_SOURCE_REPORTED_DIGITIZED' if doi==NEW_DOIS['shen'] else '48H_EXPERIMENTAL_QTOTAL_EQUILIBRIUM_NOT_SEPARATELY_PROVEN',
    mass_balance_cap_mg_g=r.C0_mg_L/r.dose_g_L,physical_cap_pass=True,
    model_use_tier='V57_RESEARCH_NOT_ACTIVE',source_url=r.source_url,
    source_value_type=r.source_value_type, source_location=r.source_location,
    source_digitized=r.digitized,raw_source_row_number=i+2,
    original_qe_type=r.qe_type,source_notes=r.notes,raw_source_name='Kiro_approved_experimental_qe_rows.csv')
  rows.append(row)
 add=pd.DataFrame(rows)
 assert add.observation_id.is_unique and not set(add.doi)&set(base.doi)
 if set(add.doi)&set(pd.read_csv(root/V55_172, usecols=['doi']).doi):
  raise ValueError('Source DOI already present in cumulative original v55; verify duplicates')
 add.to_csv(ddir/'Stage2_New36_Harmonized_RESEARCH.csv',index=False)
 # 4 unique material property records; same BET repeated over C0 values must not become 29 independent Stage1 rows.
 sh=raw[raw.source_doi==NEW_DOIS['shen']].drop_duplicates(['source_doi','material_id']).copy()
 assert len(sh)==4
 propbase=pd.read_csv(root/S1_446,low_memory=False)
 prows=[]
 for r in sh.itertuples(index=False):
  row={c:np.nan for c in propbase.columns}
  row.update(material_id=f'V57_SHEN_{r.material_id}',doi=r.source_doi,original_sample_name=r.material_id,
   feedstock='wheat straw pellets' if r.feedstock=='Wheat straw pellets' else 'rice husk',
   feedstock_group='wheat straw pellets' if r.feedstock=='Wheat straw pellets' else 'rice husk',
   pyrolysis_temperature_C=r.pyrolysis_temp_C,pyrolysis_hold_min=np.nan,
   pyrolysis_atmosphere=np.nan,heating_rate_C_min=np.nan,
   activation_method='none',activation_agent='none',process_stage='single_stage_carbonization',
   BET_m2_g_training_candidate=r.BET_m2_g,TPV_cm3_g_training_candidate=np.nan,
   source_reference=r.source_url,source_pdf_page='p2 Table 1',
   v57_release_status='STAGE1_PROPERTY_SOURCE_REPORTED_STAGING',
   input_verification_limitations='Hold time, heating rate, atmosphere not inferred; biochar analysis is not feedstock elemental input')
  prows.append(row)
 prop=pd.DataFrame(prows)
 prop.to_csv(ddir/'Stage1_New4_BET_RESEARCH.csv',index=False)
 pd.concat([propbase,prop],ignore_index=True,sort=False).to_csv(ddir/'Stage1_450_RESEARCH_NOT_ACTIVE.csv',index=False)
 return raw,add,prop

def fixed140_eval(root,addon, existing):
 base=pd.read_csv(root/BASE140,low_memory=False).reset_index(drop=True)
 assert len(base)==140 and base.doi.nunique()==32
 # Source-out folded test set remains the original old140. Extra studies must not overlap any held-out DOI.
 scenarios={'old140':base.iloc[:0], 'v55_152':existing['v55_152'].iloc[140:],
   'v55_156_conditional':existing['v55_156'].iloc[140:],
   'v57_159_cao_only':pd.concat([existing['v55_152'].iloc[140:],addon[addon.doi==NEW_DOIS['cao']]],ignore_index=True,sort=False),
   'v57_181_shen_only':pd.concat([existing['v55_152'].iloc[140:],addon[addon.doi==NEW_DOIS['shen']]],ignore_index=True,sort=False),
   'v57_188_both':pd.concat([existing['v55_152'].iloc[140:],addon],ignore_index=True,sort=False),
   'v57_192_cocoa_conditional':pd.concat([existing['v55_156'].iloc[140:],addon],ignore_index=True,sort=False)}
 predictions=base[['observation_id','doi','target_value','C0_mg_L','dose_g_L']].copy()
 for tr,te in GroupKFold(n_splits=5).split(base,groups=base.doi):
  train,test=base.iloc[tr],base.iloc[te]; held=set(test.doi)
  for name,extra in scenarios.items():
   extra=extra[~extra.doi.isin(held)]
   assert not held.intersection(set(extra.doi))
   source_train=pd.concat([train,extra],ignore_index=True,sort=False)
   for variant in ('A','B'):
    p,raw,_=pipefit(source_train,test,variant)
    key=f'{variant}__{name}'
    if key not in predictions:
     predictions[key]=np.nan;predictions[key+'_raw']=np.nan
    predictions.loc[te,key]=p;predictions.loc[te,key+'_raw']=raw
 out=[]
 for name,extra in scenarios.items():
  for variant in ['A','B']:
   key=f'{variant}__{name}'
   y=base.target_value.to_numpy(float)
   p=predictions[key].to_numpy(float);pr=predictions[key+'_raw'].to_numpy(float)
   assert np.isfinite(p).all()
   cap=base.C0_mg_L.to_numpy(float)/base.dose_g_L.to_numpy(float)
   out.append(dict(model=variant,scenario=name,comparison='EXACT_FIXED140_32DOI_GROUPKFOLD5_POSTHOC',
    training_addon_count=int(len(extra)),new_source_DOI_count=int(extra.doi.nunique()),
    raw_cap_exceeded=int((pr>cap+1e-8).sum()),raw_negative=int((pr<0).sum()),
    physical_post_clip_violations=int(((p<0)|(p>cap+1e-8)).sum()),**score(y,p)))
 return pd.DataFrame(out),predictions

def new_doi_sourceout(root,existing,addon):
 results=[];preds=[];ranks=[]
 scenarios={'v55_152':existing['v55_152'],'v55_156_conditional':existing['v55_156']}
 for label,train in scenarios.items():
  for name,doi in NEW_DOIS.items():
   test=addon[addon.doi==doi].copy().reset_index(drop=True)
   assert doi not in set(train.doi)
   for variant in ['A','B']:
    p,raw,_=pipefit(train,test,variant)
    results.append(dict(source=name,source_doi=doi,scenario=label,model=variant,
     protocol='PRE_INCLUSION_SOURCE_OUT_POSTHOC_NOT_FUTURE_BLIND',
     zero_mae=float(np.mean(np.abs(test.target_value))),raw_cap_violations=int((raw>test.C0_mg_L/test.dose_g_L+1e-8).sum()),
     **score(test.target_value,p)))
    one=test[['observation_id','doi','material_uid','pollutant','pyrolysis_temperature_C','C0_mg_L','target_value']].copy()
    one['pred']=p;one['model']=variant;one['scenario']=label;one['source']=name
    preds.append(one)
    if name=='cao' and variant=='B':
     from scipy.stats import spearmanr
     if np.ptp(p)>1e-9:rho=float(spearmanr(test.target_value,p).statistic)
     else:rho=None
     winners=np.flatnonzero(np.isclose(p,p.max(),atol=1e-8))
     rank_ambig=len(winners)!=1
     ranks.append({'scenario':label,'source':'cao','n_temperature_conditions':len(test),'spearman_rho':rho,
                   'predicted_top1_temp_C':float(test.iloc[winners[0]].pyrolysis_temperature_C) if not rank_ambig else None,
                   'top1_regret_mg_g':float(test.target_value.max()-test.iloc[winners[0]].target_value) if not rank_ambig else None,
                   'prediction_top_tied':rank_ambig})
 return pd.DataFrame(results),pd.concat(preds,ignore_index=True),pd.DataFrame(ranks)

def stage1_bet_fixed(root,prop):
 old=pd.read_csv(root/'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv',low_memory=False)
 v55=pd.read_csv(root/S1_446,low_memory=False)
 d=old.loc[old.BET_m2_g_training_candidate.notna()].reset_index(drop=True)
 extra55=v55.iloc[417:]
 addsets={'old417':old.iloc[:0],'v55_446':extra55,'v57_450':pd.concat([extra55,prop],ignore_index=True,sort=False)}
 oof=d[['material_id','doi','BET_m2_g_training_candidate']].copy()
 for tr,te in GroupKFold(n_splits=5).split(d,groups=d.doi):
  for key,addon in addsets.items():
   train=pd.concat([d.iloc[tr],addon[~addon.doi.isin(set(d.iloc[te].doi))]],ignore_index=True,sort=False)
   mod,cols,nums,cats=s1_model()
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',FutureWarning)
    mod.fit(features(train,cols,nums,cats),pd.to_numeric(train.BET_m2_g_training_candidate).to_numpy(float))
    p=np.maximum(0,np.asarray(mod.predict(features(d.iloc[te],cols,nums,cats)),float))
   oof.loc[te,key]=p
 results=[]
 for key in addsets:
  y=d.BET_m2_g_training_candidate.to_numpy(float);p=oof[key].to_numpy(float)
  results.append({'scenario':key,'model':'Stage1_BET','evaluation':'SAME_ORIGINAL_BET_5GROUP_POSTHOC','n':len(y),
   'mae_m2_g':float(mean_absolute_error(y,p)),'rmse_m2_g':float(np.sqrt(mean_squared_error(y,p)))})
 return pd.DataFrame(results),oof

def paired_bootstrap(pred,meta,iterations=1200):
 rs=np.random.default_rng(20261031);t=pred.target_value.to_numpy(float)
 groupidx={g:np.flatnonzero(pred.doi.to_numpy()==g) for g in pred.doi.unique()}
 gs=np.array(list(groupidx));out=[]
 for m in ['A','B']:
  baseline=f'{m}__v55_152';p0=pred[baseline].to_numpy(float)
  for s in ['v57_159_cao_only','v57_181_shen_only','v57_188_both','v57_192_cocoa_conditional']:
   p1=pred[f'{m}__{s}'].to_numpy(float)
   improvement=np.abs(t-p0)-np.abs(t-p1)
   dist=[]
   for _ in range(iterations):
    sampled=rs.choice(gs,size=len(gs),replace=True);ix=np.concatenate([groupidx[g] for g in sampled]);dist.append(float(improvement[ix].mean()))
   low,high=np.quantile(dist,[.025,.975]);out.append({'model':m,'vs':'v55_152','new_scenario':s,'paired_improvement_positive_better_mg_g':float(improvement.mean()),
                 'bootstrap95_low':float(low),'bootstrap95_high':float(high),
                 'posthoc_exploratory':True,'bootstrap_unit':'DOI'})
 return pd.DataFrame(out)

def main(rootarg):
 root=Path(rootarg).resolve(); out=root/REPORT; out.mkdir(parents=True,exist_ok=True)
 dep=root/DEP;dep.mkdir(parents=True,exist_ok=True)
 raw,addon,props=ingest(root)
 old=pd.read_csv(root/BASE140,low_memory=False)
 existing={x:pd.read_csv(root/p,low_memory=False) for x,p in [('v55_152',V55_152),('v55_156',V55_156),('v55_168',V55_168),('v55_172',V55_172)]}
 for k,fr in existing.items():
  assert len(fr)==int(k.split('_')[1]),'prior dataset mismatch'
  assert fr.observation_id.is_unique
  assert fr.iloc[:140].observation_id.tolist()==old.observation_id.tolist()
 assert len(addon)==36 and addon.doi.nunique()==2
 ddir=root/DATA
 training={'v57_159_cao_only':pd.concat([existing['v55_152'],addon[addon.doi==NEW_DOIS['cao']]],ignore_index=True,sort=False),
  'v57_181_shen_only':pd.concat([existing['v55_152'],addon[addon.doi==NEW_DOIS['shen']]],ignore_index=True,sort=False),
  'v57_188_both':pd.concat([existing['v55_152'],addon],ignore_index=True,sort=False),
  'v57_192_cocoa_sensitivity':pd.concat([existing['v55_156'],addon],ignore_index=True,sort=False),
  'v57_204_apple24_sensitivity':pd.concat([existing['v55_168'],addon],ignore_index=True,sort=False),
  'v57_208_apple24_cocoa_sensitivity':pd.concat([existing['v55_172'],addon],ignore_index=True,sort=False)}
 for name,fr in training.items():
  assert len(fr)==int(name.split('_')[1]), name
  assert fr.observation_id.is_unique
  assert ((fr.target_value>=0)&(fr.target_value<=fr.C0_mg_L/fr.dose_g_L+1e-7)).all()
  fr.to_csv(ddir/f'Stage2_{name}_RESEARCH_NOT_ACTIVE.csv',index=False)
 m,pred=fixed140_eval(root,addon,existing)
 m.to_csv(out/'stage2_fixed140_sameDOI_metrics.csv',index=False)
 pred.to_csv(out/'stage2_fixed140_sameDOI_predictions.csv',index=False)
 boot=paired_bootstrap(pred,m)
 boot.to_csv(out/'stage2_paired_group_bootstrap.csv',index=False)
 src,pp,ranks=new_doi_sourceout(root,existing,addon)
 src.to_csv(out/'stage2_new_DOI_sourceout_metrics.csv',index=False)
 pp.to_csv(out/'stage2_new_DOI_sourceout_predictions.csv',index=False)
 ranks.to_csv(out/'cao_temperature_sourceout_rank.csv',index=False)
 s1m,s1pred=stage1_bet_fixed(root,props)
 s1m.to_csv(out/'stage1_BET_fixed_existing_metrics.csv',index=False)
 s1pred.to_csv(out/'stage1_BET_fixed_existing_predictions.csv',index=False)
 artifacts=[]
 for name,frame in training.items():
  for variant in ['A','B']:
   p,_,mod=pipefit(frame,frame.head(2),variant)
   dest=dep/f'{variant}_{name}_RESEARCH_NOT_ACTIVE.joblib'
   joblib.dump(mod,dest,compress=3)
   loaded=joblib.load(dest); cols=s2_model(variant)[1:]
   assert np.allclose(loaded.predict(features(frame.head(2),*cols)),mod.predict(features(frame.head(2),*cols)),atol=1e-8)
   artifacts.append({'model':variant,'scenario':name,'path':str(dest.relative_to(root)).replace('\\','/'),'sha256':sha(dest)})
 # one full cumulative exploratory Stage1 model; BET from 4 material rows, not 29 duplicated adsorption points.
 s1all=pd.read_csv(root/DATA/'Stage1_450_RESEARCH_NOT_ACTIVE.csv',low_memory=False)
 s1train=s1all[s1all.BET_m2_g_training_candidate.notna()]
 b,cols,num,cat=s1_model()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',FutureWarning)
  b.fit(features(s1train,cols,num,cat),s1train.BET_m2_g_training_candidate.to_numpy(float))
 dest=dep/'Stage1_BET_450_RESEARCH_NOT_ACTIVE.joblib'; joblib.dump(b,dest,compress=3)
 artifacts.append({'model':'Stage1_BET','scenario':'v57_450','path':str(dest.relative_to(root)),'sha256':sha(dest)})
 orig=root/DATA/'Kiro_approved_experimental_qe_rows_SOURCE_ORIGINAL.csv'
 manifest={'version':'v57_k_handoff_36_cumulative','active':False,'research_only':True,
  'source_batch_new_QE_36':36,'source_count':2,'new_experimental_graph_29':29,'new_experimental_table_7':7,
  'added_stage1_unique_BET_materials':4,'new_TPV':0,'source_hash_sha256':sha(orig),
  'baseline_v55_152_sha256':sha(root/V55_152), 'baseline_140_sha256':sha(root/BASE140),
  'legacy_active_sha256':sha(root/'deployment/stage2_models/active.json'),
  'python':platform.python_version(),'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,
  'note':'Same140 estimates posthoc; sourceout uses models trained before addition but sources were not hidden; no blind validation; optional new data do not imply performance improvements',
  'artifacts':artifacts}
 (dep/'manifest_v57.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 summary={'stage2_fixed140':m.to_dict('records'),'new_doi_sourceout':src.to_dict('records'),
   'ranking':ranks.to_dict('records'),'stage1_bet_fixed':s1m.to_dict('records'),
   'paired_bootstrap':boot.to_dict('records'),'manifest':manifest}
 (out/'V57_SUMMARY.json').write_text(json.dumps(sanitize_json(summary),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print('V57 COMPLETED',len(artifacts),'artifacts')
 print(m[['model','scenario','n','mae_mg_g','rmse_mg_g','raw_cap_exceeded']].to_string(index=False))
 print('SOURCEOUT\n',src[['source','scenario','model','n','mae_mg_g','zero_mae']].to_string(index=False))
 print('STAGE1\n',s1m.to_string(index=False))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.'); a=p.parse_args();main(a.root)
