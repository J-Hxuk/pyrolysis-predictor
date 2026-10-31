"""v55: cumulative source evidence, targets kept separate, unchanged DOI-fold evaluation.

Run: python -m scripts.run_v55_source_cumulative_refit --root .
Research-only; never overwrites active model or legacy datasets.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn, xgboost
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.build_v46_phase8 import s2_model, s1_model,features

DOI={'cocoa':'10.1007/s13399-026-07198-4','ragweed':'10.1016/j.biortech.2020.124011',
 'rice':'10.1155/2018/2684962','sugar':'10.1016/j.chemosphere.2013.12.042',
 'sludge':'10.1016/j.biortech.2013.08.082'}
FILE={'cocoa':'PRIORITY_CocoaPodHull_Biochar_Cd_DOI_10.1007-s13399-026-07198-4_main.pdf',
 'ragweed':'PRIORITY_Ragweed_Horseweed_Biochar_Cd_Pb_DOI_10.1016-j.biortech.2020.124011_main.pdf',
 'rice':'RiceStraw_DigestedRiceStraw_Biochar_Pb_DOI_10.1155-2018-2684962_main.pdf',
 'sugar':'SugarcaneBagasse_Biochar_Pb_DOI_10.1016-j.chemosphere.2013.12.042_main.pdf',
 'sludge':'MunicipalWastewaterSludge_Biochar_Pb_Cr_DOI_10.1016-j.biortech.2013.08.082_main.pdf'}
BASE2='data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
APPLE='data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv'
BASE1='data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
QBASE='data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/08_Langmuir_Qmax_Fitted_Separate_32.csv'
MATS='data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/01_Manufacturing_And_Feedstock_Materials_441.csv'
OUT='reports/v55_cumulative_source_refit';DATA='data/imports/v55_cumulative_sources';DEP='deployment/research_v55_NOT_ACTIVE'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rmse(y,p):return float(np.sqrt(mean_squared_error(y,p)))
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 return {'n':len(y),'mae':float(mean_absolute_error(y,p)),'rmse':rmse(y,p),'r2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None}

def build_rows(root):
 source=root/'data/imports/v55_original_sources/pdfs'
 for key,fn in FILE.items():
  assert (source/fn).exists(),f'Original PDF missing {fn}'
 # Source registry: retain even when tabular observations are not machine-recoverable.
 registry=[{'doi':d,'source_key':key,'source_file':FILE[key], 'sha256':sha(source/FILE[key]),'original_source_archived':True,
   'source_status':('QUANTITATIVE_QE_AND_STAGE1' if key in ['cocoa','sugar'] else 'QUANTITATIVE_QMAX_AND_STAGE1' if key=='ragweed' else 'NUMERICAL_STAGE1_BET_AND_OTHER_SOURCE_STAGED'),
   'notes':('Rice: eight BET values, no endpoint-specific tabular qe from supplied PDF' if key=='rice' else 'Sludge: seven BET values and fourteen printed-unit ambiguous Langmuir Qmax; fitted kinetic qe is not experimental qe' if key=='sludge' else '')} for key,d in DOI.items()]
 # New observed uptake: sugarcane: 80 mg/L, 0.1 g/50 mL, 24h; page 4 body paragraph. Four reported values.
 uptake=[]
 for T,v,bet in zip([250,400,500,600],[15.9,10.7,11.4,8.3],[.56,.78,1.98,14.1]):
  uptake.append({'source_key':'sugar','observation_id':f'V55_DING_BC{T}_PB_C0_80','doi':DOI['sugar'],'pollutant':'Pb(II)',
   'sample':'BC'+str(T),'feedstock_label_normalized':'sugarcane bagasse','feedstock_raw':'sugarcane bagasse',
   'feedstock_group_for_novelty_check':'sugarcane bagasse','manufacturing_route':'SINGLE_STAGE_PYROLYSIS',
   'pyrolysis_temperature_C':T,'pyrolysis_hold_min':np.nan,'heating_rate_C_min':10,'pyrolysis_atmosphere':'N2',
   'C0_mg_L':80,'dose_g_L':2,'initial_pH':np.nan,'adsorption_T_C':22,'contact_time_min':1440,
   'target_value':v,'target_name':'qe_reported_mg_g','target_origin':'DIRECT_TEXT_24H_PB_UPTAKE',
   'qa_tier':'PAPER_REPORTED_24H_EQUILIBRIUM','source_page':'p4 (journal page 71) text, §3.4',
   'source_url':'https://doi.org/'+DOI['sugar'],'BET_measured_m2_g':bet,'TPV_measured_cm3_g':np.nan,
   'C0_condition_status':'DIRECTLY_REPORTED','target_evidence':'DIRECT_TEXT',
   'analysis_role':'S2_MAIN_CANDIDATE'})
 # Cocoa Table3 qeexp and removal % imply C0=10 mg/L with 50 mg/50mL. Methods kinetic paragraph states 5 mg/L.
 # Include in raw evidence and sensitivity model, never silently call direct verified C0.
 for sample,T,v,sd,remove,bet,tpv in zip(['CPHB400','CPHB650','CPHB900','CPHB900A'],[400,650,900,900],
                                        [5.93,2.47,2.80,3.27],[.36,.09,.27,.09],[59.33,24.67,28,32.67],
                                        [3,5,3,178],[.03,.01,.01,.09]):
  uptake.append({'source_key':'cocoa','observation_id':f'V55_COCOA_{sample}_CD_TABLE3','doi':DOI['cocoa'],'pollutant':'Cd(II)',
   'sample':sample,'feedstock_label_normalized':'cocoa pod hull','feedstock_raw':'cocoa pod hull',
   'feedstock_group_for_novelty_check':'cocoa pod hull',
   'manufacturing_route':'MULTI_STEP_OR_MODIFIED' if sample.endswith('A') else 'SINGLE_STAGE_PYROLYSIS',
   'activation_or_modification':'CO2_activated' if sample.endswith('A') else np.nan,
   'secondary_temperature_C':800 if sample.endswith('A') else np.nan,'secondary_hold_min':60 if sample.endswith('A') else np.nan,
   'pyrolysis_temperature_C':T,'pyrolysis_hold_min':60,'heating_rate_C_min':10,'pyrolysis_atmosphere':'N2',
   'C0_mg_L':10,'dose_g_L':1,'initial_pH':6,'adsorption_T_C':22,'contact_time_min':2880,
   'target_value':v,'source_numeric_sd_mg_g':sd,'target_name':'qe_exp_kinetic_mg_g',
   'target_origin':'TABLE3_EXPERIMENTAL_KINETIC_QE','qa_tier':'C0_INFERRED_FROM_TABLE3_PERCENT_NOT_DIRECT_CONFIRMED',
   'source_page':'p12 Table 3 / p4 Methods 2.2.4 / p1 abstract','source_url':'https://doi.org/'+DOI['cocoa'],
   'BET_measured_m2_g':bet,'TPV_measured_cm3_g':tpv,
   'C0_condition_status':'10_INFERRED_FROM_QE_AND_REMOVAL; 5_MG_L_METHODS_KINETIC_CONFLICT',
   'removal_percent_table3':remove,'target_evidence':'DIRECT_TABLE3','analysis_role':'S2_SENSITIVITY_ONLY'})
 u=pd.DataFrame(uptake)
 u['mass_balance_cap_mg_g']=u.C0_mg_L/u.dose_g_L
 assert (u.target_value<=u.mass_balance_cap_mg_g+1e-8).all()
 # Cropping source notes do not enter model features.
 qrows=[]
 for metal,vals in [('Cd(II)',[68.816,139.032,86.545,43.393,82.768,60.493]),
                     ('Pb(II)',[124.377,358.780,258.756,126.339,209.798,184.084])]:
  for sample,T,feed,v in zip(['RB350','RB450','RB550','HB350','HB450','HB550'],[350,450,550]*2,
                             ['ragweed']*3+['horseweed']*3,vals):
   qrows.append({'source_key':'ragweed','doi':DOI['ragweed'],'observation_id':f'V55_LIAN_{sample}_{metal[:2].upper()}_QMAX',
    'material_uid':f'V55_LIAN_{sample}','pollutant':metal,'feedstock_raw':feed,'pyrolysis_temperature_C':T,
    'target_name':'Langmuir_qmax_fitted_mg_g','target_value':v,
    'source_page':'p6 Table 3','fit_origin':'LANGMUIR_FITTED_ISOTHERM_NOT_EXPERIMENTAL_QE',
    'source_url':'https://doi.org/'+DOI['ragweed']})
 # Cocoa Table 4 fitted Langmuir values from abstract. They are not qeexp.
 for sample,T,v in zip(['CPHB400','CPHB650','CPHB900','CPHB900A'],[400,650,900,900],[23.58,9.78,10.28,10.19]):
  qrows.append({'source_key':'cocoa','doi':DOI['cocoa'],'observation_id':f'V55_COCOA_{sample}_CD_QMAX',
   'material_uid':f'V55_COCOA_{sample}','pollutant':'Cd(II)','feedstock_raw':'cocoa pod hull',
   'pyrolysis_temperature_C':T,'target_name':'Langmuir_qmax_fitted_mg_g','target_value':v,
   'source_page':'p1 abstract / p14 Table 4','fit_origin':'LANGMUIR_FITTED_ISOTHERM_NOT_EXPERIMENTAL_QE',
   'source_url':'https://doi.org/'+DOI['cocoa']})
 q=pd.DataFrame(qrows)
 # Physical property records: 14 BET and 10 TPV directly reported, not post-pyrolysis elemental analysis as raw features.
 props=[]
 def addp(key,sample,feed,T,bet,tpv,hold,heat,atm,activation='none',act_t=np.nan,act_time=np.nan):
  props.append({'material_id':f'V55_{key.upper()}_{sample}','doi':DOI[key],
   'original_sample_name':sample,'feedstock':feed,'feedstock_group':feed,
   'pyrolysis_temperature_C':T,'pyrolysis_hold_min':hold,'heating_rate_C_min':heat,
   'pyrolysis_atmosphere':atm,'process_stage':'carbonization_plus_activation' if activation!='none' else 'single_stage_carbonization',
   'activation_method':activation,'activation_agent':'CO2' if activation!='none' else 'none',
   'activation_temperature_C':act_t,'activation_time_min':act_time,
   'BET_m2_g_training_candidate':bet,'TPV_cm3_g_training_candidate':tpv,
   'source_pdf_page':'p5 Table 1' if key=='cocoa' else 'p3 Table 1' if key=='ragweed' else 'p3 Table 1',
   'source_reference':'https://doi.org/'+DOI[key],'v55_release_status':'RESEARCH_PROPERTY_SOURCE_VERIFIED',
   'input_verification_limitations':'Raw feedstock C/H/O not imputed from biochar analysis; absent recipe variables remain missing', 'source_key':key})
 for sample,t,bet,tpv in zip(['CPHB400','CPHB650','CPHB900','CPHB900A'],[400,650,900,900],[3,5,3,178],[.03,.01,.01,.09]):
  addp('cocoa',sample,'cocoa pod hull',t,bet,tpv,60,10,'N2','CO2 activation' if sample.endswith('A') else 'none',800 if sample.endswith('A') else np.nan,60 if sample.endswith('A') else np.nan)
 for sample,t,feed,bet,tpv in zip(['RB350','RB450','RB550','HB350','HB450','HB550'],[350,450,550]*2,['ragweed']*3+['horseweed']*3,[8.1,26.8,25.6,4.3,4.4,15.6],[.0439,.0610,.0576,.0148,.0106,.0335]):
  addp('ragweed',sample,feed,t,bet,tpv,np.nan,np.nan,np.nan)
 for t,bet in zip([250,400,500,600],[.56,.78,1.98,14.1]):
  addp('sugar','BC'+str(t),'sugarcane bagasse',t,bet,np.nan,np.nan,10,'N2')
 # Rice straw and anaerobically digested rice straw: 8 numerical BET entries from Table 1 p4;
 # N2 0.1 m3/h = 1666.67 mL/min, 10 °C/min, 60 min from Methods p3.
 for feedcode,feed,values in [('RS','rice straw',[270.84,314.86,465.21,404.97]),
                              ('DRS','anaerobically digested rice straw',[117.42,137.50,162.00,213.27])]:
  for T,bet in zip([400,500,600,700],values):
   addp('rice',f'{feedcode}-{T}',feed,T,bet,np.nan,60,10,'N2')
 # Municipal wastewater sludge Table 1 p4, seven temperature/time conditions,
 # no TPV numerical values in this table. Sample 1 through 7 only; no invented 300°C 1h.
 for (number,T,hold,bet) in [(1,300,120,7.90),(2,400,60,14.14),(3,400,120,23.72),
                             (4,500,60,17.19),(5,500,120,20.12),(6,600,60,13.16),(7,600,120,16.00)]:
  addp('sludge',f'SDBC_{number}', 'municipal wastewater sludge',T,bet,np.nan,hold,np.nan,np.nan)
 props=pd.DataFrame(props)
 assert len(u)==8 and len(q)==16 and len(props)==29
 return pd.DataFrame(registry),u,q,props

def to_stage2(u,template):
 rows=[]
 for rec in u.to_dict('records'):
  row={col:np.nan for col in template.columns}
  row.update(rec)
  row.update(doi_group=rec['doi'],material_uid=rec['sample']+'::'+rec['doi'],sample_group=rec['sample']+'::'+rec['doi'],
             experiment_series_id=rec['doi']+'::'+rec['pollutant']+'::'+rec['sample'],
             target_unit='mg/g',physical_cap_pass=True,process_step_count=2 if rec['sample']=='CPHB900A' else 1,
             model_use_tier='V55_RESEARCH_NOT_ACTIVE',target_qualification=rec['qa_tier'])
  rows.append(row)
 return pd.DataFrame(rows)

def stage2_fit(train,test,variant):
 pipe,cols,nums,cats=s2_model(variant)
 pipe.fit(features(train,cols,nums,cats),train.target_value.to_numpy(float))
 raw=np.asarray(pipe.predict(features(test,cols,nums,cats)),float)
 cap=test.C0_mg_L.to_numpy(float)/test.dose_g_L.to_numpy(float)
 return np.clip(raw,0,cap),raw,pipe

def fixed_eval(base,addons,variants=('A','B'),n_folds=5):
 d=base.reset_index(drop=True)
 out=d[['observation_id','doi','target_value','C0_mg_L','dose_g_L']].copy();records=[]
 groups=GroupKFold(n_splits=n_folds).split(d,groups=d.doi)
 for fold,(tr,te) in enumerate(groups,1):
  oldtrain,test=d.iloc[tr],d.iloc[te]
  for scenario,extra in addons.items():
   assert set(extra.doi.dropna()).isdisjoint(set(test.doi))
   train=pd.concat([oldtrain,extra],ignore_index=True,sort=False)
   for variant in variants:
    pred,raw,_=stage2_fit(train,test,variant)
    name=f'{variant}__{scenario}'
    if name not in out:out[name]=np.nan;out[name+'_raw']=np.nan
    out.loc[te,name]=pred;out.loc[te,name+'_raw']=raw
 for variant in variants:
  for scenario,extra in addons.items():
   col=f'{variant}__{scenario}';pred=out[col].to_numpy(float);raw=out[col+'_raw'].to_numpy(float)
   assert np.isfinite(pred).all()
   cap=d.C0_mg_L.to_numpy(float)/d.dose_g_L.to_numpy(float)
   records.append({'model':variant,'scenario':scenario,'evaluation':'SAME_ORIGINAL_140_32DOI_GROUPKFOLD5',
    'training_addon_rows':len(extra),'raw_upper_bound_violations':int((raw>cap+1e-8).sum()),
    'raw_lower_bound_violations':int((raw<0).sum()),'physical_post_clip_violations':int(((pred>cap+1e-8)|(pred<0)).sum()),
    **metrics(d.target_value,pred)})
 return pd.DataFrame(records),out

def external_group_eval(base,addons):
 out=[]
 for source in ['sugar','cocoa']:
  target=addons[addons.source_key==source].copy()
  for v in ['A','B']:
   pred,raw,_=stage2_fit(base,target,v)
   out.append({'model':v,'source':source,'protocol':'NEW_DOI_PRE_INCLUSION_BASE140_TRAIN_ONLY',**metrics(target.target_value,pred)})
 return pd.DataFrame(out)

def qmax_eval(base,qnew):
 from sklearn.compose import ColumnTransformer
 from sklearn.pipeline import Pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import OneHotEncoder
 from xgboost import XGBRegressor
 pcols=['pollutant','feedstock_raw','pyrolysis_temperature_C','manufacturing_route']
 def pipe():
  return Pipeline([('pre',ColumnTransformer([('numeric',SimpleImputer(strategy='median'),['pyrolysis_temperature_C']),
   ('categorical',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='unknown')),
    ('ohe',OneHotEncoder(handle_unknown='ignore'))]),['pollutant','feedstock_raw','manufacturing_route'])])),
   ('model',XGBRegressor(n_estimators=150,max_depth=2,learning_rate=.06,reg_lambda=8,min_child_weight=4,n_jobs=1,random_state=42,tree_method='hist'))])
 base=base.copy().reset_index(drop=True);qnew=qnew.copy()
 mats=pd.read_csv(ROOT/MATS,low_memory=False)
 mcols=['material_uid','manufacturing_route','pyrolysis_temperature_C']
 m=mats[mcols].drop_duplicates('material_uid')
 base=base.merge(m,on='material_uid',how='left',validate='many_to_one')
 qnew['manufacturing_route']=np.where(qnew.material_uid.str.contains('CPHB900A'),'MULTI_STEP_OR_MODIFIED','SINGLE_STAGE_PYROLYSIS')
 b=base.rename(columns={'doi_group':'doi'})
 out=b[['material_uid','doi','target_value']].copy();results=[]
 addons={'qmax_old32':qnew.iloc[:0], 'qmax_plus12_ragweed':qnew[qnew.source_key=='ragweed'], 'qmax_plus16_5_doi':qnew}
 for fold,(tr,te) in enumerate(GroupKFold(n_splits=3).split(b,groups=b.doi)):
  for scenario,addon in addons.items():
   train=pd.concat([b.iloc[tr],addon],ignore_index=True,sort=False)
   fit=pipe().fit(train.reindex(columns=pcols),train.target_value.to_numpy(float))
   out.loc[te,scenario]=np.maximum(0,fit.predict(b.iloc[te].reindex(columns=pcols)))
 for name,addon in addons.items():
  results.append({'scenario':name,'evaluation':'SAME_ORIGINAL_32_3DOI_GROUPKFOLD3','extra_rows':len(addon),**metrics(b.target_value,out[name])})
 # Fit full distinct qmax models, never merge qmax into qe training.
 mod=rootdep/'qmax';mod.mkdir(parents=True,exist_ok=True)
 for name,addon in [('qmax_plus12_ragweed',addons['qmax_plus12_ragweed']),('qmax_plus16_5_doi',qnew)]:
  train=pd.concat([b,addon],ignore_index=True,sort=False)
  fit=pipe().fit(train.reindex(columns=pcols),train.target_value.to_numpy(float))
  joblib.dump(fit,mod/(name+'.joblib'),compress=3)
 return pd.DataFrame(results),out

def stage1_eval(base,extra):
 from sklearn.metrics import mean_absolute_error
 results=[];preds={}
 for target,label in [('BET_m2_g_training_candidate','BET'),('TPV_cm3_g_training_candidate','TPV')]:
  d=base.loc[pd.to_numeric(base[target],errors='coerce').notna()].reset_index(drop=True)
  addon=extra.loc[extra[target].notna()].copy()
  assert set(addon.doi).isdisjoint(set(d.doi)),'Stage1 DOI shared: check inherited exposure'
  if d.doi.nunique()<5:continue
  out=d[['material_id','doi',target]].copy()
  addsets={'baseline':addon.iloc[:0],
           'cocoa_ragweed_sugar14':addon[addon.source_key.isin(['cocoa','ragweed','sugar'])],
           'with_rice22':addon[addon.source_key.isin(['cocoa','ragweed','sugar','rice'])],
           'with_sludge21':addon[addon.source_key.isin(['cocoa','ragweed','sugar','sludge'])],
           'all_new_source29':addon}
  for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(d,groups=d.doi)):
   for scen,ad in addsets.items():
    train=pd.concat([d.iloc[tr],ad],ignore_index=True,sort=False)
    model,cols,nums,cats=s1_model()
    model.fit(features(train,cols,nums,cats),train[target].to_numpy(float))
    y=np.maximum(0,np.asarray(model.predict(features(d.iloc[te],cols,nums,cats)),float))
    out.loc[te,scen]=y
  for scen,ad in addsets.items():
   results.append({'target':label,'scenario':scen,'evaluation':'SAME_ORIGINAL_STAGE1_5DOI_GROUPKFOLD5','new_rows':len(ad),
                   'doi_count':d.doi.nunique(),**metrics(d[target],out[scen])})
  preds[label]=out
  train=pd.concat([d,addon],ignore_index=True,sort=False)
  model,cols,nums,cats=s1_model();model.fit(features(train,cols,nums,cats),train[target].to_numpy(float))
  joblib.dump(model,rootdep/f'STAGE1_{label}_{len(train)}_RESEARCH_NOT_ACTIVE.joblib',compress=3)
  if label=='BET':
   bestadd=addon[addon.source_key.isin(['cocoa','ragweed','sugar'])]
   selected=pd.concat([d,bestadd],ignore_index=True,sort=False)
   selected_model,cols,nums,cats=s1_model()
   selected_model.fit(features(selected,cols,nums,cats),selected[target].to_numpy(float))
   joblib.dump(selected_model,rootdep/f'STAGE1_BET_{len(selected)}_SELECTED14_EXPLORATORY_NOT_ACTIVE.joblib',compress=3)
 return pd.DataFrame(results),preds

def main(root):
 global ROOT,rootdep
 ROOT=Path(root).resolve();rootdep=ROOT/DEP;rootdep.mkdir(parents=True,exist_ok=True)
 report=ROOT/OUT;report.mkdir(parents=True,exist_ok=True)
 ddir=ROOT/DATA;ddir.mkdir(parents=True,exist_ok=True)
 registry,u,q,props=build_rows(ROOT)
 base=pd.read_csv(ROOT/BASE2,low_memory=False)
 apple=pd.read_csv(ROOT/APPLE,low_memory=False)
 assert len(base)==140 and base.doi.nunique()==32 and len(apple)==24
 assert not set(base.doi.str.lower())&set(DOI.values())
 assert set(DOI.values()).isdisjoint(set(apple.doi.str.lower()))
 assert not set(q.doi)&set(base.doi)
 for name,frame in [('Source_Registry_5',registry),('New_Reported_QE_8',u),('New_Fitted_Qmax_16',q),('New_Stage1_Properties_29',props)]:frame.to_csv(ddir/(name+'.csv'),index=False)
 new=to_stage2(u,base)
 apple8=apple[apple.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP'].copy()
 sugar=new[new.source_key=='sugar'].copy();cocoa=new[new.source_key=='cocoa'].copy()
 addons={'old140':new.iloc[:0],'v53_148':apple8,
   'v53_164_sensitivity':apple,'v55_152_plus_sugar':pd.concat([apple8,sugar],ignore_index=True),
   'v55_156_plus_cocoa_conditional':pd.concat([apple8,sugar,cocoa],ignore_index=True),
   'v55_168_all24_plus_sugar_sensitivity':pd.concat([apple,sugar],ignore_index=True),
   'v55_172_all24_plus_cocoa_sensitivity':pd.concat([apple,sugar,cocoa],ignore_index=True)}
 for scenario,addon in addons.items():
  a=pd.concat([base,addon],ignore_index=True,sort=False)
  assert a.observation_id.is_unique
  assert (a.target_value<=a.C0_mg_L/a.dose_g_L+1e-8).all()
  a.to_csv(ddir/('Stage2_'+str(len(a))+'_'+scenario+'_RESEARCH_NOT_ACTIVE.csv'),index=False)
 results,pred=fixed_eval(base,addons)
 results.to_csv(report/'stage2_same140_group_oof_metrics.csv',index=False)
 pred.to_csv(report/'stage2_same140_group_oof_predictions.csv',index=False)
 ext=external_group_eval(base,new);ext.to_csv(report/'stage2_new_doi_pre_inclusion_challenge.csv',index=False)
 artifacts=[]
 for scenario,addon in addons.items():
  if scenario=='old140':continue
  full=pd.concat([base,addon],ignore_index=True,sort=False)
  for variant in ['A','B']:
   pipe,cols,nums,cats=s2_model(variant)
   pipe.fit(features(full,cols,nums,cats),full.target_value.to_numpy(float))
   filename=f'STAGE2_{variant}_{scenario}_{len(full)}_RESEARCH_NOT_ACTIVE.joblib'
   joblib.dump(pipe,rootdep/filename,compress=3)
   loaded=joblib.load(rootdep/filename)
   assert np.isfinite(loaded.predict(features(full.head(2),cols,nums,cats))).all()
   artifacts.append({'model':variant,'scenario':scenario,'path':f'{DEP}/{filename}','sha256':sha(rootdep/filename)})
 # Separate Langmuir Qmax dataset: numeric qmax only, keep target types distinct.
 qbase=pd.read_csv(ROOT/QBASE)
 qmet,qpred=qmax_eval(qbase,q)
 qmet.to_csv(report/'qmax_same32_group_oof_metrics.csv',index=False)
 qpred.to_csv(report/'qmax_same32_group_oof_predictions.csv',index=False)
 pd.concat([qbase.rename(columns={'doi_group':'doi'}),q],ignore_index=True,sort=False).to_csv(ddir/'QMAX_48_FITTED_SEPARATE.csv',index=False)
 # Additional Stage1 source properties on top of current 417-row exploratory research master.
 s1base=pd.read_csv(ROOT/BASE1,low_memory=False)
 assert len(s1base)==417
 s1met,s1pred=stage1_eval(s1base,props)
 s1met.to_csv(report/'stage1_same_original_group_oof_metrics.csv',index=False)
 for k,predictions in s1pred.items():predictions.to_csv(report/f'stage1_{k}_oof_predictions.csv',index=False)
 pd.concat([s1base,props],ignore_index=True,sort=False).to_csv(ddir/'Stage1_446_RESEARCH_NOT_ACTIVE.csv',index=False)
 # Source-presentation provenance; old active untouched.
 manifest={'version':'v55_source_cumulative_research','status':'RESEARCH_ONLY_NOT_ACTIVE',
  'stage2_new_reported_qe':len(new),'stage2_new_source_strict_candidate':len(sugar),
  'stage2_new_source_conditional_candidate':len(cocoa),'qmax_new_fitted':len(q),
  'stage1_new_BET':int(props.BET_m2_g_training_candidate.notna().sum()),
  'stage1_new_TPV':int(props.TPV_cm3_g_training_candidate.notna().sum()),
  'source_registry_entries':len(registry),'python':platform.python_version(),'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,
  'source_data_hashes':{'base_stage2_140':sha(ROOT/BASE2),'v52_apple24':sha(ROOT/APPLE),'base_stage1_417':sha(ROOT/BASE1)},
  'models':artifacts,'source_doi':DOI,'note':'Evaluate same original DOIs; new sources not in original folds; new DOI challenge only pre-inclusion; no independent test after model selection'}
 (rootdep/'manifest_v55.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 summary={'manifest':manifest,'stage2':results.to_dict('records'),'qmax':qmet.to_dict('records'),
          'stage1':s1met.to_dict('records'),'new_doi_pre_inclusion':ext.to_dict('records')}
 (report/'V55_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print(json.dumps({'stage2':results.to_dict('records'),'qmax':qmet.to_dict('records'),'stage1':s1met.to_dict('records'),
  'new_doi_pre_inclusion':ext.to_dict('records')},ensure_ascii=False,indent=2))
 return summary

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
 args=parser.parse_args();main(args.root)
