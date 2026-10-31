"""v34: audited literature staging + research-only training; never changes deployment/active.json.
Run: python -m scripts.build_v34_temperature
Requires v33 installed, core98 Phase8 and v15/v16 model modules.
"""
from __future__ import annotations
import json, hashlib, warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,r2_score
from scripts.compare_phase8_v23 import stage2_targets,_stage2_fit_predict,cluster_delta_ci
from pyrolysis.biochar_stage2.v15_features import model_inputs
from pyrolysis.biochar_stage2.v15_models import build_pipeline
from pyrolysis.biochar_stage2.v16_validate import validation_algorithms

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v34_temperature'
DATA=ROOT/'data/imports/v34_temperature'
V33=ROOT/'data/imports/v33_five_papers'
SEED=20261030
DOI={
 'zhou':'10.1038/s41598-022-17901-w',
 'li':'10.3390/plants11233234',
 'shen':'10.1016/j.scitotenv.2018.11.282',
 'cedar':'10.1007/s11356-017-1047-2',
 'jia_duplicate':'10.3390/app8071019'}
# The 17 rows copied verbatim from Zhou 2022, Table 2, PDF p5, in coded X1,X2,X3.
DOE_ROWS=[
 (1,0,1,1,16.07,17.03),(2,-1,0,-1,15.75,16.56),(3,1,0,-1,16.14,17.83),
 (4,0,0,0,17.83,19.14),(5,1,1,0,16.23,17.13),(6,1,-1,0,13.73,14.16),
 (7,-1,1,0,15.57,16.94),(8,-1,-1,0,15.04,13.84),(9,0,1,-1,15.86,16.09),
 (10,0,0,0,17.72,19.61),(11,-1,0,1,14.49,17.19),(12,0,-1,1,14.23,14.54),
 (13,0,0,0,18.17,19.64),(14,0,0,0,18.20,19.49),(15,0,0,0,18.12,19.29),
 (16,1,0,1,15.69,18.51),(17,0,-1,-1,14.53,14.68)]
# Cedar 2018 Table 2 (PDF p5): fitted Langmuir Csm, not measured uptake.
CEDAR=[(250,2.73,1.65),(300,1.50,1.14),(350,12.80,.75),(400,14.66,1.17),
 (450,16.72,10.64),(500,21.23,12.01),(550,27.17,13.93),(600,30.58,13.51),
 (650,23.92,15.20),(700,28.65,15.06)]


def audit_source_registry() -> pd.DataFrame:
 return pd.DataFrame([
  dict(doi=DOI['zhou'],title='Optimization of target biochar for the adsorption of target heavy metal ion',source='WaterHyacinth_Biochar_CuZn_main.pdf',source_figure='Table 1 p4; Table 2 p5',role='qt_240min_BBD_separate',note='17 runs per ion; 5 center-point replicates; C0=50 mg/L dose=2 g/L; 240 min; pH≈5.5; do not mix with qe'),
  dict(doi=DOI['li'],title='Effect of Pyrolysis Temperature on the Sorption of Cd(II) and Se(IV) by Rice Husk Biochar',source='RiceHusk_Biochar_CdSe_main.pdf',source_figure='Table 1 p5; Figures 2–5',role='stage1_BET_and_graph_hold',note='R300,R500,R700; kinetic/isotherm fitted parameters excluded from direct qe'),
  dict(doi=DOI['shen'],title='Effect of production temperature on lead removal mechanisms by rice straw biochars',source='RiceStraw_Biochar_Pb_main.pdf',source_figure='Table 1 p3; Figure 4; Tables 2–3',role='stage1_BET_and_graph_hold',note='RSB300,RSB500,RSB700; Table 2 kinetic fitted qe and Table 3 qmax not direct experimental qe'),
  dict(doi=DOI['cedar'],title='Predicting Cu and Zn sorption capacity of biochar from feedstock C/N ratio and pyrolysis temperature',source='CedarWood_Biochar_CuZn_main.pdf',source_figure='Table 2 p5; supporting isotherms',role='qmax_separate',note='Ten temperature levels, fitted Langmuir Csm; some low R2; distinct from qe'),
  dict(doi=DOI['jia_duplicate'],title='Study of the Effect of Pyrolysis Temperature on the Cd2+ Adsorption Characteristics of Biochar',source='RiceHusk_CottonStraw_Biochar_Cd_main.pdf',source_figure='v33 already registered',role='DUPLICATE_EXCLUDE',note='Existing v33 DOI; never append a second time'),
 ])


def new_bet() -> pd.DataFrame:
 rows=[]
 for doi,feed,prefix,temps,values,hold,rate,atm,pdf,table in [
  (DOI['li'],'rice husk','R',[300,500,700],[64.78,257.5,330.0],120,5,'NR','RiceHusk_Biochar_CdSe_main.pdf','Table 1 p5'),
  (DOI['shen'],'rice straw','RSB',[300,500,700],[6.766,22.383,115.465],60,10,'oxygen_limited','RiceStraw_Biochar_Pb_main.pdf','Table 1 p3'),
 ]:
  for t,v in zip(temps,values):
   rows.append(dict(material_id=f"V34_{prefix}_{t}_{doi.split('/')[-1]}",doi=doi,feedstock=feed,sample_name=f'{prefix}{t}',pyrolysis_temperature_C=t,
                    pyrolysis_hold_min=hold,heating_rate_C_min=rate,pyrolysis_atmosphere=atm,BET_m2_g=v,
                    TPV_cm3_g=np.nan,pore_volume_type='NR',feedstock_composition_basis='NOT_REPORTED',
                    source_pdf=pdf,source_table=table,approval='RESEARCH_STAGING_BET_ONLY',source_target='biochar_BET_not_feedstock_composition'))
 return pd.DataFrame(rows)


def cedar_qmax() -> pd.DataFrame:
 rows=[]
 # Table 2 fit R2; low-quality fits retained but explicitly flagged for exclusion.
 fit_r2={250:(1.00,.95),300:(.94,.96),350:(.20,.30),400:(.54,.99),450:(.97,.97),
         500:(.99,.85),550:(1.00,.77),600:(.99,.88),650:(.99,.84),700:(.99,.86)}
 for t,cu,zn in CEDAR:
  for pol,val in [('Cu(II)',cu),('Zn(II)',zn)]:
   rows.append(dict(doi=DOI['cedar'],feedstock='cedar wood',pyrolysis_temperature_C=t,pollutant=pol,qmax_langmuir_mg_g=val,fit_r2=fit_r2[t][0 if pol=='Cu(II)' else 1],
                    fit_qa=('LOW_FIT_R2_EXCLUDE_FROM_TRAINING' if fit_r2[t][0 if pol=='Cu(II)' else 1] <.8 else 'RESEARCH_ONLY_FIT'),
                    target_type='LANGMUIR_CSM_NOT_QE',source='CedarWood_Biochar_CuZn_main.pdf Table 2 p5',approval='SEPARATE_QMAX_RESEARCH_ONLY'))
 return pd.DataFrame(rows)


def doeraw() -> pd.DataFrame:
 out=[]
 for run,a,b,c,cu,zn in DOE_ROWS:
  for pol,q in [('Cu(II)',cu),('Zn(II)',zn)]:
   if pol=='Cu(II)':
    hold_h={-1:2,0:3,1:4}[a]; rate={-1:15,0:20,1:25}[c]
   else:
    hold_h={-1:1,0:2,1:3}[a]; rate={-1:10,0:15,1:20}[c]
   temp={-1:300,0:400,1:500}[b]
   out.append(dict(doi=DOI['zhou'],run=run,pollutant=pol,feedstock='water hyacinth',sample_id=f'ZHOU22_{pol}_{run:02d}',
                   x1=a,x2=b,x3=c,pyrolysis_temperature_C=temp,pyrolysis_hold_min=hold_h*60,heating_rate_C_min=rate,
                   pyrolysis_atmosphere='N2',C0_mg_L=50.,initial_pH=5.5,dose_g_L=2.,adsorption_T_C=25.,contact_time_min=240.,
                   qt_240min_mg_g=q,physical_cap_mg_g=25.,target_type='QT_240MIN_NOT_CERTIFIED_QE',
                   provenance='Zhou 2022 Table 2 p5; Methods p2-3',qa_status='SOURCE_TABLE_QT_RESEARCH_ONLY'))
 return pd.DataFrame(out)


def dedup_doe(df:pd.DataFrame) -> pd.DataFrame:
 keys=['pollutant','pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min']
 d=df.groupby(keys,as_index=False).agg(qt_240min_mg_g=('qt_240min_mg_g','mean'),n_replicates=('run','size'),
                                         qt_std_mg_g=('qt_240min_mg_g','std'))
 return d.sort_values(keys).reset_index(drop=True)


def make_v33_training(core:pd.DataFrame, v33_root:Path=V33) -> pd.DataFrame:
 ads=pd.read_csv(v33_root/'stage2_new15_single_metal_measured_or_derived_RESEARCH_STAGING.csv')
 mats=pd.read_csv(v33_root/'stage1_new30_temperature_series_RESEARCH_STAGING.csv')
 if len(ads)!=15 or ads.doi.nunique()!=2:raise RuntimeError('V33 adsorption inputs changed')
 if ads.observation_id.duplicated().any():raise RuntimeError('V33 observation IDs duplicated')
 if ads.material_id.isna().any():raise RuntimeError('Unlinked material')
 mcols=['material_id','feedstock','pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min','pyrolysis_atmosphere']
 ads=ads.merge(mats[mcols],on='material_id',validate='many_to_one')
 if len(ads)!=15:raise RuntimeError('Joining materials changed observation count')
 frame=pd.DataFrame(index=ads.index)
 frame['observation_id']=ads.observation_id
 frame['doi']=ads.doi
 frame['target_value']=ads.qe_mg_g
 frame['target_name']='qe_reported_mg_g'
 frame['pollutant']=ads.pollutant
 frame['C0_mg_L']=ads.C0_mg_L
 frame['dose_g_L']=ads.dose_g_L
 frame['initial_pH_raw_reported']=ads.initial_pH.astype(str)
 # Source pH reported as an interval (e.g. Wang pH 4.5–5.0): do not fabricate a precise pH=4.75.
 # Non-numeric reported ranges are preserved in the provenance column, numeric model input stays NA.
 frame['initial_pH']=pd.to_numeric(ads.initial_pH,errors='coerce')
 frame['adsorption_T_C']=ads.adsorption_temperature_C
 frame['contact_time_min']=ads.contact_time_min
 frame['feedstock_label_normalized']=ads.feedstock
 frame['manufacturing_route']='SINGLE_STAGE_PYROLYSIS'
 frame['process_step_count']=1
 frame['pyrolysis_temperature_C']=ads.pyrolysis_temperature_C
 frame['pyrolysis_hold_min']=ads.pyrolysis_hold_min
 frame['heating_rate_C_min']=ads.heating_rate_C_min
 frame['pyrolysis_atmosphere']=ads.pyrolysis_atmosphere
 frame['activation_or_modification']='none'
 frame['secondary_temperature_C']=np.nan
 frame['secondary_hold_min']=np.nan
 frame['mass_balance_cap_mg_g']=ads.C0_mg_L/ads.dose_g_L
 frame['source_target_status']=ads.qe_origin
 frame['model_use_tier']='V34_RESEARCH_CONDITIONAL_NOT_CERTIFIED'
 frame['source_material_id']=ads.material_id
 frame['source_pdf']=ads.source_pdf
 frame['source_page']=ads.source_pdf_page
 frame['raw_composition_basis']='NR'
 if frame['target_value'].isna().any() or (frame.target_value<0).any() or (frame.target_value>frame.mass_balance_cap_mg_g+1e-6).any():
  raise RuntimeError('V33 targets invalid physical condition')
 assert not (set(frame.doi) & set(core.doi))
 return frame


def clean_xy(df:pd.DataFrame,features:list[str]) -> pd.DataFrame:
 X=df.reindex(columns=features).copy()
 cats={'pollutant','feedstock_label_normalized','manufacturing_route','pyrolysis_atmosphere','activation_or_modification'}
 for col in X:
  if col in cats:
   X[col]=X[col].astype(object)
   X[col]=X[col].where(X[col].notna() & X[col].astype(str).str.strip().ne(''),np.nan)
  else:X[col]=pd.to_numeric(X[col],errors='coerce')
 return X


def train_refit_model(df:pd.DataFrame,model:str,algo:str):
 features=model_inputs(model)
 reg=validation_algorithms()[algo]()
 pipe=build_pipeline(reg,features)
 pipe.fit(clean_xy(df,features),df.target_value.to_numpy(float))
 return pipe


def cv_existing(core:pd.DataFrame,expanded:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
 rows=[]
 schemes=[('A','XGBoost'),('B','Ridge'),('B','XGBoost')]
 for model,algo in schemes:
  for doi in sorted(core.doi.unique()):
   test=core.loc[core.doi==doi]
   old=core.loc[core.doi!=doi]
   new=expanded.loc[expanded.doi!=doi]
   assert not (set(test.doi)&set(old.doi)) and not (set(test.doi)&set(new.doi))
   for kind,tr in [('old98',old),('expanded113',new)]:
    p=_stage2_fit_predict(tr,test,model,algo)
    for r,pr in zip(test.itertuples(index=False),p):
     rows.append(dict(model=model,algo=algo,doi=doi,observation_id=r.observation_id,training=kind,y=float(r.target_value),
                      pred_raw=float(pr),cap=float(r.C0_mg_L)/float(r.dose_g_L)))
 out=pd.DataFrame(rows)
 stat={}
 for (model,algo,kind),sub in out.groupby(['model','algo','training']):
  y=sub.y.to_numpy(float); p=sub.pred_raw.to_numpy(float)
  stat[f'{model}_{algo}_{kind}']=dict(n=len(sub),mae_mg_g=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),
                              physical_violations=int(((p<0)|(p>sub.cap)).sum()))
 for model,algo in schemes:
  sub=out[(out.model==model)&(out.algo==algo)]
  pv=sub.pivot(index=['doi','observation_id','y','cap'],columns='training',values='pred_raw').reset_index()
  pv.columns.name=None
  pv['err_original']=abs(pv.y-pv.old98);pv['err_expanded']=abs(pv.y-pv.expanded113)
  stat[f'{model}_{algo}_paired_bootstrap']=cluster_delta_ci(pv,reps=1000)
 return out,stat


def external_newdoi(core:pd.DataFrame,new15:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
 rows=[]
 for model,algo in [('A','XGBoost'),('B','Ridge'),('B','XGBoost')]:
  for doi in sorted(new15.doi.unique()):
   test=new15.loc[new15.doi==doi]
   for kind,train in [('old98_only',core),('other_new_doi_added',pd.concat([core,new15.loc[new15.doi!=doi]],ignore_index=True))]:
    assert doi not in set(train.doi)
    p=_stage2_fit_predict(train,test,model,algo)
    for row,pred in zip(test.itertuples(index=False),p):
     rows.append(dict(model=model,algo=algo,doi=doi,observation_id=row.observation_id,training=kind,
                      pollutant=row.pollutant,C0_mg_L=float(row.C0_mg_L),temperature_C=int(row.pyrolysis_temperature_C),
                      y=float(row.target_value),pred_raw=float(pred),cap=float(row.C0_mg_L)/float(row.dose_g_L)))
 out=pd.DataFrame(rows); metrics={}
 for (model,algo,train),sub in out.groupby(['model','algo','training']):
  y=sub.y.to_numpy(float);pred=sub.pred_raw.to_numpy(float)
  scores=[]
  for (doi,c0),g in sub.groupby(['doi','C0_mg_L']):
   if g.temperature_C.nunique()<2:continue
   gs=g.sort_values('temperature_C')
   if np.ptp(gs.pred_raw.values)<1e-9: top=None
   else:top=int(gs.iloc[np.argmax(gs.pred_raw)].temperature_C)
   actual=int(gs.iloc[np.argmax(gs.y)].temperature_C)
   regret=float(gs.y.max() - gs.loc[gs.temperature_C.eq(top),'y'].iloc[0]) if top is not None else None
   scores.append(dict(doi=doi,c0=c0,actual_top_temperature_C=actual,pred_top_temperature_C=top,top1=(actual==top if top is not None else None),regret_mg_g=regret))
  metrics[f'{model}_{algo}_{train}']=dict(n=len(sub),mae_mg_g=float(mean_absolute_error(y,pred)),
     violations=int(((pred<0)|(pred>sub.cap)).sum()),temp_group_count=len(scores),
     top1_correct=sum(s['top1'] is True for s in scores),top1_evaluable=sum(s['top1'] is not None for s in scores),
     rankings=scores)
 return out,metrics


def bbd_loocv(unique:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
 rows=[]; stats={}; fields=['pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min']
 for pol,sub in unique.groupby('pollutant',sort=True):
  sub=sub.reset_index(drop=True); x=sub[fields].to_numpy(float);y=sub.qt_240min_mg_g.to_numpy(float)
  for algo in ['Mean','LinearRidge','QuadraticRidge']:
   for j in range(len(sub)):
    tr=np.arange(len(sub))!=j
    if algo=='Mean':pred=y[tr].mean()
    else:
     if algo=='LinearRidge':reg=make_pipeline(StandardScaler(),Ridge(alpha=1.))
     else:reg=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=1.))
     reg.fit(x[tr],y[tr]);pred=float(reg.predict(x[[j]])[0])
    r=sub.iloc[j]
    rows.append(dict(pollutant=pol,algorithm=algo,temperature_C=float(r.pyrolysis_temperature_C),hold_min=float(r.pyrolysis_hold_min),
      heating_rate_C_min=float(r.heating_rate_C_min),qt_obs=float(y[j]),qt_pred_loocv=float(pred),
      n_replicates=int(r.n_replicates)))
 out=pd.DataFrame(rows)
 for (pol,algo),g in out.groupby(['pollutant','algorithm']):
  yy=g.qt_obs.to_numpy(float);pp=g.qt_pred_loocv.to_numpy(float)
  stats[f'{pol}_{algo}']=dict(n=len(g),MAE=float(mean_absolute_error(yy,pp)),R2=float(r2_score(yy,pp)),
   top1_actual_C=float(g.iloc[yy.argmax()].temperature_C),top1_pred_C=float(g.iloc[pp.argmax()].temperature_C),
   physical_violations=int(((pp<0)|(pp>25)).sum()))
 return out,stats


def main(root:Path=ROOT) -> dict:
 global OUT,DATA,V33
 root=Path(root);OUT=root/'reports/v34_temperature';DATA=root/'data/imports/v34_temperature';V33=root/'data/imports/v33_five_papers'
 OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
 _,core=stage2_targets()
 if len(core)!=98 or core.doi.nunique()!=26:raise RuntimeError('Core98 must remain unchanged')
 registry=audit_source_registry()
 if len(set(registry.doi)&set(core.doi)) !=0:raise RuntimeError('Duplicate DOI in source registry/core98')
 registry.to_csv(DATA/'source_registry.csv',index=False)
 bet=new_bet();bet.to_csv(DATA/'stage1_new6_BET_only_RESEARCH_STAGING.csv',index=False)
 # The v33 327-row master remains the master source. Append six BET-only rows, no inferred TPV.
 v33master=pd.read_csv(V33/'Stage1_Master_327_RESEARCH_STAGING_NOT_TRAINED.csv',low_memory=False)
 if len(v33master)!=327 or v33master.material_id.duplicated().any():raise RuntimeError('v33 Stage1 baseline inconsistent')
 if set(v33master.doi)&set(bet.doi):raise RuntimeError('v34 Stage1 DOI unexpectedly already exists in v33')
 aligned=pd.DataFrame(index=bet.index,columns=v33master.columns)
 for k in ['material_id','doi','feedstock','pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min','pyrolysis_atmosphere']:
  aligned[k]=bet[k]
 aligned['feedstock_group']=bet.feedstock
 aligned['original_sample_name']=bet.sample_name
 aligned['BET_m2_g_original_source']=bet.BET_m2_g
 aligned['BET_m2_g_training_candidate']=bet.BET_m2_g
 aligned['BET_candidate_flag']='PENDING_SOURCE_APPROVAL'
 aligned['TPV_candidate_flag']='NOT_REPORTED'
 aligned['release_status']='RESEARCH_STAGING_NOT_TRAINED'
 aligned['source_stage']='V34_RESEARCH_STAGING'
 aligned['source_reference']=bet.source_table
 aligned['source_pdf_page']=bet.source_table
 aligned['input_verification_limitations']='Feedstock raw composition NR; biochar elemental not substituted; TPV NR'
 master333=pd.concat([v33master,aligned],ignore_index=True)
 assert len(master333)==333 and master333.material_id.is_unique
 master333.to_csv(DATA/'Stage1_Master_333_RESEARCH_STAGING_NOT_TRAINED.csv',index=False)
 cedar=cedar_qmax();cedar.to_csv(DATA/'cedar_qmax20_LANGMUIR_SEPARATE.csv',index=False)
 raw=doeraw();raw.to_csv(DATA/'zhou_BBD_raw34_QT_240min_NOT_QE.csv',index=False)
 unique=dedup_doe(raw);unique.to_csv(DATA/'zhou_BBD_unique26_QT_240min_NOT_QE.csv',index=False)
 assert len(raw)==34 and len(unique)==26 and unique.groupby('pollutant').size().eq(13).all()
 assert ((raw.qt_240min_mg_g>=0)&(raw.qt_240min_mg_g<=raw.physical_cap_mg_g)).all()
 new15=make_v33_training(core)
 new15.to_csv(DATA/'v33_new15_model_B_feature_bridge_RESEARCH.csv',index=False)
 expanded=pd.concat([core,new15],ignore_index=True,sort=False)
 assert len(expanded)==113 and expanded.doi.nunique()==28 and expanded.observation_id.is_unique
 expanded.to_csv(DATA/'stage2_core98_plus_v33_15_RESEARCH_NOT_APPROVED.csv',index=False)
 cv,paired=cv_existing(core,expanded)
 cv.to_csv(OUT/'core98_paired_DOI_OOF.csv',index=False)
 external,extstat=external_newdoi(core,new15)
 external.to_csv(OUT/'new_DOI_external_temperature_predictions.csv',index=False)
 bbdpred,bbdstats=bbd_loocv(unique)
 bbdpred.to_csv(OUT/'Zhou_BBD_unique26_LOO_pred.csv',index=False)
 research=root/'deployment/research_v34';research.mkdir(parents=True,exist_ok=True)
 # Explicitly inactive research model files only. sklearn model is not portable across versions.
 for model,algo in [('A','XGBoost'),('B','Ridge'),('B','XGBoost')]:
  pipe=train_refit_model(expanded,model,algo)
  joblib.dump(pipe,research/f'stage2_{model}_{algo}_113_RESEARCH_ONLY.joblib',compress=3)
 meta={'version':'v34','training_rows':113,'doi':28,'targets':'reported experimental uptake only; strict qe not certified',
       'no_zhou_qt_mixing':True,'deployed':False,'active_json_unchanged':True,
       'sklearn_version':__import__('sklearn').__version__,'algorithm_recipe':'v16_validate.validation_algorithms()'
       }
 (research/'RESEARCH_ONLY_MODEL_METADATA.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 # Train the standalone BBD quadratic surrogate on the 13 UNIQUE designs/ion, not raw repeats.
 doe_models={}
 for pol,g in unique.groupby('pollutant'):
  p=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=1.))
  p.fit(g[['pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min']].to_numpy(float),g.qt_240min_mg_g.to_numpy(float))
  doe_models[pol]=p
 joblib.dump(doe_models,research/'Zhou2022_BBD_QT240min_SURROGATES_DOI_INTERNAL_ONLY.joblib',compress=3)
 result={'v34_status':'RESEARCH_TRAINING_COMPLETED_NOT_DEPLOYED',
   'uploaded_zip_doifiles':5,'new_doi_in_zip':4,'duplicate_jia':DOI['jia_duplicate'],
   'new_stage1_bet':6,'new_stage1_tpv':0,'new_cedar_qmax':20,
   'zhou_raw_qt':34,'zhou_unique_design_qt':26,'zhou_independent_doi':1,
   'core_baseline':{'rows':98,'doi':26},'stage2_expanded_qe_research':{'rows':113,'doi':28,'v33_added_rows':15,'v34_qt_added_to_qe':0},
   'paired_existing_doi':paired,'new_doi_external':extstat,'zhou_within_DOI_BBD_LOO':bbdstats,
   'deployed_model_updated':False,'notes':['No duplicate Jia','Not equivalent to independently certified qe','BBD 240-min uptake trained SEPARATELY',
   'No choice of best model based on these external DOI evaluations','Model files require compatible sklearn runtime']}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 return result

if __name__=='__main__':
 result=main()
 print(json.dumps({'stage2':result['stage2_expanded_qe_research'],'paired':result['paired_existing_doi'],'new_doi':result['new_doi_external'],'bbd':result['zhou_within_DOI_BBD_LOO']},ensure_ascii=False,indent=2))
