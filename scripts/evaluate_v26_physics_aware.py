"""v26 preregistered research comparison: log target Stage 1 and bounded f Stage 2.

No model deployment, no v24 source training, no model selection from external DOI.
Run: python -m scripts.evaluate_v26_physics_aware
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from scripts.compare_phase8_v23 import (
    stage1_targets, stage1_pipe, input_frame, S1_NUMERIC, S1_CATEGORICAL,
    stage2_targets, _stage2_fit_predict, cluster_delta_ci,
)
from scripts.evaluate_v24_v25 import v24_external_stage1, v24_external_stage2, scores
from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2.v15_models import build_pipeline

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v26_physics_aware'
STAGE1_TARGETS={'BET': ('BET_training_use','BET_m2_g_training_candidate','BET_m2_g_original_source',1.0),
                'TPV': ('TPV_training_use','TPV_cm3_g_training_candidate','TPV_cm3_g_original_source',100.0)}


def safe_f(data: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    c0=pd.to_numeric(data[fc.C0_COL],errors='raise').to_numpy(dtype=float)
    dose=pd.to_numeric(data[fc.DOSE_COL],errors='raise').to_numpy(dtype=float)
    if not (np.all(c0>0) and np.all(dose>0)):
        raise ValueError('Non-positive physical cap components')
    cap=c0/dose
    y=pd.to_numeric(data[fc.TARGET_COLUMN],errors='raise').to_numpy(dtype=float)
    f=y/cap
    if not np.all(np.isfinite(f)) or np.any((f < -1e-8)|(f>1+1e-8)):
        raise ValueError('Unphysical training target: exclude or audit source first')
    return np.clip(f,0,1),cap


def log_target_model(scale:float):
    return TransformedTargetRegressor(
        regressor=stage1_pipe(),
        func=lambda y: np.log1p(np.asarray(y,dtype=float)*scale),
        inverse_func=lambda z: np.expm1(np.asarray(z,dtype=float))/scale,
        check_inverse=True,
    )


def prepared_stage1_external(root:Path) -> pd.DataFrame:
    # Reuse v25's explicit auditable route mapping, without labels entering training.
    src=root/'data/imports/v24_source_audit'
    new=pd.read_csv(src/'stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    master=pd.read_csv(src/'Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv',low_memory=False)
    ext=master[master.material_id.astype(str).str.startswith('S1ADD24_')].copy()
    lookup=new[['doi','sample_name','manufacturing_route']].rename(columns={'sample_name':'original_sample_name'})
    ext=ext.merge(lookup,on=['doi','original_sample_name'],validate='one_to_one')
    assert len(ext)==33 and ext.material_id.is_unique and ext.doi.nunique()==6
    route=ext['manufacturing_route'].fillna('').astype(str)
    ext.loc[route.eq('single_stage_pyrolysis'),'process_stage']='single_stage_pyrolysis'
    ext.loc[route.eq('single_stage_pyrolysis'),'v59_route_family']='none'
    ext.loc[route.eq('single_stage_pyrolysis'),'activation_method']='none'
    ext.loc[route.eq('co_pyrolysis_biomass_plastic'),'process_stage']='co_pyrolysis_biomass_plastic'
    ext.loc[route.eq('co_pyrolysis_biomass_plastic'),'v59_route_family']='none'
    ext.loc[route.eq('co_pyrolysis_biomass_plastic'),'activation_method']='none'
    ext.loc[route.eq('post_pyrolysis_modification'),'process_stage']='post_pyrolysis_modification'
    ext.loc[route.eq('post_pyrolysis_modification'),'v59_route_family']='chemical'
    ext.loc[route.eq('post_pyrolysis_modification'),'activation_method']='post_pyrolysis_chemical_modification'
    for col in ['process_stage','v59_route_family','activation_method']:
        ext.loc[route.str.contains('hydrothermal|HTC',case=False),col]='__UNSUPPORTED_HTC__'
    return ext


def stage1_comparison(root:Path):
    _,data=stage1_targets()
    ext=prepared_stage1_external(root)
    assert not (set(data.doi)&set(ext.doi))
    rows=[]; ext_rows=[]; totals={}
    for target,(role,ycol,external_col,scale) in STAGE1_TARGETS.items():
        tr=data[data[role].eq(True)&pd.to_numeric(data[ycol],errors='coerce').notna()].copy()
        xext=input_frame(ext,S1_NUMERIC,S1_CATEGORICAL)
        for doi in sorted(tr.doi.unique()):
            train=tr[tr.doi.ne(doi)]; test=tr[tr.doi.eq(doi)]
            assert not(set(train.doi)&set(test.doi))
            xx=input_frame(train,S1_NUMERIC,S1_CATEGORICAL)
            xt=input_frame(test,S1_NUMERIC,S1_CATEGORICAL)
            ref=stage1_pipe().fit(xx,train[ycol].astype(float)).predict(xt)
            logged=log_target_model(scale).fit(xx,train[ycol].astype(float).to_numpy()).predict(xt)
            for idx,row in enumerate(test.itertuples(index=False)):
                rows.append({'target':target,'doi':doi,'material_id':row.material_id,
                             'actual':float(getattr(row,ycol)),'raw_rf':float(ref[idx]),'log_rf':float(logged[idx])})
        xx=input_frame(tr,S1_NUMERIC,S1_CATEGORICAL)
        log_model=log_target_model(scale).fit(xx,tr[ycol].astype(float).to_numpy())
        pr=log_model.predict(xext)
        y=pd.to_numeric(ext[external_col],errors='raise').to_numpy(dtype=float)
        for ix,(_,row) in enumerate(ext.iterrows()):
            ext_rows.append({'target':target,'doi':row.doi,'sample':row.original_sample_name,
                             'route':row.manufacturing_route,'is_single_stage':row.manufacturing_route=='single_stage_pyrolysis',
                             'actual':float(y[ix]),'log_rf':float(pr[ix])})
    oof=pd.DataFrame(rows); external=pd.DataFrame(ext_rows)
    ref_external,_=v24_external_stage1(root)
    for t in STAGE1_TARGETS:
        ref=ref_external[ref_external.target.eq(t)]
        old=oof[oof.target.eq(t)].copy()
        old['err_original']=(old.actual-old.raw_rf).abs()
        old['err_expanded']=(old.actual-old.log_rf).abs()
        totals[t]={
            'lodo_raw':scores(old.actual,old.raw_rf),'lodo_log':scores(old.actual,old.log_rf),
            'doi_cluster_bootstrap_delta_log_minus_raw':cluster_delta_ci(old),
            'lodo_low_range':{'threshold':30 if t=='BET' else 0.03,
                              'raw':scores(old.loc[old.actual <= (30 if t=='BET' else .03),'actual'],old.loc[old.actual <= (30 if t=='BET' else .03),'raw_rf']),
                              'log':scores(old.loc[old.actual <= (30 if t=='BET' else .03),'actual'],old.loc[old.actual <= (30 if t=='BET' else .03),'log_rf'])},
        }
        matched=external[external.target.eq(t)].merge(ref[['doi','sample','pred']].rename(columns={'pred':'raw_rf_v25'}),on=['doi','sample'],validate='one_to_one')
        totals[t]['external33_raw']=scores(matched.actual,matched.raw_rf_v25)
        totals[t]['external33_log']=scores(matched.actual,matched.log_rf)
        solo=matched[matched.is_single_stage]
        totals[t]['external_single_stage_raw']=scores(solo.actual,solo.raw_rf_v25)
        totals[t]['external_single_stage_log']=scores(solo.actual,solo.log_rf)
        totals[t]['external_low_range']={
            'threshold':30 if t=='BET' else .03,
            'raw':scores(matched.loc[matched.actual <= (30 if t=='BET' else .03),'actual'], matched.loc[matched.actual <= (30 if t=='BET' else .03),'raw_rf_v25']),
            'log':scores(matched.loc[matched.actual <= (30 if t=='BET' else .03),'actual'], matched.loc[matched.actual <= (30 if t=='BET' else .03),'log_rf']),
        }
    # Use trusted v25 benchmark aligned by material identity rather than dataframe order.
    external=external.merge(ref_external[['target','doi','sample','pred']].rename(columns={'pred':'raw_rf_v25'}),on=['target','doi','sample'],validate='one_to_one')
    return oof,external,totals


def inputs_for_model_a(train:pd.DataFrame,test:pd.DataFrame):
    names=fc.model_inputs('A')
    xtrain=train.reindex(columns=names).copy()
    xtest=test.reindex(columns=names).copy()
    for col in names:
        if col=='pollutant':
            xtrain[col]=xtrain[col].replace('',np.nan).astype('object')
            xtest[col]=xtest[col].replace('',np.nan).astype('object')
        else:
            xtrain[col]=pd.to_numeric(xtrain[col],errors='coerce')
            xtest[col]=pd.to_numeric(xtest[col],errors='coerce')
    return xtrain,xtest,names


def bounded_predictions(train:pd.DataFrame,test:pd.DataFrame):
    ftr,_=safe_f(train)
    xtr,xte,names=inputs_for_model_a(train,test)
    # Prespecified fractional-link model; no information from the holdout fit or preprocess.
    epsilon=.01
    z=np.log(np.clip(ftr,epsilon,1-epsilon)/(1-np.clip(ftr,epsilon,1-epsilon)))
    est=build_pipeline(Ridge(alpha=10.0),names)
    est.fit(xtr,z)
    pred_z=np.asarray(est.predict(xte),dtype=float)
    prob=np.where(pred_z>=0,1/(1+np.exp(-np.minimum(pred_z,700))),np.exp(np.maximum(pred_z,-700))/(1+np.exp(np.maximum(pred_z,-700))))
    _,cap_te=safe_f(test)
    return {'fraction_logit_ridge':cap_te*prob,
            'fraction_mean_baseline':cap_te*float(ftr.mean()),
            'full_removal_baseline':cap_te,
            'predicted_fraction':prob}


def stage2_comparison(root:Path):
    _,train=stage2_targets()
    staged=pd.read_csv(root/'data/imports/v24_source_audit/Stage2_Core_167_RESEARCH_STAGING_NOT_TRAINED.csv')
    ext=staged[staged.observation_id.astype(str).str.startswith('S2ADD24_') & staged.doi.eq('10.3390/ma14216566')].copy()
    assert len(ext)==51 and not (set(train.doi)&set(ext.doi))
    out=[]
    for doi in sorted(train.doi.unique()):
        tr=train[train.doi.ne(doi)]; te=train[train.doi.eq(doi)]
        assert len(tr)>0 and len(te)>0
        pr=bounded_predictions(tr,te)
        direct=_stage2_fit_predict(tr,te,'A','XGBoost')
        y=pd.to_numeric(te.target_value).to_numpy(dtype=float)
        f,cap=safe_f(te)
        for i,(_,row) in enumerate(te.iterrows()):
            out.append({'doi':doi,'observation_id':row.observation_id,'cohort':'lodo_core98',
                        'pollutant':row.pollutant,'actual':float(y[i]),'cap':float(cap[i]),
                        'f_actual':float(f[i]),'direct_xgb_raw':float(direct[i]),
                        'direct_xgb_clipped':float(np.clip(direct[i],0,cap[i])),
                        'fraction_logit_ridge':float(pr['fraction_logit_ridge'][i]),
                        'fraction_mean_baseline':float(pr['fraction_mean_baseline'][i]),
                        'full_removal_baseline':float(pr['full_removal_baseline'][i]),
                        'f_predicted':float(pr['predicted_fraction'][i]),'dose_g_L':float(row.dose_g_L)})
    lodo=pd.DataFrame(out)
    pr=bounded_predictions(train,ext)
    prior=v24_external_stage2(root)[0]
    prior=prior[prior.model.eq('A')][['observation_id','pred_raw_mg_g']].rename(columns={'pred_raw_mg_g':'direct_xgb_raw'})
    ee=ext.merge(prior,on='observation_id',validate='one_to_one')
    _,cap=safe_f(ee)
    external=[]
    for i,(_,row) in enumerate(ee.iterrows()):
        y=float(row.target_value)
        external.append({'doi':row.doi,'observation_id':row.observation_id,'cohort':'poultry51_ood_not_generalization',
                         'pollutant':row.pollutant,'actual':y,'cap':float(cap[i]),'f_actual':y/cap[i],
                         'direct_xgb_raw':float(row.direct_xgb_raw),
                         'direct_xgb_clipped':float(np.clip(row.direct_xgb_raw,0,cap[i])),
                         'fraction_logit_ridge':float(pr['fraction_logit_ridge'][i]),
                         'fraction_mean_baseline':float(pr['fraction_mean_baseline'][i]),
                         'full_removal_baseline':float(pr['full_removal_baseline'][i]),
                         'f_predicted':float(pr['predicted_fraction'][i]),'dose_g_L':float(row.dose_g_L)})
    external=pd.DataFrame(external)
    summary={}
    models=['direct_xgb_raw','direct_xgb_clipped','fraction_logit_ridge','fraction_mean_baseline','full_removal_baseline']
    for name,df in [('lodo_core98',lodo),('external_poultry51_ood',external)]:
        summary[name]={'rows':len(df),'dois':int(df.doi.nunique()),
                       'methods':{m:{**scores(df.actual,df[m]),
                            'f_mae_pct_points':float(100*np.mean(np.abs(df.actual/df.cap-df[m]/df.cap))),
                            'physical_violation_pct':float(100*np.mean((df[m]<0)|(df[m]>df.cap)))} for m in models},
                       'bound_warning': 'Poultry is one DOI, 100 g/L, untrained feedstock and Zn pollutant' if name.startswith('external') else 'Target not strict equilibrium certified'}
    return lodo,external,summary


def run(root=ROOT,out=OUT):
    root=Path(root); out=Path(out)
    s1,ext1,sum1=stage1_comparison(root)
    s2,ext2,sum2=stage2_comparison(root)
    out.mkdir(parents=True,exist_ok=True)
    s1.to_csv(out/'stage1_lodo_log_vs_raw.csv',index=False)
    ext1.to_csv(out/'stage1_external33_log_vs_raw.csv',index=False)
    s2.to_csv(out/'stage2_lodo_bounded_fraction.csv',index=False)
    ext2.to_csv(out/'stage2_external51_bounded_fraction.csv',index=False)
    summary={'scope':'v26 research-only; no deployment/retraining/staging approval',
             'stage1':sum1,'stage2':sum2,
             'interpretation':['Externally evaluated alternatives cannot be advertised as validated after exploratory selection',
                               'v24 data remains excluded from training',
                               'New raw composition not used because basis incomplete',
                               'DOI group is unit of uncertainty; repeated rows not independent',
                               'Clipped qe is not proof of predictive skill; always show full removal baseline',
                               'Stage1->2 model D remains inactive']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return summary

if __name__=='__main__':
    res=run()
    print(json.dumps({'Stage1':{t:{k:round(v['mae'],4) if isinstance(v,dict) and v.get('mae') is not None else None for k,v in d.items() if k in ('lodo_raw','lodo_log','external33_raw','external33_log')} for t,d in res['stage1'].items()},
                       'Stage2':{k:{m:round(v['mae'],4) for m,v in d['methods'].items()} for k,d in res['stage2'].items()}},ensure_ascii=False,indent=2))
