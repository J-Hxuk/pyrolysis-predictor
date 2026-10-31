"""v58 research-only diagnostic: why v57 36-row augmentation transfers poorly.
Run from v47 project root: python -m scripts.run_v58_failure_diagnostics --root .
No active model, source CSV, or pre-existing result is overwritten.
"""
from __future__ import annotations
import argparse
import json
import warnings
import hashlib
import platform
import joblib
import sklearn
import xgboost
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.build_v46_phase8 import s2_model, features

ROOT_OLD="data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv"
ROOT_V55="data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv"
ROOT_NEW="data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv"
ROOT_OOF="reports/v57_handoff_cumulative_refit/stage2_fixed140_sameDOI_predictions.csv"
ROOT_SOURCE_OOF="reports/v57_handoff_cumulative_refit/stage2_new_DOI_sourceout_predictions.csv"
REPORT="reports/v58_failure_diagnostics"
SEED=20261031

def safe_float(x):
    v=float(x)
    return None if not np.isfinite(v) else v

def derive(frame,kind):
    df=frame.copy()
    if kind!='baseline':
        c0=pd.to_numeric(df.C0_mg_L,errors='coerce').to_numpy(float)
        dose=pd.to_numeric(df.dose_g_L,errors='coerce').to_numpy(float)
        if np.any(dose<=0) or np.any(c0<=0):
            raise ValueError('Nonpositive C0/dose supplied to physical feature')
        df['physical_cap_mg_g']=c0/dose
        if kind=='cap_log':
            df['log_physical_cap']=np.log1p(df['physical_cap_mg_g'])
    return df

def build(variant,kind):
    pipe,cols,nums,cats=s2_model(variant)
    if kind!='baseline':
        extra=['physical_cap_mg_g']
        if kind=='cap_log': extra.append('log_physical_cap')
        transformers=pipe.named_steps['pre'].transformers
        pre=ColumnTransformer([
            ('num',transformers[0][1],nums+extra),
            ('cat',transformers[1][1],cats)
        ])
        pipe.set_params(pre=pre)
        cols=cols+extra
        nums=nums+extra
    return pipe,cols,nums,cats

def fit_predict(train,test,variant,kind):
    tr=derive(train,kind);te=derive(test,kind)
    pipe,cols,nums,cats=build(variant,kind)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=FutureWarning)
        pipe.fit(features(tr,cols,nums,cats),tr.target_value.to_numpy(float))
        raw=np.asarray(pipe.predict(features(te,cols,nums,cats)),float)
    cap=te.C0_mg_L.to_numpy(float)/te.dose_g_L.to_numpy(float)
    return np.clip(raw,0,cap),raw

def metric(y,p):
    return {'mae_mg_g':float(mean_absolute_error(y,p)), 'rmse_mg_g':float(np.sqrt(mean_squared_error(y,p)))}

def confidence(x,p0,p1,groups,boot_n=1600):
    errors=np.abs(x-p0)-np.abs(x-p1)
    group_ids=np.unique(groups)
    ii={g:np.flatnonzero(groups==g) for g in group_ids}
    rng=np.random.default_rng(SEED)
    res=[]
    for i in range(boot_n):
        sample=rng.choice(group_ids,len(group_ids),replace=True)
        index=np.concatenate([ii[g] for g in sample])
        res.append(float(np.mean(errors[index])))
    return float(np.quantile(res,.025)),float(np.quantile(res,.975))

def main(root):
    root=Path(root).resolve()
    out=root/REPORT
    out.mkdir(exist_ok=True,parents=True)
    base=pd.read_csv(root/ROOT_OLD,low_memory=False).reset_index(drop=True)
    v55=pd.read_csv(root/ROOT_V55,low_memory=False).reset_index(drop=True)
    new=pd.read_csv(root/ROOT_NEW,low_memory=False).reset_index(drop=True)
    oof=pd.read_csv(root/ROOT_OOF,low_memory=False).reset_index(drop=True)
    src=pd.read_csv(root/ROOT_SOURCE_OOF,low_memory=False)
    assert len(base)==140 and len(v55)==152 and len(new)==36 and base.doi.nunique()==32
    assert list(v55.iloc[:140].observation_id)==list(base.observation_id)
    assert list(base.observation_id)==list(oof.observation_id)
    assert not set(new.doi)&set(v55.doi)
    assert new.doi.value_counts().sort_values().tolist()==[7,29]
    assert set(src.scenario.unique())>= {'v55_152'}
    addon=v55.iloc[140:]
    existing_cols=[]
    for m in ['A','B']:
        for sce in ['v55_152','v57_159_cao_only','v57_181_shen_only','v57_188_both']:
            existing_cols.append(f'{m}__{sce}')
    assert all(c in oof for c in existing_cols)

    # 1. Real, archived OOF diagnostic: quantify which old DOI and pollutant cohorts lose accuracy.
    errors=[]
    for m in ['A','B']:
        a=oof[f'{m}__v55_152'].to_numpy(float)
        b=oof[f'{m}__v57_188_both'].to_numpy(float)
        for i,r in base.iterrows():
            errors.append({'model':m,'observation_id':r.observation_id,'doi':r.doi,
             'pollutant':r.pollutant,'true_qe_mg_g':r.target_value,'old_pred':a[i],
             'all36_pred':b[i],'old_ae':abs(r.target_value-a[i]),'all36_ae':abs(r.target_value-b[i]),
             'change_ae_positive_is_worse':abs(r.target_value-b[i])-abs(r.target_value-a[i])})
    err=pd.DataFrame(errors);err.to_csv(out/'archived_row_error_deltas.csv',index=False)
    doi=err.groupby(['model','doi'],as_index=False).agg(n=('observation_id','size'),old_mae=('old_ae','mean'),
        all36_mae=('all36_ae','mean'),delta_total=('change_ae_positive_is_worse','sum'),delta_avg=('change_ae_positive_is_worse','mean'))
    doi.sort_values(['model','delta_total'],ascending=[True,False]).to_csv(out/'error_by_DOI.csv',index=False)
    pollutant=err.groupby(['model','pollutant'],as_index=False).agg(n=('observation_id','size'),old_mae=('old_ae','mean'),all36_mae=('all36_ae','mean'))
    pollutant['delta']=pollutant.all36_mae-pollutant.old_mae
    pollutant.to_csv(out/'error_by_pollutant.csv',index=False)
    material_profile=new.groupby(['doi','material_uid'],as_index=False).agg(
        n=('observation_id','size'),target_mean=('target_value','mean'),C0_min=('C0_mg_L','min'),
        C0_max=('C0_mg_L','max'))
    material_profile.to_csv(out/'new_material_repeated_measurements.csv',index=False)

    srcbase=src[src.scenario=='v55_152'].copy()
    srcbase=srcbase.merge(new[['observation_id','dose_g_L']],on='observation_id',validate='many_to_one')
    srcbase['cap']=srcbase.C0_mg_L/srcbase.dose_g_L
    srcbase['is_capped']=np.isclose(srcbase.pred,srcbase.cap,atol=1e-6,rtol=0)
    src_metrics=[]
    for (source,m),t in srcbase.groupby(['source','model']):
        src_metrics.append(dict(source=source,model=m,n=len(t),materials=int(t.material_uid.nunique()),
          capped_count=int(t.is_capped.sum()),mae_mg_g=float(np.abs(t.pred-t.target_value).mean()),
          zero_baseline_mae=float(np.abs(t.target_value).mean()),
          target_mean=float(t.target_value.mean()),pred_mean=float(t.pred.mean()),
          pred_min=float(t.pred.min()),pred_max=float(t.pred.max())))
    src_df=pd.DataFrame(src_metrics)
    src_df.to_csv(out/'new_DOI_sourceout_saturation.csv',index=False)

    # 2. Physical feature ablation: cap and log(cap) only use prior-known lab inputs, no fitted qe.
    variants=['A','B']
    kinds=['cap','cap_log']
    scenarios={'v55_152':addon,'v57_188_both':pd.concat([addon,new],ignore_index=True)}
    pred=base[['observation_id','doi','pollutant','target_value','C0_mg_L','dose_g_L']].copy()
    for tr,te in GroupKFold(n_splits=5).split(base,groups=base.doi):
        held=set(base.iloc[te].doi)
        for sc,extra in scenarios.items():
            e=extra[~extra.doi.isin(held)]
            assert not set(e.doi)&held
            train=pd.concat([base.iloc[tr],e],ignore_index=True,sort=False)
            for m in variants:
                for k in kinds:
                    label=f'{m}__{sc}__{k}'
                    pp,raw=fit_predict(train,base.iloc[te],m,k)
                    if label not in pred: pred[label]=np.nan;pred[label+'_raw']=np.nan
                    pred.loc[te,label]=pp
                    pred.loc[te,label+'_raw']=raw
    pred.to_csv(out/'derived_cap_feature_fixed140_oof.csv',index=False)
    baseline={m:oof[f'{m}__v55_152'].to_numpy(float) for m in variants}
    rows=[]
    truth=base.target_value.to_numpy(float)
    cap=(base.C0_mg_L/base.dose_g_L).to_numpy(float)
    for m in variants:
        rows.append(dict(model=m,scenario='v55_152',feature_set='original_archived',n=len(base),
            raw_cap_exceeded=int((oof[f'{m}__v55_152_raw']>cap).sum()),
            bootstrap_delta95_low=None,bootstrap_delta95_high=None,**metric(truth,baseline[m])))
        for sc in scenarios:
            for k in kinds:
                col=f'{m}__{sc}__{k}'
                p=pred[col].to_numpy(float);raw=pred[col+'_raw'].to_numpy(float)
                low,high=confidence(truth,baseline[m],p,base.doi.to_numpy())
                rows.append(dict(model=m,scenario=sc,feature_set=k,n=len(base),
                   raw_cap_exceeded=int((raw>cap).sum()),bootstrap_delta95_low=low,
                   bootstrap_delta95_high=high,**metric(truth,p)))
    tbl=pd.DataFrame(rows)
    tbl.to_csv(out/'derived_cap_feature_fixed140_metrics.csv',index=False)

    # 3. Source-out ablation: no observation from the tested DOI in training.
    rows=[];sop=[]
    for source,doi_value in [('shen','10.1007/s11356-017-8847-2'),('cao','10.1016/j.scitotenv.2019.07.102')]:
        test=new[new.doi==doi_value].copy()
        other=new[new.doi!=doi_value].copy()
        for m in variants:
            for k in kinds:
                for sc,train in [('v55_152',v55),('v55_152_plus_other_DOI',pd.concat([v55,other],ignore_index=True))]:
                    assert doi_value not in set(train.doi)
                    p,raw=fit_predict(train,test,m,k)
                    capacity=(test.C0_mg_L/test.dose_g_L).to_numpy(float)
                    rows.append(dict(source=source,model=m,feature_set=k,training=sc,n=len(test),
                        cap_clipped=int(np.isclose(p,capacity,atol=1e-6).sum()),
                        raw_cap_exceeded=int((raw>capacity).sum()),
                        zero_baseline_mae=float(np.abs(test.target_value).mean()),**metric(test.target_value,p)))
                    for id,y,pred0,raw0 in zip(test.observation_id,test.target_value,p,raw):
                        sop.append(dict(source=source,model=m,feature_set=k,training=sc,observation_id=id,
                            target_value=y,pred=pred0,pred_raw=raw0))
    so_df=pd.DataFrame(rows);so_df.to_csv(out/'cap_feature_new_DOI_sourceout_metrics.csv',index=False)
    pd.DataFrame(sop).to_csv(out/'cap_feature_new_DOI_sourceout_predictions.csv',index=False)

    # 4. Independent target-type: in-paper 16 manufacturing settings (never claim external or blind).
    from scripts.run_v54_bounded_rank import chen_features, rank_metrics, CHEN
    chen=pd.read_csv(root/CHEN)
    cinp=chen_features(chen)  # pH remains NR (median-imputed inside fold training pipeline)
    def strict_top_metrics(ids, y, yhat):
        y=np.asarray(y,float); yhat=np.asarray(yhat,float)
        mask=np.isclose(yhat,np.max(yhat),atol=1e-8,rtol=1e-8)
        n=int(mask.sum())
        best=float(np.max(y))
        return dict(top_tie_count=n,
                    strict_unique_top1_id=ids[int(np.argmax(yhat))] if n==1 else None,
                    strict_top1_regret=(best-float(y[mask][0])) if n==1 else None,
                    tied_top_regret_min=best-float(np.max(y[mask])),
                    tied_top_regret_max=best-float(np.min(y[mask])),
                    top1_unresolved_due_to_tie=bool(n>1))
    rank_rows=[]
    rank_predictions=chen[['material_uid','reported_removal_pct','temperature_C','hold_min','heating_rate_C_min']].copy()
    for m in variants:
        for k in ['baseline']+kinds:
            trained=derive(v55,k); evaluated=derive(cinp,k)
            pipeline,cols,nums,cats=build(m,k)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',FutureWarning)
                pipeline.fit(features(trained,cols,nums,cats),trained.target_value.to_numpy(float))
                raw=np.asarray(pipeline.predict(features(evaluated,cols,nums,cats)),float)
            cap=(cinp.C0_mg_L/cinp.dose_g_L).to_numpy(float)
            pred=np.clip(raw,0,cap)
            estimated=100*pred/cap
            rank_predictions[f'{m}__{k}']=estimated
            rm=rank_metrics(chen.material_uid.to_list(),chen.reported_removal_pct.to_numpy(float),estimated)
            strict=strict_top_metrics(chen.material_uid.to_list(),chen.reported_removal_pct.to_numpy(float),estimated)
            rank_rows.append(dict(model=m,feature_set=k,training='v55_152',
              status='POSTHOC_IN_PAPER_RANK_PH_NOT_REPORTED_SOURCE_DOI_NOT_BUNDLED_FOR_STRICT_TEST',**rm,**strict))
    pd.DataFrame(rank_rows).to_csv(out/'chen16_rank_feature_comparison.csv',index=False)
    rank_predictions.to_csv(out/'chen16_rank_feature_predictions.csv',index=False)

    # 5. Gate by available pollutant for posthoc sensitivity; never use this as a promoted model metric.
    gate=[]
    for m in variants:
        g=oof[f'{m}__v55_152'].to_numpy(float).copy()
        mask_ni=base.pollutant=='Ni(II)';mask_pb=base.pollutant=='Pb(II)'
        g[mask_ni]=oof.loc[mask_ni,f'{m}__v57_181_shen_only'].to_numpy(float)
        g[mask_pb]=oof.loc[mask_pb,f'{m}__v57_159_cao_only'].to_numpy(float)
        gate.append(dict(model=m,scenario='pollutant_gated_posthoc',**metric(truth,g)))
    pd.DataFrame(gate).to_csv(out/'pollutant_gated_posthoc.csv',index=False)

    # 6. Preserve exploratory full-fit candidates separately. No activation/overwriting.
    dest=root/'deployment/research_v58_NOT_ACTIVE'
    dest.mkdir(exist_ok=True,parents=True)
    saved=[]
    for variant,scenario,kind in [('A','v55_152','cap'),('B','v55_152','cap'),
                                  ('A','v57_188_both','cap_log'),('B','v57_188_both','cap_log')]:
        data=v55 if scenario=='v55_152' else pd.concat([v55,new],ignore_index=True,sort=False)
        data=derive(data,kind)
        model,cols,nums,cats=build(variant,kind)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning)
            model.fit(features(data,cols,nums,cats),data.target_value.to_numpy(float))
        path=dest/f'STAGE2_{variant}_{scenario}_{kind}_RESEARCH_NOT_ACTIVE.joblib'
        joblib.dump(model,path,compress=3)
        loaded=joblib.load(path)
        sample=features(data.iloc[:5],cols,nums,cats)
        assert np.allclose(model.predict(sample),loaded.predict(sample),rtol=1e-8,atol=1e-8)
        saved.append({'model':variant,'scenario':scenario,'feature_set':kind,'n_training':len(data),
            'n_DOI':data.doi.nunique(),'features':cols,'numeric':nums,'categorical':cats,
            'model_path':str(path.relative_to(root)).replace('\\','/'),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,
            'active':False})
    (dest/'manifest_v58.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')

    manifest={'policy':'RESEARCH_ANALYSIS_ONLY_NEVER_ACTIVATE','n_original_test':len(base),
      'n_original_doi':int(base.doi.nunique()),'n_new_rows':len(new),'n_new_DOI':int(new.doi.nunique()),
      'source_hashes_sha256':{x:hashlib.sha256((root/y).read_bytes()).hexdigest() for x,y in [('v46_original_140',ROOT_OLD),('v55_152',ROOT_V55),('v57_new36',ROOT_NEW),('v57_original_OOF',ROOT_OOF)]},
      'feature_set':['cap','cap_log'],'group_unit':'DOI',
      'limitations':['The same original DOI cohort has been inspected for model development; all score comparisons are exploratory.',
       'Source-out new DOI results are retrospective, not genuinely prospective.',
       'Extra 29 isotherm concentration observations come from 4 materials within one DOI.',
       'Raw-source qe/qt/qmax qualifications are unchanged.',
       'No active artifact, existing CSV, or existing model is replaced.']}
    (out/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',out)
    print('Fixed140 physical feature ablation:')
    print(tbl[['model','scenario','feature_set','mae_mg_g','raw_cap_exceeded']].to_string(index=False))
    print('New DOI source-out physical features:')
    print(so_df[['source','model','feature_set','training','mae_mg_g','cap_clipped']].to_string(index=False))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    args=parser.parse_args()
    main(args.root)
