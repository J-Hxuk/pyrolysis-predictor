"""v39: positivity-aware Stage1 and conditional-quality Stage2 audits.

NO active deployment modification.  Source candidates (not all approved) are used
for exploratory evaluation.  Input feature contract is inherited from v37/v15.
Run: python -m scripts.build_v39_quality_positivity --root .
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scripts.build_v37_full_refit import (
    S1_NUM,S1_CAT,S1_TARGETS,S2_CAT,input_frame,pipeline,audit_s1,audit_s2
)
from pyrolysis.biochar_stage2.v15_features import model_inputs

S1='reports/v38_new_dois/Stage1_Master_351_RESEARCH_STAGING.csv'
S2='reports/v38_new_dois/Stage2_reported_uptake_125_RESEARCH_CONDITIONAL.csv'
OUT='reports/v39_quality_and_positivity'
TIER_CORE={'CORE_EXPLORATORY','PHASE8_RESEARCH_CANDIDATE_NOT_TRAINED'}
TIER_V34={'V34_RESEARCH_CONDITIONAL_NOT_CERTIFIED'}
TIER_V38={'V38_RESEARCH_CONDITIONAL_NOT_CERTIFIED'}


def poisson_model():
    """Poisson log-link gives positive prediction, including when y is tiny."""
    num=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))])
    cat=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                  ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    pre=ColumnTransformer([('num',num,S1_NUM),('cat',cat,S1_CAT)],sparse_threshold=0)
    reg=HistGradientBoostingRegressor(loss='poisson',max_iter=140,max_leaf_nodes=9,
                min_samples_leaf=10,learning_rate=.065,l2_regularization=5,random_state=42)
    return Pipeline([('pre',pre),('model',reg)])


def stage1_factory(variant):
    if variant=='Ridge':return pipeline(S1_NUM,S1_CAT,'Ridge')
    if variant=='RF':return pipeline(S1_NUM,S1_CAT,'RF')
    if variant=='RF_log1p':return pipeline(S1_NUM,S1_CAT,'RF',log_target=True)
    if variant=='Poisson_HGB':return poisson_model()
    raise ValueError(variant)


def input_audit(df):
    if df.material_id.duplicated().any():raise ValueError('material_id duplicate')
    if df.doi.isna().any():raise ValueError('DOI missing')
    audit_s1(df)


def doi_bootstrap_delta(groups, abs_base, abs_new, *, n=1200,seed=39):
    grp=np.asarray(groups).astype(str); uniq=np.unique(grp)
    by={d:np.flatnonzero(grp==d) for d in uniq}
    delta=np.asarray(abs_new)-np.asarray(abs_base); rng=np.random.default_rng(seed)
    samples=np.empty(n)
    for j in range(n):
        ids=np.concatenate([by[d] for d in rng.choice(uniq,size=len(uniq),replace=True)])
        samples[j]=np.mean(delta[ids])
    return tuple(float(x) for x in np.percentile(samples,[2.5,97.5]))


def fixed_folds(df):
    groups=df.doi.astype(str).to_numpy()
    return list(GroupKFold(n_splits=5).split(df,np.zeros(len(df)),groups))


def stage1_evaluate(s1):
    metrics=[]; rows=[]; preds={}
    for target,col in S1_TARGETS.items():
        d=s1.copy();d[col]=pd.to_numeric(d[col],errors='coerce');d=d[d[col].notna()].reset_index(drop=True)
        if d.doi.nunique()<5:raise ValueError('Stage1 insufficient independent DOIs')
        y=d[col].to_numpy(float);x=input_frame(d,S1_NUM,S1_CAT);splits=fixed_folds(d)
        for variant in ('Ridge','RF','RF_log1p','Poisson_HGB'):
            p=np.full(len(d),np.nan)
            for tr,te in splits:
                model=stage1_factory(variant);model.fit(x.iloc[tr],y[tr]);p[te]=model.predict(x.iloc[te])
            if not np.isfinite(p).all():raise ValueError('nan OOF predictions')
            preds[variant]=p
            lo=y<=25 if target=='BET' else y<=0.05
            metrics.append(dict(target=target,variant=variant,n_rows=len(d),n_doi=d.doi.nunique(),
                MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)),negative_raw=int((p<0).sum()),
                MAE_low=float(np.mean(np.abs(p[lo]-y[lo]))),n_low=int(lo.sum()),
                n_low_doi=int(d.loc[lo,'doi'].nunique()),
                median_AE_low=float(np.median(np.abs(p[lo]-y[lo])))))
            for i in range(len(d)):
                rows.append(dict(target=target,doi=d.iloc[i].doi,material_id=d.iloc[i].material_id,
                    actual=y[i],variant=variant,pred_raw=p[i],abs_error=abs(y[i]-p[i]),
                    true_low=bool(lo[i])))
        baseline=preds['Ridge']
        for row in metrics:
            if row['target']==target:
                pp=preds[row['variant']]; ci=doi_bootstrap_delta(d.doi,np.abs(y-baseline),np.abs(y-pp))
                row.update(delta_MAE_vs_Ridge=float(np.mean(np.abs(y-pp)-np.abs(y-baseline))),
                           delta_ci_low=ci[0],delta_ci_high=ci[1])
    return pd.DataFrame(metrics),pd.DataFrame(rows)


def stage2_cohorts(s2):
    if not set(s2.model_use_tier).issubset(TIER_CORE|TIER_V34|TIER_V38):
        raise ValueError('unexpected quality tier; do not silently drop')
    core=s2[s2.model_use_tier.isin(TIER_CORE)].copy().reset_index(drop=True)
    old=s2[s2.model_use_tier.isin(TIER_V34)].copy()
    new=s2[s2.model_use_tier.isin(TIER_V38)].copy()
    if set(core.doi)&set(old.doi) or set(core.doi)&set(new.doi) or set(old.doi)&set(new.doi):
        raise ValueError('quality cohorts DOI overlap; leakage hazard')
    return core,old,new


def stage2_quality_eval(s2):
    core,v34,v38=stage2_cohorts(s2)
    cohorts={'Core_98':core,'Core_plus_v34_113':pd.concat([core,v34],ignore_index=True),
        'All_research_125':pd.concat([core,v34,v38],ignore_index=True)}
    splits=fixed_folds(core);res=[];preds=[]
    for model_name in ('A','B'):
        inp=model_inputs(model_name);cats=[f for f in inp if f in S2_CAT];nums=[f for f in inp if f not in cats]
        xcore=input_frame(core,nums,cats); y=core.target_value.to_numpy(float)
        for label,trainpool in cohorts.items():
            all_extra=trainpool[~trainpool.observation_id.isin(core.observation_id)].copy()
            xextra=input_frame(all_extra,nums,cats)
            p=np.full(len(core),np.nan)
            for tr,te in splits:
                m=pipeline(nums,cats,'XGBoost')
                fitx=pd.concat([xcore.iloc[tr],xextra],ignore_index=True)
                fity=np.concatenate([y[tr],all_extra.target_value.to_numpy(float)])
                m.fit(fitx,fity);p[te]=m.predict(xcore.iloc[te])
            cap=core.C0_mg_L.to_numpy(float)/core.dose_g_L.to_numpy(float)
            res.append(dict(model=model_name,cohort=label,eval_rows=len(core),eval_doi=core.doi.nunique(),
                train_extra_rows=len(all_extra),train_extra_doi=all_extra.doi.nunique(),
                MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)),
                negative_raw=int((p<0).sum()),exceed_cap=int((p>cap).sum()),
                physics_violation=int(((p<0)|(p>cap)).sum())))
            for i in range(len(core)):
                preds.append(dict(model=model_name,train_cohort=label,doi=core.iloc[i].doi,
                      observation_id=core.iloc[i].observation_id,actual=y[i],pred_raw=p[i],cap=cap[i]))
    met=pd.DataFrame(res);pr=pd.DataFrame(preds)
    for model_name in ('A','B'):
        base=pr[(pr.model==model_name)&(pr.train_cohort=='Core_98')].sort_values('observation_id')
        for label in cohorts:
            row=met[(met.model==model_name)&(met.cohort==label)].index[0]
            v=pr[(pr.model==model_name)&(pr.train_cohort==label)].sort_values('observation_id')
            assert np.array_equal(base.observation_id.values,v.observation_id.values)
            ci=doi_bootstrap_delta(base.doi,np.abs(base.actual.values-base.pred_raw.values),
                                            np.abs(v.actual.values-v.pred_raw.values))
            met.loc[row,'delta_MAE_vs_core']=float((np.abs(v.actual.values-v.pred_raw.values)-
                           np.abs(base.actual.values-base.pred_raw.values)).mean())
            met.loc[row,'delta_ci_low']=ci[0];met.loc[row,'delta_ci_high']=ci[1]
    return met,pr


def historical_new_doi_diagnostic(root):
    """DESCRIPTIVE ONLY: new coffee/wheat DOIs influenced v39 hypothesis selection."""
    old_path=root/'data/imports/v34_temperature/Stage1_Master_333_RESEARCH_STAGING_NOT_TRAINED.csv'
    new_path=root/'data/imports/v38_new_dois/new_materials_18_SOURCE_STAGING.csv'
    if not(old_path.is_file() and new_path.is_file()):
        return pd.DataFrame(), pd.DataFrame()
    old=pd.read_csv(old_path,low_memory=False)
    fresh=pd.read_csv(new_path,low_memory=False)
    if set(old.doi)&set(fresh.doi):raise ValueError('new DOI challenge leakage')
    rows=[];scores=[]
    for target,col in S1_TARGETS.items():
        tr=old[old[col].notna()].copy();te=fresh[fresh[col].notna()].copy()
        y=te[col].to_numpy(float)
        for variant in ('Ridge','RF_log1p','Poisson_HGB'):
            m=stage1_factory(variant)
            m.fit(input_frame(tr,S1_NUM,S1_CAT),tr[col].to_numpy(float))
            pred=m.predict(input_frame(te,S1_NUM,S1_CAT))
            scores.append(dict(target=target,variant=variant,n_rows=len(te),n_doi=te.doi.nunique(),
                     MAE=float(mean_absolute_error(y,pred)),negative_raw=int((pred<0).sum())))
            for id,doi,actual,pred_i in zip(te.material_id,te.doi,y,pred):
                rows.append(dict(target=target,variant=variant,material_id=id,doi=doi,
                         actual=actual,pred_raw=pred_i))
    return pd.DataFrame(scores),pd.DataFrame(rows)


def coverage(s1,s2):
    rows=[]
    for target,col in S1_TARGETS.items():
        d=s1[s1[col].notna()]
        for c in ('pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min',
                  'feedstock_C_pct_original','feedstock_VM_pct_original','feedstock_FC_pct_original'):
            rows.append(dict(stage='Stage1',target=target,variable=c,n=len(d),reported=int(d[c].notna().sum()),
                  n_doi_reporting=int(d.loc[d[c].notna(),'doi'].nunique())))
    for label,d in s2.groupby('model_use_tier',dropna=False):
        for c in ('initial_pH','contact_time_min','adsorption_T_C','pyrolysis_temperature_C',
                  'pyrolysis_hold_min','heating_rate_C_min','raw_VM_wt_pct','raw_FC_wt_pct'):
            rows.append(dict(stage='Stage2',target=label,variable=c,n=len(d),
                  reported=int(d[c].notna().sum()),n_doi_reporting=int(d.loc[d[c].notna(),'doi'].nunique())))
    return pd.DataFrame(rows)


def run(root:Path,save_model=True):
    root=Path(root).resolve();out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    s1=pd.read_csv(root/S1,low_memory=False);s2=pd.read_csv(root/S2,low_memory=False)
    input_audit(s1);audit_s2(s2)
    stage1,stage1pred=stage1_evaluate(s1)
    stage2,stage2pred=stage2_quality_eval(s2)
    cov=coverage(s1,s2)
    new_diag,new_diag_rows=historical_new_doi_diagnostic(root)
    stage1.to_csv(out/'stage1_model_comparison.csv',index=False)
    stage1pred.to_csv(out/'stage1_oof_predictions.csv',index=False)
    stage2.to_csv(out/'stage2_quality_sensitivity.csv',index=False)
    stage2pred.to_csv(out/'stage2_oof_predictions.csv',index=False)
    cov.to_csv(out/'missingness_by_source_tier.csv',index=False)
    if len(new_diag):
        new_diag.to_csv(out/'v38_new_doi_historical_diagnostic.csv',index=False)
        new_diag_rows.to_csv(out/'v38_new_doi_historical_predictions.csv',index=False)
    # Fit a priori positive research model on all currently eligible numeric candidates.
    # This is NOT selected from OOF winner or activated.
    if save_model:
        d=s1[s1[S1_TARGETS['BET']].notna()].copy();x=input_frame(d,S1_NUM,S1_CAT)
        m=poisson_model();m.fit(x,d[S1_TARGETS['BET']].to_numpy(float))
        folder=root/'deployment/research_v39_NOT_ACTIVE';folder.mkdir(parents=True,exist_ok=True)
        joblib.dump(m,folder/'Stage1_BET_Poisson_HGB_RESEARCH_ONLY.joblib')
    summary={'Stage1':stage1.to_dict(orient='records'), 'Stage2':stage2.to_dict(orient='records'),
             'v38_new_doi_historical_diagnostic':new_diag.to_dict(orient='records'),
             'n_stage1':len(s1),'n_stage2':len(s2),'n_stage1_doi':int(s1.doi.nunique()),
             'n_stage2_doi':int(s2.doi.nunique()),
             'method':'fixed DOI GroupKFold fivefold, nonnested; source quality cohorts DOI disjoint; no tuning on diagnostics',
             'caveats':['research staging observations not all deployment approved','reported uptake not strict equilibrium certified',
                        'same historical DOIs used in iterative exploratory development; not future locked validation',
                        'raw positivity, no clipping credit','no active model edits']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--no-save-model',action='store_true')
    args=ap.parse_args(); ret=run(args.root,not args.no_save_model)
    print('Stage1:',pd.DataFrame(ret['Stage1']).to_string(index=False))
    print('Stage2:',pd.DataFrame(ret['Stage2']).to_string(index=False))
