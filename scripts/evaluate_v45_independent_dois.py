"""v45: compare old-only, true XGB continuation, and pooled FROM-SCRATCH refit.
Research sensitivity evaluation of manually read figure values; NO automatic promotion.
Run from root: python -m scripts.evaluate_v45_independent_dois --root .
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GroupKFold
from scripts.build_v37_full_refit import S1_NUM,S1_CAT,S2_CAT,S1_TARGETS,pipeline,input_frame
from scripts.evaluate_v40_research import positive_model
from pyrolysis.biochar_stage2.v15_features import model_inputs
from scripts.build_v45_independent_dois import run as build,OLD_S1,OLD_S2,OUT

OUTPUT='reports/v45_independent_dois'
DEPLOY='deployment/research_v45_NOT_ACTIVE'

def feature_set(stage,variant):
    if stage=='Stage1':return S1_NUM,S1_CAT
    features=model_inputs(variant)
    return [c for c in features if c not in S2_CAT],[c for c in features if c in S2_CAT]

def model_maker(stage,variant):
    nums,cats=feature_set(stage,variant)
    return positive_model(nums,cats) if stage=='Stage1' else pipeline(nums,cats,'XGBoost')

def trainfit(m,d,stage,variant,target):
    nums,cats=feature_set(stage,variant)
    m.fit(input_frame(d,nums,cats),d[target].to_numpy(float))
    return m

def predict(m,d,stage,variant):
    nums,cats=feature_set(stage,variant)
    return np.asarray(m.predict(input_frame(d,nums,cats)),float)

def continuation_predict(fit,new,test,variant,target):
    """Genuine continuation of XGBoost booster on new paper rows with OLD preprocessor frozen.
    40 added boosting trees; not scientifically equivalent to full pooled re-fit.
    """
    old_model=trainfit(model_maker('Stage2',variant),fit,'Stage2',variant,target)
    pre=old_model.named_steps['pre']
    xnew=pre.transform(input_frame(new,*feature_set('Stage2',variant)))
    xtest=pre.transform(input_frame(test,*feature_set('Stage2',variant)))
    from xgboost import XGBRegressor
    # Start from old booster; extra 40 trees can overfit to only 10 new observations.
    r=XGBRegressor(n_estimators=40,max_depth=2,learning_rate=.05,subsample=.8,
       colsample_bytree=.8,reg_lambda=5,min_child_weight=5,random_state=42,n_jobs=1,tree_method='hist')
    r.fit(xnew,new[target].to_numpy(float),xgb_model=old_model.named_steps['model'].get_booster())
    return r.predict(xtest)

def score(y,p,stage,data):
    d=dict(n=len(y),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),
      negative_predictions=int((p<0).sum()))
    if stage=='Stage2':
        cap=data.C0_mg_L.to_numpy(float)/data.dose_g_L.to_numpy(float)
        d.update(physical_violation_count=int(((p<0)|(p>cap)).sum()),
            physical_clipped_mae=float(mean_absolute_error(y,np.clip(p,0,cap))),
            upper_cap_hit_count=int((p>cap).sum()))
    return d

def paired_evaluate(old,new,stage,target,variant,continued=False):
    old=old.reset_index(drop=True);new=new.reset_index(drop=True)
    assert set(old.doi).isdisjoint(set(new.doi))
    p0=np.full(len(old),np.nan); pa=np.full(len(old),np.nan)
    pc=np.full(len(old),np.nan) if continued else None
    nums,cats=feature_set(stage,variant)
    for tr,te in GroupKFold(n_splits=5).split(old,groups=old.doi):
        fit=old.iloc[tr]; test=old.iloc[te]
        p0[te]=predict(trainfit(model_maker(stage,variant),fit,stage,variant,target),test,stage,variant)
        pool=pd.concat([fit,new],ignore_index=True)
        pa[te]=predict(trainfit(model_maker(stage,variant),pool,stage,variant,target),test,stage,variant)
        if continued:pc[te]=continuation_predict(fit,new,test,variant,target)
    data=[('old_only_refit',p0),('pooled_from_scratch_refit',pa)]
    if continued:data.append(('warm_continue_on_new_only',pc))
    return data

def all_oof(df,stage,target,variant):
    df=df.reset_index(drop=True);y=df[target].to_numpy(float);p=np.full(len(df),np.nan)
    for tr,te in GroupKFold(n_splits=5).split(df,groups=df.doi):
        m=trainfit(model_maker(stage,variant),df.iloc[tr],stage,variant,target)
        p[te]=predict(m,df.iloc[te],stage,variant)
    return p

def run(root:Path):
    root=Path(root);build(root)
    out=root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    md=root/DEPLOY;md.mkdir(parents=True,exist_ok=True)
    old1=pd.read_csv(root/OLD_S1,low_memory=False)
    old2=pd.read_csv(root/OLD_S2,low_memory=False)
    new1=pd.read_csv(root/OUT/'new_stage1_23_SOURCE_STAGING.csv',low_memory=False)
    new2=pd.read_csv(root/OUT/'new_stage2_10_FIGURE_APPROX_RESEARCH_ONLY.csv',low_memory=False)
    combined1=pd.read_csv(root/OUT/'Stage1_Master_404_RESEARCH_STAGING.csv',low_memory=False)
    combined2=pd.read_csv(root/OUT/'Stage2_reported_uptake_138_RESEARCH_CONDITIONAL.csv',low_memory=False)
    metrics=[]; predictions=[]; prospect=[]
    for stage,old,new,combined,target,variants in [
        ('Stage1',old1,new1,combined1,S1_TARGETS['BET'],['Poisson']),
        ('Stage1',old1,new1,combined1,S1_TARGETS['TPV'],['Poisson']),
        ('Stage2',old2,new2,combined2,'target_value',['A','B'])]:
        old=old[old[target].notna()].reset_index(drop=True)
        new=new[new[target].notna()].reset_index(drop=True)
        combined=combined[combined[target].notna()].reset_index(drop=True)
        for variant in variants:
            # Paired old DOI CV: fixed test rows, new DOIs added to TRAINING ONLY.
            compares=paired_evaluate(old,new,stage,target,variant,continued=(stage=='Stage2'))
            for label,pred in compares:
                sc=score(old[target].to_numpy(float),pred,stage,old)
                metrics.append(dict(stage=stage,target=target,model=variant,
                    evaluation='old_DOI_fixed_5fold',strategy=label,**sc))
                for j,(doi,v,yhat) in enumerate(zip(old.doi,old[target],pred)):
                    predictions.append(dict(stage=stage,model=variant,evaluation='old_DOI_fixed_5fold',strategy=label,
                            doi=doi,record_index=j,actual=v,prediction=float(yhat)))
            # Full newly merged DOIs evaluation. Is a different test population, not directly comparable to old-only.
            yp=all_oof(combined,stage,target,variant)
            sc=score(combined[target].to_numpy(float),yp,stage,combined)
            metrics.append(dict(stage=stage,target=target,model=variant,
                evaluation='all_DOI_group5',strategy='pooled_from_scratch_refit',**sc))
            for j,(doi,v,yhat) in enumerate(zip(combined.doi,combined[target],yp)):
                predictions.append(dict(stage=stage,model=variant,evaluation='all_DOI_group5',strategy='pooled_from_scratch_refit',
                        doi=doi,record_index=j,actual=v,prediction=float(yhat)))
            # old-only fit -> additional new DOIs prospective challenge (not blinded)
            m=trainfit(model_maker(stage,variant),old,stage,variant,target)
            pp=predict(m,new,stage,variant)
            prospect.append(dict(stage=stage,model=variant,eval='new_DOI_preexisting_old_only_model',
                           n_new=len(new),new_doi=int(new.doi.nunique()),**score(new[target].to_numpy(float),pp,stage,new)))
            # new pooled FULL FROM SCRATCH model (stored research only)
            m2=trainfit(model_maker(stage,variant),combined,stage,variant,target)
            joblib.dump(m2,md/f'V45_{stage}_{target}_{variant}_POOLED_RESEARCH_ONLY.joblib',compress=3)
    metrics_df=pd.DataFrame(metrics)
    metrics_df.to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(prospect).to_csv(out/'new_doi_old_model_challenge.csv',index=False)
    outcome=dict(new_doi=4,material_records_added=23,stage1_BET_obs=int(combined1[S1_TARGETS['BET']].notna().sum()),
       stage1_TPV_obs=int(combined1[S1_TARGETS['TPV']].notna().sum()),
       stage2_research_rows=len(combined2),added_stage2_figure_approx=len(new2),
       new_secondary_figures_not_authorized_for_active_model=True,
       definitions={
         'old_only_refit':'Same fixed old DOI CV: trained old dataset on each fold.',
         'pooled_from_scratch_refit':'Fitted anew from scratch on same old train fold + new DOIs; never incremental.',
         'warm_continue_on_new_only':'XGB booster old-fold model + 40 trees on new DOIs, preprocessing frozen from old fold; not equivalent.',
         'all_DOI_group5':'Different CV population includes new DOIs in held-out folds; do not compare raw MAE directly with old-only.'},
       metrics=metrics,prospective=prospect,active_models_changed=False)
    (out/'summary.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding='utf-8')
    return outcome

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    args=p.parse_args();ans=run(args.root)
    print(pd.DataFrame(ans['metrics']).to_string(index=False));
    print('\nNEW DOI');print(pd.DataFrame(ans['prospective']).to_string(index=False))
