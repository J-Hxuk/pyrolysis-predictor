"""v38 -- source-audited, NON-DEPLOYED full-refit and paired old-DOI evaluations.

Intended for repository with v37 script/data; never edits deployed models.
Rebuild training from all compatible observations (not incremental continuation).
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scripts.build_v37_full_refit import (S1_NUM,S1_CAT,S1_TARGETS,S2_CAT,STAGE1_PATH,STAGE2_PATH,
                                           input_frame,pipeline,audit_s1,audit_s2)
from pyrolysis.biochar_stage2 import v15_features as FC

S1_ADD='data/imports/v38_new_dois/new_materials_18_SOURCE_STAGING.csv'
S2_ADD='data/imports/v38_new_dois/coffee_singlemetal_12_RESEARCH_CONDITIONAL.csv'
MULTI='data/imports/v38_new_dois/bagasse_multimetal_6_SEPARATE_DOMAIN.csv'
OUT='reports/v38_new_dois'; OUTMODELS='deployment/research_v38_NOT_ACTIVE'


def combine(root):
    base1=pd.read_csv(root/STAGE1_PATH,low_memory=False); new1=pd.read_csv(root/S1_ADD,low_memory=False)
    base2=pd.read_csv(root/STAGE2_PATH,low_memory=False); new2=pd.read_csv(root/S2_ADD,low_memory=False)
    if new1.material_id.duplicated().any() or new2.observation_id.duplicated().any():
        raise ValueError('new source IDs duplicated')
    if set(new1.doi.astype(str)) & set(base1.doi.astype(str)):
        raise ValueError('new Stage1 paper DOI already in source; no duplicate merge')
    if set(new2.doi.astype(str)) & set(base2.doi.astype(str)):
        raise ValueError('new Stage2 paper DOI already in source; no duplicate merge')
    for df in [new1,new2]:
        assert df.doi.notna().all()
    # exact schema contract; unknown extra source annotations retained in separate CSV only
    added1=new1.reindex(columns=base1.columns)
    added2=new2.reindex(columns=base2.columns)
    s1=pd.concat([base1,added1],ignore_index=True)
    s2=pd.concat([base2,added2],ignore_index=True)
    audit_s1(s1);audit_s2(s2)
    return base1,new1,s1,base2,new2,s2


def paired_old_oof(old, extra, target, nums, cats, algorithm, log=False):
    base=old.dropna(subset=[target]).copy()
    extra=extra.dropna(subset=[target]).copy()
    assert base.doi.nunique()>=5
    y=base[target].astype(float).to_numpy(); grp=base.doi.astype(str).to_numpy()
    x=input_frame(base,nums,cats); newx=input_frame(extra,nums,cats)
    before=np.full(len(base),np.nan); after=np.full(len(base),np.nan)
    for tr,te in GroupKFold(n_splits=5).split(x,y,grp):
        mod=pipeline(nums,cats,algorithm,log_target=log);mod.fit(x.iloc[tr],y[tr]);before[te]=mod.predict(x.iloc[te])
        x_aug=pd.concat([x.iloc[tr],newx],ignore_index=True)
        y_aug=np.concatenate([y[tr],extra[target].to_numpy(float)])
        aug=pipeline(nums,cats,algorithm,log_target=log); aug.fit(x_aug,y_aug);after[te]=aug.predict(x.iloc[te])
    return base,y,before,after


def bstrap_ci(doi, before_abs, after_abs, seed=7, n=1500):
    # DOI-cluster resampling, preserving unequal numbers of observations per DOI
    groups=pd.Series(doi).astype(str).to_numpy()
    uq=np.unique(groups); idx={d:np.flatnonzero(groups==d) for d in uq}
    rng=np.random.default_rng(seed); delta=after_abs-before_abs
    diffs=[]
    for _ in range(n):
        take=rng.choice(uq,len(uq),replace=True)
        ix=np.concatenate([idx[d] for d in take]);diffs.append(float(delta[ix].mean()))
    return [float(x) for x in np.percentile(diffs,[2.5,97.5])]


def slice_audit(df, pred, target, variant):
    y=df[target].astype(float).to_numpy();abs_e=np.abs(pred-y)
    ret=[]
    # Do not use POST-pyrolysis chemical composition as predictive input.
    if variant=='Stage1_BET':
        category=pd.cut(df[target].astype(float),bins=[-1,25,200,np.inf],labels=['BET_0_25','BET_25_200','BET_above_200'])
    elif variant=='Stage1_TPV':
        category=pd.cut(df[target].astype(float),bins=[-1,0.05,0.25,np.inf],labels=['TPV_0_0.05','TPV_0.05_0.25','TPV_above_0.25'])
    else:
        category=df['pollutant'].astype(str)
    for label in pd.Series(category).dropna().unique():
        ix=(pd.Series(category).astype(str).to_numpy()==str(label))
        ret.append({'domain':variant,'slice_type':'target_band' if variant.startswith('Stage1') else 'pollutant',
                    'category':str(label),'n_rows':int(ix.sum()),'n_doi':int(df.loc[ix,'doi'].nunique()),
                    'MAE':float(abs_e[ix].mean()),'reportable':bool(df.loc[ix,'doi'].nunique()>=3 and ix.sum()>=8)})
    # Bias and missing coverage by feedstock and processing route; one-DOI groups
    # are descriptive only, never treated as independently generalized categories.
    features= [('feedstock_group','feedstock_group'),('process_stage','manufacture_route')] if variant.startswith('Stage1') else [('feedstock_label_normalized','feedstock'),('manufacturing_route','manufacture_route')]
    for col,typ in features:
        if col not in df: continue
        keys=df[col].fillna('NR').astype(str)
        for value in keys.unique():
            ix=(keys.to_numpy()==value)
            ret.append({'domain':variant,'slice_type':typ,'category':value,
                        'n_rows':int(ix.sum()),'n_doi':int(df.loc[ix,'doi'].nunique()),
                        'MAE':float(abs_e[ix].mean()),
                        'reportable':bool(df.loc[ix,'doi'].nunique()>=3 and ix.sum()>=8)})
    # DOI-aggregated error: intentionally audit concentration, not an unbiased random row split.
    for doi,g in df.assign(_err=abs_e).groupby('doi'):
        ret.append({'domain':variant,'slice_type':'doi','category':str(doi),
                    'n_rows':len(g),'n_doi':1,'MAE':float(g._err.mean()),'reportable':False})
    return ret


def main(root, output=None):
    root=Path(root).resolve(); out=Path(output).resolve() if output else root/OUT
    out.mkdir(parents=True,exist_ok=True)
    b1,n1,s1,b2,n2,s2=combine(root)
    s1.to_csv(out/'Stage1_Master_351_RESEARCH_STAGING.csv',index=False)
    s2.to_csv(out/'Stage2_reported_uptake_125_RESEARCH_CONDITIONAL.csv',index=False)
    scores=[];oof_rows=[];slices=[]
    # Paired test on EXACT SAME original DOI rows and fixed folds; new DOI always train only.
    for stage,target,algo,log in [('Stage1','BET','Ridge',False),('Stage1','TPV','RF',True)]:
        col=S1_TARGETS[target]
        old,y,p0,p1=paired_old_oof(b1,n1,col,S1_NUM,S1_CAT,algo,log=log)
        ci=bstrap_ci(old.doi,np.abs(p0-y),np.abs(p1-y))
        scores.append(dict(stage=stage,target=target,algorithm='RF_log' if log else algo,
                           tested_rows=len(old),tested_dois=old.doi.nunique(),added_train_rows=n1[col].notna().sum(),
                           MAE_old=float(mean_absolute_error(y,p0)),MAE_new=float(mean_absolute_error(y,p1)),
                           delta_new_minus_old=float(np.mean(np.abs(y-p1)-np.abs(y-p0))),ci95_low=ci[0],ci95_high=ci[1],
                           R2_old=float(r2_score(y,p0)),R2_new=float(r2_score(y,p1))))
        slices.extend(slice_audit(old,p0,col,f'Stage1_{target}'))
        for id,doi,yy,pp0,pp1 in zip(old.material_id,old.doi,y,p0,p1):
            oof_rows.append(dict(stage=stage,target=target,id=id,doi=doi,actual=yy,old_oof=pp0,plus_new_doi_oof=pp1))
        # New DOI challenge with only original data (not used for tuning): diagnostic, not untouched external.
        train=b1[b1[col].notna()].copy();test=n1[n1[col].notna()].copy()
        if len(test):
            m=pipeline(S1_NUM,S1_CAT,algo,log_target=log)
            m.fit(input_frame(train,S1_NUM,S1_CAT),train[col].to_numpy(float))
            pred=m.predict(input_frame(test,S1_NUM,S1_CAT))
            for id,doi,yy,pp in zip(test.material_id,test.doi,test[col],pred):
                oof_rows.append(dict(stage=stage,target=target,id=id,doi=doi,actual=yy,old_oof=pp,plus_new_doi_oof=np.nan,
                                     evaluation_role='NEW_DOI_DIAGNOSTIC_HELD_FROM_TRAIN'))
    for mdl in ['A','B']:
        inp=FC.model_inputs(mdl);cats=[c for c in inp if c in S2_CAT];nums=[c for c in inp if c not in cats]
        old,y,p0,p1=paired_old_oof(b2,n2,'target_value',nums,cats,'XGBoost')
        ci=bstrap_ci(old.doi,np.abs(y-p0),np.abs(y-p1))
        cap=old.C0_mg_L.to_numpy(float)/old.dose_g_L.to_numpy(float)
        scores.append(dict(stage='Stage2',target=mdl,algorithm='XGBoost',tested_rows=len(old),
                           tested_dois=old.doi.nunique(),added_train_rows=len(n2),
                           MAE_old=float(mean_absolute_error(y,p0)),MAE_new=float(mean_absolute_error(y,p1)),
                           delta_new_minus_old=float(np.mean(np.abs(y-p1)-np.abs(y-p0))),ci95_low=ci[0],ci95_high=ci[1],
                           R2_old=float(r2_score(y,p0)),R2_new=float(r2_score(y,p1)),
                           raw_violations_old=int(((p0<0)|(p0>cap)).sum()),raw_violations_new=int(((p1<0)|(p1>cap)).sum())))
        slices.extend(slice_audit(old,p0,'target_value','Stage2_'+mdl))
        for id,doi,yy,pp0,pp1 in zip(old.observation_id,old.doi,y,p0,p1):
            oof_rows.append(dict(stage='Stage2',target=mdl,id=id,doi=doi,actual=yy,old_oof=pp0,plus_new_doi_oof=pp1))
        m=pipeline(nums,cats,'XGBoost')
        m.fit(input_frame(b2,nums,cats),b2.target_value.to_numpy(float))
        pred=m.predict(input_frame(n2,nums,cats))
        for id,doi,yy,pp in zip(n2.observation_id,n2.doi,n2.target_value,pred):
            oof_rows.append(dict(stage='Stage2',target=mdl,id=id,doi=doi,actual=yy,old_oof=pp,plus_new_doi_oof=np.nan,
                                 evaluation_role='NEW_DOI_DIAGNOSTIC_HELD_FROM_TRAIN'))
    pd.DataFrame(scores).to_csv(out/'paired_old_DOI_evaluation.csv',index=False)
    pd.DataFrame(oof_rows).to_csv(out/'paired_predictions_and_new_DOI.csv',index=False)
    pd.DataFrame(slices).to_csv(out/'error_slices.csv',index=False)
    # Real full refit on all compatible research candidates, never active deployment.
    moddir=root/OUTMODELS;moddir.mkdir(parents=True,exist_ok=True)
    for target,algo,log in [('BET','Ridge',False),('TPV','RF',True)]:
        col=S1_TARGETS[target];sub=s1[s1[col].notna()].copy()
        m=pipeline(S1_NUM,S1_CAT,algo,log_target=log);m.fit(input_frame(sub,S1_NUM,S1_CAT),sub[col].to_numpy(float))
        joblib.dump(m,moddir/f'v38_Stage1_{target}_ALL_RESEARCH_ONLY.joblib')
    for mdl in ['A','B']:
        inp=FC.model_inputs(mdl);cats=[c for c in inp if c in S2_CAT];nums=[c for c in inp if c not in cats]
        m=pipeline(nums,cats,'XGBoost');m.fit(input_frame(s2,nums,cats),s2.target_value.to_numpy(float))
        joblib.dump(m,moddir/f'v38_Stage2_{mdl}_ALL_RESEARCH_ONLY.joblib')
    audit={'input':{'stage1_before':len(b1),'stage1_added':len(n1),'stage1_after':len(s1),
                    'stage1_bet':int(s1[S1_TARGETS['BET']].notna().sum()),'stage1_tpv':int(s1[S1_TARGETS['TPV']].notna().sum()),
                    'stage2_before':len(b2),'stage2_conditional_added':len(n2),'stage2_after':len(s2),
                    'stage2_new_pH_NR':int(n2.initial_pH.isna().sum()),
                    'stage2_new_source_DOI':int(n2.doi.nunique())},
           'evaluation':'exact pre-existing DOI test rows and same GroupKFold 5 split; new DOI train-only in augmentation; new DOI challenge diagnostic',
           'policy':'NO automatic model activation; source-derived reported uptake provisional; not certified strict equilibrium; qmax & qt & mixed metal excluded',
           'python_versions':{'scikit_learn':__import__('sklearn').__version__},
           'results':scores}
    (out/'summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=lambda o: o.item() if isinstance(o,np.generic) else str(o)),encoding='utf-8')
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(main(a.root,a.output),ensure_ascii=False,indent=2,default=lambda o: o.item() if isinstance(o,np.generic) else str(o)))
