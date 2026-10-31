"""v59: strictly research-only DOI/material hierarchical sample weighting.
No source CSV, active model, or prior research model is changed.
Run: python -m scripts.run_v59_doi_material_weighting --root .
"""
from __future__ import annotations
import argparse, hashlib, json, platform, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn, xgboost
from scripts.build_v46_phase8 import s2_model, features
from scripts.run_v58_failure_diagnostics import build,derive
from scripts.run_v54_bounded_rank import chen_features, CHEN

OLD='data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
PRE='data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv'
NEW='data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv'
REPORT='reports/v59_doi_material_weighting'
DEPLOY='deployment/research_v59_NOT_ACTIVE'
POLICIES={
 'row_equal':(1.,1.),
 'doi_sqrt_row_equal':(.5,1.),
 'doi_sqrt_material_equal':(.5,0.),
 'doi_equal_material_equal':(0.,0.),
}
SEED=20261031

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def material_key(frame):
    """Use source material UID, then conservatively parse three legacy ID structures.
    Unrecognised missing material UIDs remain separate observations (no invented batch link).
    """
    keys=[]; src=[]
    for r in frame.itertuples(index=False):
        uid=getattr(r,'material_uid',None)
        if isinstance(uid,str) and uid.strip():
            keys.append(str(r.doi)+'::'+uid.strip());src.append('material_uid');continue
        obs=str(r.observation_id)
        doi=str(r.doi)
        if doi=='10.3390/ijerph16050845' and obs.startswith('V33_WANG19_') and '_Cd(II)_' in obs:
            sub=obs.split('_Cd(II)_',1)[0];src.append('legacy_id_wang19')
        elif doi=='10.3390/su14159022' and obs.startswith('V33_HU22_') and '_Cd(II)_' in obs:
            sub=obs.split('_Cd(II)_',1)[0];src.append('legacy_id_hu22')
        elif doi=='10.1016/j.chemosphere.2022.134051' and obs.startswith('V38_V38_') and (obs.endswith('_CdII') or obs.endswith('_ZnII')):
            sub=obs.rsplit('_',1)[0];src.append('legacy_id_v38')
        else:
            sub='UNRESOLVED_OBS::'+obs;src.append('unresolved_unique_observation')
        keys.append(doi+'::'+sub)
    return pd.Series(keys,index=frame.index,dtype='string'),pd.Series(src,index=frame.index,dtype='string')

def weights_for(frame,policy):
    if policy not in POLICIES:raise ValueError('Unsupported policy')
    if frame.doi.isna().any() or not frame.observation_id.is_unique:raise ValueError('Nonunique IDs or missing DOI')
    alpha,beta=POLICIES[policy]
    if policy=='row_equal':return np.ones(len(frame),float)
    keys,_=material_key(frame)
    dd=pd.DataFrame({'doi':frame['doi'].astype(str).to_numpy(),'material':keys.astype(str).to_numpy()})
    doi_counts=dd.groupby('doi')['material'].transform('size').astype(float).to_numpy()
    mat_counts=dd.groupby(['doi','material'])['material'].transform('size').astype(float).to_numpy()
    # DOI total weight is n_doi**alpha. Within each DOI, material shares
    # proportional to n_material_observations**beta. beta=0 balances materials.
    # Count material-group membership only inside the same DOI.
    pair_counts=dd.groupby(['doi','material']).size()
    denom={doi:float(sum(float(n)**beta for (group, _),n in pair_counts.items() if group==doi)) for doi in dd.doi.unique()}
    w=np.array([n**alpha*(nm**beta)/denom[doi]/nm for doi,n,nm in zip(dd.doi,doi_counts,mat_counts)],float)
    # Keep effective total model regularization comparable to unweighted model.
    w*=len(w)/w.sum()
    if not np.isfinite(w).all() or (w<=0).any() or not np.isclose(w.mean(),1):raise ValueError('Bad sample weights')
    return w

def fit_predict(train,test,model_type,feature_set,policy,return_model=False):
    tr=derive(train,feature_set);te=derive(test,feature_set)
    model,cols,nums,cats=build(model_type,feature_set)
    wt=weights_for(train,policy)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        if policy=='row_equal':model.fit(features(tr,cols,nums,cats),tr.target_value.to_numpy(float))
        else:model.fit(features(tr,cols,nums,cats),tr.target_value.to_numpy(float),model__sample_weight=wt)
        raw=np.asarray(model.predict(features(te,cols,nums,cats)),float)
    cap=(te.C0_mg_L/te.dose_g_L).to_numpy(float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite predictions')
    pred=np.clip(raw,0,cap)
    if return_model:return pred,raw,model,cols,nums,cats,wt
    return pred,raw

def boot_ci(y,reference,candidate,doi,reps=2000):
    ids=np.unique(doi); ix=[np.flatnonzero(doi==d) for d in ids]
    rng=np.random.default_rng(SEED); gains=[]
    err=np.abs(y-reference)-np.abs(y-candidate)
    for _ in range(reps):
        selected=rng.integers(0,len(ids),len(ids))
        choose=np.concatenate([ix[j] for j in selected]);gains.append(float(err[choose].mean()))
    return tuple(float(q) for q in np.quantile(gains,[.025,.975]))

def summarize_weights(frame,scenario,policy,fold):
    w=weights_for(frame,policy); d=frame[['doi','observation_id']].copy()
    keys,kind=material_key(frame)
    d['material_group']=keys.to_numpy();d['material_group_origin']=kind.to_numpy();d['weight']=w
    grp=d.groupby('doi',as_index=False).agg(n=('observation_id','size'),materials=('material_group','nunique'),weight_sum=('weight','sum'))
    grp['weight_fraction']=grp.weight_sum/w.sum()
    grp['policy']=policy;grp['scenario']=scenario;grp['fold']=fold
    return grp,d

def main(root):
    root=Path(root).resolve(); dst=root/REPORT;dst.mkdir(exist_ok=True,parents=True)
    base=pd.read_csv(root/OLD,low_memory=False).reset_index(drop=True)
    old152=pd.read_csv(root/PRE,low_memory=False).reset_index(drop=True)
    new36=pd.read_csv(root/NEW,low_memory=False).reset_index(drop=True)
    assert len(base)==140 and len(old152)==152 and len(new36)==36 and base.doi.nunique()==32
    assert list(old152.iloc[:140].observation_id)==list(base.observation_id)
    assert not set(new36.doi)&set(old152.doi)
    assert len(pd.concat([old152,new36]).observation_id.unique())==188
    datasets={'v55_152':old152,'v57_188':pd.concat([old152,new36],ignore_index=True,sort=False)}
    output=base[['observation_id','doi','pollutant','target_value','C0_mg_L','dose_g_L']].copy()
    fold_idx=np.full(len(base),0,int)
    weight_details=[];folds=[]
    # Selection/ablations deliberately use EXACT same old140 held-out DOI folds.
    for fold,(tr,te) in enumerate(GroupKFold(5).split(base,groups=base.doi),1):
        fold_idx[te]=fold;held=set(base.iloc[te].doi)
        folds.append(dict(fold=fold,test_dois=sorted(held),test_n=len(te)))
        for scenario,data in datasets.items():
            extra=data.iloc[140:];extra=extra[~extra.doi.isin(held)]
            training=pd.concat([base.iloc[tr],extra],ignore_index=True,sort=False)
            assert not set(training.doi)&held
            if fold==1:
                for pol in POLICIES:
                    grp,_=summarize_weights(training,scenario,pol,fold);weight_details.append(grp)
            for variant in ['A','B']:
                for feat in ['baseline','cap']:
                    for pol in POLICIES:
                        name=f'{variant}__{scenario}__{feat}__{pol}'
                        pred,raw=fit_predict(training,base.iloc[te],variant,feat,pol)
                        if name not in output:
                            output[name]=np.nan;output[name+'__raw']=np.nan
                        output.loc[te,name]=pred
                        output.loc[te,name+'__raw']=raw
    output['fold']=fold_idx
    output.to_csv(dst/'fixed140_weighted_oof.csv',index=False)
    records=[];y=base.target_value.to_numpy(float); doi=base.doi.to_numpy(str);cap=(base.C0_mg_L/base.dose_g_L).to_numpy(float)
    for scenario in datasets:
        for variant in ['A','B']:
            for feat in ['baseline','cap']:
                ref=f'{variant}__{scenario}__{feat}__row_equal'
                for pol in POLICIES:
                    col=f'{variant}__{scenario}__{feat}__{pol}'
                    p=output[col].to_numpy(float);raw=output[col+'__raw'].to_numpy(float)
                    if not np.isfinite(p).all():raise ValueError('Missing predictions '+col)
                    ci=(None,None) if pol=='row_equal' else boot_ci(y,output[ref].to_numpy(float),p,doi)
                    bydoi=pd.DataFrame({'doi':doi,'ae':np.abs(y-p)}).groupby('doi').ae.mean()
                    records.append(dict(model=variant,scenario=scenario,feature_set=feat,weight_policy=pol,n_test=len(y),n_test_doi=len(set(doi)),
                        mae_mg_g=float(mean_absolute_error(y,p)),rmse_mg_g=float(np.sqrt(mean_squared_error(y,p))),
                        doi_macro_mae=float(bydoi.mean()),raw_upper_cap_violation=int((raw>cap+1e-8).sum()),
                        at_upper_cap_count=int(np.isclose(p,cap,atol=1e-8,rtol=0).sum()),
                        mae_gain_over_equal_row=float(np.mean(np.abs(y-output[ref].to_numpy(float))-np.abs(y-p))),
                        gain_ci95_low=ci[0],gain_ci95_high=ci[1]))
    met=pd.DataFrame(records);met.to_csv(dst/'same140_weighted_metrics.csv',index=False)
    pd.concat(weight_details,ignore_index=True).to_csv(dst/'doi_weight_profiles_fold1.csv',index=False)
    (dst/'eval_folds.json').write_text(json.dumps(folds,ensure_ascii=False,indent=2),encoding='utf-8')
    # Novel source-out; this is still retrospective, previously examined DOIs.
    source=[];sourcepred=[]
    for source_doi in sorted(new36.doi.unique()):
        test=new36[new36.doi==source_doi];other=new36[new36.doi!=source_doi]
        assert source_doi not in set(old152.doi)
        for training_label,train in [('v55_152',old152),('v55_152_plus_other_DOI',pd.concat([old152,other],ignore_index=True,sort=False))]:
            assert source_doi not in set(train.doi)
            for variant in ['A','B']:
                for feat in ['baseline','cap']:
                    for pol in POLICIES:
                        pp,raw=fit_predict(train,test,variant,feat,pol)
                        yy=test.target_value.to_numpy(float);lim=(test.C0_mg_L/test.dose_g_L).to_numpy(float)
                        source.append(dict(heldout_doi=source_doi,source_n=len(test),train_scenario=training_label,model=variant,
                            feature_set=feat,weight_policy=pol,mae_mg_g=float(np.abs(yy-pp).mean()),
                            raw_upper_violation=int((raw>lim+1e-8).sum()),upper_cap_count=int(np.isclose(pp,lim,atol=1e-8,rtol=0).sum()),
                            zero_baseline_mae=float(np.abs(yy).mean())))
                        for oid,yv,pv in zip(test.observation_id,yy,pp):
                            sourcepred.append(dict(heldout_doi=source_doi,train_scenario=training_label,model=variant,feature_set=feat,
                                weight_policy=pol,observation_id=oid,target_value=yv,pred=pv))
    pd.DataFrame(source).to_csv(dst/'sourceout_weighted_metrics.csv',index=False)
    pd.DataFrame(sourcepred).to_csv(dst/'sourceout_weighted_predictions.csv',index=False)
    # Chen 16-condition analysis: DO NOT call prospective independent verification;
    # all capped predictions sharing 100% are unresolved ties.
    from scipy.stats import spearmanr
    chen=pd.read_csv(root/CHEN)
    xin=chen_features(chen);ychen=chen.reported_removal_pct.to_numpy(float)
    rank=[];rankpred=chen[['material_uid','reported_removal_pct']].copy()
    for scenario,train in datasets.items():
        for feat in ['baseline','cap']:
            for pol in POLICIES:
                p,raw=fit_predict(train,xin,'B',feat,pol)
                pct=p/(xin.C0_mg_L/xin.dose_g_L).to_numpy(float)*100
                mask=np.isclose(pct,np.max(pct),atol=1e-8,rtol=1e-8)
                rho=spearmanr(ychen,pct).statistic if len(np.unique(pct))>1 else np.nan
                rank.append(dict(scenario=scenario,feature_set=feat,weight_policy=pol,
                    n_top_tied=int(mask.sum()),unique_top1=bool(mask.sum()==1),
                    top1_id=(chen.material_uid.iloc[np.argmax(pct)] if mask.sum()==1 else None),
                    regret_if_unique=(float(np.max(ychen)-ychen[np.argmax(pct)]) if mask.sum()==1 else None),
                    tied_regret_worst=float(np.max(ychen)-np.min(ychen[mask])),spearman_rho=(float(rho) if np.isfinite(rho) else None),
                    context='RETROSPECTIVE_CHEN_IN_PAPER_PH_MISSING'))
                rankpred[f'{scenario}__{feat}__{pol}']=pct
    pd.DataFrame(rank).to_csv(dst/'chen16_weighted_rank.csv',index=False)
    rankpred.to_csv(dst/'chen16_weighted_predictions.csv',index=False)
    # Save preregistered candidate: sqrt DOI + equal material on ALL 188, not posthoc best.
    deploy=root/DEPLOY;deploy.mkdir(exist_ok=True,parents=True)
    saved=[]
    for variant in ['A','B']:
        for feat in ['baseline','cap']:
            _,_,mdl,cols,nums,cats,wt=fit_predict(datasets['v57_188'],datasets['v57_188'].iloc[:2],variant,feat,'doi_sqrt_material_equal',True)
            name=f'{variant}_188_{feat}_doi_sqrt_material_equal_RESEARCH_NOT_ACTIVE.joblib'
            path=deploy/name; joblib.dump(mdl,path,compress=3)
            saved.append({'model':variant,'feature_set':feat,'weights':'doi_sqrt_material_equal','path':str(path.relative_to(root)),'sha256':sha(path),
                          'required_feature_columns':cols,'training_rows':188,'training_dois':int(datasets['v57_188'].doi.nunique()),
                          'sample_weight_sum':float(wt.sum())})
    # Report-driven sensitivity candidates. Exploratory only: selected on the
    # same historical OOF cohort, hence not valid confirmation evidence.
    for source_scenario,variant,feat,policy in [
        ('v55_152','B','cap','doi_sqrt_material_equal'),
        ('v57_188','B','cap','doi_sqrt_row_equal'),
    ]:
        full_data=datasets[source_scenario]
        _,_,mdl,cols,nums,cats,wt=fit_predict(full_data,full_data.iloc[:2],variant,feat,policy,True)
        name=f'{variant}_{len(full_data)}_{feat}_{policy}_POSTHOC_RESEARCH_NOT_ACTIVE.joblib'
        path=deploy/name;joblib.dump(mdl,path,compress=3)
        saved.append({'model':variant,'feature_set':feat,'weights':policy,
                      'path':str(path.relative_to(root)),'sha256':sha(path),
                      'required_feature_columns':cols,'training_rows':len(full_data),
                      'training_dois':int(full_data.doi.nunique()),
                      'selection':'POSTHOC_HISTORICAL_OLD140_NOT_INDEPENDENT',
                      'sample_weight_sum':float(wt.sum())})
    # Additional group inventory; source IDs are evidence labels, not extra features.
    keys,keys_source=material_key(datasets['v57_188'])
    groups=datasets['v57_188'][['observation_id','doi','material_uid','pollutant','C0_mg_L']].copy()
    groups['weight_material_group']=keys;groups['material_group_source']=keys_source
    groups.to_csv(dst/'material_id_mapping_audit.csv',index=False)
    config={'version':'v59','run_type':'RETROSPECTIVE_WEIGHTING_RESEARCH_NOT_ACTIVE','base_rows':140,'base_doi':32,
            'weights_definition':'doi_group_mass = n_doi ** alpha; material_share within doi proportional to n_material ** beta; row_share = material_share/n_material; then normalized to mean 1 per training fold',
            'policies':POLICIES,'no_target_used_for_weighting':True,'test_DOI_excluded_from_weight_computation':True,
            'preregistered_candidates_saved_without_score_selection':True,'posthoc_sensitivity_candidates_also_saved':True,'training_data_sha256':{k:sha(root/val) for k,val in [('old',OLD),('v55',PRE),('new',NEW)]},
            'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},'models':saved,
            'historical_sourceout_not_prospective':True}
    (dst/'run_manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nFIXED OLD140 DOI split weighted comparisons (MAE):')
    print(met[['model','scenario','feature_set','weight_policy','mae_mg_g','doi_macro_mae','gain_ci95_low','gain_ci95_high']].to_string(index=False))
    print('\nSaved',len(saved),'research-only model artifacts')
    return met

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');opt=ap.parse_args();main(opt.root)
