"""v53 Stage2 evidence-tiered research-only refit. NEVER changes active deployment.

Usage: python -m scripts.run_v53_research_refit --root .
The vNext10 + vNext11 extracts are audits, not new automatically approved data.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.build_v46_phase8 import s2_model, features

DOI_APPLE='10.3390/en13133498'
BASEREL='data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
EXTRAREL='data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv'
MERGEDREL='data/imports/v52_apple_series/Stage2_164_RESEARCH_NOT_ACTIVE.csv'
REPORTREL='reports/v53_evidence_tier_refit'
MODELREL='deployment/research_v53_NOT_ACTIVE'
VERIFYREL='data/imports/v53_source_review'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normdoi(s):return str(s).strip().lower()

def feature_matrix(d, cols,nums,cats):return features(d,cols,nums,cats)

def verify_and_load(root):
    base=pd.read_csv(root/BASEREL,low_memory=False)
    extra=pd.read_csv(root/EXTRAREL,low_memory=False)
    merged=pd.read_csv(root/MERGEDREL,low_memory=False)
    assert len(base)==140 and base.doi.nunique()==32,(len(base),base.doi.nunique())
    assert len(extra)==24 and extra.doi.nunique()==1 and extra.doi.iloc[0].lower()==DOI_APPLE
    assert len(merged)==164 and merged.doi.nunique()==33
    assert len(base.observation_id.unique())==140 and len(extra.observation_id.unique())==24
    assert not(set(base.observation_id)&set(extra.observation_id))
    assert set(merged.observation_id)==set(base.observation_id)|set(extra.observation_id)
    assert not set(base.doi.str.lower())&set(extra.doi.str.lower())
    assert (extra.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP').sum()==8
    assert (extra.target_origin=='V52_ZHAO2020_SECTION3_1_TEXT_EXPERIMENT1').sum()==16
    for d in [base,extra,merged]:
        assert d.observation_id.is_unique
        qe=pd.to_numeric(d.target_value,errors='coerce').to_numpy(float)
        c0=pd.to_numeric(d.C0_mg_L,errors='coerce').to_numpy(float)
        dose=pd.to_numeric(d.dose_g_L,errors='coerce').to_numpy(float)
        assert np.isfinite(qe).all() and np.isfinite(c0).all() and np.isfinite(dose).all()
        assert (dose>0).all() and (c0>0).all() and (qe>=0).all() and (qe<=c0/dose+1e-6).all()
    apple8=extra.loc[extra.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP'].copy()
    apple16=extra.loc[extra.target_origin!='V52_ZHAO2020_TABLE3_QE_EXP'].copy()
    src10=root/VERIFYREL/'AppleTree_CuZn_Observed_qe_and_Linkage.csv'
    assert src10.exists(),'vNext10 source audit CSV missing: '+str(src10)
    q=pd.read_csv(src10,low_memory=False)
    assert len(q)==8
    assert q[['doi','pollutant','dose_g_L','recipe_pyrolysis_temp_C']].duplicated().sum()==0
    matches=[]
    for _,r in apple8.iterrows():
        z=q.loc[(q.doi.astype(str).str.lower()==normdoi(r.doi))&
            (q.pollutant==r.pollutant)&
            np.isclose(pd.to_numeric(q.dose_g_L),float(r.dose_g_L))&
            np.isclose(pd.to_numeric(q.recipe_pyrolysis_temp_C),float(r.pyrolysis_temperature_C))]
        assert len(z)==1,(r.observation_id,len(z))
        found=z.iloc[0]
        assert np.isclose(float(found.qe_standardized_mg_g),float(r.target_value),atol=1e-8)
        assert np.isclose(float(found.C0_mg_L),float(r.C0_mg_L),atol=1e-8)
        assert np.isclose(float(found.initial_pH),float(r.initial_pH),atol=1e-8)
        matches.append({'v52_id':r.observation_id,'vNext10_id':found.observation_id,'doi':r.doi,
            'pollutant':r.pollutant,'temperature_C':float(r.pyrolysis_temperature_C),
            'dose_g_L':float(r.dose_g_L),'qe_mg_g':float(r.target_value),
            'action':'EXISTING_V52_SAME_OBSERVATION_DO_NOT_APPEND',
            'vNext10_training_eligible_flag':str(found.TRAINING_ELIGIBLE)})
    q11=root/VERIFYREL/'TeaBranch_Strict_qe_Review_Resolution.csv'
    assert q11.exists(),'vNext11 CSV missing: '+str(q11)
    tea=pd.read_csv(q11,low_memory=False)
    assert len(tea)==5
    assert all(normdoi(x)=='10.1039/d1ra04235a' for x in tea.doi)
    if any(normdoi(x)=='10.1039/d1ra04235a' for x in base.doi):
        raise ValueError('Tea 5 unexpectedly in v46 base; manually reconcile')
    assert not tea['TRAINING_ELIGIBLE'].astype(str).str.lower().isin(['true','yes','approved']).any(), 'Tea review flags unexpected approval'
    return base,apple8,apple16,extra,pd.DataFrame(matches),tea


def fit_pred(train,test,variant):
    pipe,cols,nums,cats=s2_model(variant)
    pipe.fit(feature_matrix(train,cols,nums,cats),pd.to_numeric(train.target_value).to_numpy(float))
    raw=np.asarray(pipe.predict(feature_matrix(test,cols,nums,cats)),float)
    cap=(test.C0_mg_L/test.dose_g_L).to_numpy(float)
    return np.clip(raw,0,cap),raw


def score(df,pred,raw=None):
    y=df.target_value.to_numpy(float);cap=(df.C0_mg_L/df.dose_g_L).to_numpy(float)
    return {'n':int(len(df)),'doi_count':int(df.doi.nunique()),'mae_mg_g':float(mean_absolute_error(y,pred)),
        'rmse_mg_g':float(np.sqrt(mean_squared_error(y,pred))),
        'r2':float(r2_score(y,pred)) if len(df)>1 else None,
        'physical_violations_after_clip':int(((pred<0)|(pred>cap+1e-8)).sum()),
        'cap_hits_after_clip':int(np.isclose(pred,cap,atol=1e-8).sum()),
        'lower_bound_hits_after_clip':int(np.isclose(pred,0,atol=1e-8).sum()),
        'raw_upper_violations':int((raw>cap+1e-8).sum()) if raw is not None else None,
        'raw_negative_violations':int((raw<0).sum()) if raw is not None else None}


def fixed_original_evaluation(base,apple8,apple24):
    base=base.reset_index(drop=True)
    scenarios={'base140':pd.DataFrame(), 'plus8':apple8,'plus24':apple24}
    out=base[['observation_id','doi','pollutant','target_value','C0_mg_L','dose_g_L']].copy()
    out['cap_mg_g']=base.C0_mg_L/base.dose_g_L
    folds=[]
    for s in scenarios:out['fold_'+s]=0
    for variant in ('A','B'):
        for s in scenarios:
            out[f'{variant}_{s}']=np.nan
            out[f'{variant}_{s}_raw']=np.nan
    for f,(tr,te) in enumerate(GroupKFold(n_splits=5).split(base,groups=base.doi),1):
        train0,test=base.iloc[tr],base.iloc[te]
        assert set(train0.doi).isdisjoint(test.doi)
        for s,addon in scenarios.items():
            if len(addon): assert set(addon.doi).isdisjoint(test.doi)
            t=pd.concat([train0,addon],ignore_index=True,sort=False) if len(addon) else train0
            out.loc[te,'fold_'+s]=f
            for variant in ('A','B'):
                pred,raw=fit_pred(t,test,variant)
                out.loc[te,f'{variant}_{s}']=pred
                out.loc[te,f'{variant}_{s}_raw']=raw
        folds.append({'fold':f,'train_old_rows':int(len(tr)),'test_old_rows':int(len(te)),
            'test_dois':';'.join(sorted(test.doi.unique())), 'added_table3_rows':8,'added_allapple_rows':24})
    rows=[]
    for v in ('A','B'):
        for s in scenarios:
            pp=out[f'{v}_{s}'].to_numpy(float)
            raw=out[f'{v}_{s}_raw'].to_numpy(float)
            assert np.isfinite(pp).all()
            rows.append({'model':v,'scenario':s,'evaluation':'SAME_ORIGINAL_140_GROUPKFOLD5','training_extra_rows':int(len(scenarios[s])),**score(base,pp,raw)})
    return pd.DataFrame(rows),out,pd.DataFrame(folds)


def apple_preinclusion_test(base,apple24):
    out=apple24[['observation_id','doi','pollutant','target_value','dose_g_L','pyrolysis_temperature_C','target_origin']].copy()
    summary=[]
    for v in ('A','B'):
        pp,raw=fit_pred(base,apple24,v)
        out[v+'_pred']=pp
        for kind,mask in [('strict8',out.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP'),
                          ('qt16',out.target_origin=='V52_ZHAO2020_SECTION3_1_TEXT_EXPERIMENT1')]:
            d=apple24.loc[mask]
            p=pp[np.asarray(mask)]
            summary.append({'model':v,'subset':kind,'evaluation':'PRE_INCLUSION_SINGLE_DOI_CHALLENGE_NOT_INDEPENDENT',**score(d,p,raw[np.asarray(mask)])})
    return pd.DataFrame(summary),out


def paired_bootstrap(pred,seed=53,n_boot=3000):
    rng=np.random.default_rng(seed)
    group=[g for _,g in pred.groupby('doi',sort=True)]
    diffs=[]
    for v in ('A','B'):
        for comp in ('plus8','plus24'):
            y=pred.target_value.to_numpy(float)
            a=np.abs(y-pred[f'{v}_base140'].to_numpy(float))
            b=np.abs(y-pred[f'{v}_{comp}'].to_numpy(float))
            delta=float(np.mean(a-b)) # improvement > 0
            boot=np.empty(n_boot)
            groups=list(pred.groupby('doi',sort=True).indices.values())
            for i in range(n_boot):
                chosen=rng.integers(0,len(groups),size=len(groups))
                inds=np.concatenate([groups[k] for k in chosen])
                boot[i]=float(np.mean(a[inds]-b[inds]))
            diffs.append({'model':v,'augmented':comp,'mae_improvement_mg_g':delta,
                'doi_cluster_bootstrap_ci_low':float(np.quantile(boot,.025)),
                'doi_cluster_bootstrap_ci_high':float(np.quantile(boot,.975)),
                'doi_groups':len(groups),'bootstrap_replicates':n_boot,
                'caveat':'Fixed original 140 DOI folds; training augmentation study not a new independent external holdout'})
    return pd.DataFrame(diffs)


def save_models(root,base,apple8,apple24):
    target_dir=root/MODELREL;target_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for label,addon in [('strict8_sensitivity',apple8),('all24_sensitivity',apple24)]:
        combined=pd.concat([base,addon],ignore_index=True,sort=False)
        for v in ('A','B'):
            model,cols,nums,cats=s2_model(v)
            model.fit(features(combined,cols,nums,cats),combined.target_value.to_numpy(float))
            path=target_dir/f'V53_STAGE2_{v}_{len(combined)}_{label}_NOT_ACTIVE.joblib'
            joblib.dump(model,path,compress=3)
            reload=joblib.load(path)
            vals=reload.predict(features(combined.head(3),cols,nums,cats))
            assert np.isfinite(vals).all()
            records.append({'model':v,'scenario':label,'path':str(path.relative_to(root)),
                'sha256':digest(path),'nrows':len(combined),'doi_count':int(combined.doi.nunique()),
                'features':cols,'status':'RESEARCH_ONLY_NOT_APPROVED_NOT_ACTIVE'})
    (target_dir/'manifest_v53.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    return records


def main(root):
    root=Path(root).resolve()
    base,apple8,apple16,apple24,matches,tea=verify_and_load(root)
    out=root/REPORTREL;out.mkdir(parents=True,exist_ok=True)
    matches.to_csv(out/'vNext10_v52_eight_duplicate_match.csv',index=False)
    pd.concat([base,apple8],ignore_index=True,sort=False).to_csv(root/VERIFYREL/'Stage2_148_RESEARCH_NOT_ACTIVE.csv',index=False)
    m,p,folds=fixed_original_evaluation(base,apple8,apple24)
    m.to_csv(out/'v53_fixed140_metrics.csv',index=False)
    p.to_csv(out/'v53_fixed140_predictions.csv',index=False)
    folds.to_csv(out/'v53_fixed140_fold_audit.csv',index=False)
    h,apple_pred=apple_preinclusion_test(base,apple24)
    h.to_csv(out/'apple_preinclusion_single_doi_metrics.csv',index=False)
    apple_pred.to_csv(out/'apple_preinclusion_single_doi_predictions.csv',index=False)
    ci=paired_bootstrap(p)
    ci.to_csv(out/'doi_cluster_paired_bootstrap.csv',index=False)
    status={'input_rows_baseline':len(base),'new_approved_training_rows_from_vNext10':0,
        'new_approved_training_rows_from_vNext11':0,
        'overlap_vNext10_v52_table3_rows':len(matches),'v52_table3_qe_8':len(apple8),
        'v52_24h_qt_16':len(apple16),'vNext11_existing_tea_strict_qe_5_not_added':len(tea),
        'vNext11_tea_in_baseline_140':0,'candidate_148_research_only':len(base)+len(apple8),
        'candidate_164_sensitivity_only':len(base)+len(apple24),
        'baseline_warning':'Core140 includes prior exploratory, approximate, conditional targets; it is NOT an all-strict-qe benchmark.',
        'vNext10_training_eligible':'PENDING_CROSS_SOURCE_MATERIAL_PROPERTY_REVIEW; 148 RESEARCH_ONLY, not approved',
        'provenance':{'stage2_140_sha256':digest(root/BASEREL),'v52_apple24_sha256':digest(root/EXTRAREL),
            'v52_164_sha256':digest(root/MERGEDREL),
            'vNext10_audit_csv_sha256':digest(root/VERIFYREL/'AppleTree_CuZn_Observed_qe_and_Linkage.csv'),
            'vNext11_tea_review_sha256':digest(root/VERIFYREL/'TeaBranch_Strict_qe_Review_Resolution.csv')},
        'python_version':platform.python_version(),'sklearn_version':sklearn.__version__,
        'xgboost_version':xgboost.__version__,'pandas_version':pd.__version__,
        'source_data_unchanged':True,'active_model_unchanged':True}
    models=save_models(root,base,apple8,apple24)
    report={'status':status,'metrics':m.to_dict('records'),'apple_preinclusion':h.to_dict('records'),
        'bootstrap':ci.to_dict('records'),'models':models}
    (out/'v53_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'metrics':m.to_dict('records'),'bootstrap':ci.to_dict('records'),
                      'new_training_rows':0,'model_files':len(models)},ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('.'));args=a.parse_args()
    main(args.root)
