"""v54 research-only bounded adsorption model and manufacturing rank benchmark.

Data provenance: retains v46 140 rows; adds v52 Zhao Table 3 eight strict source-labelled qe
for research sensitivity only. Its material-property link still awaits review.
No active model, UI, Stage1, raw data, or v52/v53 research model is changed.

    python -m scripts.run_v54_bounded_rank --root .
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy, sklearn, xgboost
from scipy.special import expit,logit
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold
from scripts.build_v46_phase8 import s2_model,features
from scripts.run_v53_research_refit import verify_and_load

N_SPLITS=5
LOGIT_EPS=1e-4
ROOT_REPORT='reports/v54_bounded_ranking'
ROOT_DEPLOY='deployment/research_v54_NOT_ACTIVE'
CHEN='data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/09_Manufacturing_Recommendation_Benchmark_Chen16.csv'
DOI_CHEN='10.1038/s41598-022-22714-y'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cap_of(frame):
    c0=pd.to_numeric(frame['C0_mg_L'],errors='raise').to_numpy(float)
    dose=pd.to_numeric(frame['dose_g_L'],errors='raise').to_numpy(float)
    if not(np.isfinite(c0).all() and np.isfinite(dose).all() and (c0>0).all() and (dose>0).all()):
        raise ValueError('Invalid C0 or dose: refuse physical model, do not silently impute')
    return c0/dose

def check_target(frame):
    y=pd.to_numeric(frame['target_value'],errors='raise').to_numpy(float)
    cap=cap_of(frame)
    if not(np.isfinite(y).all() and (y>=0).all() and (y<=cap+1e-8).all()):
        raise ValueError('Target outside physical mass-balance bounds')
    return y,cap

def fit_model(data, variant, method):
    y,cap=check_target(data)
    model,cols,nums,cats=s2_model(variant)
    if method=='logit_fraction':
        frac=np.clip(y/cap,LOGIT_EPS,1-LOGIT_EPS)
        target=logit(frac)
    elif method=='logistic_fraction':
        model.set_params(model__objective='reg:logistic',model__base_score=0.5)
        target=y/cap
    elif method=='direct_clip':target=y
    else: raise ValueError(method)
    model.fit(features(data,cols,nums,cats),target)
    return model,cols,nums,cats

def infer(model, cols,nums,cats, data,method):
    cap=cap_of(data)
    raw=np.asarray(model.predict(features(data,cols,nums,cats)),dtype=float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite model output')
    if method=='logit_fraction':
        fraction=expit(raw)
        pred=cap*fraction
        if not ((pred>=0).all() and (pred<=cap+1e-8).all()):raise ValueError('Bound violation in bounded model')
        return pred,raw,fraction
    if method=='logistic_fraction':
        fraction=raw
        pred=cap*fraction
        if not ((pred>=0).all() and (pred<=cap+1e-8).all()):raise ValueError('Bound violation in logistic objective model')
        return pred,raw,fraction
    if method=='direct_clip':
        return np.clip(raw,0,cap),raw,np.clip(raw/cap,0,1)
    raise ValueError(method)

def metric(y,p):
    return {'mae_mg_g':float(mean_absolute_error(y,p)),
            'rmse_mg_g':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)) if len(y)>1 and not np.isclose(np.var(y),0) else None}

def eval_same140(base,apple8,apple24):
    base=base.reset_index(drop=True)
    scenarios={'base140':base.iloc[:0].copy(),'plus8':apple8,'plus24':apple24}
    methods=['direct_clip','logit_fraction','logistic_fraction']
    variants=['A','B']
    base_y,base_cap=check_target(base)
    pred=base[['observation_id','doi','pollutant','target_value','C0_mg_L','dose_g_L']].copy()
    pred['mass_balance_cap_mg_g']=base_cap
    pred['fold']=0
    for v in variants:
        for scenario in scenarios:
            for method in methods:
                prefix=f'{v}__{scenario}__{method}'
                pred[prefix]=np.nan
                pred[prefix+'__raw']=np.nan
    folds=[]
    for fold,(i_train,i_test) in enumerate(GroupKFold(n_splits=N_SPLITS).split(base,groups=base.doi),1):
        tr,te=base.iloc[i_train],base.iloc[i_test]
        if not set(tr.doi).isdisjoint(set(te.doi)):raise AssertionError('DOI leakage')
        pred.loc[i_test,'fold']=fold
        folds.append({'fold':fold,'old_train_rows':len(tr),'old_test_rows':len(te),
                      'test_dois':';'.join(sorted(te.doi.unique()))})
        for scenario,addon in scenarios.items():
            if not set(addon.doi).isdisjoint(set(te.doi)):raise AssertionError('New DOI leakage')
            train=pd.concat([tr,addon],ignore_index=True,sort=False) if len(addon) else tr
            for v in variants:
                for method in methods:
                    model,cols,nums,cats=fit_model(train,v,method)
                    pr,raw,fraction=infer(model,cols,nums,cats,te,method)
                    prefix=f'{v}__{scenario}__{method}'
                    pred.loc[i_test,prefix]=pr
                    pred.loc[i_test,prefix+'__raw']=raw
    scores=[]
    for v in variants:
        for scenario,addon in scenarios.items():
            for method in methods:
                prefix=f'{v}__{scenario}__{method}'
                pp=pred[prefix].to_numpy(float)
                rr=pred[prefix+'__raw'].to_numpy(float)
                if not np.isfinite(pp).all():raise AssertionError('Missing OOF predictions')
                nrawupper=int((rr>base_cap+1e-8).sum()) if method=='direct_clip' else 0
                nrawlower=int((rr<0).sum()) if method=='direct_clip' else 0
                scores.append({'variant':v,'training_scenario':scenario,'method':method,
                               'evaluation':'SAME_140_DOI_GROUPKFOLD5','evaluation_rows':len(base),
                               'evaluation_dois':base.doi.nunique(),'train_extra_rows':len(addon),
                               'predicted_upper_violation':int((pp>base_cap+1e-8).sum()),
                               'predicted_lower_violation':int((pp<0).sum()),
                               'direct_unconstrained_upper_violation':nrawupper,
                               'direct_unconstrained_negative_violation':nrawlower,
                               'fraction_edge_hits':int((np.isclose(pp,base_cap,rtol=0,atol=1e-8)|np.isclose(pp,0,rtol=0,atol=1e-8)).sum()),
                               **metric(base_y,pp)})
    return pd.DataFrame(scores),pred,pd.DataFrame(folds)

def paired_bootstrap(pred, n_boot=2000,seed=54):
    groups=[inds for inds in pred.groupby('doi',sort=True).indices.values()]
    rng=np.random.default_rng(seed)
    boot_lookups=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(n_boot)]
    y=pred['target_value'].to_numpy(float)
    rows=[]
    for v in ['A','B']:
        for scenario in ['base140','plus8','plus24']:
            direct=pred[f'{v}__{scenario}__direct_clip'].to_numpy(float)
            for method in ['logit_fraction','logistic_fraction']:
                bounded=pred[f'{v}__{scenario}__{method}'].to_numpy(float)
                delta=np.abs(y-direct)-np.abs(y-bounded)
                boot=np.asarray([delta[i].mean() for i in boot_lookups])
                rows.append({'variant':v,'training_scenario':scenario,'method':method,
                         'mae_improvement_bounded_over_direct_mg_g':float(delta.mean()),
                         'ci95_low':float(np.quantile(boot,.025)),
                         'ci95_high':float(np.quantile(boot,.975)),
                         'n_doi':len(groups),'bootstrap_repeats':n_boot,
                         'warning':'Observational DOI-cluster interval; not causal proof; model selection on same folds incurs optimism'})
    return pd.DataFrame(rows)

def rank_metrics(ids,observed,estimated):
    observed=np.asarray(observed,float);estimated=np.asarray(estimated,float)
    if not (np.isfinite(observed).all() and np.isfinite(estimated).all()):raise ValueError('Nonfinite ranking data')
    if len(ids)!=len(observed):raise ValueError('Rank series length mismatch')
    ties=bool(np.isclose(np.ptp(estimated),0,atol=1e-10,rtol=0))
    order=np.argsort(-estimated,kind='stable')
    best=float(np.max(observed))
    top1=observed[order[0]]
    best3=float(np.max(observed[order[:min(3,len(order))]]))
    rho=spearmanr(observed,estimated).statistic if not ties and np.ptp(observed)>0 else np.nan
    tau=kendalltau(observed,estimated).statistic if not ties and np.ptp(observed)>0 else np.nan
    return {'n':len(ids),'spearman_rho':float(rho) if np.isfinite(rho) else None,
            'kendall_tau':float(tau) if np.isfinite(tau) else None,
            'all_prediction_tie':ties,'observed_best_id':ids[int(np.argmax(observed))],
            'observed_best_value':best,
            'predicted_top1_id':None if ties else ids[order[0]],
            'top1_regret':None if ties else float(best-top1),
            'top3_regret':None if ties else float(best-best3),
            'warning':'Top-1/3 withheld when tied; arbitrary first-file-order recommendation prohibited' if ties else ''}

def apple_rank_test(base,apple8):
    # Strict eight belong to two fixed-condition in-paper 4-temperature ranking series.
    # The apple DOI is not used in Stage2 training for this benchmark, BUT its related
    # 2017 physical-property DOI was historically exposed to other Stage1 models.
    train=base
    out=apple8[['observation_id','doi','pollutant','target_value','C0_mg_L','dose_g_L','pyrolysis_temperature_C']].copy()
    result=[]
    for v in ['A','B']:
        for method in ['direct_clip','logit_fraction','logistic_fraction']:
            model,cols,nums,cats=fit_model(train,v,method)
            pr,raw,frac=infer(model,cols,nums,cats,apple8,method)
            out[f'{v}__{method}']=pr
            for pollutant,g in out.groupby('pollutant',sort=True):
                scores=rank_metrics(g.observation_id.to_list(),g.target_value.to_numpy(float),
                                    g[f'{v}__{method}'].to_numpy(float))
                result.append({'benchmark':'ZHAO2020_STRICT_TABLE3','variant':v,'method':method,
                               'pollutant':pollutant,'measure':'qe_mg_g','test_doi':str(g.doi.iloc[0]),
                               'status':'SINGLE_DOI_PREINCLUSION_RELATED_STAGE1_EXPOSED_NOT_EXTERNAL',**scores})
    return pd.DataFrame(result),out

def chen_features(chen, assumed_pH=None):
    if not (len(chen)==16 and chen.material_uid.is_unique and chen.paper_group.nunique()==1 and
            str(chen.paper_group.iloc[0]).lower()==DOI_CHEN):raise ValueError('Chen reference group mismatch')
    # Source establishes Cd2+ and original unmodified char; paper does NOT give
    # a pH measurement corresponding to these 16 comparison trials. Keep NR.
    # No manual mapping of the adsorption benchmark to experimental qe labels.
    df=pd.DataFrame(index=chen.index)
    df['pollutant']='Cd(II)'
    df['C0_mg_L']=chen.C0_mg_L
    df['initial_pH']=np.nan if assumed_pH is None else float(assumed_pH)
    df['dose_g_L']=chen.dose_g_L
    df['adsorption_T_C']=chen.adsorption_T_C
    df['contact_time_min']=chen.contact_time_min
    df['feedstock_label_normalized']='corn stover'
    df['manufacturing_route']='SINGLE_STAGE_PYROLYSIS'
    df['process_step_count']=1
    df['pyrolysis_temperature_C']=chen.temperature_C
    df['pyrolysis_hold_min']=chen.hold_min
    df['heating_rate_C_min']=chen.heating_rate_C_min
    df['pyrolysis_atmosphere']=chen.atmosphere
    df['activation_or_modification']='none'
    df['secondary_temperature_C']=np.nan
    df['secondary_hold_min']=np.nan
    return df

def chen_rank_test(base,chen):
    out=chen[['material_uid','paper_group','reported_removal_pct','observed_rank_in_16',
              'temperature_C','hold_min','heating_rate_C_min','initial_pH']].copy()
    inp=chen_features(chen)
    results=[]
    for v in ['A','B']:
        for method in ['direct_clip','logit_fraction','logistic_fraction']:
            model,cols,nums,cats=fit_model(base,v,method)
            pr,raw,frac=infer(model,cols,nums,cats,inp,method)
            # Since C0 and dose fixed within this paper, qe rank == removal % rank.
            removal_pred=100*frac
            out[f'{v}__{method}__predicted_removal_pct']=removal_pred
            scores=rank_metrics(out.material_uid.to_list(),out.reported_removal_pct.to_numpy(float), removal_pred)
            results.append({'benchmark':'CHEN2022_16_ORTHOGONAL','variant':v,'method':method,
                            'pollutant':'Cd(II)','measure':'removal_pct','test_doi':DOI_CHEN,
                            'status':'PAPER_DOI_ABSENT_STAGE2_BUT_UNVERIFIED_OTHER_STAGE_EXPOSURE;_PH_NR_IMPUTED;_RANK_ONLY',
                            **scores})
    return pd.DataFrame(results),out

def chen_ph_sensitivity(base,chen):
    """Hypothetical pH scenarios, not new source measurements."""
    summary=[]
    for pHscenario in [None,4.0,6.0,7.0]:
        inp=chen_features(chen,pHscenario)
        for method in ['direct_clip','logit_fraction','logistic_fraction']:
            model,cols,nums,cats=fit_model(base,'B',method)
            pr,raw,frac=infer(model,cols,nums,cats,inp,method)
            res=rank_metrics(chen.material_uid.to_list(),chen.reported_removal_pct.to_numpy(float),100*frac)
            summary.append({'model':'B','method':method,
                'pH_scenario':'NR_MODEL_MEDIAN_IMPUTED' if pHscenario is None else 'HYPOTHETICAL_'+str(pHscenario),
                'source_pH':'NOT_REPORTED_FOR_16_EXPERIMENTS',**res})
    return pd.DataFrame(summary)

def save_models(base,apple8,apple24,root):
    dest=root/ROOT_DEPLOY;dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for scenario,addon in [('plus8',apple8),('plus24',apple24)]:
        tr=pd.concat([base,addon],ignore_index=True,sort=False)
        for v in ['A','B']:
            for method in ['logit_fraction','logistic_fraction']:
                m,cols,nums,cats=fit_model(tr,v,method)
                path=dest/f'V54_STAGE2_{v}_{len(tr)}_{method}_{scenario}_RESEARCH_NOT_ACTIVE.joblib'
                joblib.dump(m,path,compress=3)
                loaded=joblib.load(path)
                p1,_,_=infer(m,cols,nums,cats,tr.head(5),method)
                p2,_,_=infer(loaded,cols,nums,cats,tr.head(5),method)
                if not np.allclose(p1,p2,rtol=1e-7):raise AssertionError('Reloaded model differs')
                records.append({'variant':v,'scenario':scenario,'method':method,
                    'inference':'qe=(C0/dose)*expit(model.predict(X))' if method=='logit_fraction' else 'qe=(C0/dose)*model.predict(X)',
                    'logit_epsilon':LOGIT_EPS if method=='logit_fraction' else None,
                    'model_path':str(path.relative_to(root)).replace('\\','/'),'sha256':digest(path),
                    'train_rows':len(tr),'train_dois':tr.doi.nunique(),
                    'features':cols,'status':'RESEARCH_ONLY_NOT_ACTIVE'})
    (dest/'manifest_v54.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    return records

def main(root):
    root=Path(root).resolve()
    base,apple8,apple16,apple24,matches,tea=verify_and_load(root)
    chen=pd.read_csv(root/CHEN,low_memory=False)
    assert DOI_CHEN not in set(base.doi.astype(str).str.lower())
    assert len(chen)==16 and chen.reported_removal_pct.between(0,100).all()
    output=root/ROOT_REPORT;output.mkdir(parents=True,exist_ok=True)
    scores,pred,folds=eval_same140(base,apple8,apple24)
    boot=paired_bootstrap(pred)
    rankapple,rankapple_rows=apple_rank_test(base,apple8)
    rankchen,rankchen_rows=chen_rank_test(base,chen)
    phsensitivity=chen_ph_sensitivity(base,chen)
    records=save_models(base,apple8,apple24,root)
    for filename,data in [('v54_same140_metrics.csv',scores),('v54_same140_oof_predictions.csv',pred),
        ('v54_fold_audit.csv',folds),('v54_paired_bootstrap.csv',boot),
        ('v54_apple_ranking_metrics.csv',rankapple),('v54_apple_ranking_predictions.csv',rankapple_rows),
        ('v54_chen_ranking_metrics.csv',rankchen),('v54_chen_ranking_predictions.csv',rankchen_rows),
        ('v54_chen_pH_sensitivity.csv',phsensitivity)]:
        data.to_csv(output/filename,index=False,encoding='utf-8-sig')
    results={'data':{'base_rows':len(base),'apple_strict8':len(apple8),
        'apple_qt16_only_sensitivity':len(apple16),'chen_removal_benchmark':len(chen),
        'vNext11_new_training_rows':0,'study':'research_only_not_deployable',
        'stage1_cross_source_exposure_not_full_audited':True,
        'target_approval_note':'Baseline140 is not all strict qe. Apple8 remains cross-source property-link pending approval.'},
        'software_versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,
                             'xgboost':xgboost.__version__,'scipy':scipy.__version__,
                             'pandas':pd.__version__},
        'source_hashes':{'base140':digest(root/'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'),
                'apple24':digest(root/'data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv'),
                'chen16':digest(root/CHEN)},
        'models':records,'same140_metrics':scores.to_dict('records'),'paired_bootstrap':boot.to_dict('records'),
        'apple_rank':rankapple.where(pd.notna(rankapple),None).to_dict('records'),
        'chen_rank':rankchen.where(pd.notna(rankchen),None).to_dict('records'),
        'chen_pH_sensitivity':phsensitivity.where(pd.notna(phsensitivity),None).to_dict('records')}
    def clean(obj):
        if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
        if isinstance(obj,(tuple,list)):return [clean(v) for v in obj]
        if isinstance(obj,(int,float,np.integer,np.floating)):
            x=float(obj)
            return None if not np.isfinite(x) else (int(obj) if isinstance(obj,(int,np.integer)) else x)
        return obj
    (output/'v54_summary.json').write_text(json.dumps(clean(results),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'model_count':len(records),'same140_metrics':scores[['variant','training_scenario','method','mae_mg_g','predicted_upper_violation','direct_unconstrained_upper_violation']].to_dict('records'),
                      'apple_rank':rankapple[['variant','method','pollutant','spearman_rho','top1_regret']].to_dict('records'),
                      'chen_rank':rankchen[['variant','method','spearman_rho','top1_regret']].to_dict('records')},ensure_ascii=False,indent=2))
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.')
    args=ap.parse_args();main(args.root)
