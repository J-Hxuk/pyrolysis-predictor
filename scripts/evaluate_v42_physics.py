"""v42: DOI-held-out evaluation of raw vs clipped vs intrinsically bounded qe.

Runs without modifying active model, active.json, source dataset or user UI.
This is an exploratory, repeatedly-consulted DOI cohort, NOT blind validation.

python -m scripts.evaluate_v42_physics --root .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from pyrolysis.biochar_stage2.v15_features import model_inputs
from pyrolysis.biochar_stage2.v42_physics import BoundedUptakeRegressor, physical_cap, clip_raw_predictions
from scripts.build_v37_full_refit import S2_CAT, audit_s2, input_frame, pipeline

DATA_REL = 'reports/v40_new9/Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv'
OUTPUT_REL = 'reports/v42_physics'
MODEL_REL = 'deployment/research_v42_NOT_ACTIVE'
STRATEGIES=('XGB_raw','XGB_clipped','Logit_Ridge_fraction','RF_fraction','FoldMean_fraction','FoldFullRemoval_cap')


def _xgb(model):
    fields=model_inputs(model)
    cats=[x for x in fields if x in S2_CAT]
    nums=[x for x in fields if x not in cats]
    return pipeline(nums,cats,'XGBoost'),nums,cats


def predict_oof(data):
    d=data.reset_index(drop=True).copy()
    audit_s2(d)
    if d.doi.nunique() < 5:
        raise ValueError('At least 5 DOI groups required')
    cap=physical_cap(d)
    yy=d.target_value.to_numpy(float)
    truth_frac=yy/cap
    if ((truth_frac<0)|(truth_frac>1+1e-8)).any():
        raise ValueError('Target does not respect mass-balance cap')
    groups=d.doi.astype(str).to_numpy()
    # Same, deterministic GroupKFold splits across all strategies and models.
    fold_ids=np.full(len(d),-1,dtype=int)
    answers={(model,method):np.full(len(d),np.nan) for model in ('A','B') for method in STRATEGIES}
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(d,yy,groups)):
        fold_ids[te]=fold
        for model in ('A','B'):
            fitted,nums,cats=_xgb(model)
            fitted.fit(input_frame(d.iloc[tr],nums,cats),yy[tr])
            raw=fitted.predict(input_frame(d.iloc[te],nums,cats))
            answers[(model,'XGB_raw')][te]=raw
            answers[(model,'XGB_clipped')][te]=clip_raw_predictions(raw,d.iloc[te])
            for kind,name in [('logit_ridge','Logit_Ridge_fraction'),('fraction_forest','RF_fraction')]:
                estimator=BoundedUptakeRegressor(model=model,kind=kind)
                estimator.fit(d.iloc[tr],yy[tr])
                answers[(model,name)][te]=estimator.predict(d.iloc[te])
            # Data-derived fraction baseline (not test-derived): no target leakage.
            train_fraction=(yy[tr]/cap[tr]).mean()
            answers[(model,'FoldMean_fraction')][te]=cap[te]*train_fraction
            answers[(model,'FoldFullRemoval_cap')][te]=cap[te]
    if (fold_ids<0).any() or any(not np.isfinite(x).all() for x in answers.values()):
        raise AssertionError('Invalid OOF coverage')
    records=d[['observation_id','doi','pollutant','feedstock_label_normalized','model_use_tier','target_origin']].copy()
    records['fold']=fold_ids
    records['qe_actual_mg_g']=yy
    records['cap_mg_g']=cap
    records['fraction_actual']=truth_frac
    long=[];metrics=[];doi_metrics=[]
    for (model,method),pred in answers.items():
        frac=pred/cap
        violation=(pred<-1e-9)|(pred>cap+1e-9)
        metric={'model':model,'method':method,'n':len(d),'n_doi':d.doi.nunique(),
                'qe_mae_mg_g':float(mean_absolute_error(yy,pred)),
                'qe_r2':float(r2_score(yy,pred)),
                'fraction_mae':float(mean_absolute_error(truth_frac,frac)),
                'raw_mass_balance_violations':int(violation.sum()),
                'at_cap_count':int(np.count_nonzero(np.isclose(pred,cap,atol=1e-8,rtol=1e-8))),
                'mean_pred_fraction':float(frac.mean())}
        for doi,g in d.groupby('doi',sort=True):
            ids=g.index.to_numpy()
            doi_metrics.append(dict(model=model,method=method,doi=doi,n_rows=len(ids),
                   qe_mae_mg_g=float(np.abs(yy[ids]-pred[ids]).mean()),
                   fraction_mae=float(np.abs(truth_frac[ids]-frac[ids]).mean()),
                   violates=int(violation[ids].sum())))
        metric['doi_macro_mae_mg_g']=float(np.mean([r['qe_mae_mg_g'] for r in doi_metrics if r['model']==model and r['method']==method]))
        metrics.append(metric)
        for ix in range(len(d)):
            long.append(dict(observation_id=d.at[ix,'observation_id'],doi=d.at[ix,'doi'],fold=int(fold_ids[ix]),
                 model=model,method=method,qe_actual_mg_g=float(yy[ix]),cap_mg_g=float(cap[ix]),
                 predicted_qe_mg_g=float(pred[ix]),actual_fraction=float(truth_frac[ix]),
                 pred_fraction=float(frac[ix]),abs_error_mg_g=float(abs(yy[ix]-pred[ix])),
                 raw_violation=bool(violation[ix]),conditional=bool(d.at[ix,'model_use_tier'] != 'CORE_EXPLORATORY')))
    return d, pd.DataFrame(long),pd.DataFrame(metrics),pd.DataFrame(doi_metrics)


def doi_bootstrap(long,model,comparison,baseline='XGB_raw',seed=42,n_boot=3000):
    df=long[long.model.eq(model)]
    a=df[df.method.eq(comparison)].set_index('observation_id')
    b=df[df.method.eq(baseline)].set_index('observation_id')
    ids=a.index.intersection(b.index)
    if len(ids)!=len(a) or len(ids)!=len(b):raise ValueError('row mismatch')
    # Negative is improvement relative to baseline.
    tmp=a.loc[ids][['doi','abs_error_mg_g']].copy()
    tmp['delta']=a.loc[ids,'abs_error_mg_g']-b.loc[ids,'abs_error_mg_g']
    group={doi:x.delta.to_numpy(float) for doi,x in tmp.groupby('doi')}
    dois=list(group)
    rng=np.random.default_rng(seed)
    dist=[]
    for _ in range(n_boot):
        select=rng.choice(len(dois),size=len(dois),replace=True)
        pooled=np.concatenate([group[dois[j]] for j in select])
        dist.append(float(pooled.mean()))
    return dict(model=model,baseline=baseline,comparison=comparison,
           delta_mae_mg_g=float(tmp.delta.mean()),bootstrap_CI95_mg_g=[float(x) for x in np.quantile(dist,[.025,.975])],
           n_doi=len(dois),bootstrap_reps=n_boot,unit='DOI cluster, sampled with replacement',
           note='Exploratory repeated-cohort comparison, NOT a confirmatory significance test')


def stress_panels(d,long):
    """Only exact matched, completely reported temperature comparisons.

    They are illustrative and not a validated recommendation score.
    """
    fields=['doi','feedstock_label_normalized','pollutant','C0_mg_L','dose_g_L',
            'initial_pH','adsorption_T_C','contact_time_min','pyrolysis_hold_min',
            'heating_rate_C_min','pyrolysis_atmosphere','activation_or_modification']
    if any(x not in d for x in fields+['pyrolysis_temperature_C']):return []
    keep=d.dropna(subset=['pyrolysis_temperature_C','C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min'])
    # Unknown manufacturing controls cannot be assumed identical; fields may be missing across all rows.
    out=[]
    if keep.empty:return out
    for key,grp in keep.groupby(fields,dropna=False):
        if len(grp)<3 or grp.pyrolysis_temperature_C.nunique()<3:continue
        panelid=grp.observation_id.tolist()
        if grp.target_value.max()-grp.target_value.min()<1e-6:continue
        for model in ('A','B'):
            for method in ('XGB_raw','XGB_clipped','Logit_Ridge_fraction','RF_fraction'):
                current=long[long.model.eq(model)&long.method.eq(method)&long.observation_id.isin(panelid)]
                actual=grp.set_index('observation_id').target_value
                pred=current.set_index('observation_id').predicted_qe_mg_g
                if len(current)!=len(grp):continue
                best_true=actual.idxmax();best_pred=pred.idxmax()
                out.append(dict(doi=grp.doi.iloc[0],n_temperatures=grp.pyrolysis_temperature_C.nunique(),
                      model=model,method=method,actual_best_temperature_C=float(grp.set_index('observation_id').loc[best_true,'pyrolysis_temperature_C']),
                      predicted_best_temperature_C=float(grp.set_index('observation_id').loc[best_pred,'pyrolysis_temperature_C']),
                      top1=int(best_true==best_pred),regret_mg_g=float(actual.max()-actual.loc[best_pred]),
                      warning='Single DOI matched panel, reused in training-design research; descriptive only'))
    return out


def run(root:Path,*,fit_research=True):
    root=Path(root);src=root/DATA_REL
    if not src.is_file():raise FileNotFoundError(f'v40 integrated source missing: {src}')
    data=pd.read_csv(src,low_memory=False)
    data,long,metrics,doi_metrics=predict_oof(data)
    out=root/OUTPUT_REL;out.mkdir(parents=True,exist_ok=True)
    long.to_csv(out/'doi_oof_predictions_long.csv',index=False)
    metrics.to_csv(out/'doi_group5_metrics.csv',index=False)
    doi_metrics.to_csv(out/'per_doi_mae.csv',index=False)
    # Since all observed y are within [0, cap], projection of predictions
    # onto [0, cap] can never worsen absolute error for any one observation.
    # Improved MAE after clipping is therefore algebraic, NOT new learning.
    for model in ('A','B'):
        x=long[(long.model==model)&(long.method=='XGB_raw')].set_index('observation_id')
        c=long[(long.model==model)&(long.method=='XGB_clipped')].set_index('observation_id')
        if not (c.abs_error_mg_g <= x.abs_error_mg_g+1e-8).all():
            raise AssertionError('Projection bound violated')
    deltas=[]
    for model in ('A','B'):
        for name in ('XGB_clipped','Logit_Ridge_fraction','RF_fraction','FoldMean_fraction','FoldFullRemoval_cap'):
            deltas.append(doi_bootstrap(long,model,name))
    (out/'paired_doi_bootstrap.json').write_text(json.dumps(deltas,ensure_ascii=False,indent=2),encoding='utf-8')
    # Same OOF observations and DOI clusters: B minus A, not independent CV scores.
    ab_comparison=[]
    for method in ('XGB_raw','XGB_clipped','RF_fraction','Logit_Ridge_fraction'):
        m=long[long.method.eq(method)]
        a=m[m.model.eq('A')].set_index('observation_id')
        b=m[m.model.eq('B')].set_index('observation_id')
        ix=a.index.intersection(b.index)
        dd=(b.loc[ix,'abs_error_mg_g']-a.loc[ix,'abs_error_mg_g'])
        grouped={doi:dd.loc[a.loc[ix,'doi'].eq(doi)].to_numpy(float) for doi in sorted(a.doi.unique())}
        rng=np.random.default_rng(2026)
        keys=list(grouped)
        bootstrap=[]
        for _ in range(3000):
            selection=rng.choice(keys,len(keys),replace=True)
            bootstrap.append(float(np.concatenate([grouped[k] for k in selection]).mean()))
        ab_comparison.append(dict(method=method,delta_MAE_B_minus_A=float(dd.mean()),
            bootstrap_CI95=[float(z) for z in np.quantile(bootstrap,[.025,.975])],n_doi=len(keys)))
    (out/'A_B_paired_doi_comparison.json').write_text(json.dumps(ab_comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    panels=stress_panels(data,long)
    pd.DataFrame(panels).to_csv(out/'matched_temperature_panels_DESCRIPTIVE_ONLY.csv',index=False)
    # Freeze OOF predictions, then compare their errors by source QA tier.
    # This is diagnostic stratification, NOT a fair CV across dataset revisions.
    tier_map=data.set_index('observation_id')['model_use_tier'].to_dict()
    long['source_tier']=long.observation_id.map(tier_map)
    sensitivity=[]
    for (model,method,tier),part in long.groupby(['model','method','source_tier']):
        sensitivity.append(dict(model=model,method=method,tier=tier,n=len(part),n_doi=part.doi.nunique(),
            MAE_mg_g=float(part.abs_error_mg_g.mean()),
            raw_violations=int(part.raw_violation.sum()),
            at_upper_cap=int(np.isclose(part.predicted_qe_mg_g,part.cap_mg_g,rtol=1e-8,atol=1e-8).sum())))
    pd.DataFrame(sensitivity).to_csv(out/'qa_tier_oof_diagnostics.csv',index=False)
    summary={'source':DATA_REL,'rows':len(data),'doi':int(data.doi.nunique()),
       'conditional_rows':int(data.model_use_tier.ne('CORE_EXPLORATORY').sum()),
       'real_at_or_above_99pct_cap':int((data.target_value.to_numpy(float)/physical_cap(data)>=0.99).sum()),
       'validation':'GroupKFold DOI=5 (research, repeated selection; nonnested)',
       'deploy':False,'methods':list(STRATEGIES),'metrics':metrics.to_dict('records'),
       'bootstrap':deltas,'A_vs_B':ab_comparison,'temperature_panels':len(panels),
       'by_source_tier':sensitivity,
       'limitations':['qe_reported is not independently certified equilibrium qe',
       'Mass balance cap does not prove adsorption; precipitation etc may contribute',
       'Clipping can improve MAE without learning adsorption physics',
       'Hyperparameters fixed for exploratory research, reused DOI evidence not blinded',
       'Missing pH and time for 12 coffee observations; conditional rows need quality sensitivity',
       'Few independent DOIs per pollutant and feedstock combination']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if fit_research:
        dest=root/MODEL_REL;dest.mkdir(parents=True,exist_ok=True)
        y=data.target_value.to_numpy(float)
        for model in ('A','B'):
            for kind in ('logit_ridge','fraction_forest'):
                clf=BoundedUptakeRegressor(model=model,kind=kind).fit(data,y)
                joblib.dump(clf,dest/f'V42_{model}_{kind}_128_RESEARCH_ONLY.joblib')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=Path(__file__).resolve().parents[1],type=Path)
    ap.add_argument('--no-fit',action='store_true')
    args=ap.parse_args()
    summary=run(args.root,fit_research=not args.no_fit)
    print(json.dumps({'rows':summary['rows'],'doi':summary['doi'],'saved_research':not args.no_fit,
       'metrics':summary['metrics'],'temperature_panels':summary['temperature_panels']},ensure_ascii=False,indent=2))
