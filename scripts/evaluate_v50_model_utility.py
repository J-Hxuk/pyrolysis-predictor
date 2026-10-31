"""v50: fixed-DOI-fold, common-row comparison of adsorption models.

Research-only; NO deploy, overwrite, or synthetic target generation.
Use: python -m scripts.evaluate_v50_model_utility --root .
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.special import expit, logit
from scripts.build_v46_phase8 import S2_A, S2_B, S2_CATS, features, s2_model
from pyrolysis.biochar_stage2.v15_features import model_inputs

DATA = 'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
OUT = 'reports/v50_model_utility'
SEED=42
SPECIALISTS={'Cd(II)','Cu(II)'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model(cols):
    """Identical XGB hyperparameters, fold-fitted preprocessing for all variants."""
    kind='A' if cols==S2_A else 'B'
    pipe,_,_,_=s2_model(kind)
    nums=[k for k in cols if k not in S2_CATS]
    cats=[k for k in cols if k in S2_CATS]
    # Rebuild the preprocessor with *exactly* the requested feature selection,
    # avoiding property information leaking into A/B.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    pre=ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),nums),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)
    ])
    pipe.set_params(pre=pre)
    return pipe


def model_predict(fit,test,cols, target='target_value'):
    reg=model(cols)
    nums=[c for c in cols if c not in S2_CATS]
    cats=[c for c in cols if c in S2_CATS]
    reg.fit(features(fit,cols,nums,cats),fit[target].to_numpy(float))
    return np.asarray(reg.predict(features(test,cols,nums,cats)),dtype=float)


def physical_clip(p, test):
    cap=(test.C0_mg_L.to_numpy(float)/test.dose_g_L.to_numpy(float))
    if (cap<=0).any() or not np.isfinite(cap).all():
        raise ValueError('Invalid batch mass-balance cap')
    return np.clip(np.asarray(p,float),0,cap)


def evaluate_global(df):
    """All variants evaluated on the *same* globally predetermined held-out DOI folds."""
    df=df.reset_index(drop=True)
    if not df.observation_id.is_unique:raise ValueError('duplicate observations')
    if df.doi.nunique()<5: raise ValueError('too few independent DOI')
    cap=df.C0_mg_L.to_numpy(float)/df.dose_g_L.to_numpy(float)
    y=df.target_value.to_numpy(float)
    if not (np.isfinite(y).all() and ((y>=0)&(y<=cap+1e-6)).all()):raise ValueError('Data have invalid targets or caps')
    models=['A_raw','B_raw','A_clipped','B_clipped','A_bounded_fraction','B_bounded_fraction','pollutant_fraction_baseline',
            'CdCu_local_A_clipped','CdCu_local_B_clipped']
    pred={k:np.full(len(df),np.nan) for k in models}
    audits=[]
    gkf=GroupKFold(n_splits=5)
    for fold,(tr,te) in enumerate(gkf.split(df,groups=df.doi),start=1):
        train,test=df.iloc[tr],df.iloc[te]
        held=set(test.doi)
        assert held.isdisjoint(set(train.doi))
        a=model_predict(train,test,S2_A)
        b=model_predict(train,test,S2_B)
        pred['A_raw'][te]=a; pred['B_raw'][te]=b
        pred['A_clipped'][te]=physical_clip(a,test)
        pred['B_clipped'][te]=physical_clip(b,test)
        # Directly learn bounded fractional uptake; avoids post-hoc saturation at the cap.
        # Input features remain A/B only (never derived cap, observed qe, DOI or QC flags).
        fraction=(train.target_value*train.dose_g_L/train.C0_mg_L).clip(1e-4,1-1e-4)
        fit_fraction=train.copy()
        fit_fraction['_logit_fraction_target']=logit(fraction.to_numpy(float))
        test_cap=test.C0_mg_L.to_numpy(float)/test.dose_g_L.to_numpy(float)
        for key,cols in [('A_bounded_fraction',S2_A),('B_bounded_fraction',S2_B)]:
            pred[key][te]=test_cap*expit(model_predict(fit_fraction,test,cols,target='_logit_fraction_target'))
        frac=(train.target_value*train.dose_g_L/train.C0_mg_L).clip(0,1)
        pooled_median=float(frac.median())
        for pollutant,tt in test.groupby('pollutant'):
            local_indices=np.array([df.index.get_loc(i) for i in tt.index])
            training_poll=train.loc[train.pollutant==pollutant]
            ndoi=training_poll.doi.nunique()
            if ndoi>=3 and len(training_poll)>=6:
                rate=float((training_poll.target_value*training_poll.dose_g_L/training_poll.C0_mg_L).clip(0,1).median())
            else:rate=pooled_median
            pred['pollutant_fraction_baseline'][local_indices]=rate*(tt.C0_mg_L.to_numpy(float)/tt.dose_g_L.to_numpy(float))
            # Only pre-declared Cd & Cu, supported by ≥5 other DOI per fold.
            for key,base,cols in [('CdCu_local_A_clipped',a,S2_A),('CdCu_local_B_clipped',b,S2_B)]:
                local=np.array(base[np.isin(test.index.to_numpy(),tt.index.to_numpy())],dtype=float)
                if pollutant in SPECIALISTS and ndoi>=5 and len(training_poll)>=12:
                    local=model_predict(training_poll,tt,cols)
                    audits.append(dict(fold=fold,pollutant=pollutant,test_n=len(tt),held_doi=';'.join(sorted(held)),
                                       training_doi=int(ndoi),trained_specialist=key))
                pred[key][local_indices]=physical_clip(local,tt)
    if any(not np.isfinite(x).all() for x in pred.values()): raise ValueError('Missing OOF prediction')
    scores=[]
    for name,vals in pred.items():
        scores.append(dict(scope='all140',model=name,n=len(df),doi=int(df.doi.nunique()),
                           mae=float(mean_absolute_error(y,vals)),r2=float(r2_score(y,vals)),
                           physical_violation=int(((vals<0)|(vals>cap)).sum()),
                           cap_hits=int(np.isclose(vals,cap,rtol=1e-8,atol=1e-10).sum())))
    prediction=pd.DataFrame({'observation_id':df.observation_id,'doi':df.doi,'pollutant':df.pollutant,
                             'feedstock':df.feedstock_label_normalized,'observed_qe':y,'physical_cap':cap,
                             'pyrolysis_temperature_C':df.pyrolysis_temperature_C,
                             'model_use_tier':df.model_use_tier})
    for name,vals in pred.items():prediction[name]=vals
    return pd.DataFrame(scores),prediction,pd.DataFrame(audits)


def evaluate_properties(df):
    """Pairwise A/B/C on SAME 103 material-characterized rows/DOI group split.

    Observed BET/TPV are input features, NOT a claim that Stage1 predictions can replace them.
    """
    mask=df[['BET_measured_m2_g','TPV_measured_cm3_g']].notna().all(axis=1)
    subset=df.loc[mask].reset_index(drop=True)
    assert len(subset)>=30 and subset.doi.nunique()>=5
    variants={'A':list(S2_A),'B':list(S2_B),'C_measured_BET_TPV':model_inputs('C')}
    y=subset.target_value.to_numpy(float)
    cap=subset.C0_mg_L.to_numpy(float)/subset.dose_g_L.to_numpy(float)
    predictions={name:np.full(len(subset),np.nan) for name in variants}
    for tr,te in GroupKFold(5).split(subset,groups=subset.doi):
        fit,test=subset.iloc[tr],subset.iloc[te]
        assert set(fit.doi).isdisjoint(test.doi)
        for label,cols in variants.items():
            p=model_predict(fit,test,cols)
            predictions[label][te]=physical_clip(p,test)
    stats=[]
    output=pd.DataFrame({'observation_id':subset.observation_id,'doi':subset.doi,'pollutant':subset.pollutant,
                         'observed_qe':y,'physical_cap':cap})
    for name,values in predictions.items():
        assert np.isfinite(values).all()
        stats.append(dict(scope='same_measured_property_rows',model=name+'_clipped',n=len(subset),doi=int(subset.doi.nunique()),
                          mae=float(mean_absolute_error(y,values)),r2=float(r2_score(y,values)),
                          physical_violation=int(((values<0)|(values>cap)).sum()),cap_hits=int(np.isclose(values,cap).sum())))
        output[name+'_clipped']=values
    return pd.DataFrame(stats),output


def subgroups(df, preds):
    entries=[]
    for pol,sub in preds.groupby('pollutant'):
        for name in ['A_clipped','B_clipped','A_bounded_fraction','B_bounded_fraction','CdCu_local_A_clipped','CdCu_local_B_clipped','pollutant_fraction_baseline']:
            entries.append(dict(pollutant=pol,n=len(sub),doi=sub.doi.nunique(),model=name,
               mae=float(mean_absolute_error(sub.observed_qe,sub[name]))))
    return pd.DataFrame(entries)


def temperature_panels(p):
    """Strict matched experimental-condition panel; no mixing qt/qmax or noncomparable tests."""
    src=p.copy()
    if 'source' in src:pass
    condition=['doi','feedstock','pollutant','manufacturing_route', 'activation_or_modification',
               'C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']
    # Work with merged source columns in caller instead of rounded or substituted conditions.
    rows=[]
    for group, g in src.groupby(condition,dropna=False):
        same=g.dropna(subset=['pyrolysis_temperature_C'])
        # Never arbitrarily choose one of multiple observations at same temperature.
        if same.pyrolysis_temperature_C.duplicated().any(): continue
        if len(same)<3:continue
        for variant in ['A_clipped','B_clipped','CdCu_local_B_clipped','B_bounded_fraction']:
            best_true=same.loc[same.observed_qe.idxmax(),'pyrolysis_temperature_C']
            values=same[variant].to_numpy(float)
            can_rank=(np.ptp(values)>1e-8)
            best_pred=same.loc[same[variant].idxmax(),'pyrolysis_temperature_C'] if can_rank else float('nan')
            pairs=0;correct=0;ties=0
            for i in range(len(same)):
                for j in range(i+1,len(same)):
                    ya,yb=same.iloc[i].observed_qe,same.iloc[j].observed_qe
                    pa,pb=same.iloc[i][variant],same.iloc[j][variant]
                    if abs(ya-yb)>1e-8:
                        pairs+=1
                        if abs(pa-pb)<=1e-8:ties+=1
                        else:correct+=int(np.sign(ya-yb)==np.sign(pa-pb))
            rows.append(dict(doi=group[0],feedstock=group[1],pollutant=group[2],n_temps=len(same),
                             model=variant,observed_best_T_C=best_true,predicted_best_T_C=best_pred,
                             top1_correct=(bool(best_true==best_pred) if can_rank else None),ranking_defined=bool(can_rank),pairs=pairs,correct_pairs=correct,tied_pairs=ties,
                             note='OOF values from same held-out DOI; predictions may tie when features do not vary'))
    return pd.DataFrame(rows)


def run(root):
    root=Path(root)
    src=root/DATA
    if not src.exists():raise FileNotFoundError(src)
    dataset_sha=digest(src)
    df=pd.read_csv(src,low_memory=False)
    if not(df.observation_id.is_unique and df.doi.nunique()==32 and len(df)==140):
        raise ValueError('Unexpected v46 Stage2 input; stop rather than silently merge')
    globals_,preds,audit=evaluate_global(df)
    properties,props_pred=evaluate_properties(df)
    per_pol=subgroups(df,preds)
    panels=temperature_panels(preds.merge(df[['observation_id','manufacturing_route','activation_or_modification',
                       'C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']],on='observation_id',validate='one_to_one'))
    out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    globals_.to_csv(out/'global_model_comparison.csv',index=False)
    preds.to_csv(out/'doi_oof_predictions.csv',index=False)
    per_pol.to_csv(out/'per_pollutant_comparison.csv',index=False)
    properties.to_csv(out/'measured_BET_TPV_paired_comparison.csv',index=False)
    props_pred.to_csv(out/'measured_BET_TPV_oof_predictions.csv',index=False)
    audit.to_csv(out/'specialist_fold_audit.csv',index=False)
    panels.to_csv(out/'matched_temperature_panels.csv',index=False)
    summary=dict(dataset=DATA,sha256_before=dataset_sha,sha256_after=digest(src),n_rows=len(df),doi=int(df.doi.nunique()),
        n_matched_measured_properties=int(df[['BET_measured_m2_g','TPV_measured_cm3_g']].notna().all(axis=1).sum()),
        global_models=globals_.to_dict(orient='records'),
        measured_property_same_sample=properties.to_dict(orient='records'),
        temperature_panels=int(len(panels)/4),
        temperature_panel_results=panels.astype(object).where(pd.notna(panels),None).to_dict(orient='records'),
        research_only=True,existing_active_models_changed=False,
        notes=['All CV splits group by DOI; comparisons must use common sample subsets.',
          'Clipping is deterministic and not proof of physics learned.',
          'C requires independently measured material properties; NOT a deployable two-stage model.',
          'Core 97, conditional QA, graph approximations are combined for exploratory comparison only.',
          'This is reused-study cross-validation, not a blinded external DOI test.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if summary['sha256_before']!=summary['sha256_after']:raise RuntimeError('Dataset changed during evaluation')
    print(globals_.to_string(index=False))
    print('\nPaired property study\n',properties.to_string(index=False))
    print('\nPer pollutant\n',per_pol.to_string(index=False))
    print('\nStrict temperature panels',len(panels)//4)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd())
    run(parser.parse_args().root)
