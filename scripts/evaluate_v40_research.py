"""v40 independent DOI-group research comparison and nonactive full-data refit.
No model promotion. Preprocessing remains inside each train fold. For exploratory use only.
Run: python -m scripts.evaluate_v40_research --root .
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scripts.build_v37_full_refit import S1_NUM,S1_CAT,S1_TARGETS,S2_CAT,input_frame,pipeline,audit_s1,audit_s2
from pyrolysis.biochar_stage2.v15_features import model_inputs

DATA='reports/v40_new9'; MODELS='deployment/research_v40_NOT_ACTIVE'
COMPOSITION=['feedstock_VM_pct_original','feedstock_FC_pct_original', 'feedstock_ash_pct_original',
            'feedstock_C_pct_original','feedstock_H_pct_original', 'feedstock_O_pct_original','feedstock_N_pct_original']

def positive_model(nums,cats):
    # train only on positive target measurements, log-link keeps predictions >=0
    pre=ColumnTransformer([
        ('n',Pipeline([('fill',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]),nums),
        ('c',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                      ('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)],sparse_threshold=0)
    return Pipeline([('pre',pre),('reg',HistGradientBoostingRegressor(loss='poisson',max_iter=140,
      max_leaf_nodes=9,min_samples_leaf=10,learning_rate=.065,l2_regularization=5,random_state=42))])

def maker(stage,target,variant):
    if stage=='Stage1':
        if variant=='Ridge': return pipeline(S1_NUM,S1_CAT,'Ridge'),S1_NUM,S1_CAT
        if variant=='Poisson_Process': return positive_model(S1_NUM,S1_CAT),S1_NUM,S1_CAT
        if variant=='Poisson_Composition':return positive_model(S1_NUM+COMPOSITION,S1_CAT),S1_NUM+COMPOSITION,S1_CAT
    else:
        inputs=model_inputs(variant)
        cats=[c for c in inputs if c in S2_CAT];nums=[c for c in inputs if c not in cats]
        return pipeline(nums,cats,'XGBoost'),nums,cats
    raise ValueError((stage,variant))

def oof(d,stage,target,variant):
    d=d.reset_index(drop=True)
    if d.doi.nunique()<5:raise ValueError('need at least 5 independent DOIs')
    y=d[target].to_numpy(float)
    pred=np.full(len(d),np.nan)
    for tr,te in GroupKFold(n_splits=5).split(d,groups=d.doi):
        model,nums,cats=maker(stage,target,variant)
        x=input_frame(d,nums,cats)
        model.fit(x.iloc[tr],y[tr]);pred[te]=model.predict(x.iloc[te])
    assert np.isfinite(pred).all()
    return pred

def paired_old_eval(old,extra,stage,target,variant):
    """Evaluate old DOI rows with new DOI added to training, fixed GroupKFold old rows.
    This is exploratory, NOT a blinded prospective test of generalization.
    """
    old=old.reset_index(drop=True);extra=extra.reset_index(drop=True)
    assert set(old.doi).isdisjoint(set(extra.doi))
    y=old[target].to_numpy(float); ey=extra[target].to_numpy(float)
    pred_old=np.empty(len(old));pred_aug=np.empty(len(old))
    for tr,te in GroupKFold(n_splits=5).split(old,groups=old.doi):
        for aug,out in [(False,pred_old),(True,pred_aug)]:
            m,nums,cats=maker(stage,target,variant)
            fit=pd.concat([old.iloc[tr],extra],ignore_index=True) if aug else old.iloc[tr]
            m.fit(input_frame(fit,nums,cats),fit[target].to_numpy(float))
            out[te]=m.predict(input_frame(old.iloc[te],nums,cats))
    return pred_old,pred_aug

def summary_score(stage,target,variant,subset,y,p,doi):
    data=dict(stage=stage,target=target,model=variant,subset=subset,n=len(y),n_doi=int(pd.Series(doi).nunique()),
        MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)),raw_negative=int((p<0).sum()))
    if stage=='Stage2':
        cap=y*0 # supplied below in evaluation logic
    return data

def doi_bootstrap(groups,truth,base,aug,seed=40):
    g=np.asarray(groups).astype(str);uniq=np.unique(g);inds={d:np.flatnonzero(g==d) for d in uniq}
    deltas=np.abs(truth-aug)-np.abs(truth-base);rng=np.random.default_rng(seed)
    z=[]
    for _ in range(1000):
        ix=np.concatenate([inds[d] for d in rng.choice(uniq,len(uniq),replace=True)])
        z.append(float(deltas[ix].mean()))
    return [float(v) for v in np.quantile(z,[.025,.975])]

def run(root:Path):
    root=Path(root);folder=root/DATA;out=root/'reports/v40_new9/model_checks';out.mkdir(parents=True,exist_ok=True)
    s1=pd.read_csv(folder/'Stage1_Master_381_RESEARCH_STAGING.csv',low_memory=False)
    s2=pd.read_csv(folder/'Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv',low_memory=False)
    audit_s1(s1);audit_s2(s2)
    # Numeric target presence, not metadata-only rows. DOIs from same paper never split.
    allrows=[];predrows=[];paired=[];newchallenge=[]
    for stage,frame,target,variants,newdoi in [
      ('Stage1',s1,S1_TARGETS['BET'],['Ridge','Poisson_Process','Poisson_Composition'],
       {'10.3390/ma17123030','10.3390/app11199223','10.1016/j.biteb.2021.100920','10.1016/j.egyr.2021.07.032'}),
      ('Stage1',s1,S1_TARGETS['TPV'],['Ridge','Poisson_Process','Poisson_Composition'],
       {'10.3390/app11199223','10.1016/j.biteb.2021.100920','10.1016/j.egyr.2021.07.032'}),
      ('Stage2',s2,'target_value',['A','B'],{'10.3390/app11199223'})
    ]:
        f=frame[frame[target].notna()].copy().reset_index(drop=True)
        if not (pd.to_numeric(f[target],errors='coerce').ge(0).all()):raise ValueError(target)
        old=f[~f.doi.isin(newdoi)].reset_index(drop=True); extra=f[f.doi.isin(newdoi)].reset_index(drop=True)
        for variant in variants:
            p=oof(f,stage,target,variant)
            row=summary_score(stage,target,variant,'all_doi_group5',f[target].to_numpy(float),p,f.doi)
            if stage=='Stage2':
                cap=f.C0_mg_L.to_numpy(float)/f.dose_g_L.to_numpy(float)
                row['cap_exceeded']=int((p>cap).sum());row['raw_physics_violations']=int(((p<0)|(p>cap)).sum())
            allrows.append(row)
            for id,doi,a,v in zip(f.iloc[:,0],f.doi,f[target],p):
                predrows.append(dict(stage=stage,target=target,model=variant,record_id=id,doi=doi,actual=a,pred=v))
            if len(extra):
                old_p,aug_p=paired_old_eval(old,extra,stage,target,variant)
                delta=float(np.mean(np.abs(old[target]-aug_p)-np.abs(old[target]-old_p)))
                pair=dict(stage=stage,target=target,model=variant,old_rows=len(old),old_doi=int(old.doi.nunique()),
                   new_train_rows=len(extra),new_train_doi=int(extra.doi.nunique()),
                   old_train_MAE=float(mean_absolute_error(old[target],old_p)),
                   augmented_train_MAE=float(mean_absolute_error(old[target],aug_p)),
                   delta_MAE=delta,DOI_bootstrap_CI95=doi_bootstrap(old.doi,old[target].to_numpy(float),old_p,aug_p))
                paired.append(pair)
                # frozen old-only fit and one-time diagnostic on the new DOI; no reusing new DOI for selection.
                m,nums,cats=maker(stage,target,variant)
                m.fit(input_frame(old,nums,cats),old[target].to_numpy(float))
                xnew=m.predict(input_frame(extra,nums,cats))
                score=summary_score(stage,target,variant,'prospective_new_DOI_NOT_BLINDED',extra[target].to_numpy(float),xnew,extra.doi)
                if stage=='Stage2':
                    cap=extra.C0_mg_L.to_numpy(float)/extra.dose_g_L.to_numpy(float)
                    score['cap_exceeded']=int((xnew>cap).sum());score['raw_physics_violations']=int(((xnew<0)|(xnew>cap)).sum())
                newchallenge.append(score)
        # Full refit fixed configuration, never active.
        for variant in variants:
            if stage=='Stage1' and variant=='Ridge':continue
            m,nums,cats=maker(stage,target,variant)
            m.fit(input_frame(f,nums,cats),f[target].to_numpy(float))
            md=root/MODELS;md.mkdir(parents=True,exist_ok=True)
            joblib.dump(m,md/f'V40_{stage}_{"QE" if stage=="Stage2" else target.split("_")[0]}_{variant}_RESEARCH_ONLY.joblib')
    pd.DataFrame(allrows).to_csv(out/'all_DOI_OOF_metrics.csv',index=False)
    pd.DataFrame(predrows).to_csv(out/'all_DOI_OOF_predictions.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_old_DOI_with_new_training.csv',index=False)
    pd.DataFrame(newchallenge).to_csv(out/'new_DOI_prospective_challenge.csv',index=False)
    report=dict(n_stage1=len(s1),n_stage2=len(s2),new_DOI_added=7,
     model_metrics=allrows,paired_old_doi=paired,prospective_new_doi=newchallenge,
     methodology='DOI GroupKFold 5-fold, predefined hyperparameters; full model fit non-active; historical source QA',
     limitations=['No blinded independent DOI: uploaded sources already inspected for model planning',
                  'Not all observational rows production approved, graph-derived not auto-enrolled',
                  'QE reported experimental 24h cannot certify strict equilibrium',
                  'Source-specific raw feedstock composition not imputed across DOI',
                  'Foldwise preprocessing, no credit for physical cap clipping'])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);v=a.parse_args()
    results=run(v.root)
    print(pd.DataFrame(results['model_metrics']).to_string(index=False))
    print('PAIRED');print(pd.DataFrame(results['paired_old_doi']).to_string(index=False))
