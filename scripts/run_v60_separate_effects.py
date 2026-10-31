"""v60: separate adsorption-condition response and within-context manufacturing ranking.
Research-only; nested DOI grouped OOF. NO active model/data writes.
Run: python -m scripts.run_v60_separate_effects --root .
"""
from __future__ import annotations
import argparse, hashlib, json, warnings, platform
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import sklearn, xgboost
from scripts.run_v59_doi_material_weighting import fit_predict as direct_fit_predict
from scripts.run_v59_doi_material_weighting import weights_for, OLD, PRE, NEW
from scripts.build_v46_phase8 import S2_B, S2_CATS, features
from scripts.run_v58_failure_diagnostics import build, derive

REPORT='reports/v60_separate_effects'; DEPLOY='deployment/research_v60_NOT_ACTIVE'
ADS=['pollutant','C0_mg_L','initial_pH','dose_g_L','adsorption_T_C','contact_time_min']
PANEL=[ 'doi','pollutant','C0_mg_L','initial_pH','dose_g_L','adsorption_T_C','contact_time_min',
        'feedstock_label_normalized','manufacturing_route','activation_or_modification']
CONC=['doi','material_group','pollutant','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']
MFG=['pollutant']+[c for c in S2_B[6:] if c not in ('pollutant',)]
MFG_CATS=[x for x in MFG if x in S2_CATS]; MFG_NUM=[x for x in MFG if x not in MFG_CATS]
SEED=20261031

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _safe(x):
    x=float(x)
    return x if np.isfinite(x) else None

def material_groups(df):
    """Only source material UID or source sample group. Never infer from measurements."""
    return df['material_uid'].fillna(df['sample_group']).fillna(df['observation_id']).astype(str)

def panel_inventory(frame):
    """Exact reported adsorption context + same feedstock/route/modification; >=3 distinct temperatures/materials.
    Missing values match only missing values; never impute panel match keys.
    """
    d=frame.copy().reset_index(drop=True)
    d['material_group']=material_groups(d)
    d['panel_id']=-1
    groups=[]
    for _,sub in d.groupby(PANEL,sort=True,dropna=False):
        if len(sub)<3 or sub['pyrolysis_temperature_C'].nunique(dropna=True)<3 or sub.material_group.nunique()<3:
            continue
        # Avoid hidden duplicates if identical material/temperature rows masquerade as candidates.
        if sub.material_group.duplicated().any():
            continue
        ix=len(groups); d.loc[sub.index,'panel_id']=ix
        groups.append({'panel_id':ix,'doi':str(sub.doi.iloc[0]),'pollutant':str(sub.pollutant.iloc[0]),
                       'feedstock':str(sub.feedstock_label_normalized.iloc[0]),
                       'n':int(len(sub)),'n_materials':int(sub.material_group.nunique()),
                       'n_temp':int(sub.pyrolysis_temperature_C.nunique()),
                       'n_missing_ads_context':int(sub[['initial_pH','adsorption_T_C','contact_time_min']].isna().any(axis=1).sum()),
                       'obs_ids':'|'.join(sub.observation_id.astype(str).tolist())})
    return d,pd.DataFrame(groups)

def concentration_inventory(frame):
    d=frame.copy().reset_index(drop=True)
    d['material_group']=material_groups(d); d['conc_series_id']=-1
    groups=[]
    for _,sub in d.groupby(CONC,sort=True,dropna=False):
        if len(sub)<3 or sub.C0_mg_L.nunique()<3:continue
        # No duplicate C0 in an exact manufactured-material series.
        if sub.C0_mg_L.duplicated().any(): continue
        k=len(groups); d.loc[sub.index,'conc_series_id']=k
        groups.append({'conc_series_id':k,'doi':str(sub.doi.iloc[0]),
                       'material_group':str(sub.material_group.iloc[0]),'n':int(len(sub)),
                       'n_c0':int(sub.C0_mg_L.nunique()), 'obs_ids':'|'.join(sub.observation_id.astype(str).tolist())})
    return d,pd.DataFrame(groups)

def make_rank_model(kind):
    num=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                  ('scale',StandardScaler())]); cat=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                                 ('ohe',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',num,MFG_NUM),('cat',cat,MFG_CATS)])
    if kind=='ridge':reg=Ridge(alpha=10.0)
    elif kind=='forest':reg=RandomForestRegressor(n_estimators=160,max_depth=3,min_samples_leaf=2,
                                                  max_features=.9,random_state=42,n_jobs=1)
    else:raise ValueError(kind)
    return Pipeline([('pre',pre),('model',reg)])

def panel_centered_training(train):
    t,_=panel_inventory(train)
    t=t[t.panel_id>=0].copy()
    if t.empty:return t,np.array([]),np.array([])
    cap=t.C0_mg_L.to_numpy(float)/t.dose_g_L.to_numpy(float)
    y=100*t.target_value.to_numpy(float)/cap
    grp=pd.Series(y,index=t.index).groupby(t.panel_id).transform('mean')
    centered=y-grp.to_numpy()
    assert np.allclose(pd.DataFrame({'p':t.panel_id,'r':centered}).groupby('p').r.mean(),0,atol=1e-9)
    # DOI-balanced total mass ~ sqrt(number of panels per DOI), each panel equal within DOI.
    npan=t.groupby('doi').panel_id.nunique().to_dict()
    dd=t.groupby(['doi','panel_id']).size().to_dict()
    wt=np.array([1/(np.sqrt(npan[d])*dd[d,int(p)]) for d,p in zip(t.doi,t.panel_id)],float)
    wt*=len(wt)/wt.sum()
    return t,centered,wt

def fit_ranker(train,kind):
    t,y,wt=panel_centered_training(train)
    if t.doi.nunique()<2 or len(t)<8:return None,len(t),t.doi.nunique()
    m=make_rank_model(kind)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=FutureWarning)
        m.fit(features(t,MFG,MFG_NUM,MFG_CATS),y,model__sample_weight=wt)
    return m,len(t),t.doi.nunique()

def rho(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.ptp(a)<1e-10 or np.ptp(b)<1e-10:return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _safe(spearmanr(a,b).statistic)

def pairwise(y,s,eps=1e-8):
    """Ties in the prediction get 0.5 credit; true ties are excluded."""
    total=hit=tie=0
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            dy=float(y[i]-y[j]);ds=float(s[i]-s[j])
            if abs(dy)<=eps:continue
            total+=1
            if abs(ds)<=eps:hit+=.5;tie+=1
            elif np.sign(dy)==np.sign(ds):hit+=1
    return (hit/total if total else None),total,tie

def rank_stats(d,score_col,panel_type,kind):
    key='panel_id' if panel_type=='manufacturing' else 'conc_series_id'
    vals=[]
    for pid,g in d[d[key]>=0].groupby(key):
        y=g.target_value.to_numpy(float); p=g[score_col].to_numpy(float)
        if not np.isfinite(p).all():continue
        if panel_type=='concentration':
            order=np.argsort(g.C0_mg_L.to_numpy(float))
            mono=int((np.diff(p[order])< -1e-8).sum());true_mono=int((np.diff(y[order])< -1e-8).sum())
        else:mono=None;true_mono=None
        top=np.flatnonzero(np.isclose(p,p.max(),atol=1e-8,rtol=0))
        concord,npair,ntie=pairwise(y,p)
        vals.append(dict(panel_type=panel_type,panel_id=int(pid),doi=str(g.doi.iloc[0]),
                    feedstock=str(g.feedstock_label_normalized.iloc[0]),n=int(len(g)),
                    model=kind,rho=rho(y,p),concordance=concord,
                    n_pairs=npair,pred_tied_pairs=ntie,n_tied_top=int(len(top)),
                    top1_regret_worst=float(y.max()-y[top].min()),
                    unique_top1=bool(len(top)==1),
                    monotonic_violations=mono,observed_monotonic_violations=true_mono))
    return vals

def aggregate(g):
    if g.empty:return {}
    ranks=g[g.panel_type=='manufacturing']
    conc=g[g.panel_type=='concentration']
    out=[]
    for (sc,model,typ),t in g.groupby(['scenario','model','panel_type']):
        e=t.groupby('doi',as_index=False).agg(rho=('rho','mean'),concordance=('concordance','mean'))
        out.append({'scenario':sc,'model':model,'panel_type':typ,'panels':len(t),'dois':t.doi.nunique(),
                    'mean_rho_panel':_safe(t.rho.mean()),'doi_macro_rho':_safe(e.rho.mean()),
                    'mean_pairwise_concordance':_safe(t.concordance.mean()),
                    'n_unique_top1':int(t.unique_top1.sum()),'mean_worst_tie_regret_mg_g':_safe(t.top1_regret_worst.mean()),
                    'monotonic_violations':int(t.monotonic_violations.sum()) if typ=='concentration' else None,
                    'observed_monotonic_violations':int(t.observed_monotonic_violations.sum()) if typ=='concentration' else None})
    return out

def main(root):
    root=Path(root).resolve(); dst=root/REPORT;dst.mkdir(parents=True,exist_ok=True)
    orig=pd.read_csv(root/OLD,low_memory=False);v152=pd.read_csv(root/PRE,low_memory=False);new=pd.read_csv(root/NEW,low_memory=False)
    assert len(orig)==140 and len(v152)==152 and len(new)==36
    assert list(orig.observation_id)==list(v152.iloc[:140].observation_id)
    assert not (set(v152.doi)&set(new.doi))
    d=pd.concat([v152,new],ignore_index=True,sort=False)
    assert len(d)==188 and d.doi.nunique()==36 and d.observation_id.is_unique
    p,p_inventory=panel_inventory(d);c,c_inventory=concentration_inventory(d)
    if len(p_inventory)==0 or len(c_inventory)==0:raise RuntimeError('No measured panels for tests')
    p_inventory.to_csv(dst/'manufacturing_panels.csv',index=False)
    c_inventory.to_csv(dst/'concentration_series.csv',index=False)
    pred=d[['observation_id','doi','pollutant','material_uid','feedstock_label_normalized','pyrolysis_temperature_C',
            'C0_mg_L','dose_g_L','target_value']].copy()
    pred['panel_id']=p.panel_id;pred['conc_series_id']=c.conc_series_id
    folds=[]; train_panels=[]
    for fold,(it,iv) in enumerate(GroupKFold(n_splits=5).split(d,groups=d.doi),1):
        held=set(d.iloc[iv].doi)
        folds.append({'fold':fold,'dois':sorted(held),'n_test':len(iv)})
        for label,data in [('152',v152),('188',d)]:
            tr=data[~data.doi.isin(held)].copy()
            te=d.iloc[iv].copy()
            assert set(tr.doi).isdisjoint(held)
            # Disjoint paper-group for every estimator, including ranking submodel.
            for mod,variant,policy in [('A_row','A','row_equal'),('B_row','B','row_equal'),
                                        ('B_doi_material','B','doi_sqrt_material_equal')]:
                pr,raw=direct_fit_predict(tr,te,variant,'cap',policy)
                for nm,v in [(f'{label}__{mod}',pr),(f'{label}__{mod}__raw',raw)]:
                    if nm not in pred:pred[nm]=np.nan
                    pred.loc[iv,nm]=v
            for kind in ('ridge','forest'):
                model,nrow,ngroups=fit_ranker(tr,kind)
                train_panels.append(dict(fold=fold,scenario=label,kind=kind,train_panels=nrow,train_DOIs=ngroups))
                nm=f'{label}__mfg_{kind}'
                if nm not in pred:pred[nm]=np.nan
                if model is not None:
                    pp=model.predict(features(te,MFG,MFG_NUM,MFG_CATS))
                    pred.loc[iv,nm]=pp
    pred['fold']=0
    for i,f in enumerate(folds,1):pred.loc[pred.doi.isin(f['dois']),'fold']=i
    if not (pred.fold>0).all():raise RuntimeError('Failed to predict all folds')
    pred.to_csv(dst/'all188_doi_oof_predictions.csv',index=False)
    # All-research: EVALUATION IS POSTHOC. Same 188 held-out rows and identical DOI folds per scenario.
    scoring=[];metr=[]
    for label in ('152','188'):
        for mod in ('A_row','B_row','B_doi_material'):
            col=f'{label}__{mod}'
            yp=pred[col].to_numpy(float);y=pred.target_value.to_numpy(float)
            metr.append(dict(scenario=label,model=mod,n=188,n_doi=36,
                mae_mg_g=float(mean_absolute_error(y,yp)),rmse_mg_g=float(np.sqrt(mean_squared_error(y,yp))),
                mae_old140=float(mean_absolute_error(y[:140],yp[:140])),
                mae_new36=float(mean_absolute_error(y[152:],yp[152:])),
                raw_upper_cap_violation=int((pred[f'{col}__raw']>(pred.C0_mg_L/pred.dose_g_L)+1e-8).sum())))
            for typ in ('manufacturing','concentration'):
                scores=rank_stats(pred,col,typ,mod)
                for row in scores:row.update(scenario=label)
                scoring.extend(scores)
            if mod!='A_row':
                for typ in ('manufacturing',):
                    scores=rank_stats(pred,col+'__raw',typ,mod+'_unclipped_DIAGNOSTIC')
                    for row in scores:row.update(scenario=label)
                    scoring.extend(scores)
        for kind in ('ridge','forest'):
            col=f'{label}__mfg_{kind}'
            rows=rank_stats(pred,col,'manufacturing','mfg_'+kind+'_RELATIVE_SCORE_NOT_QE')
            for row in rows:row.update(scenario=label)
            scoring.extend(rows)
    score=pd.DataFrame(scoring); score.to_csv(dst/'per_panel_rank_results.csv',index=False)
    smry=pd.DataFrame(aggregate(score));smry.to_csv(dst/'rank_summary.csv',index=False)
    pd.DataFrame(metr).to_csv(dst/'same188_mae_comparison.csv',index=False)
    pd.DataFrame(train_panels).to_csv(dst/'training_panel_counts.csv',index=False)
    (dst/'doi_folds.json').write_text(json.dumps(folds,ensure_ascii=False,indent=2),encoding='utf-8')
    # full research models: selected by fixed architecture, not posthoc best
    deploy=root/DEPLOY;deploy.mkdir(parents=True,exist_ok=True)
    saved=[]
    for mod,variant,policy in [('A_row','A','row_equal'),('B_doi_material','B','doi_sqrt_material_equal')]:
        pr,raw,model,cols,nums,cats,wt=direct_fit_predict(d,d.iloc[:2],variant,'cap',policy,True)
        fp=deploy/f'{mod}_188_cap_RESEARCH_NOT_ACTIVE.joblib';joblib.dump(model,fp,compress=3)
        saved.append(dict(model=mod,artifact=str(fp.relative_to(root)),sha256=sha(fp),required_features=cols))
    for kind in ('ridge','forest'):
        model,n,ng=fit_ranker(d,kind)
        fp=deploy/f'mfg_{kind}_within_panel_188_RELATIVE_SCORE_NOT_QE.joblib';joblib.dump(model,fp,compress=3)
        saved.append(dict(model='mfg_'+kind,artifact=str(fp.relative_to(root)),sha256=sha(fp),
                          required_features=MFG,training_rows=n,training_dois=ng,
                          output_units='within-context centered removal percentage-points; NOT qe'))
    run=dict(version='v60',description='Adsorption-condition vs manufacturing-context separated retrospective study',
      role='RESEARCH_ONLY_NOT_ACTIVE',created='2026-10-31',fixed_rows=188,fixed_groups=36,
      study_data={'old140':{'file':OLD,'sha256':sha(root/OLD)},'v55_152':{'file':PRE,'sha256':sha(root/PRE)},
                  'new36':{'file':NEW,'sha256':sha(root/NEW)}},
      design={'manufacturing_panel':'exact adsorption context+feedstock+route+modification; >=3 distinct materials and pyro temperatures',
              'concentration_panel':'same material, pollutant, dose, pH, T, contact time; >=3 distinct C0',
              'preprocessing':'All fit only in DOI nonheld training folds','ranker_target':'observed removal percentage centered within TRAINING panel only',
              'ranking_model':'Ridge and RandomForest; training DOI excluded from test',
              'same188_comparison':'Test all 188 with same 5 group folds for 152 and 188 models; posthoc',
              'no_causal_claim':True},
      scipy=None,python=platform.python_version(),sklearn=sklearn.__version__,xgboost=xgboost.__version__,
      panels={'manufacturing':len(p_inventory),'manufacturing_doi':int(p_inventory.doi.nunique()),
              'concentration':len(c_inventory),'concentration_doi':int(c_inventory.doi.nunique())},
      saved=saved)
    (dst/'run_manifest.json').write_text(json.dumps(run,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('V60 finished, panels:',run['panels']);print(pd.DataFrame(metr).round(3).to_string(index=False));
    print(smry[['scenario','model','panel_type','panels','mean_rho_panel','mean_pairwise_concordance','n_unique_top1']].round(3).to_string(index=False))
    return run

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');a=ap.parse_args();main(a.root)
