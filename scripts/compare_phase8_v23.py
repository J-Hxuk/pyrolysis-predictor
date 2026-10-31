"""Phase 8 incremental-value audit: paired DOI holdouts, no deployment change.

Entry: python -m scripts.compare_phase8_v23
Uses fixed, *research-only* models; NOT a rerun of deployed v60 or nested-v16.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2.v15_models import build_pipeline as stage2_pipe
from pyrolysis.biochar_stage2.phase8_evidence import audit_phase8, build_stage2_core98_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = PROJECT_ROOT / 'data/imports/Biochar_Phase8_Integrated_Data_Package/data'
REPORT = PROJECT_ROOT / 'reports/phase8_v23'
SEED = 20261030
S1_NUMERIC = [
    'pyrolysis_temperature_C', 'pyrolysis_hold_min', 'heating_rate_C_min',
    'pyrolysis_N2_flow_mL_min', 'activation_temperature_C', 'activation_time_min',
    'activation_ratio_value', 'activation_steam_fraction_pct', 'activation_CO2_fraction_pct',
]
S1_CATEGORICAL = [
    'feedstock_group', 'activation_method', 'activation_agent',
    'pyrolysis_atmosphere', 'process_stage', 'v59_route_family',
]
S1_FEATURES = S1_NUMERIC + S1_CATEGORICAL


def input_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Normalize missing to explicit unknown categories, without filling numeric from test data."""
    x = df.reindex(columns=numeric + categorical).copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    for col in categorical:
        x[col] = x[col].fillna('__UNREPORTED__').astype(str).replace({'': '__UNREPORTED__', 'nan': '__UNREPORTED__'})
    return x


def stage1_pipe() -> Pipeline:
    num = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                    ('scale', StandardScaler())])
    cat = OneHotEncoder(handle_unknown='ignore')
    pre = ColumnTransformer([('num', num, S1_NUMERIC), ('cat', cat, S1_CATEGORICAL)])
    return Pipeline([('pre', pre), ('estimator', RandomForestRegressor(
        n_estimators=120, max_depth=7, min_samples_leaf=3, max_features=0.85,
        random_state=SEED, n_jobs=1))])


def observed_metrics(rows: pd.DataFrame, truth='actual', prediction='pred') -> dict:
    if rows.empty:
        return {'n': 0, 'r2': None, 'mae': None, 'rmse': None}
    y = rows[truth].to_numpy(dtype=float)
    p = rows[prediction].to_numpy(dtype=float)
    return {
        'n': len(rows), 'r2': float(r2_score(y, p)) if len(y)>1 and np.var(y)>0 else None,
        'mae': float(mean_absolute_error(y, p)),
        'rmse': float(np.sqrt(mean_squared_error(y, p))),
    }


def cluster_delta_ci(df: pd.DataFrame, seed=SEED, reps=2000) -> dict:
    """Cluster bootstrap of per-row paired abs error difference; DOI sampled with replacement."""
    if df.empty:
        return {'delta_mae': None, 'ci95': [None, None], 'doi_count': 0}
    d = (df['err_expanded'] - df['err_original']).to_numpy(dtype=float)
    groups = df['doi'].astype(str).to_numpy()
    keys = np.unique(groups)
    index = [np.flatnonzero(groups == g) for g in keys]
    rng = np.random.default_rng(seed)
    boot=[]
    for _ in range(reps):
        sample = rng.integers(0,len(keys),size=len(keys))
        ids = np.concatenate([index[k] for k in sample])
        boot.append(float(d[ids].mean()))
    return {'delta_mae': float(d.mean()), 'ci95': [float(x) for x in np.quantile(boot,[.025,.975])],
            'doi_count': len(keys), 'unit': 'expanded_minus_original; negative favors expanded'}


def stage1_targets() -> tuple[pd.DataFrame,pd.DataFrame]:
    old = pd.read_csv(PHASE/'stage1/Stage1_v60_256_UNCHANGED.csv', low_memory=False)
    new = pd.read_csv(PHASE/'stage1/Stage1_Master_264_UNCHANGED.csv', low_memory=False)
    roles = pd.read_csv(PHASE/'stage1/Stage1_Target_Training_Roles_UNCHANGED.csv')
    role_keys = ['material_id','BET_training_use','TPV_training_use']
    new = new.merge(roles[role_keys], on='material_id', validate='one_to_one')
    old = old.merge(roles[role_keys], on='material_id', validate='one_to_one')
    if len(old)!=256 or len(new)!=264 or new.material_id.nunique()!=264:
        raise ValueError('Stage1 integration shape mismatch')
    if set(old.material_id) - set(new.material_id):
        raise ValueError('Phase8 deleted Stage1 identities')
    if len(set(new.doi) - set(old.doi)) != 2:
        raise ValueError('Expected precisely two new Stage1 DOI')
    return old,new


def _evaluate_stage1(target: str,old: pd.DataFrame,new: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    role, ycol = {'BET': ('BET_training_use','BET_m2_g_training_candidate'),
                  'TPV': ('TPV_training_use','TPV_cm3_g_training_candidate')}[target]
    allowed = lambda d: d.loc[d[role].eq(True) & pd.to_numeric(d[ycol],errors='coerce').notna()].copy()
    a,b=allowed(old),allowed(new)
    if a.material_id.duplicated().any() or b.material_id.duplicated().any():
        raise ValueError('Duplicate Stage1 material_id in evaluation cohort')
    old_dois = set(a.doi)
    if len(b)-len(a)!=8:
        raise ValueError(f'Unexpected {target} role growth: {len(a)} => {len(b)}')
    fixed=[]
    for doi in sorted(old_dois):
        te=a.loc[a.doi.eq(doi)]
        tr_old=a.loc[~a.doi.eq(doi)]
        tr_new=b.loc[~b.doi.eq(doi)]
        # held DOI must be absent from both training cohorts
        assert doi not in set(tr_old.doi) and doi not in set(tr_new.doi)
        for version,train in [('original',tr_old),('expanded',tr_new)]:
            xtr=input_frame(train,S1_NUMERIC,S1_CATEGORICAL)
            xte=input_frame(te,S1_NUMERIC,S1_CATEGORICAL)
            model=stage1_pipe().fit(xtr,train[ycol].astype(float))
            preds=model.predict(xte)
            for row,p in zip(te.itertuples(index=False),preds):
                fixed.append({'target':target,'material_id':row.material_id,'doi':doi,
                              'version':version,'actual':float(getattr(row,ycol)),'pred':float(p),
                              'trained_rows':len(train),'evaluation':'paired_old_DOI'})
    fixed=pd.DataFrame(fixed)
    wide=fixed.pivot(index=['material_id','doi','actual'],columns='version',values='pred').reset_index()
    wide['target']=target
    wide['err_original']=(wide.actual-wide.original).abs()
    wide['err_expanded']=(wide.actual-wide.expanded).abs()
    old_new_group=set(b.doi)-set(a.doi)
    cold=[]
    for doi in sorted(old_new_group):
        te=b.loc[b.doi.eq(doi)]
        for version,train in [('original',a),('other_new_DOI',b.loc[~b.doi.eq(doi)])]:
            assert doi not in set(train.doi)
            model=stage1_pipe().fit(input_frame(train,S1_NUMERIC,S1_CATEGORICAL),train[ycol].astype(float))
            preds=model.predict(input_frame(te,S1_NUMERIC,S1_CATEGORICAL))
            for row,p in zip(te.itertuples(index=False),preds):
                cold.append({'target':target,'material_id':row.material_id,'doi':doi,
                             'version':version,'actual':float(getattr(row,ycol)),'pred':float(p),
                             'trained_rows':len(train),'evaluation':'new_DOI_challenge'})
    challenge=pd.DataFrame(cold)
    stats={
        'target':target,'original_rows':len(a),'expanded_rows':len(b),
        'old_doi_count':len(old_dois),'new_doi_count':len(old_new_group),
        'paired_original':observed_metrics(fixed.loc[fixed.version.eq('original')]),
        'paired_expanded':observed_metrics(fixed.loc[fixed.version.eq('expanded')]),
        'paired_delta':cluster_delta_ci(wide),
        'new_doi_old_training':observed_metrics(challenge.loc[challenge.version.eq('original')]),
        'new_doi_other_new_training':observed_metrics(challenge.loc[challenge.version.eq('other_new_DOI')]),
    }
    return wide,challenge,stats


def stage2_targets() -> tuple[pd.DataFrame,pd.DataFrame]:
    original_file = PROJECT_ROOT/'data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/03_Stage2_QE_Core_97.csv'
    old=pd.read_csv(original_file,low_memory=False)
    new=pd.DataFrame(build_stage2_core98_candidate(PROJECT_ROOT))
    if len(old)!=97 or len(new)!=98 or new['observation_id'].duplicated().any():
        raise ValueError('Stage2 provenance/row integrity mismatch')
    assert set(old.observation_id).issubset(set(new.observation_id))
    if len(set(new.doi)-set(old.doi)) != 1:
        raise ValueError('Expected one new DOI in Stage2')
    return old,new


def _stage2_fit_predict(train:pd.DataFrame, test:pd.DataFrame, model_name:str, algo:str) -> np.ndarray:
    from pyrolysis.biochar_stage2.v16_validate import validation_algorithms
    factory=validation_algorithms()[algo]
    features=fc.model_inputs(model_name)
    xxtr=train.reindex(columns=features).copy()
    xxte=test.reindex(columns=features).copy()
    nums=[c for c in features if c not in {'pollutant','feedstock_label_normalized','manufacturing_route','pyrolysis_atmosphere','activation_or_modification'}]
    cats=[c for c in features if c not in nums]
    for col in nums:
        xxtr[col]=pd.to_numeric(xxtr[col],errors='coerce')
        xxte[col]=pd.to_numeric(xxte[col],errors='coerce')
    for col in cats:
        xxtr[col]=xxtr[col].where(xxtr[col].notna() & xxtr[col].ne(''), np.nan).astype('object')
        xxte[col]=xxte[col].where(xxte[col].notna() & xxte[col].ne(''), np.nan).astype('object')
    model=stage2_pipe(factory(),features)
    model.fit(xxtr,pd.to_numeric(train[fc.TARGET_COLUMN],errors='raise'))
    return np.asarray(model.predict(xxte),dtype=float)


def _evaluate_stage2(model_name: str, algo: str, old:pd.DataFrame, new:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    if set(old.doi) & (set(new.doi)-set(old.doi)):
        raise AssertionError('new DOI unexpectedly old')
    old_groups=sorted(set(old.doi))
    records=[]
    for doi in old_groups:
        test=old.loc[old.doi.eq(doi)]
        for version,train in [('original',old.loc[~old.doi.eq(doi)]),
                              ('expanded',new.loc[~new.doi.eq(doi)])]:
            assert doi not in set(train.doi)
            preds=_stage2_fit_predict(train,test,model_name,algo)
            for r,p in zip(test.itertuples(index=False),preds):
                records.append({'model':model_name,'algorithm':algo,'observation_id':r.observation_id,
                                'doi':doi,'actual':float(r.target_value),'pred':float(p),'version':version,
                                'c0':float(r.C0_mg_L),'dose':float(r.dose_g_L),'evaluation':'paired_old_DOI'})
    fixed=pd.DataFrame(records)
    wide=fixed.pivot(index=['observation_id','doi','actual','c0','dose'],columns='version',values='pred').reset_index()
    wide['model']=model_name
    wide['algorithm']=algo
    wide['err_original']=(wide.actual-wide.original).abs()
    wide['err_expanded']=(wide.actual-wide.expanded).abs()
    newdois=set(new.doi)-set(old.doi)
    challenge=[]
    for doi in newdois:
        test=new.loc[new.doi.eq(doi)]
        pred=_stage2_fit_predict(old,test,model_name,algo)
        for r,p in zip(test.itertuples(index=False),pred):
            challenge.append({'model':model_name,'algorithm':algo,'observation_id':r.observation_id,
                              'doi':doi,'actual':float(r.target_value),'pred':float(p),
                              'c0':float(r.C0_mg_L),'dose':float(r.dose_g_L),'evaluation':'new_DOI_challenge'})
    cold=pd.DataFrame(challenge)
    for ver in ('original','expanded'):
        wide[f'{ver}_clipped']=np.clip(wide[ver].to_numpy(dtype=float),0,wide.c0.to_numpy(dtype=float)/wide.dose.to_numpy(dtype=float))
    stats={'model':model_name,'algorithm':algo,'original_rows':len(old),'expanded_rows':len(new),
           'paired_original':observed_metrics(fixed.loc[fixed.version.eq('original')]),
           'paired_expanded':observed_metrics(fixed.loc[fixed.version.eq('expanded')]),
           'paired_delta':cluster_delta_ci(wide),
           'new_DOI_challenge':observed_metrics(cold),
           'new_DOI_absolute_error':float(abs(cold.iloc[0].actual-cold.iloc[0].pred)) if len(cold)==1 else None,
           'clipped_original':observed_metrics(wide.rename(columns={'original_clipped':'pred'})),
           'clipped_expanded':observed_metrics(wide.rename(columns={'expanded_clipped':'pred'})),
           'raw_violation_original_pct':float(100*np.mean((wide.original<0)|(wide.original>wide.c0/wide.dose))),
           'raw_violation_expanded_pct':float(100*np.mean((wide.expanded<0)|(wide.expanded>wide.c0/wide.dose))),
          }
    return wide,cold,stats


def run(base=PROJECT_ROOT, out=REPORT) -> dict:
    global PROJECT_ROOT,PHASE
    PROJECT_ROOT=Path(base)
    PHASE=PROJECT_ROOT/'data/imports/Biochar_Phase8_Integrated_Data_Package/data'
    out=Path(out)
    audit_phase8(PROJECT_ROOT)
    out.mkdir(parents=True,exist_ok=True)
    summary={'evaluation_name':'Phase 8 fixed-pipeline, DOI-paired incremental audit',
             'deployed_model_unchanged':True,'same_doi_test_rows_for_old_cohort':True,
             'not_v60_deployed_architecture':True,
             'not_nested_v16_selection':True,
             'new_doi_used_for_old_doi_training_only':True,'models':{}}
    stage1_old,stage1_new=stage1_targets()
    for target in ['BET','TPV']:
        paired,challenge,stats=_evaluate_stage1(target,stage1_old,stage1_new)
        paired.to_csv(out/f'stage1_{target.lower()}_paired_old_DOI.csv',index=False)
        challenge.to_csv(out/f'stage1_{target.lower()}_new_DOI_challenge.csv',index=False)
        summary['models'][f'Stage1_{target}']=stats
    stage2_old,stage2_new=stage2_targets()
    for label,algo in [('A','XGBoost'),('B','Ridge')]:
        paired,challenge,stats=_evaluate_stage2(label,algo,stage2_old,stage2_new)
        paired.to_csv(out/f'stage2_{label}_paired_old_DOI.csv',index=False)
        challenge.to_csv(out/f'stage2_{label}_new_DOI_challenge.csv',index=False)
        summary['models'][f'Stage2_{label}_{algo}']=stats
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    result=run()
    print(json.dumps(result['models'],indent=2,ensure_ascii=False))
    print('RESULT DIR:',REPORT)
