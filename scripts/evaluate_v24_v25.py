"""v25: research-only external DOI challenge on v24 SOURCE STAGING.

No refitting on staged data; no deployment writes. Invalid/out-of-contract source
observations are reported separately rather than silently repaired.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.compare_phase8_v23 import (
    stage1_targets, stage1_pipe, input_frame, S1_NUMERIC, S1_CATEGORICAL,
    _stage2_fit_predict, stage2_targets,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/imports/v24_source_audit'
DEST = ROOT / 'reports/v25_v24_external_challenge'
COLS = {'BET': 'BET_m2_g_training_candidate', 'TPV': 'TPV_cm3_g_training_candidate'}


def scores(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    return {'n': int(len(y)), 'mae': float(mean_absolute_error(y, p)) if len(y) else None,
            'rmse': float(np.sqrt(mean_squared_error(y, p))) if len(y) else None,
            'r2': float(r2_score(y, p)) if len(y)>1 and np.var(y)>0 else None}


def v24_external_stage1(base=ROOT):
    root = Path(base)
    old, phase8 = stage1_targets()
    src = root / 'data/imports/v24_source_audit'
    new = pd.read_csv(src/'stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    assert len(new)==33 and new.doi.nunique()==6
    assert not (set(new.doi)&set(phase8.doi)), 'External DOI leakage / duplicate'
    # Reuse mapped research-staging features rather than reconstruct from PDF tables.
    master = pd.read_csv(src/'Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv',low_memory=False)
    external = master.loc[master.material_id.astype(str).str.startswith('S1ADD24_')].copy()
    assert len(external)==33 and external.material_id.is_unique
    assert set(external.doi)==set(new.doi)
    # Restore auditable manufacturing metadata only from v24 source table: v24 master leaves
    # several feature columns empty. This is an evaluation transformation, not a source edit.
    lookup = new[['doi','sample_name','manufacturing_route']].rename(columns={'sample_name':'original_sample_name'})
    external=external.merge(lookup,on=['doi','original_sample_name'],validate='one_to_one')
    route=external['manufacturing_route'].fillna('').astype(str)
    external.loc[route.eq('single_stage_pyrolysis'),'process_stage']='single_stage_pyrolysis'
    external.loc[route.eq('single_stage_pyrolysis'),'v59_route_family']='none'
    external.loc[route.eq('single_stage_pyrolysis'),'activation_method']='none'
    external.loc[route.eq('co_pyrolysis_biomass_plastic'),'process_stage']='co_pyrolysis_biomass_plastic'
    external.loc[route.eq('co_pyrolysis_biomass_plastic'),'v59_route_family']='none'
    external.loc[route.eq('co_pyrolysis_biomass_plastic'),'activation_method']='none'
    external.loc[route.eq('post_pyrolysis_modification'),'process_stage']='post_pyrolysis_modification'
    external.loc[route.eq('post_pyrolysis_modification'),'v59_route_family']='chemical'
    external.loc[route.eq('post_pyrolysis_modification'),'activation_method']='post_pyrolysis_chemical_modification'
    # HTC is structurally different from dry pyrolysis; report outcomes separately as OOD.
    for column in ['process_stage','v59_route_family','activation_method']:
        external.loc[route.str.contains('hydrothermal|HTC'),column]='__UNSUPPORTED_HTC__'
    # Require direct source targets. Staging training columns must be blank by design.
    assert external['BET_m2_g_training_candidate'].isna().all()
    assert external['TPV_cm3_g_training_candidate'].isna().all()
    audit=[]; predictions=[]
    for target,col in COLS.items():
        # v23 cohort and same feature engineering, no model selection on external cohort.
        roles = pd.read_csv(root/'data/imports/Biochar_Phase8_Integrated_Data_Package/data/stage1/Stage1_Target_Training_Roles_UNCHANGED.csv')
        old_train = phase8.merge(roles[['material_id',f'{target}_training_use']],on='material_id',validate='one_to_one') if f'{target}_training_use' not in phase8.columns else phase8
        tr=old_train.loc[old_train[f'{target}_training_use'].eq(True) & pd.to_numeric(old_train[col],errors='coerce').notna()]
        model=stage1_pipe().fit(input_frame(tr,S1_NUMERIC,S1_CATEGORICAL),tr[col].astype(float))
        pr=model.predict(input_frame(external,S1_NUMERIC,S1_CATEGORICAL))
        actual=pd.to_numeric(external['BET_m2_g_original_source' if target=='BET' else 'TPV_cm3_g_original_source'],errors='coerce')
        baseline=float(tr[col].astype(float).median())
        for (_, row), y, p in zip(external.iterrows(),actual,pr):
            preds=dict(target=target,doi=row['doi'],material_id=row['material_id'],sample=row['original_sample_name'],
                       route=row['manufacturing_route'],
                       source_status=row.get('release_status'),
                       train_n=len(tr),actual=float(y),pred=float(p),dummy_median=baseline,
                       abs_error=float(abs(y-p)),relative_error_pct=float(100*abs(y-p)/max(abs(y),1e-6)),
                       is_single_stage=(row.get('manufacturing_route')=='single_stage_pyrolysis'))
            predictions.append(preds)
        audit.append({'target':target,'trained_rows':len(tr), 'train_doi':tr.doi.nunique()})
    result=pd.DataFrame(predictions)
    summary={}
    for target in COLS:
        a=result.loc[result.target.eq(target)]
        summary[target]={'all':scores(a.actual,a.pred),'median_baseline':scores(a.actual,a.dummy_median),
                         'single_stage':scores(a.loc[a.is_single_stage,'actual'],a.loc[a.is_single_stage,'pred']),
                         'by_route':{str(route):scores(g.actual,g.pred) for route,g in a.groupby('route',dropna=False)},
                         'by_doi':{doi:scores(g.actual,g.pred) for doi,g in a.groupby('doi')},
                         'training':next(x for x in audit if x['target']==target)}
    return result,summary


def v24_external_stage2(base=ROOT):
    root=Path(base)
    old, phase8 = stage2_targets()
    assert len(phase8)==98
    source=pd.read_csv(root/'data/imports/v24_source_audit/stage2_new72_source_reported_QE_STAGING.csv')
    assert len(source)==72 and len(source[source.physical_cap_pass==True])==69
    # Source contract: 18 pomelo missing initial pH; reject despite source unit correction.
    missing=source.loc[source.initial_pH.isna() | source.initial_pH.astype(str).str.strip().eq('')]
    assert len(missing)==18 and set(missing.doi)=={'10.3390/w13060868'}
    fails=source.loc[~source.physical_cap_pass.eq(True)]
    assert len(fails)==3
    eligible=source.loc[source.physical_cap_pass.eq(True) & source.initial_pH.notna() & source.doi.eq('10.3390/ma14216566')].copy()
    assert len(eligible)==51
    staged=pd.read_csv(root/'data/imports/v24_source_audit/Stage2_Core_167_RESEARCH_STAGING_NOT_TRAINED.csv')
    test=staged.loc[staged.observation_id.astype(str).str.startswith('S2ADD24_') & staged.doi.eq('10.3390/ma14216566')].copy()
    assert len(test)==51 and not (set(test.doi)&set(phase8.doi))
    assert test.target_qualification.astype(str).str.contains('NOT_CERTIFIED').all()
    results=[]
    for label,algo in [('A','XGBoost'),('B','Ridge')]:
        pred=_stage2_fit_predict(phase8,test,label,algo)
        for (_, row), p in zip(test.iterrows(),pred):
            y=float(row.target_value); cap=float(row.C0_mg_L)/float(row.dose_g_L)
            results.append({'model':label,'algorithm':algo,'doi':row.doi,'observation_id':row.observation_id,
                            'pollutant':row.pollutant,'sample':row.sample_group,
                            'C0_mg_L':float(row.C0_mg_L),'dose_g_L':float(row.dose_g_L),
                            'actual_mg_g':y,'pred_raw_mg_g':float(p),'pred_physically_clipped_mg_g':float(np.clip(p,0,cap)),
                            'physical_cap_mg_g':cap,'raw_violation':bool(p<0 or p>cap),
                            'actual_fraction':y/cap,'pred_fraction_raw':float(p/cap),
                            'has_unknown_pollutant':row.pollutant not in set(phase8.pollutant),
                            'external_challenge_only':True})
    res=pd.DataFrame(results)
    total={}
    for (label,algo), g in res.groupby(['model','algorithm']):
        total[label]={'algorithm':algo,'all':scores(g.actual_mg_g,g.pred_raw_mg_g),
                      'clipped':scores(g.actual_mg_g,g.pred_physically_clipped_mg_g),
                      'fraction':scores(g.actual_fraction,g.pred_fraction_raw),
                      'raw_violation_pct':float(100*g.raw_violation.mean()),
                      'pure_cap_baseline':scores(g.actual_mg_g,g.physical_cap_mg_g),
                      'raw_negative_count':int((g.pred_raw_mg_g<0).sum()),
                      'raw_above_cap_count':int((g.pred_raw_mg_g>g.physical_cap_mg_g).sum()),
                      'by_pollutant':{pol:{'raw':scores(v.actual_mg_g,v.pred_raw_mg_g),
                                           'clipped':scores(v.actual_mg_g,v.pred_physically_clipped_mg_g),
                                           'raw_violation_pct':float(100*v.raw_violation.mean()),
                                           'unseen_pollutant':bool(v.has_unknown_pollutant.any())}
                                      for pol,v in g.groupby('pollutant')}}
    return res, {'input_exclusions':{'pomelo_missing_initial_pH':18,'poultry_physical_cap_failed':3,
                                     'poultry_cold_challenge_only':51,'training_approved_new_rows':0,
                                     'training_dose_max':float(pd.to_numeric(phase8.dose_g_L).max()),'new_dose_g_L':100,
                                     'unseen_pollutant': 'Zn(II)'},'models':total}


def run(base=ROOT, output=DEST):
    global ROOT
    ROOT=Path(base)
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    s1,m1=v24_external_stage1(ROOT)
    s2,m2=v24_external_stage2(ROOT)
    s1.to_csv(output/'stage1_external33_predictions.csv',index=False)
    s2.to_csv(output/'stage2_poultry51_stress_predictions.csv',index=False)
    payload={'type':'research_only_v25_external_DOI_challenge','not_model_upgrade':True,
             'not_training_on_v24':True,'cannot_establish_independent_DOI_generalization_from_one_poultry_DOI':True,
             'stage1':m1,'stage2':m2}
    (output/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return payload

if __name__=='__main__':
    out=run()
    print(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False))
