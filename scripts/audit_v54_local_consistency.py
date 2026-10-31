"""Reproducibility audit for v53/v54 research-only reports.

Never changes raw data, trained artifacts or active deployment.
Old MAE values are HISTORICAL reference numbers, not cross-version exact assertions.
Usage:
    python -m scripts.audit_v54_local_consistency --root .
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

R53 = Path('reports/v53_evidence_tier_refit')
R54 = Path('reports/v54_bounded_ranking')
BASE = Path('data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv')
HISTORICAL = {'A_base140':17.246810957570563, 'A_plus8':17.189799, 'A_plus24':16.760127406300303,
              'B_base140':17.690416, 'B_plus8':17.642990, 'B_plus24':17.760963}
TOL=1e-7


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(root):
    root=Path(root).resolve()
    r53 = json.loads((root/R53/'v53_summary.json').read_text(encoding='utf-8'))
    r54 = json.loads((root/R54/'v54_summary.json').read_text(encoding='utf-8'))
    return root, r53, r54


def ensure_current_runtime(r53,r54):
    actual={'sklearn':sklearn.__version__, 'xgboost':xgboost.__version__}
    prev={'sklearn':str(r53['status']['sklearn_version']),
          'xgboost':str(r53['status']['xgboost_version'])}
    curr={k:str(r54['software_versions'][k]) for k in actual}
    require(prev==actual,
        f'v53 report/model was saved with {prev}, current runtime {actual}; rerun v53 then v54, do NOT edit the MAE expected value')
    require(curr==actual,
        f'v54 report/model was saved with {curr}, current runtime {actual}; rerun v54 after v53')
    require(str(r53['status']['python_version']).split('.')[:2]==platform.python_version().split('.')[:2],
        'v53 report was made with different Python minor version: re-run under current interpreter')
    require(str(r54['software_versions']['python']).split('.')[:2]==platform.python_version().split('.')[:2],
        'v54 report was made with different Python minor version: re-run under current interpreter')


def metrics(y,p):
    return {'mae_mg_g':float(mean_absolute_error(y,p)),
            'rmse_mg_g':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p))}


def validate_base(root, oof):
    base=pd.read_csv(root/BASE,low_memory=False)
    require(len(base)==len(oof)==140, 'Expected 140 original evaluation rows')
    require(base.observation_id.is_unique and oof.observation_id.is_unique, 'Duplicate observation ID')
    require(base.doi.nunique()==32 and oof.doi.nunique()==32, 'Expected 32 evaluation DOI groups')
    require(list(base.observation_id.astype(str))==list(oof.observation_id.astype(str)),
            'OOF observation ordering differs from original; check v46 source and regeneration')
    for col in ['doi','C0_mg_L','dose_g_L','target_value']:
        a=base[col].to_numpy(); b=oof[col].to_numpy()
        if col=='doi':require((a==b).all(), 'OOF DOI values differ from original')
        else:require(np.allclose(a.astype(float),b.astype(float),atol=1e-10,rtol=0),f'OOF {col} differs from original')
    cap=base.C0_mg_L.to_numpy(float)/base.dose_g_L.to_numpy(float)
    return base,cap


def verify_v53_report(root):
    root=Path(root)
    tab=pd.read_csv(root/R53/'v53_fixed140_metrics.csv')
    oof=pd.read_csv(root/R53/'v53_fixed140_predictions.csv')
    base,cap=validate_base(root,oof)
    require(len(tab)==6 and set(tab.model)=={'A','B'} and set(tab.scenario)=={'base140','plus8','plus24'},'v53 score contract changed')
    for v in 'AB':
        for scenario in ['base140','plus8','plus24']:
            row=tab[(tab.model==v)&(tab.scenario==scenario)]
            require(len(row)==1,'v53 score row missing or duplicated')
            row=row.iloc[0]
            pred=oof[f'{v}_{scenario}'].to_numpy(float)
            raw=oof[f'{v}_{scenario}_raw'].to_numpy(float)
            require(np.isfinite(pred).all() and np.isfinite(raw).all(),'Invalid v53 OOF predictions')
            require(np.allclose(np.clip(raw,0,cap),pred,rtol=0,atol=TOL), 'v53 clipped OOF differs from raw predictions')
            m=metrics(oof.target_value.to_numpy(float),pred)
            for k,value in m.items():
                require(abs(float(row[k])-value)<TOL,f'v53 {v}/{scenario}/{k} does not match saved OOF predictions')
            require(int(row.raw_upper_violations)==int((raw>cap+1e-8).sum()), 'v53 raw upper violations inconsistent')
            require(int(row.n)==140 and int(row.doi_count)==32, 'v53 evaluation sample count incorrect')
            fold=oof[f'fold_{scenario}'].to_numpy(int)
            require(set(fold)=={1,2,3,4,5},'v53 invalid folds')
            require(oof.groupby('doi')[f'fold_{scenario}'].nunique().eq(1).all(), 'DOI appears in multiple heldout folds')
    return tab,oof


def verify_v54_report(root):
    root=Path(root)
    tab=pd.read_csv(root/R54/'v54_same140_metrics.csv')
    oof=pd.read_csv(root/R54/'v54_same140_oof_predictions.csv')
    base,cap=validate_base(root,oof)
    require(len(tab)==18,'v54 expected 18 OOF score rows')
    require(set(oof.fold.astype(int))=={1,2,3,4,5},'v54 fold contract failed')
    require(oof.groupby('doi').fold.nunique().eq(1).all(),'v54 DOI held-out groups mixed')
    y=base.target_value.to_numpy(float)
    for v in 'AB':
        for scenario in ['base140','plus8','plus24']:
            for method in ['direct_clip','logit_fraction','logistic_fraction']:
                sel=tab[(tab.variant==v)&(tab.training_scenario==scenario)&(tab.method==method)]
                require(len(sel)==1,f'v54 missing/duplicate row {v}/{scenario}/{method}')
                row=sel.iloc[0]
                col=f'{v}__{scenario}__{method}'
                p=oof[col].to_numpy(float);raw=oof[col+'__raw'].to_numpy(float)
                require(np.isfinite(p).all() and np.isfinite(raw).all(),f'Nonfinite OOF {col}')
                require((p>=-TOL).all() and (p<=cap+TOL).all(),f'Mass balance violation after {method}')
                if method=='direct_clip':
                    require(np.allclose(np.clip(raw,0,cap),p,atol=TOL,rtol=0),f'Raw clip mismatch {col}')
                    require(int(row.direct_unconstrained_upper_violation)==int((raw>cap+1e-8).sum()),
                            'Raw cap count inconsistent')
                m=metrics(y,p)
                for k,value in m.items():
                    require(abs(float(row[k])-value)<TOL,f'v54 {col}/{k} not supported by saved predictions')
    return tab,oof


def verify_cross_report(root):
    root=Path(root)
    _,r53,r54=read(root)
    ensure_current_runtime(r53,r54)
    t53,p53=verify_v53_report(root)
    t54,p54=verify_v54_report(root)
    for v in 'AB':
        for scenario in ['base140','plus8','plus24']:
            a=p53[f'{v}_{scenario}'].to_numpy(float)
            b=p54[f'{v}__{scenario}__direct_clip'].to_numpy(float)
            require(np.allclose(a,b,atol=TOL,rtol=0),
                    f'v53/v54 OOF differs for {v}/{scenario}; rerun v53 then v54 using SAME interpreter')
    require(file_sha(root/BASE)==r54['source_hashes']['base140'], 'Baseline data hash mismatch in v54 summary')
    require(file_sha(root/BASE)==r53['status']['provenance']['stage2_140_sha256'],
            'Baseline data hash mismatch in v53 summary')
    require(file_sha(root/'data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv')==r54['source_hashes']['apple24'],
            'Apple24 source data differs from v54 report')
    require(file_sha(root/'data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv')==r53['status']['provenance']['v52_apple24_sha256'],
            'Apple24 source data differs from v53 report')
    # JSON, CSV, and raw OOF should all describe the same LOCAL training run.
    for _,line in t53.iterrows():
        entries=[z for z in r53['metrics'] if z['model']==line['model'] and z['scenario']==line['scenario']]
        require(len(entries)==1 and abs(float(entries[0]['mae_mg_g'])-float(line['mae_mg_g']))<TOL,
                'v53 summary JSON differs from current metrics CSV')
    for _,line in t54.iterrows():
        entries=[z for z in r54['same140_metrics'] if z['variant']==line['variant']
                 and z['training_scenario']==line['training_scenario'] and z['method']==line['method']]
        require(len(entries)==1 and abs(float(entries[0]['mae_mg_g'])-float(line['mae_mg_g']))<TOL,
                'v54 summary JSON differs from current metrics CSV')
    for rec in r53['models']:
        path=root/rec['path']
        require('NOT_ACTIVE' in rec['path'] and path.exists() and file_sha(path)==rec['sha256'],
            'v53 research model file missing or checksum mismatch')
    for rec in r54['models']:
        path=root/rec['model_path']
        require('NOT_ACTIVE' in rec['model_path'] and path.exists() and file_sha(path)==rec['sha256'],
            'v54 research model file missing or checksum mismatch')
    return t53,t54,r53,r54


def audit(root):
    tab53,tab54,r53,r54=verify_cross_report(root)
    rows=[]
    for v in 'AB':
        for s in ['base140','plus8','plus24']:
            old=HISTORICAL[f'{v}_{s}']
            new=float(tab53[(tab53.model==v)&(tab53.scenario==s)].iloc[0]['mae_mg_g'])
            rows.append({'model':v,'scenario':s,'historical_2026_10_31_mae':old,
                         'local_retrained_mae':new,'delta_local_minus_historical':new-old})
    summary={'status':'PASS_LOCAL_REPORT_COHERENCE_NOT_EXTERNAL_GENERALIZATION_PROOF',
             'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
             'note':'Historical figures not exact contracts across Python/sklearn/xgboost versions. Metric deltas may also have other causes and need investigation.',
             'n_original_test_rows':140,'n_original_test_dois':32,'comparison':rows,
             'v53_report':'reports/v53_evidence_tier_refit/v53_fixed140_metrics.csv',
             'v54_report':'reports/v54_bounded_ranking/v54_same140_metrics.csv',
             'deployment':'RESEARCH_ONLY_NOT_ACTIVE'}
    target=Path(root)/'reports/v54_local_repro_audit'
    target.mkdir(parents=True,exist_ok=True)
    (target/'LOCAL_ENV_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    args=ap.parse_args()
    audit(Path(args.root).resolve())
