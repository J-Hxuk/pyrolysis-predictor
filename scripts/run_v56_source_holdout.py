"""v56: source-excluded retrospective challenge + future DOI model freeze.

Usage:
    python -m scripts.run_v56_source_holdout --root . --mode retrospective
    python -m scripts.run_v56_source_holdout --root . --mode seal
    python -m scripts.run_v56_source_holdout --root . --mode prospective --candidate path/to/new_verified_qe.csv

No DB/model update. No active model writes. Existing DOI experiments are RETROSPECTIVE ONLY.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from scripts.build_v46_phase8 import s2_model, s1_model, features
from scripts.run_v55_source_cumulative_refit import to_stage2, BASE2, APPLE, BASE1, QBASE, MATS, DOI

OUT='reports/v56_source_out_prospective_seal'
SEAL='deployment/research_v56_FROZEN_EVALUATION_NOT_ACTIVE'
V55='data/imports/v55_cumulative_sources'
DEP55='deployment/research_v55_NOT_ACTIVE'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(s):return str(s).strip().lower().replace('https://doi.org/','').replace('http://doi.org/','').replace('doi:','').strip().rstrip('/')
def all_normalized(a):return set(normalize(x) for x in a if pd.notna(x) and normalize(x) not in ('','nan','nr'))
def score(y, p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('nonfinite / empty target or prediction')
    return dict(n=int(len(y)),mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),
                r2=float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>1e-12 else None)

def stage2_predict(tr,te,variant):
    assert all_normalized(tr.doi).isdisjoint(all_normalized(te.doi)), 'DOI leakage: refused'
    m,cols,nums,cats=s2_model(variant)
    m.fit(features(tr,cols,nums,cats),tr.target_value.to_numpy(float))
    raw=np.asarray(m.predict(features(te,cols,nums,cats)),float)
    cap=te.C0_mg_L.to_numpy(float)/te.dose_g_L.to_numpy(float)
    if not ((np.isfinite(cap)&(cap>0)).all() and np.isfinite(raw).all()): raise ValueError('invalid cap/raw')
    pred=np.clip(raw,0,cap)
    return pred,raw

def load_s2(root):
    base=pd.read_csv(root/BASE2,low_memory=False)
    apple=pd.read_csv(root/APPLE,low_memory=False)
    raw=pd.read_csv(root/V55/'New_Reported_QE_8.csv')
    new=to_stage2(raw,base)
    apple8=apple.loc[apple.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP'].copy()
    batches={'apple':apple8,'sugar':new.loc[new.source_key=='sugar'].copy(),
             'cocoa_conditional':new.loc[new.source_key=='cocoa'].copy()}
    assert len(base)==140 and len(batches['apple'])==8 and len(batches['sugar'])==4 and len(batches['cocoa_conditional'])==4
    assert len({normalize(x) for a in batches.values() for x in a.doi})==3
    for key,b in batches.items():
        assert set(b.observation_id).isdisjoint(set(base.observation_id))
        assert all_normalized(b.doi).isdisjoint(all_normalized(base.doi))
        assert (b.target_value>=0).all() and (b.target_value<=b.C0_mg_L/b.dose_g_L+1e-8).all()
    return base,batches

def retro_stage2(root):
    base,src=load_s2(root)
    rec=[];pred=[]
    for testkey,test in src.items():
        plans={
           'historical140_only':[],
           'other_verified_sources':[k for k in ('apple','sugar') if k!=testkey],
           'other_verified_plus_conditional':[k for k in src if k!=testkey],
        }
        for plan,keys in plans.items():
            # Never add the same observations twice (last two plans can coincide).
            tr=pd.concat([base]+[src[k] for k in keys],ignore_index=True,sort=False)
            for variant in ('A','B'):
                est,raw=stage2_predict(tr,test,variant)
                ref=score(test.target_value,est)
                cap=(test.C0_mg_L/test.dose_g_L).to_numpy(float)
                rec.append(dict(target='qe_reported',heldout_source=testkey, heldout_doi=normalize(test.doi.iloc[0]),
                                evidence='C0_INFERRED_CONFLICT_RETROSPECTIVE_ONLY' if testkey=='cocoa_conditional' else 'QE_SOURCE_REPORTED_RETROSPECTIVE_ONLY',
                                plan=plan,other_train_sources=';'.join(keys),train_rows=len(tr),model=variant,
                                raw_upper=int((raw>cap+1e-8).sum()),raw_negative=int((raw<0).sum()),
                                zero_predictor_mae=float(np.mean(np.abs(test.target_value.to_numpy(float)))),
                                beats_zero_predictor=bool(ref['mae']<np.mean(np.abs(test.target_value.to_numpy(float)))),
                                **ref))
                for row, pr, rr, cp in zip(test.itertuples(index=False),est,raw,cap):
                    pred.append(dict(target='qe_reported',heldout_source=testkey,heldout_doi=normalize(row.doi),
                        observation_id=row.observation_id,pollutant=row.pollutant,variant=variant,plan=plan,true=row.target_value,
                        predicted=float(pr),raw_predicted=float(rr),cap=float(cp),abs_error=float(abs(row.target_value-pr))))
    return pd.DataFrame(rec),pd.DataFrame(pred)

def retro_rank(pred):
    rows=[]
    for (source,metal,variant,plan),grp in pred.groupby(['heldout_source','pollutant','variant','plan'],sort=True):
        observed=grp.true.to_numpy(float);estimated=grp.predicted.to_numpy(float)
        tied=bool(np.ptp(estimated)<1e-10)
        rho=None if tied or np.ptp(observed)<1e-10 else float(spearmanr(observed,estimated).statistic)
        obsbest=int(np.argmax(observed)); top=int(np.argmax(estimated)) if not tied else None
        rows.append({'heldout_source':source,'pollutant':metal,'variant':variant,'plan':plan,'n':len(grp),
            'all_predictions_tied':tied,'spearman_rho':rho,'top1_regret_mg_g':None if tied else float(observed[obsbest]-observed[top]),
            'observed_best_id':grp.iloc[obsbest].observation_id,
            'predicted_best_id':None if tied else grp.iloc[top].observation_id,
            'status':'RETROSPECTIVE_SOURCE_OUT_NOT_PROSPECTIVE'})
    return pd.DataFrame(rows)

def retro_stage1(root):
    original=pd.read_csv(root/BASE1,low_memory=False)
    extra=pd.read_csv(root/V55/'New_Stage1_Properties_29.csv')
    assert len(original)==417 and len(extra)==29
    out=[]; preds=[]
    for target,key in [('BET_m2_g_training_candidate','BET'),('TPV_cm3_g_training_candidate','TPV')]:
        old=original.loc[pd.to_numeric(original[target],errors='coerce').notna()].copy()
        novel=extra.loc[pd.to_numeric(extra[target],errors='coerce').notna()].copy()
        for s,g in novel.groupby('source_key',sort=True):
            for plan,adds in [('historical_only',novel.iloc[0:0]),('other_new_DOIs',novel.loc[novel.source_key!=s])]:
                train=pd.concat([old,adds],ignore_index=True,sort=False)
                assert all_normalized(train.doi).isdisjoint(all_normalized(g.doi)),f'Stage1 DOI leakage {s}'
                m,cols,nums,cats=s1_model()
                m.fit(features(train,cols,nums,cats),train[target].to_numpy(float))
                raw=np.asarray(m.predict(features(g,cols,nums,cats)),float)
                pp=np.maximum(0,raw)
                out.append(dict(target=key,heldout_source=s,heldout_doi=normalize(g.doi.iloc[0]),plan=plan,
                     train_rows=len(train),zero_predictor_mae=float(np.mean(np.abs(g[target].to_numpy(float)))),
                     **score(g[target],pp)))
                for id_,actual,guess in zip(g.material_id,g[target],pp):
                    preds.append(dict(target=key,heldout_source=s,heldout_doi=normalize(g.doi.iloc[0]),
                                      material_id=id_,plan=plan,true=float(actual),predicted=float(guess),abs_error=float(abs(actual-guess))))
    return pd.DataFrame(out),pd.DataFrame(preds)

def retro_qmax(root):
    """Hold out all fitted-qmax rows of one new DOI, never merge with qe."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBRegressor
    old=pd.read_csv(root/QBASE).rename(columns={'doi_group':'doi'})
    mats=pd.read_csv(root/MATS,usecols=['material_uid','manufacturing_route','pyrolysis_temperature_C'])
    old=old.merge(mats.drop_duplicates('material_uid'),on='material_uid',how='left',validate='many_to_one')
    new=pd.read_csv(root/V55/'New_Fitted_Qmax_16.csv')
    new['manufacturing_route']=np.where(new.material_uid.str.contains('CPHB900A'),'MULTI_STEP_OR_MODIFIED','SINGLE_STAGE_PYROLYSIS')
    assert len(old)==32 and len(new)==16
    assert set(old.doi).isdisjoint(set(new.doi))
    pcols=['pollutant','feedstock_raw','pyrolysis_temperature_C','manufacturing_route']
    def make_model():
        return Pipeline([('pre',ColumnTransformer([
            ('numeric',SimpleImputer(strategy='median'),['pyrolysis_temperature_C']),
            ('categorical',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='unknown')),
                 ('ohe',OneHotEncoder(handle_unknown='ignore'))]),['pollutant','feedstock_raw','manufacturing_route'])])),
            ('model',XGBRegressor(n_estimators=150,max_depth=2,learning_rate=.06,reg_lambda=8,
                min_child_weight=4,n_jobs=1,random_state=42,tree_method='hist'))])
    rec=[];pred=[]
    for source,g in new.groupby('source_key',sort=True):
        for plan,addon in [('old32_only',new.iloc[:0]),('other_qmax_new_DOI',new.loc[new.source_key!=source])]:
            tr=pd.concat([old,addon],ignore_index=True,sort=False)
            assert all_normalized(tr.doi).isdisjoint(all_normalized(g.doi))
            m=make_model().fit(tr.reindex(columns=pcols),tr.target_value.to_numpy(float))
            pp=np.maximum(0,np.asarray(m.predict(g.reindex(columns=pcols)),float))
            rec.append(dict(target='LANGMUIR_FITTED_QMAX_NOT_QE',heldout_source=source,
                heldout_doi=normalize(g.doi.iloc[0]),plan=plan,train_rows=len(tr),
                zero_predictor_mae=float(np.mean(np.abs(g.target_value.to_numpy(float)))),**score(g.target_value,pp)))
            for row,guess in zip(g.itertuples(index=False),pp):
                pred.append(dict(target='LANGMUIR_FITTED_QMAX_NOT_QE',heldout_source=source,
                    heldout_doi=normalize(row.doi),observation_id=row.observation_id,
                    plan=plan,true=float(row.target_value),predicted=float(guess),abs_error=float(abs(guess-row.target_value))))
    return pd.DataFrame(rec),pd.DataFrame(pred)

def historical_dois(root):
    candidates=[(BASE2,'doi'),(APPLE,'doi'),(BASE1,'doi'),(V55+'/New_Reported_QE_8.csv','doi'),
                (V55+'/New_Stage1_Properties_29.csv','doi'),(V55+'/QMAX_48_FITTED_SEPARATE.csv','doi'),
                (V55+'/Source_Registry_5.csv','doi'),
                ('data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/10_Primary_Source_DOI_Registry_61.csv','doi'),
                ('data/imports/Biochar_Stage1_Expansion_v5/biochar/data/Stage2_qe.csv','original_experiment_doi'),
                ('data/imports/Biochar_Stage1_Expansion_v5/biochar/data/Stage2_qe.csv','source_doi')]
    result=set();paths={}
    for path,col in candidates:
        p=root/path
        frame=pd.read_csv(p,low_memory=False)
        if col not in frame.columns: raise ValueError(f'Invalid provenance file: {path} missing {col}')
        result |=all_normalized(frame[col].tolist())
        paths[path]=sha(p)
    # DOI historically exposed to prior Stage1 artifacts; see vNext10 2017 linkage.
    result |= {'10.3390/en10091293'}
    return sorted(result),paths

def frozen_models(root):
    p=root/DEP55
    names={
      'stage2_A_source152':'STAGE2_A_v55_152_plus_sugar_152_RESEARCH_NOT_ACTIVE.joblib',
      'stage2_B_source148':'STAGE2_B_v53_148_148_RESEARCH_NOT_ACTIVE.joblib',
      'stage1_BET_selected14':'STAGE1_BET_375_SELECTED14_EXPLORATORY_NOT_ACTIVE.joblib',
      'stage1_TPV_all10':'STAGE1_TPV_241_RESEARCH_NOT_ACTIVE.joblib',
    }
    return {k:{'path':f'{DEP55}/{v}','sha256':sha(p/v)} for k,v in names.items()}

def seal(root):
    import sklearn, xgboost
    dois,inputs=historical_dois(root); models=frozen_models(root)
    # This is a registry of already-observed DOI, not evidence of what was truly blind earlier.
    man={'protocol':'v56_frozen_futures_only','created_date':'2026-10-31',
         'historical_DOIs_including_known_lineage':dois,'historical_DOI_count':len(dois),
         'input_hashes':inputs,'frozen_candidate_models':models,
         'model_selection_status':'RETROSPECTIVELY_SELECTED_V55; future only considered prospective if never examined before seal',
         'future_test_conditions':['DOI not in historical_DOIs','source verified equilibrium qe, not fitted qmax/qt',
             'complete C0_mg_L and dose_g_L physical validation',
             'no refit/tuning/selection using the new DOI or related sample source',
             'sample_batch_independent and source_lineage_doi verified; related 2017 DOI disallowed',
             'model and training-input hash match; if not, reseal before testing'],
         'software_versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
         'caveat':'Seal produced AFTER v55 model selection; source-out tests of already seen DOI are retrospective, never blind.'}
    d=root/SEAL;d.mkdir(parents=True,exist_ok=True)
    f=d/'FUTURE_DOI_SEAL.json'
    if f.exists():
        prev=json.loads(f.read_text(encoding='utf-8'))
        if prev['input_hashes']!=inputs or prev['frozen_candidate_models']!=models:
            raise ValueError('SEAL INTEGRITY ERROR: underlying data/model changed. Do not overwrite historical seal; archive and issue new seal.')
        return prev
    f.write_text(json.dumps(man,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return man

def prospective(root,candidate):
    d=root/SEAL/'FUTURE_DOI_SEAL.json'
    if not d.exists():raise FileNotFoundError('Run --mode seal BEFORE collecting future source data.')
    man=json.loads(d.read_text(encoding='utf-8'))
    for path,expected in man['input_hashes'].items():
        if sha(root/path)!=expected:raise ValueError(f'Presealed training data changed: {path}')
    for spec in man['frozen_candidate_models'].values():
        if sha(root/spec['path'])!=spec['sha256']:raise ValueError('Sealed model changed: '+spec['path'])
    df=pd.read_csv(candidate)
    required=['doi','observation_id','target_value','C0_mg_L','dose_g_L','qe_source_evidence_status',
              'target_kind','source_lineage_doi','sample_batch_independent','pollutant','contact_time_min']
    miss=set(required)-set(df.columns)
    if miss:raise ValueError('Missing required columns: '+str(sorted(miss)))
    if df.observation_id.duplicated().any():raise ValueError('Duplicate observation_id in new source')
    seen=set(man['historical_DOIs_including_known_lineage'])
    got=all_normalized(df.doi)
    if not got:raise ValueError('Missing DOI')
    shared=seen&got
    if shared:raise ValueError(f'Previously exposed DOI: {sorted(shared)}; NOT independent external test')
    if set(df.target_kind.astype(str))!={'EXPERIMENTAL_EQUILIBRIUM_QE_MG_G'}:
        raise ValueError('target_kind must be experimental equilibrium qe, not qt/qmax')
    batch=df.sample_batch_independent.astype(str).str.strip().str.lower()
    if not batch.isin(['true','yes','1']).all():
        raise ValueError('Independent physical batch lineage has not been established')
    related=set()
    for item in df.source_lineage_doi.astype(str):
        if item.strip().upper()=='NONE_VERIFIED':continue
        related.update(all_normalized(item.split(';')))
    if seen&related:raise ValueError('Related historically exposed sample lineage DOI: '+str(sorted(seen&related)))
    statuses=set(df.qe_source_evidence_status.astype(str))
    if statuses!={'SOURCE_VERIFIED_EXPERIMENTAL_EQUILIBRIUM_QE'}:
        raise ValueError('Not strictly verified equilibrium experimental qe: '+str(statuses))
    c0=pd.to_numeric(df.C0_mg_L,errors='raise');dose=pd.to_numeric(df.dose_g_L,errors='raise')
    y=pd.to_numeric(df.target_value,errors='raise')
    if not ((np.isfinite(c0)&np.isfinite(dose)&np.isfinite(y)&(c0>0)&(dose>0)&(y>=0)&(y<=c0/dose+1e-8)).all()):
        raise ValueError('Mass balance or finite input failure. Never silently repair.')
    import joblib
    out=[];preds=[]
    for name,spec in man['frozen_candidate_models'].items():
        if not name.startswith('stage2_'):continue
        v='A' if name.startswith('stage2_A') else 'B'
        _,cols,nums,cats=s2_model(v)
        # source evidence/provenance fields are not model features
        fit=joblib.load(root/spec['path'])
        raw=np.asarray(fit.predict(features(df,cols,nums,cats)),float)
        p=np.clip(raw,0,(c0/dose).to_numpy(float))
        out.append(dict(model=name,n_doi=len(got),**score(y,p)))
        for i,actual,guess in zip(df.observation_id,y,p):preds.append(dict(observation_id=i,model=name,true=float(actual),predicted=float(guess)))
    report=root/OUT/'future_evaluation';report.mkdir(parents=True,exist_ok=True)
    suffix=sha(candidate)[:12]
    pd.DataFrame(out).to_csv(report/f'prospective_{suffix}_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(report/f'prospective_{suffix}_predictions.csv',index=False)
    return out

def retrospective(root):
    root=Path(root).resolve(); d=root/OUT;d.mkdir(parents=True,exist_ok=True)
    s2,p2=retro_stage2(root);s1,p1=retro_stage1(root);q2,pq=retro_qmax(root)
    ranks=retro_rank(p2)
    ranks.to_csv(d/'stage2_retrospective_source_out_rank.csv',index=False)
    s2.to_csv(d/'stage2_retrospective_source_out_metrics.csv',index=False)
    p2.to_csv(d/'stage2_retrospective_source_out_predictions.csv',index=False)
    q2.to_csv(d/'qmax_retrospective_source_out_metrics.csv',index=False)
    pq.to_csv(d/'qmax_retrospective_source_out_predictions.csv',index=False)
    s1.to_csv(d/'stage1_retrospective_source_out_metrics.csv',index=False)
    p1.to_csv(d/'stage1_retrospective_source_out_predictions.csv',index=False)
    summary={'retrospective_stage2':s2.to_dict('records'),'retrospective_stage2_rank':json.loads(ranks.to_json(orient='records')),'retrospective_stage1':s1.to_dict('records'), 'retrospective_qmax':q2.to_dict('records'),
      'method_limits':['Already examined DOI: not pre-registered blind','Small groups 4-8 Stage2 observations',
       'Cocoa C0 estimated and conflicts with kinetics methods','No selection on retrospective held-out performance',
       'Do not mix qmax or qt with experimental qe','No external fold can include its own DOI during training']}
    (d/'V56_RESEARCH_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return s2,s1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--mode',choices=['retrospective','seal','prospective'],required=True)
    ap.add_argument('--candidate',default=None);args=ap.parse_args();root=Path(args.root).resolve()
    if args.mode=='retrospective':
        a,b=retrospective(root);print('=== Stage2 source-out ===');print(a.to_string(index=False));print('=== Stage1 source-out ===');print(b.to_string(index=False))
    elif args.mode=='seal':
        m=seal(root);print('Frozen future DOI seal:',len(m['historical_DOIs_including_known_lineage']),'known DOI')
    else:
        if not args.candidate:ap.error('--candidate required for prospective evaluation')
        print(json.dumps(prospective(root,Path(args.candidate)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
