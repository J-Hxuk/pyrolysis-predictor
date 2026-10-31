#!/usr/bin/env python3
"""v53 offline-only Stage2 audit and controlled sensitivity evaluation.

NO deployment, NO model promotion, NO DB writes. Uses only supplied local CSVs.
All vNext10 links and cross-source properties remain evidence, NOT new training rows.
"""
from __future__ import annotations
import argparse, hashlib, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parents[1]
ZHAO='10.3390/en13133498'
SRC=ROOT/'evidence'/'vNext10_source.zip'
REF=ROOT/'evidence'/'Zhao2020_24row_AUDIT_REFERENCE_ONLY.csv'
FIELDS=['doi','metal','dose','temp','qe','c0']
MODEL_NAMES=('A','B')
SCHEMA_COLS={k:k for k in FIELDS}

class AuditError(RuntimeError):pass

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def doi(x):
    s=str(x).strip().lower()
    return re.sub(r'^https?://(?:dx\.)?doi\.org/', '',s).strip().rstrip('/')

def metal(x):
    s=str(x).strip().lower()
    if re.search(r'(?:^|[^a-z])cu(?:\b|\()', s):return 'cu'
    if re.search(r'(?:^|[^a-z])zn(?:\b|\()', s):return 'zn'
    return re.sub(r'\s+','',s)

def temp(x):
    try:return int(round(float(x)))
    except (ValueError,TypeError):
        m=re.search(r'bc\s*([3456]00)\b',str(x),re.I)
        return int(m.group(1)) if m else None

def key(row,c):
    # Dose values are stored with 6-decimal canonical precision; no row-index matching.
    raw=pd.to_numeric(row[c['dose']],errors='coerce')
    return (doi(row[c['doi']]),temp(row[c['temp']]),metal(row[c['metal']]),
            round(float(raw),6) if pd.notna(raw) else None)

def load_ref():
    r=pd.read_csv(REF)
    if len(r)!=24:raise AuditError('Reference is not exactly 24 observations')
    r['__key']=[key(row,dict(doi='doi',temp='pyrolysis_temp_C',metal='pollutant',dose='dose_g_L')) for _,row in r.iterrows()]
    if r['__key'].duplicated().any():raise AuditError('Duplicate keys in 24-row reference')
    strict=r[r.equilibrium_evidence_status.eq('TABLE3_STRICT_QE_SOURCE_VERIFIED')]
    if len(strict)!=8 or not strict.dose_g_L.eq(5).all():raise AuditError('Strict-8 cohort definition failed')
    return r

def input_evidence():
    with zipfile.ZipFile(SRC) as z:
        r=pd.read_csv(io.BytesIO(z.read('AppleTree_CuZn_Observed_qe_and_Linkage.csv')))
        b=pd.read_csv(io.BytesIO(z.read('Bamboo_SI_Recovery_Audit.csv')))
    if len(r)!=8 or len(b)!=10:raise AuditError('vNext10 row-count inconsistency')
    if not (r['STRICT_QE_VERIFIED'].eq('YES').all() and r['EXTERNAL_HOLDOUT'].eq('NO').all()):
        raise AuditError('vNext10 evidence status changed')
    # Do NOT accept an inferred batch linkage as training eligibility.
    if r['TRAINING_ELIGIBLE'].eq('YES').any():raise AuditError('Unexpected training eligibility in source')
    return r,b

def source_audit():
    r,b=input_evidence(); ref=load_ref()
    strict=ref[ref.equilibrium_evidence_status.eq('TABLE3_STRICT_QE_SOURCE_VERIFIED')]
    mism=[]
    for _,rec in r.iterrows():
        k=(doi(rec.doi),temp(rec.recipe_pyrolysis_temp_C),metal(rec.pollutant),float(rec.dose_g_L))
        m=strict[strict['__key'].apply(lambda v: v==k)]
        if len(m)!=1 or abs(float(rec.qe_standardized_mg_g)-float(m.iloc[0].observed_adsorption_mg_g))>1e-6:
            mism.append({'observation_id':rec.observation_id,'key':repr(k)})
    if mism:raise AuditError('vNext10 strict8 versus prior reference mismatch '+str(mism))
    return {'source_archive_sha256':sha256(SRC),'reference24_sha256':sha256(REF),
            'strict_qe_label_verified':8,'strict_label_not_automatically_training_eligible':8,
            'dose_1_20_review_non_strict':16,'bamboo_strict_approved':0,
            'external_holdout_approved':0,'physical_material_batch_link_confirmed':False,
            'vnext10_strict8_duplicated_in_reference24':8}

def validate_csv(path):
    p=Path(path)
    if not p.exists():raise AuditError(f'File missing: {p}')
    d=pd.read_csv(p)
    if d.empty:raise AuditError(f'Empty CSV: {p}')
    return d

def assert_config(c,df,who):
    columns=c['columns']
    for k in FIELDS:
        if k not in columns or not columns[k] or columns[k].startswith('__'):
            raise AuditError(f'Config needs explicit columns.{k}')
    for k in FIELDS:
        if columns[k] not in df:raise AuditError(f'{who} missing column {columns[k]} for {k}')
    for model in MODEL_NAMES:
        spec=c['models'][model]
        feats=spec['numeric']+spec['categorical']
        if not feats:raise AuditError(f'Empty {model} features')
        if len(feats)!=len(set(feats)):raise AuditError(f'Duplicate {model} features')
        missing=[col for col in feats if col not in df.columns]
        if missing:raise AuditError(f'{who} missing features for {model}: {missing}')
        forbidden=[col for col in feats if re.search(r'(^qe|^q_e|qmax|^qt\b|removal|^ce_|observed_adsorption|_measured_qe|_fitted_q)',col,re.I)]
        if forbidden:raise AuditError(f'Target/result leakage in {model} feature list: {forbidden}')
        if columns['qe'] in feats:raise AuditError('Target included in features')

def report_cohort(v52, config):
    c=config['columns']; ref=load_ref(); by_key={k:v for k,v in zip(ref['__key'],ref['observed_adsorption_mg_g'])}
    d=v52.copy(); d['__key']=[key(row,c) for _,row in d.iterrows()]
    z=d[d[c['doi']].map(doi).eq(ZHAO)].copy()
    dup=z['__key'].duplicated(keep=False)
    data={'v52_total_rows':len(d),'Zhao2020_rows':len(z),'zhao_duplicate_keys':[],
          'zhao_unmapped_keys':[],'zhao_qe_conflicts':[],'zhao_source_matched':0,
          'zhao_strict5_matched':0,'zhao_review_dose1or20_matched':0,
          'zhao_reference_missing_keys':[],'ready_for_three_scenarios':False}
    if dup.any():data['zhao_duplicate_keys']=[repr(x) for x in z.loc[dup,'__key'].tolist()]
    present=set(z['__key'])
    data['zhao_reference_missing_keys']=[repr(x) for x in set(by_key)-present]
    for _,row in z.iterrows():
        k=row['__key']
        if k not in by_key:
            data['zhao_unmapped_keys'].append(repr(k));continue
        measured=pd.to_numeric(row[c['qe']],errors='coerce')
        if pd.isna(measured) or abs(float(measured)-float(by_key[k]))>0.0001:
            data['zhao_qe_conflicts'].append({'key':repr(k),'v52_qe':str(row[c['qe']]),'reference_qe':float(by_key[k])})
            continue
        data['zhao_source_matched']+=1
        if k[3]==5:data['zhao_strict5_matched']+=1
        else:data['zhao_review_dose1or20_matched']+=1
    data['ready_for_three_scenarios']=(len(z)==24 and data['zhao_source_matched']==24 and
         not data['zhao_duplicate_keys'] and not data['zhao_unmapped_keys'] and not data['zhao_qe_conflicts'])
    return data,z

def check_baseline(base,c,expected_rows):
    if expected_rows and len(base)!=expected_rows:
        raise AuditError(f'Baseline rows {len(base)} do not match declared {expected_rows}. Do not guess v46 version.')
    cc=c['columns']; g=base[cc['doi']].map(doi)
    if g.isna().any() or (g=='nan').any() or (g=='').any():raise AuditError('Missing DOI in baseline')
    if g.nunique()<c['n_splits']:raise AuditError('Not enough independent DOI groups')
    yy=pd.to_numeric(base[cc['qe']],errors='coerce')
    if yy.isna().any() or not np.isfinite(yy).all():raise AuditError('Bad baseline target values')
    if (yy<0).any():raise AuditError('Negative target in baseline; audit before running')
    # Key is sufficient for the Zhao2020 24-row family only; other DOIs may legitimately
    # vary C0/pH/contact time while sharing temperature, metal, and dose.
    if ZHAO in set(g):
        raise AuditError('Zhao2020 DOI already in baseline140. Cannot treat it as new source without explicit reconciliation.')
    return g

def build_model(model_name, estimator, c):
    spec=c['models'][model_name]
    numeric,categorical=spec['numeric'],spec['categorical']
    transformers=[]
    if numeric:
        transformers.append(('num',SimpleImputer(strategy='median',keep_empty_features=True),numeric))
    if categorical:
        transformers.append(('cat',Pipeline([
            ('fill',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical))
    pre=ColumnTransformer(transformers, remainder='drop')
    if estimator=='ExtraTrees': reg=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=53,n_jobs=1)
    elif estimator=='RandomForest':reg=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=53,n_jobs=1)
    elif estimator=='Mean':reg=DummyRegressor(strategy='mean')
    else:raise AuditError(f'Unsupported estimator {estimator}')
    return Pipeline([('preprocess',pre),('regressor',reg)])

def metrics(y,p,c0,dose):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    c0=np.asarray(c0,dtype=float);dose=np.asarray(dose,dtype=float)
    ub=c0/dose
    ok=np.isfinite(ub)&(dose>0)&(c0>0)
    if not np.all(ok):raise AuditError('Invalid C0 / dose for physical mass balance')
    clipped=np.minimum(np.maximum(p,0),ub)
    return {'n':len(y),'mae':float(mean_absolute_error(y,p)),
        'rmse':float(np.sqrt(mean_squared_error(y,p))),
        'r2':float(r2_score(y,p)) if len(y)>1 and not np.allclose(y,y[0]) else None,
        'prediction_physical_violation_rate':float(np.mean((p<0)|(p>ub))),
        'target_physical_violation_rate':float(np.mean((y<0)|(y>ub))),
        'mae_after_physical_clipping_sensitivity_only':float(mean_absolute_error(y,clipped)),
        'rmse_after_physical_clipping_sensitivity_only':float(np.sqrt(mean_squared_error(y,clipped)))}

def bootstrap_group_ci(out,model,scenario,baseline='140',n_boot=2000):
    x=out[(out.model==model)&(out.scenario==scenario)].copy()
    b=out[(out.model==model)&(out.scenario==baseline)].copy()
    w=x.merge(b[['row_id','abs_error']],on='row_id',suffixes=('_new','_base'),validate='one_to_one')
    if len(w)!=len(b) or len(w)!=len(x):raise AuditError('Nonidentical test rows in scenarios')
    group_ids=w.group.unique()
    if len(group_ids)<2:return None
    grouped={g:w.loc[w.group==g,['abs_error_new','abs_error_base']].to_numpy() for g in group_ids}
    rng=np.random.default_rng(20261031)
    deltas=[]
    for _ in range(n_boot):
        gidx=rng.integers(0,len(group_ids),len(group_ids))
        stacked=np.concatenate([grouped[group_ids[i]] for i in gidx],axis=0)
        deltas.append(float(np.mean(stacked[:,0]-stacked[:,1])))
    return {'delta_mae_new_minus_baseline':float(np.mean(w.abs_error_new-w.abs_error_base)),
            'paired_doi_cluster_bootstrap_95ci':[float(q) for q in np.quantile(deltas,[.025,.975])],
            'n_doi_groups':len(group_ids),'bootstrap_draws':n_boot}

def evaluate(args):
    source=source_audit(); c=json.loads(Path(args.config).read_text(encoding='utf-8'))
    base=validate_csv(args.baseline);v52=validate_csv(args.v52)
    assert_config(c,base,'baseline');assert_config(c,v52,'v52')
    groups=check_baseline(base,c,args.expected_baseline_rows)
    audit,z=report_cohort(v52,c)
    initial={'audit':source,'v52_overlap':audit,'files':{'baseline_sha256':sha256(args.baseline),'v52_sha256':sha256(args.v52),'config_sha256':sha256(args.config)},
        'no_deployed_model_changed':True,'research_only':True,'evaluation_test_rows':'baseline only'}
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'01_audit.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2),encoding='utf-8')
    if not audit['ready_for_three_scenarios']:
        raise AuditError('v52 Zhao cohort not fully reconciled. 01_audit.json saved, evaluation BLOCKED.')
    if not args.confirm_research_only:
        raise AuditError('Pass --confirm-research-only only after inspecting audit. No model training has run.')
    cc=c['columns'];base=base.reset_index(drop=True)
    z=z.drop(columns=['__key'],errors='ignore').reset_index(drop=True)
    strict=z[pd.to_numeric(z[cc['dose']],errors='coerce').eq(5)].copy()
    if len(strict)!=8:raise AuditError('Strict-8 cohort missing')
    # Existing source rows are attached only to training partitions; same DOI held out across all splits.
    if set(z[cc['doi']].map(doi))!={ZHAO}:raise AuditError('Unexpected source DOI')
    # 2017 property-source DOI shares material lineage with Zhao2020.
    # Even with different DOI labels, exclude the candidate when such related DOI
    # is on the held-out side. Additional local material links must be reviewed by Kiro.
    related=set(map(doi,c.get('zhao_related_material_dois',['10.3390/en10091293'])))
    group_split=GroupKFold(n_splits=c['n_splits'])
    tuples=list(group_split.split(base,groups=groups))
    pred=[]; foldlog=[]
    for model_name in MODEL_NAMES:
      spec=c['models'][model_name];numeric=spec['numeric'];categorical=spec['categorical'];feats=numeric+categorical
      for estimator in c.get('estimators',['Mean','RandomForest','ExtraTrees']):
       model_key=model_name+'_'+estimator
       for fold,(tr,te) in enumerate(tuples):
        trbase=base.iloc[tr].copy();test=base.iloc[te].copy()
        test_groups=set(test[cc['doi']].map(doi))
        train_groups=set(trbase[cc['doi']].map(doi))
        if test_groups & train_groups:raise AuditError('Base train/test DOI collision')
        for name,cand in [('140',z.iloc[:0]),('140_plus_strict8',strict),('140_plus_full24_sensitivity',z)]:
          source_related_heldout=bool(test_groups & related)
          cand2=cand.iloc[:0].copy() if source_related_heldout else cand[~cand[cc['doi']].map(doi).isin(test_groups)].copy()
          train=pd.concat([trbase,cand2],ignore_index=True)
          if set(train[cc['doi']].map(doi)) & test_groups:
              raise AuditError('Augmentation leaks DOI into test fold')
          # Reject duplicate observation keys across baseline+augmentation. No silent drops.
          keys=[key(r,cc) for _,r in train.iterrows()]
          if len(set(keys))!=len(keys):raise AuditError('Duplicate DOI/temp/metal/dose keys in augmented training; reconcile before evaluation')
          # Model-B conditions involving missing or unverified properties are intentionally not auto-filled.
          for col in numeric:
            train[col]=pd.to_numeric(train[col],errors='coerce')
            test[col]=pd.to_numeric(test[col],errors='coerce')
          for col in categorical:
            train[col]=train[col].astype('string').fillna('__MISSING__').astype(str)
            test[col]=test[col].astype('string').fillna('__MISSING__').astype(str)
          if len(cand2):
            cand_train=train.iloc[len(trbase):]
            for col in numeric:
              if cand_train[col].isna().any():
                raise AuditError(f'Non-numeric or missing candidate {model_key} feature {col}; no implicit cross-source imputation')
            for col in categorical:
              if cand_train[col].str.strip().str.fullmatch(r'(?i)(NR|N/?A|nan|none|unknown|__MISSING__)').any():
                raise AuditError(f'Unresolved categorical candidate feature {col}; no implicit property linkage')
          mdl=build_model(model_name,estimator,c)
          ytr=pd.to_numeric(train[cc['qe']],errors='coerce').to_numpy(dtype=float)
          if not np.isfinite(ytr).all():raise AuditError('Invalid augmentation qe')
          mdl.fit(train[feats],ytr)
          p=mdl.predict(test[feats]);y=pd.to_numeric(test[cc['qe']],errors='coerce').to_numpy(dtype=float)
          for j,(index,actual,forecast) in enumerate(zip(te,y,p)):
             pred.append({'model':model_key,'scenario':name,'fold':fold,'row_id':int(index),
                          'group':str(groups.iloc[index]),'actual':float(actual),'predicted':float(forecast),
                          'abs_error':float(abs(actual-forecast)),
                          'c0':float(test.iloc[j][cc['c0']]),'dose':float(test.iloc[j][cc['dose']])})
          foldlog.append({'model':model_key,'scenario':name,'fold':fold,'n_train_core':len(trbase),
             'n_train_added':len(cand2),'n_test':len(te),'test_dois':sorted(test_groups),
             'candidate_doi_excluded_if_heldout':bool(cand2.empty and not cand.empty),
             'related_2017_material_doi_in_test':source_related_heldout})
    out=pd.DataFrame(pred).sort_values(['model','scenario','row_id']);f=pd.DataFrame(foldlog)
    results={};bootstrap={}
    for (model,scenario),d in out.groupby(['model','scenario']):
      results.setdefault(model,{})[scenario]=metrics(d.actual,d.predicted,d.c0,d.dose)
      if scenario!='140':bootstrap.setdefault(model,{})[scenario]=bootstrap_group_ci(out,model,scenario)
    # Critical guarantee that 3 scenarios refer to exactly the same baseline test rows/DOI folds.
    for m in results:
      signatures=[]
      for s in ['140','140_plus_strict8','140_plus_full24_sensitivity']:
        a=out[(out.model==m)&(out.scenario==s)][['row_id','fold','group']]
        signatures.append(tuple(map(tuple,a.sort_values('row_id').to_numpy())))
      if len(set(signatures))!=1:raise AuditError('Evaluation signatures differ across cohorts')
    out.to_csv(outdir/'02_oof_paired.csv',index=False,encoding='utf-8-sig')
    f.to_csv(outdir/'03_fold_lineage.csv',index=False,encoding='utf-8-sig')
    summary=dict(initial,results=results,paired_group_bootstrap=bootstrap,
        model_C_D='NOT_EVALUATED_NO_BATCH_IDENTITY',reference_dose_1_20='SENSITIVITY_ONLY_NOT_STRICT_QE',
        deployment='NOT_PROMOTED_EXISTING_DEPLOYMENT_UNCHANGED',
        limitation='Evaluation is on common baseline140 DOI-group OOF; 2020 24 rows are a single DOI; test on new papers not demonstrated.')
    (outdir/'04_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(outdir),'models':list(results),'original_baseline_n':len(base),
             'verified_source_rows':len(z),'no_model_deployment':True},ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('audit',help='Validate supplied vNext10 vs 24-row reference; optional v52 overlap')
    a.add_argument('--v52',type=Path);a.add_argument('--config',type=Path);a.add_argument('--output',type=Path,default=ROOT/'reports'/'evidence_audit.json')
    e=sub.add_parser('evaluate',help='Offline paired DOI-GroupKFold comparison; NEVER replaces active model')
    e.add_argument('--baseline',required=True,type=Path)
    e.add_argument('--v52',required=True,type=Path)
    e.add_argument('--config',required=True,type=Path)
    e.add_argument('--output',type=Path,required=True)
    e.add_argument('--expected-baseline-rows',type=int,default=140)
    e.add_argument('--confirm-research-only',action='store_true')
    args=p.parse_args()
    try:
        if args.cmd=='audit':
            data={'source':source_audit()}
            if bool(args.v52) != bool(args.config):raise AuditError('Provide both --v52 and --config')
            if args.v52:
                config=json.loads(args.config.read_text(encoding='utf-8'))
                d=validate_csv(args.v52);assert_config(config,d,'v52')
                detail,_=report_cohort(d,config)
                data['v52_overlap']=detail
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(data,ensure_ascii=False,indent=2))
        else:evaluate(args)
    except (AuditError,KeyError,ValueError,zipfile.BadZipFile) as ex:
        print('V53_BLOCKED:',ex,file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
