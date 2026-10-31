"""v29 research-only source-backed manufacturing feature recovery and LODO stress-test.

Never overwrite v15/v16/v27 deployment or train on v24 research staging.
Agent signatures are extracted from existing method notes WITHOUT target, DOI, or sample labels.
Only source-reported details can enter the modeled feature; provenance remains audit-only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pyrolysis.biochar_stage2.v15_features import model_inputs
from scripts.compare_phase8_v23 import stage2_targets, _stage2_fit_predict
from scripts.evaluate_v27_recommendation import extract_comparable_groups, score_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'v29_process_feature_recovery'

# Explicit source-method reagent strings. Sequence matters: no inferred formulations.
REAGENTS: tuple[tuple[str, str], ...] = (
    ('FECL3', r'FeCl3(?:[.]6H2O)?'),
    ('FESO4', r'FeSO4(?:[.]7H2O)?'),
    ('FE_NO3_3', r'Fe\(NO3\)3'),
    ('KH2PO4', r'KH2PO4'),
    ('NA2HPO4', r'Na2HPO4'),
    ('NA3PO4', r'Na3PO4'),
    ('PHOSPHATE_ROCK', r'phosphate\s*rock'),
    ('MONTMORILLONITE', r'montmorillonite'),
    ('CO2_ACTIVATION', r'CO2\s*activation'),
    ('STEAM_ACTIVATION', r'H2O\s*activation'),
    ('KHCO3', r'KHCO3'),
)
EXPLICIT_CONTROL = re.compile(r'(?:\bunmodified\b|\bnoimpregnation\b|no additional activation|\bnone\b)',re.I)
P_LABEL_UNVERIFIED = re.compile(r'P-only labeledcontrol', re.I)


def extract_reagent_signature(material: pd.Series) -> tuple[str, str]:
    """Return model signature and evidence tier. Do not use sample_name or response."""
    text = str(material.get('source_method_process_text') or '')
    text2 = str(material.get('source_pretreatment_activation_text') or '')
    source = (text + ';' + text2).strip(';')
    source = source.replace('nan','') if source.lower()=='nan' else source
    found = [key for key,pattern in REAGENTS if re.search(pattern,source,re.I)]
    if found:
        return '|'.join(sorted(set(found))), 'SOURCE_REAGENT_EXPLICIT'
    if P_LABEL_UNVERIFIED.search(source):
        return 'P_ONLY_REPORTED_LABEL_NOT_VERIFIED', 'LABEL_ONLY_METHOD_NR'
    if EXPLICIT_CONTROL.search(source):
        return 'NO_REAGENT_REPORTED', 'SOURCE_CONTROL_EXPLICIT'
    # Use source-listed activation agent only if explicit, no unknown molecular inference.
    agent = str(material.get('reported_activation_agent') or '').strip()
    if agent and agent.lower() not in ('nan','nr','none','none_reported'):
        return f'REPORTED_AGENT:{agent.upper()}', 'REPORTED_AGENT_FIELD'
    return 'REAGENT_NOT_REPORTED', 'UNREPORTED'


def material_feature_table(core: pd.DataFrame, root:Path=ROOT) -> pd.DataFrame:
    paths=list((Path(root)/'data/imports/Stage1_Stage2_Model_Ready_v15').rglob('01_Manufacturing_And_Feedstock_Materials_441.csv'))
    if len(paths)!=1:
        raise RuntimeError('Expected exactly one immutable v15 material source')
    materials=pd.read_csv(paths[0],low_memory=False).fillna('')
    if materials.material_uid.duplicated().any():
        raise ValueError('Duplicate material UID in master')
    x=materials.loc[materials.material_uid.isin(core.material_uid)].copy()
    missing=set(core.material_uid)-set(x.material_uid)
    # Phase8 newly added material is not in the immutable old 441-material ledger.
    # Do not invent source-method details: retain as NR (not a recovered reagent).
    if missing:
        extra=pd.DataFrame([{'material_uid':r.material_uid,'doi':r.doi,
               'source_method_process_text':'','source_pretreatment_activation_text':'',
               'sample_name':'','reported_activation_agent':'',
               'reported_impregnation_agent':''} for r in
               core.loc[core.material_uid.isin(missing),['material_uid','doi']].drop_duplicates().itertuples(index=False)])
        x=pd.concat([x,extra],ignore_index=True)
    signatures=x.apply(extract_reagent_signature, axis=1)
    x['v29_reagent_signature']=[v[0] for v in signatures]
    x['v29_source_tier']=[v[1] for v in signatures]
    x['v29_source_method_note']=x.source_method_process_text  # audit only
    x['v29_sample_name']=x.sample_name  # audit only; NEVER a model input
    result=core[['observation_id','material_uid','doi']].merge(
        x[['material_uid','doi','v29_reagent_signature','v29_source_tier','v29_source_method_note','v29_sample_name']],
        on=['material_uid','doi'], validate='many_to_one',how='left'
    )
    if result.v29_reagent_signature.isna().any() or len(result)!=len(core):
        raise ValueError('Unmatched manufacturing evidence')
    if not result.observation_id.is_unique:
        raise ValueError('Repeated observation id')
    return result


def _fixed_ridge_predict(train:pd.DataFrame, test:pd.DataFrame, *, enhanced:bool) -> np.ndarray:
    feats=model_inputs('B') + (['v29_reagent_signature'] if enhanced else [])
    categorical={'pollutant','feedstock_label_normalized','manufacturing_route',
                 'pyrolysis_atmosphere','activation_or_modification','v29_reagent_signature'}
    nums=[c for c in feats if c not in categorical]
    cats=[c for c in feats if c in categorical]
    a=train[feats].copy();b=test[feats].copy()
    for f in nums:
        a[f]=pd.to_numeric(a[f],errors='coerce');b[f]=pd.to_numeric(b[f],errors='coerce')
    for f in cats:
        a[f]=a[f].where(a[f].notna() & a[f].ne(''),np.nan).astype('object')
        b[f]=b[f].where(b[f].notna() & b[f].ne(''),np.nan).astype('object')
    num_pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    cat_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                       ('onehot',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',num_pipe,nums),('cat',cat_pipe,cats)])
    pipe=Pipeline([('pre',pre),('model',Ridge(alpha=10.,random_state=42))])
    pipe.fit(a,pd.to_numeric(train.target_value,errors='raise'))
    return pipe.predict(b).astype(float)


def compare_identifiability(base:pd.DataFrame, enriched:pd.DataFrame):
    contrast=extract_comparable_groups(base)
    contrast=contrast.merge(enriched[['observation_id','v29_reagent_signature','v29_source_tier']],
                             on='observation_id',validate='one_to_one')
    rows=[]
    for cg,sub in contrast.groupby('comparison_id',sort=True):
        y=sub.target_value.to_numpy(dtype=float)
        model_cols=model_inputs('B')
        # compare *all* recorded B inputs; only manufacturing components may differ within strict group
        before=sub[model_cols].fillna('<NR>').astype(str).agg('|'.join,axis=1).tolist()
        after=(sub[model_cols+['v29_reagent_signature']].fillna('<NR>').astype(str)
               .agg('|'.join,axis=1).tolist())
        for i in range(len(sub)):
            for j in range(i+1,len(sub)):
                if np.isclose(y[i],y[j],atol=1e-9,rtol=0):continue
                rows.append({'comparison_id':cg,'doi':sub.doi.iloc[0],
                             'observation_1':sub.observation_id.iloc[i],
                             'observation_2':sub.observation_id.iloc[j],
                             'old_inputs_identical':before[i]==before[j],
                             'new_inputs_identical':after[i]==after[j],
                             'source_tier_1':sub.v29_source_tier.iloc[i],
                             'source_tier_2':sub.v29_source_tier.iloc[j],
                             'source_agent_1':sub.v29_reagent_signature.iloc[i],
                             'source_agent_2':sub.v29_reagent_signature.iloc[j],
                             'observed_target_gap_mg_g':abs(y[i]-y[j])})
    return pd.DataFrame(rows)


def lodo_compare(core:pd.DataFrame, ev:pd.DataFrame):
    base=core.merge(ev[['observation_id','v29_reagent_signature']],on='observation_id',validate='one_to_one')
    output=[]
    for doi in sorted(base.doi.unique()):
        tr=base.loc[base.doi.ne(doi)].copy();te=base.loc[base.doi.eq(doi)].copy()
        assert set(tr.doi).isdisjoint(set(te.doi))
        pred_b=_stage2_fit_predict(tr,te,'B','Ridge')
        pred_e=_fixed_ridge_predict(tr,te,enhanced=True)
        for r,p,q in zip(te.itertuples(index=False),pred_b,pred_e):
            cap=float(r.C0_mg_L)/float(r.dose_g_L)
            output.append({'observation_id':r.observation_id,'doi':doi,'material_uid':r.material_uid,
                'y':float(r.target_value),'cap':cap,'pred_baseline':float(p),'pred_enriched':float(q),
                'agent_signature':r.v29_reagent_signature})
    return pd.DataFrame(output)


def summarize_predictions(core:pd.DataFrame, predictions:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    pred=predictions.copy()
    for name,col in [('baseline','pred_baseline'),('enriched','pred_enriched')]:
        pred[f'err_{name}']=abs(pred.y-pred[col]);pred[f'violation_{name}']=(pred[col]<0)|(pred[col]>pred.cap)
    g=extract_comparable_groups(core)
    g=g.merge(pred[['observation_id','pred_baseline','pred_enriched']],on='observation_id',validate='one_to_one')
    ranked=[]
    for cid,sub in g.groupby('comparison_id',sort=True):
        for n,col in [('baseline','pred_baseline'),('enriched','pred_enriched')]:
            rank=score_group(sub.rename(columns={col:'pred_b_raw'}).drop(columns=[x for x in ['pred_baseline','pred_enriched'] if x!=col]),min_difference=0.)
            ranked.append({'comparison_id':cid,'doi':sub.doi.iloc[0],'model':n,
               'n_pairs':rank['n_pairs'],'pairwise':rank['tie_aware_pairwise'],
               'raw_violations':rank['n_raw_violations'],'prediction_ties':rank['predicted_tie_pairs'],
               'top1_expected':rank['heldout_top1_expected']})
    ranks=pd.DataFrame(ranked)
    eligible=ranks[ranks.n_pairs>0]
    summary={'records':len(pred),'DOIs':pred.doi.nunique(),
      'baseline':{'mae':float(pred.err_baseline.mean()),'r2':float(r2_score(pred.y,pred.pred_baseline)),
                  'raw_violations':int(pred.violation_baseline.sum())},
      'enriched':{'mae':float(pred.err_enriched.mean()),'r2':float(r2_score(pred.y,pred.pred_enriched)),
                  'raw_violations':int(pred.violation_enriched.sum())},
      'ranking':{n:{'groups':int(len(a)),'macro_pairwise':float(a.pairwise.mean()),
                    'pairs':int(a.n_pairs.sum()),'predicted_ties':int(a.prediction_ties.sum()),
                    'raw_violations_group_rows':int(a.raw_violations.sum())}
          for n,a in eligible.groupby('model')},
      'exploratory_only':True,
      'limitations':['Agent text extraction rules inspected after v28 collisions; evaluation is post-hoc and not independent model selection.',
        'No model or historical core rewrite; no external validation of novel agent combinations.',
        'A source note containing a named chemical does not establish complete quantitative reagent dose.',
        'P-only control label lacks separate synthesis evidence; kept as uncertainty-marked category.',
        'Raw qe is reported uptake, not universally certified equilibrium.',
        'LODO does not establish pyrolysis temperature optimization.']}
    return ranks,summary


def run(root=ROOT,out=OUT):
    root=Path(root);out=Path(out)
    _,core=stage2_targets()
    if len(core)!=98 or core.doi.nunique()!=26:raise ValueError('Incorrect immutable Core 98')
    ev=material_feature_table(core,root)
    collision=compare_identifiability(core,ev)
    pred=lodo_compare(core,ev)
    ranks,summary=summarize_predictions(core,pred)
    summary['pairwise_feature_identifiability']={
       'all_unequal_target_pairs':int(len(collision)),
       'old_collisions':int(collision.old_inputs_identical.sum()),
       'new_collisions':int(collision.new_inputs_identical.sum()),
       'resolved_collisions':int((collision.old_inputs_identical & ~collision.new_inputs_identical).sum()),
       'remaining_pairs':collision.loc[collision.new_inputs_identical,'comparison_id'].tolist()}
    summary['source_evidence_tiers']={k:int(v) for k,v in ev[['material_uid','v29_source_tier']].drop_duplicates().v29_source_tier.value_counts().items()}
    out.mkdir(parents=True,exist_ok=True)
    ev.to_csv(out/'source_agent_feature_lookup.csv',index=False)
    collision.to_csv(out/'pairwise_feature_resolution.csv',index=False)
    pred.to_csv(out/'lodo_oof_baseline_vs_enriched.csv',index=False)
    ranks.to_csv(out/'lodo_group_ranking_comparison.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return summary

if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
