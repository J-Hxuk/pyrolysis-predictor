"""Read-only audit for pyrolysis-temperature contrast availability.

Fitted qmax NEVER treated as observed qe. No ML training or ranking validation.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v29_process_feature_recovery'

def series_from_named_recipe(name:str) -> str:
    # Only separate processing series explicitly encoded in 2025 rice-husk sample names.
    n=str(name).upper().strip()
    if n.startswith('HNO3-'):return 'HNO3_POST'
    if n.startswith('KOH-'):return 'KOH_POST'
    if n.startswith('BC'):return 'UNMODIFIED'
    return 'SAME_REPORTED_ROUTE_UNVERIFIED'

def audit_temperature(root:Path=ROOT):
    base=Path(root)/'data/imports/v24_source_audit'
    s=pd.read_csv(base/'Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv',low_memory=False)
    q=pd.read_csv(base/'all_qmax_32_FITTED_NOT_QE.csv')
    s=s.loc[s.doi.isin(q.doi)].copy()
    s=s[['doi','original_sample_name','material_id','pyrolysis_temperature_C',
         'BET_m2_g_training_candidate','TPV_cm3_g_training_candidate']].copy()
    s['sample_key']=s.original_sample_name.astype(str).str.strip().str.upper()
    q=q.copy();q['sample_key']=q.sample_name.astype(str).str.strip().str.upper()
    if s.duplicated(['doi','sample_key']).any() or q.duplicated(['doi','pollutant','sample_key']).any():
        raise ValueError('Source sample key not unique; manually reconcile before joining')
    j=q.merge(s,on=['doi','sample_key'],how='left',validate='many_to_one')
    if j.material_id.isna().any():
        missing=j[j.material_id.isna()][['doi','sample_name']].to_dict('records')
        raise ValueError(f'Unlinked qmax sample(s): {missing}')
    j['recipe_series']=j.sample_name.apply(series_from_named_recipe)
    groups=[]
    for key,g in j.groupby(['doi','pollutant','recipe_series'],dropna=False):
        temps=pd.to_numeric(g.pyrolysis_temperature_C,errors='coerce')
        verified=len(g)>=3 and temps.notna().all() and temps.nunique()>=3
        groups.append({'doi':key[0],'pollutant':key[1],'recipe_series':key[2],
                       'n_samples':len(g),'reported_distinct_temperatures':int(temps.dropna().nunique()),
                       'temperature_levels_C':';'.join(str(int(t)) for t in sorted(temps.dropna().unique())),
                       'numeric_temperature_contrast':bool(verified),
                       'target_type':'LANGMUIR_QMAX_FITTED_NOT_REPORTED_QE',
                       'qe_ranking_ready':False,
                       'limitations':'Same-paper temperature series with fitted qmax only; needs directly reported qe at consistent adsorption settings' })
    groups=pd.DataFrame(groups)
    summary={'source_rows_qmax':len(j),'DOIs':int(j.doi.nunique()),
             'temp_contrast_qmax_series':int(groups.numeric_temperature_contrast.sum()),
             'temp_contrast_qmax_dois':int(groups.loc[groups.numeric_temperature_contrast,'doi'].nunique()),
             'temp_contrast_qmax_sample_rows':int(groups.loc[groups.numeric_temperature_contrast,'n_samples'].sum()),
             'qe_temperature_ranking_validated':False,
             'next_data_required':'Source-reported qe or Ce, matched C0/dose/pH/contact_time, stable contaminant and manufacturing route, >=3 temperature settings/DOI',
             'interpretation':'This is discovery of candidate cohorts for new source-based qe extraction, never a measured-qe benchmark.'}
    return j,groups,summary

def audit_external_poultry_temperature(root:Path=ROOT):
    """External *reported uptake* temperature strata (not approved equilibrium)."""
    p=Path(root)/'data/imports/v24_source_audit/stage2_new69_PHYSICAL_PASS_STILL_NOT_TRAINED.csv'
    df=pd.read_csv(p)
    df=df.loc[df.doi.eq('10.3390/ma14216566')].copy()
    fields=['doi','pollutant','C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']
    groups=[]
    for key,g in df.groupby(fields,dropna=False,sort=True):
        groups.append({**dict(zip(fields,key)), 'n_rows':len(g),
          'n_pyrolysis_temperatures':int(g.pyrolysis_temp_C.nunique()),
          'temp_values_C':';'.join(str(t) for t in sorted(g.pyrolysis_temp_C.unique())),
          'reported_qe_range_mg_g':float(g.qe_mg_g.max()-g.qe_mg_g.min()),
          'physical_cap_all_pass':bool(g.physical_cap_pass.all()),
          'origin':'TABLE_REPORTED_UPTAKE_NOT_EQUILIBRIUM_CERTIFIED',
          'qualification':'SINGLE_DOI_EXTERNAL_STRESS_ONLY_PRECIPITATION_RISK_HIGH_DOSAGE',
          'valid_new_DOI_generalization_benchmark':False})
    result=pd.DataFrame(groups)
    summary={'groups':len(result),'records':len(df),'independent_dois':int(df.doi.nunique()),
      'groups_with_3_temperatures':int((result.n_pyrolysis_temperatures==3).sum()),
      'max_within_group_qe_gap_mg_g':float(result.reported_qe_range_mg_g.max()),
      'all_100_g_L':bool(df.dose_g_L.eq(100).all()),
      'approved_for_temperature_model_training':False,
      'reason':'Only one new DOI, 100 g/L OOD, precipitation and tiny qe contrasts; do not treat technical repeated groups as independent DOIs.'}
    return result,summary

def run(root=ROOT,out=OUT):
    j,g,s=audit_temperature(root)
    pg,ps=audit_external_poultry_temperature(root)
    s["external_poultry_reported_uptake"]=ps
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    j.to_csv(out/'fitted_qmax_sample_links_NOT_QE.csv',index=False)
    g.to_csv(out/'temperature_contrast_candidates_FITTED_ONLY.csv',index=False)
    pg.to_csv(out/'external_poultry_17_temperature_strata_NOT_APPROVED.csv',index=False)
    (out/'temperature_readiness_summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    return s
if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
