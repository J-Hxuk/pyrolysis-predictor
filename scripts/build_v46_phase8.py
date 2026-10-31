"""v46 Phase8 source-verified reconciliation and fold-safe research refit.

Research data only. Original reported measurements remain untouched, qa flags not model features.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

DIR='data/imports/v46_phase8'
REPORT='reports/v46_phase8'
DEPLOY='deployment/research_v46_NOT_ACTIVE'
S1_NUM=['pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min',
 'pyrolysis_N2_flow_mL_min','activation_temperature_C','activation_time_min',
 'activation_agent_feedstock_mass_ratio','component_1_fraction','component_2_fraction']
S1_CAT=['feedstock','feedstock_group','pyrolysis_atmosphere','activation_method','activation_agent','process_stage']
S2_A=['pollutant','C0_mg_L','initial_pH','dose_g_L','adsorption_T_C','contact_time_min']
S2_B=S2_A+['feedstock_label_normalized','manufacturing_route','process_step_count',
 'pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min',
 'pyrolysis_atmosphere','activation_or_modification','secondary_temperature_C','secondary_hold_min']
S2_CATS=['pollutant','feedstock_label_normalized','manufacturing_route',
          'pyrolysis_atmosphere','activation_or_modification']


def load_inputs(root:Path):
    base=root/DIR
    d2=pd.read_csv(base/'v45_Stage2_138_BASELINE.csv',low_memory=False)
    d1=pd.read_csv(base/'v45_Stage1_404_BASELINE.csv',low_memory=False)
    v2=pd.read_csv(base/'Phase8_source_verified_equilibrium_45.csv',low_memory=False)
    s1=pd.read_csv(base/'Stage1_new_13_source_staged.csv',low_memory=False)
    return d1,d2,v2,s1


def reconcile(d1,d2,verified,s1):
    if len(d1)!=404 or len(d2)!=138 or len(verified)!=45 or len(s1)!=13:
        raise ValueError('Expected v45 404/138 and source upgrade 45/13. Version mismatch')
    if not d2.observation_id.is_unique or not d1.material_id.is_unique:
        raise ValueError('Original ID duplicates; refusing merger')
    if not verified.adsorption_id.is_unique or not s1.Stage1_source_record_id.is_unique:
        raise ValueError('Source upgrade IDs duplicated')
    original_ids=set(d2.observation_id)
    seen=verified[verified.adsorption_id.isin(original_ids)].copy()
    new=verified[~verified.adsorption_id.isin(original_ids)].copy()
    if len(seen)!=43 or len(new)!=2: raise ValueError('Unexpected source verification overlap')
    ref=d2.set_index('observation_id')
    compare_fields={'doi':'doi','pollutant':'pollutant','qe_reported_mg_g':'target_value',
                    'C0_mg_L':'C0_mg_L','dose_g_L':'dose_g_L'}
    for r in seen.itertuples(index=False):
        orig=ref.loc[r.adsorption_id]
        for source_f,target_f in compare_fields.items():
            x=getattr(r,source_f); y=orig[target_f]
            if source_f in ['doi','pollutant']:
                if str(x).lower().strip()!=str(y).lower().strip():
                    raise ValueError(f'Verified source conflicts with v45 {r.adsorption_id} {source_f}: {x} vs {y}')
            elif not np.isclose(float(x),float(y),atol=1e-7):
                raise ValueError(f'Verified source changes original {r.adsorption_id} {source_f}')
    expected_new={'UPG2_OAK_CU_PH3','UPG_NI_MSBC_KINETIC_18H'}
    if set(new.adsorption_id)!=expected_new: raise ValueError('New record identity changed; review required')
    d2=d2.copy();d2['phase8_source_verified_v2']=d2.observation_id.isin(set(verified.adsorption_id))
    d2['phase8_source_verification_class']=np.where(d2.phase8_source_verified_v2,'SOURCE_VERIFIED_CORE_V2','NOT_IN_V2_VERIFIED_COHORT')
    d2['phase8_source_qa_warnings']=''
    warnings_by_id=verified.set_index('adsorption_id')['warnings'].fillna('').to_dict()
    d2.loc[d2.phase8_source_verified_v2,'phase8_source_qa_warnings']=d2.loc[d2.phase8_source_verified_v2,'observation_id'].map(warnings_by_id)
    rows=[]
    for r in new.itertuples(index=False):
        doi=str(r.doi)
        source_same=d2[d2.doi==doi]
        if source_same.empty: raise ValueError(f'Expected existing DOI for added record: {doi}')
        key=source_same[source_same.material_uid.astype(str)==str(r.material_key)]
        if key.empty: raise ValueError(f'Expected linked original material for {r.adsorption_id}')
        template=key.iloc[0].copy().to_dict()
        template.update(observation_id=r.adsorption_id,doi=doi,doi_group=doi,
          material_uid=r.material_key,sample_group=r.material_key,
          pollutant=r.pollutant,C0_mg_L=float(r.C0_mg_L),
          initial_pH=pd.to_numeric(r.initial_pH,errors='coerce'),
          dose_g_L=float(r.dose_g_L),adsorption_T_C=pd.to_numeric(r.adsorption_T_C,errors='coerce'),
          contact_time_min=pd.to_numeric(r.contact_time_min,errors='coerce'),
          target_value=float(r.qe_reported_mg_g),target_name='qe_reported_mg_g',target_unit='mg/g',
          mass_balance_cap_mg_g=float(r.C0_mg_L)/float(r.dose_g_L),
          source_url=r.source_url,source_target_status=r.source_target_status,
          target_origin=str(r.raw_data_label),target_qualification=str(r.target_approval_scope),
          warnings=str(r.warnings),model_use_tier='V46_SOURCE_VERIFIED_RESEARCH_NOT_ACTIVE',
          phase8_source_verified_v2=True,phase8_source_verification_class='SOURCE_VERIFIED_CORE_V2',
          phase8_source_qa_warnings=str(r.warnings),
          experiment_series_id=f'{doi}::{r.source_experiment_group}',
          BET_measured_m2_g=pd.to_numeric(r.BET_measured_m2_g,errors='coerce'),
          TPV_measured_cm3_g=pd.to_numeric(r.TPV_measured_cm3_g,errors='coerce'))
        rows.append(template)
    appended=pd.DataFrame(rows)
    d2full=pd.concat([d2,appended],ignore_index=True,sort=False)
    if len(d2full)!=140 or not d2full.observation_id.is_unique or d2full.doi.nunique()!=32:
        raise ValueError('Incorrect 140/32 DOI reconciliation')
    y=pd.to_numeric(d2full.target_value,errors='coerce'); c0=pd.to_numeric(d2full.C0_mg_L,errors='coerce')
    dose=pd.to_numeric(d2full.dose_g_L,errors='coerce')
    if not ((y>=0)&(dose>0)&(c0>0)&(y<=c0/dose+1e-6)).all():
        raise ValueError('Mass balance fails in merged source rows')
    # Stage1: direct Table 1 material properties. Source explicitly says NOT auto-approved for deployed model.
    s1out=[]
    for r in s1.itertuples(index=False):
        activation=pd.to_numeric(r.activation_temperature_C,errors='coerce')
        hasactivation=pd.notna(activation)
        agent=('K2CO3+FeCl3' if str(r.sample_name).startswith('FBPC') else 'K2CO3') if hasactivation else 'none'
        route='carbonization_plus_chemical_activation' if hasactivation else 'single_stage_carbonization'
        s1out.append(dict(material_id=r.Stage1_source_record_id,doi=r.doi,
            original_sample_name=r.sample_name,feedstock=r.feedstock,
            feedstock_group='bamboo',pyrolysis_temperature_C=pd.to_numeric(r.pyrolysis_temperature_C,errors='coerce'),
            pyrolysis_hold_min=pd.to_numeric(r.pyrolysis_hold_min,errors='coerce'),
            heating_rate_C_min=pd.to_numeric(r.pyrolysis_heating_rate_C_min,errors='coerce'),
            pyrolysis_atmosphere=r.pyrolysis_atmosphere,activation_temperature_C=activation,
            activation_time_min=pd.to_numeric(r.activation_hold_min,errors='coerce'),
            activation_method=route,activation_agent=agent,process_stage=route,
            # IMPORTANT: K2CO3/char != activation agent/FEEDSTOCK; do not silently recode.
            activation_agent_feedstock_mass_ratio=np.nan,
            source_K2CO3_to_char_mass_ratio=pd.to_numeric(r.K2CO3_to_char_mass_ratio,errors='coerce'),
            source_FeCl3_to_char_mass_percent=pd.to_numeric(r.FeCl3_to_char_mass_percent,errors='coerce'),
            BET_m2_g_training_candidate=float(r.BET_measured_m2_g),
            TPV_cm3_g_training_candidate=float(r.TPV_measured_cm3_g),
            BET_m2_g_original_source=float(r.BET_measured_m2_g),
            TPV_cm3_g_original_source=float(r.TPV_measured_cm3_g),
            source_stage='V46_PHASE8_SOURCE_SPECIFIC_RESEARCH_ONLY',
            release_status='RESEARCH_SENSITIVITY_ONLY_NOT_AUTO_APPROVED',
            source_location=r.source_location,source_url=r.source_url,
            raw_feedstock_basis_note='Source is post-pyrolysis manufactured biochar; no raw composition imputed',
            source_ratio_basis_note='K2CO3_to_CHAR ratio not mapped to activation_agent_to_FEEDSTOCK ratio'))
    addon=pd.DataFrame(s1out)
    assert not set(addon.doi)&set(d1.doi), 'New Stage1 DOI already in old master; duplicate sample review needed'
    d1full=pd.concat([d1,addon],ignore_index=True,sort=False)
    assert d1full.material_id.is_unique and len(d1full)==417
    return d1full,d2full,appended,addon


def s2_model(variant):
    from xgboost import XGBRegressor
    cols=S2_A if variant=='A' else S2_B
    cats=[c for c in cols if c in S2_CATS]; nums=[c for c in cols if c not in cats]
    num=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',num,nums),('cat',cat,cats)])
    reg=XGBRegressor(n_estimators=220,max_depth=2,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                     reg_lambda=5,min_child_weight=5,random_state=42,n_jobs=1,tree_method='hist')
    return Pipeline([('pre',pre),('model',reg)]),cols,nums,cats


def s1_model():
    num=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))])
    cat=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                  ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    pre=ColumnTransformer([('num',num,S1_NUM),('cat',cat,S1_CAT)],sparse_threshold=0)
    reg=HistGradientBoostingRegressor(loss='poisson',max_iter=140,max_leaf_nodes=9,
       min_samples_leaf=10,learning_rate=.065,l2_regularization=5,random_state=42)
    return Pipeline([('pre',pre),('reg',reg)]),S1_NUM+S1_CAT,S1_NUM,S1_CAT


def features(df,cols,nums,cats):
    x=df.reindex(columns=cols).copy()
    for c in nums:x[c]=pd.to_numeric(x[c],errors='coerce')
    for c in cats:x[c]=x[c].astype(object).where(x[c].notna(),np.nan)
    return x


def fitpred(train,test,stage,variant,target):
    m,cols,nums,cats=(s1_model() if stage=='Stage1' else s2_model(variant))
    m.fit(features(train,cols,nums,cats),train[target].to_numpy(float))
    return m.predict(features(test,cols,nums,cats))


def evaluate(old,combined,stage,variant,target):
    old=old[old[target].notna()].reset_index(drop=True)
    combined=combined[combined[target].notna()].reset_index(drop=True)
    oldids=set(old.iloc[:,0].astype(str))
    # Stable explicit IDs (first column generally material_id or observation_id)
    idcol='material_id' if stage=='Stage1' else 'observation_id'
    extra=combined[~combined[idcol].isin(old[idcol])].reset_index(drop=True)
    if not set(extra.doi).issubset(set(combined.doi)): raise ValueError('unexpected DOI')
    if combined[idcol].duplicated().any():raise ValueError('Duplicate record ID')
    base=np.full(len(old),np.nan);add=np.full(len(old),np.nan)
    base_doi=set(old.doi)
    for tr,te in GroupKFold(5).split(old,groups=old.doi):
        train=old.iloc[tr];test=old.iloc[te]
        held=set(test.doi)
        # DOI-safe: never allow any additional row from test DOI into training.
        eligible=extra[~extra.doi.isin(held)]
        assert set(eligible.doi).isdisjoint(held)
        base[te]=fitpred(train,test,stage,variant,target)
        add[te]=fitpred(pd.concat([train,eligible],ignore_index=True),test,stage,variant,target)
    y=old[target].to_numpy(float)
    allp=np.full(len(combined),np.nan)
    for tr,te in GroupKFold(5).split(combined,groups=combined.doi):
        train=combined.iloc[tr];test=combined.iloc[te]
        assert not set(train.doi)&set(test.doi)
        allp[te]=fitpred(train,test,stage,variant,target)
    sc=[]
    for label,values,data in [('old_fixed_old_train',base,old),('old_fixed_source_augmented',add,old),('pooled_all_doi_group5',allp,combined)]:
        truth=data[target].to_numpy(float)
        rec=dict(stage=stage,model=variant,target=target,evaluation=label,n=len(data),doi_count=data.doi.nunique(),
                 mae=float(mean_absolute_error(truth,values)),r2=float(r2_score(truth,values)),negative_pred=int((values<0).sum()))
        if stage=='Stage2':
            cap=data.C0_mg_L.to_numpy(float)/data.dose_g_L.to_numpy(float)
            rec['mass_balance_violations']=int(((values<0)|(values>cap)).sum())
        sc.append(rec)
    preds=pd.DataFrame({'record_id':old[idcol],'doi':old.doi,'stage':stage,'model':variant,'truth':y,
                        'old_pred':base,'augmented_pred':add})
    return sc,preds


def run(root:Path):
    root=Path(root); (root/REPORT).mkdir(parents=True,exist_ok=True);(root/DEPLOY).mkdir(parents=True,exist_ok=True)
    d1,d2,v2,s1=load_inputs(root)
    merged1,merged2,extra2,extra1=reconcile(d1,d2,v2,s1)
    out=root/DIR
    merged1.to_csv(out/'Stage1_417_EXPLORATORY_UNAPPROVED.csv',index=False)
    merged2.to_csv(out/'Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv',index=False)
    extra2.to_csv(out/'Stage2_two_additional_v45_missing_rows.csv',index=False)
    extra1.to_csv(out/'Stage1_13_compatible_features_INCOMPLETE.csv',index=False)
    scores=[];preds=[]
    for stage,old,aug,target,variants in [('Stage1',d1,merged1,'BET_m2_g_training_candidate',['Poisson']),
                                         ('Stage1',d1,merged1,'TPV_cm3_g_training_candidate',['Poisson']),
                                         ('Stage2',d2,merged2,'target_value',['A','B'])]:
        for variant in variants:
            score,p=evaluate(old,aug,stage,variant,target)
            scores.extend(score);preds.append(p)
            full=aug[aug[target].notna()]
            model,cols,nums,cats=(s1_model() if stage=='Stage1' else s2_model(variant))
            model.fit(features(full,cols,nums,cats),full[target].to_numpy(float))
            joblib.dump(model,root/DEPLOY/f'V46_{stage}_{target}_{variant}_RESEARCH_NOT_ACTIVE.joblib',compress=3)
    sf=pd.DataFrame(scores)
    sf.to_csv(root/REPORT/'paired_evaluation.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(root/REPORT/'old_DOI_paired_predictions.csv',index=False)
    overview=dict(v45_stage1=len(d1),v46_stage1=len(merged1),stage1_BET=int(merged1.BET_m2_g_training_candidate.notna().sum()),
                  stage1_TPV=int(merged1.TPV_cm3_g_training_candidate.notna().sum()),
                  v45_stage2=len(d2),v46_stage2=len(merged2),stage2_doi=int(merged2.doi.nunique()),
                  phase8_verified_45=len(v2),verified_already_present=43,addition_old_recovered=1,
                  addition_new_measured=1,source_stage1_auto_train_approved=False,active_model_changed=False,
                  comparisons=scores)
    (root/REPORT/'summary.json').write_text(json.dumps(overview,ensure_ascii=False,indent=2),encoding='utf-8')
    return overview

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args()
    out=run(args.root)
    print(pd.DataFrame(out['comparisons']).to_string(index=False))
    print({k:v for k,v in out.items() if k!='comparisons'})
