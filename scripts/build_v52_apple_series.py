"""v52 -- source-verified apple-branch biochar temperature series intake and refit.

DOI 10.3390/en13133498. Source: Zhao, Ta, Wang, Energies 2020, 13, 3498.
Table 1: BC300..BC600 BET. Table 2 and Section 2.2: adsorption conditions.
Table 3: qe_exp, NOT pseudo-first/pseudo-second-order fitted qe or Langmuir qmax.
Research ONLY. No active deployment changes. Repeatable local model saving supported.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scripts.build_v46_phase8 import S2_A,S2_B,S2_CATS,features,s2_model,s1_model,S1_NUM,S1_CAT

DOI='10.3390/en13133498'
URL='https://www.mdpi.com/1996-1073/13/13/3498'
S2_SOURCE='data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'
S1_SOURCE='data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
DATA_DIR='data/imports/v52_apple_series'
REPORT='reports/v52_apple_series'
DEPLOY='deployment/research_v52_NOT_ACTIVE'


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_rows():
    """24 verified uptake values: 8 Table 3 qe_exp + 16 Section 3.1 reported 1440-min qe.
    All come from one DOI, four manufactured materials, two metals and three doses.
    """
    rows=[]
    for metal,c0,values,std in [
        ('Cu(II)',64.0,[4.96,7.56,9.70,8.87],[.18,.27,.33,.42]),
        ('Zn(II)',65.0,[4.30,5.69,9.11,7.99],[.10,.13,.28,.22]),
    ]:
        for temp,qe,sd,bet in zip([300,400,500,600],values,std,[2.39,7.00,37.25,108.59]):
            rows.append({
                'observation_id':f'V52_APPLE_{metal[:2].upper()}_{temp}_QEEXP',
                'doi':DOI,'doi_group':DOI,
                'feedstock_raw':'apple tree branches',
                'feedstock_label_normalized':'apple tree branches',
                'feedstock_group_for_novelty_check':'wood branches',
                'sample_group':f'V52_APPLE_BC{temp}',
                'material_uid':f'V52_APPLE_BC{temp}',
                'experiment_series_id':f'{DOI}::kinetics_qe_exp::{metal}::pH5_25C',
                'pollutant':metal,'C0_mg_L':c0,'initial_pH':5.0,'dose_g_L':5.0,
                'adsorption_T_C':25.0,'contact_time_min':1440.0,
                'manufacturing_route':'SINGLE_STAGE_PYROLYSIS',
                'process_step_count':1,
                'pyrolysis_temperature_C':float(temp),
                'pyrolysis_hold_min':np.nan,'heating_rate_C_min':np.nan,
                'pyrolysis_atmosphere':np.nan,'activation_or_modification':'none',
                'BET_measured_m2_g':bet,'TPV_measured_cm3_g':np.nan,
                'target_name':'qe_reported_mg_g','target_value':qe,'target_unit':'mg/g',
                'target_origin':'V52_ZHAO2020_TABLE3_QE_EXP',
                'target_qualification':'TABLE3_EXPERIMENTAL_EQUILIBRIUM_QE_EXP',
                'model_use_tier':'V52_SOURCE_VERIFIED_RESEARCH_NOT_ACTIVE',
                'source_target_status':'TABLE3_QE_EXP_NOT_FITTED',
                'physical_cap_pass':'YES','mass_balance_cap_mg_g':c0/5,
                'source_url':URL,'source_pdf':'Energies_2020_13_3498',
                'source_page':'Table 3 p10; Methods Section 2.2 pp4-5',
                'source_numeric_sd_mg_g':sd,
                'stage1_v60_eligible':'NOT_APPROVED',
                'raw_composition_basis':'NOT_REPORTED_RAW_FEEDSTOCK',
                'warnings':('Equilibrium about 1440 min per paper conclusion. '
                    'Table 3 qe_exp not kinetic fitted qe. Pyrolysis hold/heating/atmosphere not established in this source; '
                    'post-pyrolysis Table1 C/H/O MUST NOT fill raw feedstock composition. '
                    'Removal may include precipitation/ion exchange; not necessarily surface adsorption alone.'),
                'source_table_qe_type':'qe_exp',
            })
    # Direct text, Section 3.1: dosage 1 and 20 g/L, Experiment 1.
    # Experiment 1 Table 2: pH5, 25C, 1440min; same C0 as Table3, same manufactured material.
    dose_values={
        'Cu(II)':{1.0:[10.13,17.75,27.20,23.35],20.0:[2.90,2.97,3.14,3.08]},
        'Zn(II)':{1.0:[7.10,13.20,23.84,20.90],20.0:[2.80,2.84,3.01,2.98]},
    }
    for metal,block in dose_values.items():
        c0=64.0 if metal=='Cu(II)' else 65.0
        for dose,values in block.items():
            for temp,qe in zip([300,400,500,600],values):
                row=next(r for r in rows if r['pollutant']==metal and r['pyrolysis_temperature_C']==temp).copy()
                row.update({
                    'observation_id':f'V52_APPLE_{metal[:2].upper()}_{temp}_DOSE{int(dose)}_TEXT',
                    'dose_g_L':dose,'target_value':qe,'mass_balance_cap_mg_g':c0/dose,
                    'experiment_series_id':f'{DOI}::dose_effect::{metal}::pH5_25C',
                    'target_origin':'V52_ZHAO2020_SECTION3_1_TEXT_EXPERIMENT1',
                    'target_qualification':'REPORTED_1440MIN_UPTAKE_SOURCE_TEXT',
                    'source_target_status':'SECTION3_1_TEXT_REPORTED_QT_1440MIN_EQ',
                    'source_page':'Section 3.1 text / Fig 1 and Methods Experiment 1 Table 2',
                    'source_numeric_sd_mg_g':np.nan,
                    'source_table_qe_type':'reported_1440min_qt_equilibrium_per_paper',
                    'warnings':('Source text Section 3.1 dosage 1 or 20 g/L; '
                                'equilibrium at 1440 min in Section 3.3. '
                                'Same DOI and manufactured material as Table 3, other dose. '
                                'No standard deviation reported for these values.'),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def stage1_rows():
    rows=[]
    for temp,bet in zip([300,400,500,600],[2.39,7.00,37.25,108.59]):
        rows.append(dict(material_id=f'V52_APPLE_BC{temp}',doi=DOI,paper_year=2020,
            original_sample_name=f'BC{temp}',feedstock='apple tree branches',
            feedstock_group='wood branches',pyrolysis_temperature_C=float(temp),
            pyrolysis_hold_min=np.nan,heating_rate_C_min=np.nan,pyrolysis_atmosphere=np.nan,
            activation_method='none',activation_agent='none',
            process_stage='single_stage_pyrolysis',
            BET_m2_g_original_source=bet,BET_m2_g_training_candidate=bet,
            source_reference=URL,source_url=URL,primary_source_url=URL,
            source_pdf_page='Energies 2020 13 3498 Table 1',
            BET_value_evidence='Table 1 direct BET area 2.39/7.00/37.25/108.59',
            release_status='V52_RESEARCH_NOT_ACTIVE',
            source_stage='V52_PRIMARY_TABLE1_BET',
            input_verification_limitations='hold/time/atmosphere NR; char C/H/O not raw feedstock; do not impute reported conditions',
            feedstock_elemental_basis='POST_PYROLYSIS_CHAR_NOT_RAW',
            raw_feedstock_basis_note='DO NOT use post-pyrolysis char elemental/ash as raw biomass',
        ))
    return pd.DataFrame(rows)


def load_and_build(root):
    old2=pd.read_csv(root/S2_SOURCE,low_memory=False)
    old1=pd.read_csv(root/S1_SOURCE,low_memory=False)
    if len(old2)!=140 or old2.doi.nunique()!=32 or len(old1)!=417:
        raise ValueError('v46 baseline counts changed. Do not silently create v52.')
    if DOI in set(old2.doi.astype(str).str.lower()) or DOI in set(old1.doi.astype(str).str.lower()):
        raise ValueError('Apple DOI already present in baseline; prevent duplicate intake.')
    extra2=candidate_rows();extra1=stage1_rows()
    if extra2.observation_id.duplicated().any() or extra1.material_id.duplicated().any():raise ValueError('duplicate new ids')
    if not np.all(extra2.target_value>=0) or not np.all(extra2.target_value<=extra2.C0_mg_L/extra2.dose_g_L):raise ValueError('mass balance failed')
    # ALL unknown feature columns stay missing; never populate with numerical guesses.
    extra2=extra2.reindex(columns=list(dict.fromkeys(list(old2.columns)+list(extra2.columns))))
    extra1=extra1.reindex(columns=list(dict.fromkeys(list(old1.columns)+list(extra1.columns))))
    all2=pd.concat([old2,extra2],ignore_index=True,sort=False)
    all1=pd.concat([old1,extra1],ignore_index=True,sort=False)
    if len(all2)!=164 or all2.doi.nunique()!=33 or len(all1)!=421 or all1.doi.nunique()!=59:
        raise ValueError('expected 164 Stage2 /421 Stage1, new DOI unique')
    if not all2.observation_id.is_unique or not all1.material_id.is_unique:raise ValueError('duplicate IDs')
    if not np.all(pd.to_numeric(all2.target_value)>-1e-9) or not np.all(pd.to_numeric(all2.target_value)<=pd.to_numeric(all2.C0_mg_L)/pd.to_numeric(all2.dose_g_L)+1e-6):
        raise ValueError('old/new Stage2 cap errors')
    return old1,old2,extra1,extra2,all1,all2


def output_data(root,old1,old2,extra1,extra2,all1,all2):
    out=root/DATA_DIR;out.mkdir(parents=True,exist_ok=True)
    extra2.to_csv(out/'AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv',index=False)
    extra1.to_csv(out/'AppleTree_Table1_BET_4_VERIFIED.csv',index=False)
    all2.to_csv(out/'Stage2_164_RESEARCH_NOT_ACTIVE.csv',index=False)
    all1.to_csv(out/'Stage1_421_RESEARCH_NOT_ACTIVE.csv',index=False)
    return out


def clip(pred,testing):
    return np.clip(np.asarray(pred,float),0,testing.C0_mg_L.to_numpy(float)/testing.dose_g_L.to_numpy(float))


def fit_predict(train,test,variant):
    pipe,cols,nums,cats=s2_model(variant)
    pipe.fit(features(train,cols,nums,cats),train.target_value.to_numpy(float))
    return clip(pipe.predict(features(test,cols,nums,cats)),test)


def eval_same_140(old,extra):
    old=old.reset_index(drop=True); extra=extra.reset_index(drop=True)
    names=['A_base140','A_plus24','B_base140','B_plus24']
    preds={n:np.full(len(old),np.nan) for n in names}
    fold_rows=[]
    for fold,(tr,te) in enumerate(GroupKFold(5).split(old,groups=old.doi),1):
        train,test=old.iloc[tr],old.iloc[te]
        assert not set(train.doi)&set(test.doi)
        assert not set(extra.doi)&set(test.doi)
        augmented=pd.concat([train,extra],ignore_index=True,sort=False)
        for model in ['A','B']:
            preds[f'{model}_base140'][te]=fit_predict(train,test,model)
            preds[f'{model}_plus24'][te]=fit_predict(augmented,test,model)
        fold_rows.append({'fold':fold,'train_old_rows':len(train),'added_new_doi_rows':len(extra),
                          'test_old_rows':len(test),'held_doi_count':test.doi.nunique(),
                          'extra_doi_not_held_out':True})
    result=pd.DataFrame({'observation_id':old.observation_id,'doi':old.doi,'pollutant':old.pollutant,
                         'qe_observed':old.target_value,'cap_mg_g':old.C0_mg_L/old.dose_g_L})
    scores=[]
    for name,p in preds.items():
        if not np.isfinite(p).all():raise ValueError('missing original-fold predictions')
        result[name]=p
        scores.append({'evaluation':'fixed_original140_DOI_GroupKFold5','model':name,
            'rows':len(old),'doi_count':old.doi.nunique(),
            'mae_mg_g':mean_absolute_error(old.target_value,p),
            'r2':r2_score(old.target_value,p),
            'physical_violations':int(((p<0)|(p>result.cap_mg_g)).sum()),
            'cap_hits':int(np.isclose(p,result.cap_mg_g,atol=1e-8).sum())})
    return pd.DataFrame(scores),result,pd.DataFrame(fold_rows)


def eval_new_apple_as_unseen(old,extra):
    """Pre-augmentation check: original 140-only fit -> 24 NEW DOI outcomes.
    Exactly one study, so no claim of cross-DOI generalization.
    """
    assert set(old.doi).isdisjoint(set(extra.doi))
    result=extra[['observation_id','doi','pollutant','pyrolysis_temperature_C','dose_g_L','target_value']].copy()
    scores=[]
    for name in ['A','B']:
        pp=fit_predict(old,extra,name)
        result[f'{name}_base140_predict_unseen_apple']=pp
        for (pol,dose),chunk in result.groupby(['pollutant','dose_g_L']):
            x=chunk[f'{name}_base140_predict_unseen_apple']
            y=chunk['target_value']
            has_rank=np.ptp(x.to_numpy(float))>1e-8
            bestT=float(chunk.loc[x.idxmax(),'pyrolysis_temperature_C']) if has_rank else np.nan
            observedT=float(chunk.loc[y.idxmax(),'pyrolysis_temperature_C'])
            scores.append({'model':name,'pollutant':pol,'dose_g_L':dose,'rows':len(chunk),'doi_count':1,
                           'mae_mg_g':mean_absolute_error(y,x),
                           'ranking_defined':bool(has_rank),
                           'predicted_best_T_C':bestT,'observed_best_T_C':observedT,
                           'top1':bool(bestT==observedT) if has_rank else False})
    return pd.DataFrame(scores),result


def eval_stage1_fixed417(old,extra):
    target='BET_m2_g_training_candidate'
    old=old.loc[old[target].notna()].reset_index(drop=True)
    extra=extra.loc[extra[target].notna()].reset_index(drop=True)
    pp=np.full((len(old),2),np.nan)
    for tr,te in GroupKFold(5).split(old,groups=old.doi):
        train,test=old.iloc[tr],old.iloc[te]
        assert set(train.doi).isdisjoint(test.doi)
        for j,t in enumerate([train,pd.concat([train,extra],ignore_index=True)]):
            mdl,cols,nums,cats=s1_model()
            mdl.fit(features(t,cols,nums,cats),t[target].to_numpy(float))
            pp[te,j]=mdl.predict(features(test,cols,nums,cats))
    scores=[]
    for j,name in enumerate(['BET_base','BET_plus4']):
        scores.append({'model':name,'mae_m2_g':mean_absolute_error(old[target],pp[:,j]),
                       'r2':r2_score(old[target],pp[:,j]),'rows':len(old),'doi_count':old.doi.nunique()})
    return pd.DataFrame(scores)


def fit_final_research_models(root,all1,all2):
    out=root/DEPLOY;out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for model in ['A','B']:
        m,cols,nums,cats=s2_model(model)
        m.fit(features(all2,cols,nums,cats),all2.target_value.to_numpy(float))
        path=out/f'V52_STAGE2_{model}_164_NOT_ACTIVE.joblib'
        joblib.dump(m,path,compress=3)
        # Immediate local reloading catches incompatible serialized models in current environment.
        check=joblib.load(path)
        if not np.isfinite(check.predict(features(all2.head(3),cols,nums,cats))).all():raise ValueError('saved model cannot predict')
        manifest.append({'model':f'Stage2_{model}','file':path.name,'sha256':sha256(path),'nrows':len(all2)})
    for target,short in [('BET_m2_g_training_candidate','BET'),('TPV_cm3_g_training_candidate','TPV')]:
        train=all1.loc[all1[target].notna()]
        m,cols,nums,cats=s1_model()
        m.fit(features(train,cols,nums,cats),train[target].to_numpy(float))
        path=out/f'V52_STAGE1_{short}_{len(train)}_NOT_ACTIVE.joblib'
        joblib.dump(m,path,compress=3)
        check=joblib.load(path)
        if not np.isfinite(check.predict(features(train.head(3),cols,nums,cats))).all():raise ValueError('saved model cannot predict')
        manifest.append({'model':f'Stage1_{short}','file':path.name,'sha256':sha256(path),'nrows':len(train)})
    return manifest


def run(root,do_train=True):
    root=Path(root).resolve()
    old1,old2,extra1,extra2,all1,all2=load_and_build(root)
    out=output_data(root,old1,old2,extra1,extra2,all1,all2)
    summary,preds,folds=eval_same_140(old2,extra2)
    holdout,apple=eval_new_apple_as_unseen(old2,extra2)
    s1=eval_stage1_fixed417(old1,extra1)
    report=root/REPORT;report.mkdir(parents=True,exist_ok=True)
    for name,frame in [('stage2_same140_metrics.csv',summary),('stage2_same140_predictions.csv',preds),
        ('stage2_fold_audit.csv',folds),('apple_unseen_DOI_metrics.csv',holdout),
        ('apple_unseen_DOI_predictions.csv',apple),('stage1_BET_same_old_metrics.csv',s1)]:
        frame.to_csv(report/name,index=False)
    manifest=fit_final_research_models(root,all1,all2) if do_train else []
    result={'source_DOI':DOI,'source_URL':URL,'base_Stage2_rows':len(old2),
        'new_Stage2_rows':len(extra2),'new_Stage2_DOI':1,'merged_Stage2_rows':len(all2),
        'base_Stage1_rows':len(old1),'new_Stage1_BET_rows':len(extra1),
        'merged_Stage1_rows':len(all1),
        'Stage1_TPV_new_rows':0,
        'old_Stage2_sha256':sha256(root/S2_SOURCE),
        'old_Stage1_sha256':sha256(root/S1_SOURCE),
        'model_compatibility':'Models serialized in execution environment; Windows re-run training before deployment',
        'stage2_same140_scores':summary.to_dict(orient='records'),
        'stage1_BET_scores':s1.to_dict(orient='records'),
        'apple_new_DOI_scores':holdout.to_dict(orient='records'),
        'research_models':manifest,
        'model_deployed':False}
    (report/'V52_MACHINE_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--no-save-models',action='store_true')
    a=p.parse_args();run(a.root,do_train=not a.no_save_models)
