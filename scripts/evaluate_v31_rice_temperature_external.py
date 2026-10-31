"""v31 -- held-out paper temperature-ranking challenge, NOT a model release.

Do not ingest graph-digitized values into training. Fixed v23 recipe: Stage 2 A XGB,
Stage 2 B Ridge; B XGB is a pre-declared secondary sensitivity analysis.

Source: Huang et al., Molecules 2025 30 3616, Figure 9, Methods 3.1-3.4.
The graph readings are from v30, are approximate qt at 1440 min, NOT certified qe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from pyrolysis.biochar_stage2.v15_features import model_inputs
from scripts.compare_phase8_v23 import stage2_targets, _stage2_fit_predict
from scripts.evaluate_v27_recommendation import tie_aware_accuracy, expected_top1_and_regret

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'v31_rice_temperature_external'
EVIDENCE = ROOT / 'reports' / 'v30_graph_evidence' / 'rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv'
LINKED = ROOT / 'reports' / 'v30_graph_evidence' / 'rice_figure9_stage1_linked_18_QA_NOT_TRAINED.csv'
HELDOUT_DOI = '10.3390/molecules30173616'
TOL = 0.5  # v30 declared graph-reading uncertainty, not measured analytical uncertainty

# Literally reported in article's Methods sections 3.1-3.4, pp.18-19.
# 'none' represents explicitly unmodified material; deliberately do NOT map to
# 'none_reported' (the latter means no agent was reported, not unmodified).
UNMODIFIED_PROCESS = dict(
    feedstock_label_normalized='rice husk',
    manufacturing_route='SINGLE_STAGE_PYROLYSIS',
    process_step_count=1,
    pyrolysis_hold_min=120.0,
    heating_rate_C_min=10.0,
    pyrolysis_atmosphere='N2',
    activation_or_modification='none',
    secondary_temperature_C=np.nan,
    secondary_hold_min=np.nan,
)


def load_external_unmodified(source:Path=EVIDENCE, linked:Path=LINKED) -> pd.DataFrame:
    raw = pd.read_csv(source)
    if len(raw)!=18 or set(raw.doi)!={HELDOUT_DOI}:
        raise ValueError('v30 original 18-marker cohort missing or changed')
    sel = raw.loc[raw.modification_route.eq('UNMODIFIED')].copy()
    if len(sel)!=6 or sel.duplicated(['pollutant','pyrolysis_temp_C']).any():
        raise ValueError('Expected 6 unique unmodified (pollutant x temperature) markers')
    if set(sel.pollutant)!={'Cd(II)','Pb(II)'}:
        raise ValueError('Expected Cd and Pb only')
    for pol,grp in sel.groupby('pollutant'):
        if sorted(grp.pyrolysis_temp_C.tolist()) != [300,400,500]:
            raise ValueError(f'Incomplete temperature contrast {pol}')
        conditions=['C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']
        if any(grp[c].nunique(dropna=False)!=1 for c in conditions):
            raise ValueError(f'Adsorption conditions not identical within {pol}')
    if sel.training_approved.astype(str).str.lower().ne('false').any():
        raise ValueError('These graph readings must remain NOT_TRAINED')
    if not sel.physical_cap_pass.astype(str).str.lower().eq('true').all():
        raise ValueError('Unmodified external measured markers must pass physical mass balance')
    if sel.fig9_vs_table4_5_conflict.astype(str).str.lower().eq('true').any():
        raise ValueError('Selected evidence includes known figure-vs-table conflicts')
    if not sel.source_measure_type.eq('GRAPH_EXPERIMENTAL_MARKER_QT_AT_1440_MIN').all():
        raise ValueError('Incorrect or mixed target type')
    linked_df=pd.read_csv(linked)
    lookup=linked_df.loc[linked_df.modification_route.eq('UNMODIFIED'),['pollutant','pyrolysis_temp_C','sample_id','stage1_material_id']]
    if lookup.duplicated(['pollutant','pyrolysis_temp_C','sample_id']).any():
        raise ValueError('Duplicate Stage1 material join')
    sel=sel.merge(lookup,on=['pollutant','pyrolysis_temp_C','sample_id'],how='left',validate='one_to_one')
    if sel.stage1_material_id.isna().any():
        raise ValueError('Missing Stage1 sample identity')
    sel=sel.sort_values(['pollutant','pyrolysis_temp_C']).reset_index(drop=True)
    return sel


def model_input_rows(source_rows:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for r in source_rows.itertuples(index=False):
        x=dict(UNMODIFIED_PROCESS)
        x.update(pollutant=r.pollutant,C0_mg_L=float(r.C0_mg_L),
                 dose_g_L=float(r.dose_g_L),initial_pH=float(r.initial_pH),
                 adsorption_T_C=float(r.adsorption_T_C),
                 contact_time_min=float(r.contact_time_min),
                 pyrolysis_temperature_C=float(r.pyrolysis_temp_C))
        rows.append(x)
    result=pd.DataFrame(rows)
    expected=set(model_inputs('B'))
    if set(result.columns)!=expected:
        raise AssertionError(f'Model B mismatch missing={expected-set(result.columns)} extras={set(result.columns)-expected}')
    for pol,sub in result.groupby('pollutant'):
        ignore={'pyrolysis_temperature_C'}
        nonvary=[c for c in result.columns if c not in ignore]
        if any(sub[c].nunique(dropna=False)!=1 for c in nonvary):
            raise ValueError(f'Not temperature-only counterfactual: {pol}')
    return result


def validate_holdout(core:pd.DataFrame, external:pd.DataFrame, inputs:pd.DataFrame) -> dict:
    if HELDOUT_DOI in set(core.doi.astype(str).str.lower().str.strip()):
        raise ValueError('LEAKAGE: held-out DOI present in training')
    if len(core)!=98 or core.doi.nunique()!=26:
        raise ValueError('Unexpected training cohort; do not silently change denominator')
    if set(external.sample_id)&set(core.get('observation_id',pd.Series(dtype=str)).astype(str)):
        raise ValueError('Unexpected sample/observation ID overlap')
    for name in model_inputs('B'):
        if name not in inputs: raise ValueError(f'No column {name}')
    sub=core.loc[core.feedstock_label_normalized.eq('rice husk')]
    process_overlap=int(sub.manufacturing_route.eq('SINGLE_STAGE_PYROLYSIS').sum())
    # Novel feedstock-route interaction matters even though route exists globally.
    return {'train_rows':len(core),'train_dois':int(core.doi.nunique()),
            'external_doi':HELDOUT_DOI,'external_doi_seen_in_stage2_training':False,
            'rice_husk_train_rows':int(len(sub)),
            'rice_husk_single_stage_train_rows':process_overlap,
            'global_single_stage_train_rows':int(core.manufacturing_route.eq('SINGLE_STAGE_PYROLYSIS').sum()),
            'explicit_unmodified_agent_category_seen_in_training':bool(core.activation_or_modification.eq('none').any()),
            'external_target':'graph experimental qt at 1440 min; exploratory, equilibrium not certified',
            'external_qa_approved_for_training':False}


def score_one(y:np.ndarray,pred:np.ndarray,tolerance:float=TOL) -> dict:
    pairwise,npairs,nties=tie_aware_accuracy(y,pred)
    top1,regret,top_ties=expected_top1_and_regret(y,pred)
    best_idx=int(np.argmax(y))
    margins=[float(y[best_idx]-y[k]) for k in range(len(y)) if k!=best_idx]
    return dict(pairwise_tie_half=pairwise,n_pairs=npairs,prediction_ties=nties,
                top1_expected=top1,regret_mg_g=regret,top_prediction_ties=top_ties,
                observed_best_has_digitization_margin=bool(all(v>2*tolerance for v in margins)),
                obs_best_gap_to_second_mg_g=min(margins))


def run(root:Path=ROOT,out:Path=REPORT) -> dict:
    root=Path(root);out=Path(out)
    core97,core=stage2_targets()
    sel=load_external_unmodified(root/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv',
                                 root/'reports/v30_graph_evidence/rice_figure9_stage1_linked_18_QA_NOT_TRAINED.csv')
    test=model_input_rows(sel)
    audit=validate_holdout(core,sel,test)
    records=sel[['doi','pollutant','sample_id','stage1_material_id','pyrolysis_temp_C','qe_reported_approx_mg_g',
                 'digitization_tolerance_mg_g','C0_mg_L','dose_g_L','initial_pH',
                 'adsorption_T_C','contact_time_min']].copy()
    records['physical_cap_mg_g']=records.C0_mg_L/records.dose_g_L
    models={'A_XGBoost':('A','XGBoost',core), 'B_Ridge':('B','Ridge',core),
            'B_XGBoost_sensitivity':('B','XGBoost',core),
            'B_Ridge_refitCore97_sensitivity':('B','Ridge',core97),
            'B_XGBoost_refitCore97_sensitivity':('B','XGBoost',core97)}
    for key,(model,algo,train_data) in models.items():
        pred=_stage2_fit_predict(train_data,test,model,algo)
        records['pred_raw_'+key]=pred
        records['physical_violation_'+key]=(pred < 0)|(pred>records.physical_cap_mg_g.to_numpy()+1e-9)
        # For diagnostic only, never score clipped values as model performance.
        records['pred_clipped_'+key+'_DIAGNOSTIC_ONLY']=np.clip(pred,0,records.physical_cap_mg_g.to_numpy())
    ranks=[]
    for pol,group in records.groupby('pollutant',sort=True):
        y=group.qe_reported_approx_mg_g.to_numpy(dtype=float)
        cap=group.physical_cap_mg_g.to_numpy(dtype=float)
        temperatures=group.pyrolysis_temp_C.to_numpy(dtype=int)
        for key in models:
            p=group['pred_raw_'+key].to_numpy(dtype=float)
            s=score_one(y,p)
            s.update(pollutant=pol,model=key,n_rows=len(y),
                     experimental_best_C=int(temperatures[y.argmax()]),
                     predicted_best_C=int(temperatures[p.argmax()]) if not np.isclose(p.max(),p.min(),rtol=0,atol=1e-9) else None,
                     mae_mg_g=float(mean_absolute_error(y,p)),
                     raw_physical_violations=int(np.sum((p<0)|(p>cap+1e-9))),
                     all_predictions_physically_valid=bool(np.all((p>=0)&(p<=cap+1e-9))),
                     is_exploratory_model_comparison=True)
            ranks.append(s)
    ranking=pd.DataFrame(ranks)
    out.mkdir(parents=True,exist_ok=True)
    records.to_csv(out/'external_rice_predictions_6_NOT_TRAINED.csv',index=False)
    ranking.to_csv(out/'per_pollutant_ranking_and_physical_qa.csv',index=False)
    pd.DataFrame(test).assign(sample_id=sel.sample_id,doi=HELDOUT_DOI).to_csv(out/'source_backed_model_inputs_NO_TARGET.csv',index=False)
    aggregate={}
    for name in models:
        p=records['pred_raw_'+name]
        rec=ranking.loc[ranking.model.eq(name)]
        aggregate[name]={'mae_mg_g':float(mean_absolute_error(records.qe_reported_approx_mg_g,p)),
            'raw_constraint_violations':int(records['physical_violation_'+name].sum()),
            'pairwise_macro':float(rec.pairwise_tie_half.mean()),
            'top1_expected_mean':float(rec.top1_expected.mean()),
            'mean_regret_mg_g':float(rec.regret_mg_g.mean())}
    audit.update(external_rows=6, external_dois=1,external_pollutant_series=2,
        primary_fit='A XGBoost / B Ridge re-fit on Core98; active deployed B XGBoost joblib NOT loaded',
        core97_sensitivity_fit='Fixed original 97-row cohort re-fit (NOT the saved deployed estimator)',
        source_paper_method_pages=[18,19],
        contains_same_paper_stage1_BET_TPV_measurements_as_eval_predictors=False,
        train_model_uses_graph_data=False,
        observational_qt_never_used_to_fit_model=True,
        no_clipping_used_for_model_scoring=True,
        tolerance_mg_g=TOL,models=aggregate,
        independent_validation_level='ONE externally held-out DOI; 2 correlated series, NOT generalization accuracy',
        interpretation='Do not deploy global optimum / no improvement claim; paper/route OOD; 1440-min graph approximations.')
    (out/'summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return audit


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
