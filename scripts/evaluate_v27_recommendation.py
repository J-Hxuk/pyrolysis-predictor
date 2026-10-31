"""v27 research-only audit: true held-out DOI manufacturing rankings and Stage1 robustness.

Uses the 98-row Phase8 Stage2 cohort only; v24 external additions are NEVER trained on.
No hyperparameter tuning or deployment modifications are performed here.

Run: python -m scripts.evaluate_v27_recommendation
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.compare_phase8_v23 import stage2_targets, _stage2_fit_predict
from pyrolysis.biochar_stage2.v15_features import ADSORPTION_INPUTS, MANUFACTURING_INPUTS

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/v27_recommendation_audit'
GROUP_FIELDS = ['doi', 'feedstock_group_for_novelty_check', 'pollutant',
                'C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']


def extract_comparable_groups(core: pd.DataFrame) -> pd.DataFrame:
    """Find within-paper manufacturing contrasts at IDENTICAL reported adsorption inputs.

    Unreported conditions remain missing. Unknown manufacturing values must never
    be invented; raw target is forbidden from forming groups.
    """
    records = []
    needed = GROUP_FIELDS + MANUFACTURING_INPUTS + ['observation_id', 'target_value']
    if not set(needed).issubset(core.columns):
        raise ValueError('Missing strict-group input columns')
    if core.observation_id.duplicated().any():
        raise ValueError('Duplicate observation_id in evaluation cohort')
    if len(set(core['doi'].dropna())) < 2:
        raise ValueError('Need independent DOI groups')
    for key, sub in core.groupby(GROUP_FIELDS, dropna=False, sort=True):
        unique_recipes = sub[MANUFACTURING_INPUTS].drop_duplicates()
        if len(unique_recipes) < 2:
            continue
        # deterministic group ID derived from first observation order and a counter; no label leakage
        records.append({'key':key, 'rows':sub.copy(), 'n_recipes':len(unique_recipes)})
    frames = []
    for i, item in enumerate(records, 1):
        df = item['rows'].copy()
        df['comparison_id'] = f'CG{i:02d}'
        frames.append(df)
    return pd.concat(frames,ignore_index=True) if frames else core.iloc[0:0].assign(comparison_id=pd.Series(dtype=str))


def tie_aware_accuracy(y: np.ndarray, p: np.ndarray, *, min_difference: float = 0.0,
                       atol: float = 1e-9) -> tuple[float|None,int,int]:
    """Pairwise ranking probability; 0.5 for predicted ties, drop true ties/small diffs."""
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    assert y.shape==p.shape and y.ndim==1
    points=[];ties=0
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            if not (np.isfinite(y[i]) and np.isfinite(y[j]) and np.isfinite(p[i]) and np.isfinite(p[j])):
                continue
            if abs(y[i]-y[j])<=max(min_difference,atol):
                continue
            if np.isclose(p[i],p[j],atol=atol,rtol=0):
                points.append(0.5);ties+=1
            else:
                points.append(float((y[i]-y[j])*(p[i]-p[j]) > 0))
    return (float(np.mean(points)) if points else None, len(points), ties)


def expected_top1_and_regret(y: np.ndarray, p: np.ndarray, atol=1e-9) -> tuple[float,float,int]:
    """No artificial tie winner: uniformly average over tied predicted maximizers."""
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    high=np.nanmax(p)
    chosen=np.flatnonzero(np.isclose(p,high,atol=atol,rtol=0))
    truth=np.nanmax(y)
    best=np.flatnonzero(np.isclose(y,truth,atol=atol,rtol=0))
    hit=float(np.isin(chosen,best).mean())
    regret=float(np.mean(truth-y[chosen]))
    return hit,regret,len(chosen)


def score_group(group: pd.DataFrame, *, min_difference=0.0) -> dict:
    y=group['target_value'].astype(float).to_numpy()
    p=group['pred_b_raw'].to_numpy(dtype=float)
    cap=(pd.to_numeric(group['C0_mg_L'], errors='raise')/pd.to_numeric(group['dose_g_L'], errors='raise')).to_numpy(dtype=float)
    hit,regret,nties=expected_top1_and_regret(y,p)
    agreement,npairs,predict_ties=tie_aware_accuracy(y,p,min_difference=min_difference)
    valid=(np.isfinite(p)&(p>=0)&(p<=cap+1e-10))
    # Physical constraints evaluated on unmodified predictions. Do not let clipping win.
    valid_ranking_possible=bool(valid.sum()>=2 and np.any(np.diff(y[valid])!=0))
    valid_agree, valid_pairs, _=tie_aware_accuracy(y[valid],p[valid],min_difference=min_difference) if valid.sum()>=2 else (None,0,0)
    return dict(n=len(group),n_materials=int(group.material_uid.nunique()),
                observed_range_mg_g=float(np.max(y)-np.min(y)),
                n_pairs=int(npairs),tie_aware_pairwise=agreement,predicted_tie_pairs=predict_ties,
                heldout_top1_expected=hit,regret_mg_g=regret,n_predicted_top_ties=nties,
                random_baseline_top1_expected=float(np.sum(np.isclose(y,y.max()))/len(y)),
                random_baseline_regret_mg_g=float(y.max()-y.mean()),
                n_physical_predictions=int(valid.sum()),n_raw_violations=int((~valid).sum()),
                physical_eligible_ranking=valid_ranking_possible,
                physical_eligible_pairwise=valid_agree,physical_eligible_pairs=int(valid_pairs),
                all_adsorption_fields_reported=bool(group[ADSORPTION_INPUTS].notna().all().all()),
                any_manufacturing_missing=bool(group[MANUFACTURING_INPUTS].isna().any().any()))


def doi_uncertainty(groups:pd.DataFrame, *, seed:int=20261030, n_boot:int=10000) -> dict:
    """Descriptive DOI-cluster bootstrap; exact sign flip exploratory p; NOT an independent replication."""
    doi_scores=groups.groupby('held_out_doi',as_index=False)['tie_aware_pairwise'].mean().dropna()
    scores=doi_scores['tie_aware_pairwise'].to_numpy(dtype=float)
    n=len(scores)
    if n<2:
        return {'doi_count':n,'cluster_ci_95':None,'signflip_two_sided_p':None}
    rng=np.random.default_rng(seed)
    boot=scores[rng.integers(0,n,size=(n_boot,n))].mean(axis=1)
    # all 2^n sign combinations around the no-skill 0.5 threshold, valid as
    # a descriptive randomization sensitivity only when signed-effects symmetry holds.
    centered=scores-.5
    if n <= 16:
        import itertools
        null=np.array([np.mean(centered*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=n)])
        p=float(np.mean(abs(null)>=abs(centered.mean())-1e-12))
    else:
        p=None
    return {'doi_count':int(n),'macro_doi_mean':float(scores.mean()),
            'cluster_ci_95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'signflip_two_sided_p_exploratory':p,
            'interpretation':'n small; CI and signflip exploratory; groups within DOI not independent'}


def stage2_recommendation_audit() -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    _,base=stage2_targets()
    assert len(base)==98 and base.doi.nunique()==26
    comparisons=extract_comparable_groups(base)
    if comparisons.empty:
        raise RuntimeError('No comparable manufacturing groups')
    groups=[]; pred_rows=[]
    # One full-DOI held-out fit is reused for ALL comparison groups from that DOI.
    for doi,gg in comparisons.groupby('doi',sort=True):
        tr=base[base.doi.ne(doi)].copy()
        te=gg.copy()
        assert doi not in set(tr.doi) and not(set(tr.observation_id)&set(te.observation_id))
        te['pred_b_raw']=_stage2_fit_predict(tr,te,'B','Ridge')
        te['training_dois']=tr.doi.nunique()
        for comp,chunk in te.groupby('comparison_id',sort=True):
            stat=score_group(chunk)
            stats_1=score_group(chunk,min_difference=1.)
            stats_2=score_group(chunk,min_difference=2.)
            info={'comparison_id':comp,'held_out_doi':doi,
                  'feedstock':str(chunk.iloc[0]['feedstock_group_for_novelty_check']),
                  'pollutant':str(chunk.iloc[0]['pollutant']),
                  'minimum_difference_1mg_pairwise':stats_1['tie_aware_pairwise'],
                  'minimum_difference_1mg_pairs':stats_1['n_pairs'],
                  'minimum_difference_2mg_pairwise':stats_2['tie_aware_pairwise'],
                  'minimum_difference_2mg_pairs':stats_2['n_pairs'],
                  **stat}
            groups.append(info)
        pred_rows.append(te[['comparison_id','doi','observation_id','material_uid','pollutant','target_value',
                            'C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min',
                            'manufacturing_route','pyrolysis_temperature_C','pred_b_raw','training_dois']].copy())
    groups_df=pd.DataFrame(groups).sort_values('comparison_id')
    predictions=pd.concat(pred_rows,ignore_index=True)
    if predictions.observation_id.duplicated().any():
        raise AssertionError('A held-out observation was evaluated twice')
    def mean_available(field,grouped):
        a=pd.to_numeric(grouped[field],errors='coerce').dropna()
        return None if a.empty else float(a.mean())
    def stat_aggregate(x):
        if x.empty:return {}
        macro=mean_available('tie_aware_pairwise',x)
        doi_mean=x.groupby('held_out_doi',as_index=False)['tie_aware_pairwise'].mean()
        return {'n_groups':int(len(x)),'n_dois':int(x.held_out_doi.nunique()),
                'n_pairs_total':int(x.n_pairs.sum()),
                'macro_pairwise_tie_half':macro,
                'doi_macro_pairwise_tie_half':mean_available('tie_aware_pairwise',doi_mean),
                'pooled_pairwise_tie_half':(None if x.n_pairs.sum()==0 else
                        float(np.nansum(x.tie_aware_pairwise.fillna(0)*x.n_pairs)/x.n_pairs.sum())),
                'heldout_top1_expected_mean':mean_available('heldout_top1_expected',x),
                'random_top1_expected_mean':mean_available('random_baseline_top1_expected',x),
                'regret_mg_g_mean':mean_available('regret_mg_g',x),
                'random_regret_mg_g_mean':mean_available('random_baseline_regret_mg_g',x)}
    evaluable=groups_df[groups_df.n_pairs.gt(0)].copy()  # exclude flat observed response: no ranking ground truth
    usable=evaluable[evaluable.all_adsorption_fields_reported]
    screened=evaluable[evaluable.all_adsorption_fields_reported & evaluable.physical_eligible_ranking]
    summary={
       'contract':'Research-only v27, 98 rows / 26 DOI, v24 69 rows excluded, Model B Ridge fixed; no deployment',
       'found':{'groups_with_at_least_2_recipes':len(groups_df),
                'groups_with_nontied_observed_target':int(len(evaluable)),
                'rows_in_groups':len(predictions),'independent_DOIs':int(groups_df.held_out_doi.nunique()),
                'fully_reported_adsorption_groups':int(len(usable)),
                'physically_eligible_groups':int(len(screened)),
                'all_candidates_physically_valid_groups':int((groups_df.n_raw_violations.eq(0) & groups_df.n_pairs.gt(0)).sum()),
                'groups_with_predicted_ties':int(groups_df.predicted_tie_pairs.gt(0).sum()),
                'model_B_raw_violations_in_group_rows':int(sum(groups_df.n_raw_violations))},
       'all_evaluable_groups':stat_aggregate(evaluable),
       'adsorption_complete':stat_aggregate(usable),
       'physical_eligible_subset':stat_aggregate(screened),
       'all_candidates_physically_valid_subset':stat_aggregate(groups_df[groups_df.n_raw_violations.eq(0)&groups_df.n_pairs.gt(0)]),
       'margin_1mg':{'groups':int(evaluable.minimum_difference_1mg_pairwise.notna().sum()),
                     'macro_pairwise':mean_available('minimum_difference_1mg_pairwise',evaluable),
                     'pairs':int(evaluable.minimum_difference_1mg_pairs.sum())},
       'margin_2mg':{'groups':int(evaluable.minimum_difference_2mg_pairwise.notna().sum()),
                     'macro_pairwise':mean_available('minimum_difference_2mg_pairwise',evaluable),
                     'pairs':int(evaluable.minimum_difference_2mg_pairs.sum())},
       'doi_uncertainty':doi_uncertainty(evaluable),
       'limitations':['Within-DOI comparable groups are grouped at exactly same adsorption inputs, no imputed conditions',
                      'Ridge trained on other DOI only, unseen feedstocks frequently confounded with DOI',
                      'Top1 is highest within held-out candidates, NOT global optimization or 16-condition oracle',
                      'Predicted tie is assigned fair 0.5 pairwise credit; real small differences may be noise',
                      'Physical limits checked BEFORE clipping. Eligible subset is selected after prediction; not unbiased score',
                      'Observed qe is reported uptake not uniformly certified equilibrium',
                      'DOI-level inference not available at large sample n; groups per DOI dependent']}
    return groups_df,predictions,summary


def stage1_strata_audit(root=ROOT) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    src=Path(root)/'reports/v26_physics_aware'
    oof=pd.read_csv(src/'stage1_lodo_log_vs_raw.csv')
    ext=pd.read_csv(src/'stage1_external33_log_vs_raw.csv')
    rows=[]; leave=[]; summary={}
    for t in ['BET','TPV']:
        part=oof[oof.target.eq(t)].copy()
        scale=30. if t=='BET' else .03
        part['stratum']=np.where(part.actual<=scale,'low','higher')
        for stratum,ss in part.groupby('stratum'):
            rows.append({'target':t,'cohort':'core_LODO','stratum':stratum,'n':len(ss),'dois':ss.doi.nunique(),
                         'raw_mae':float(abs(ss.actual-ss.raw_rf).mean()),
                         'log_mae':float(abs(ss.actual-ss.log_rf).mean()),
                         'delta_log_minus_raw':float(abs(ss.actual-ss.log_rf).mean()-abs(ss.actual-ss.raw_rf).mean())})
        ep=ext[ext.target.eq(t)].copy()
        ep['stratum']=np.where(ep.actual<=scale,'low','higher')
        for stratum,ss in ep.groupby('stratum'):
            rows.append({'target':t,'cohort':'external_33_ALREADY_EXAMINED','stratum':stratum,'n':len(ss),'dois':ss.doi.nunique(),
                         'raw_mae':float(abs(ss.actual-ss.raw_rf_v25).mean()),
                         'log_mae':float(abs(ss.actual-ss.log_rf).mean()),
                         'delta_log_minus_raw':float(abs(ss.actual-ss.log_rf).mean()-abs(ss.actual-ss.raw_rf_v25).mean())})
        for doi,sub in part.groupby('doi'):
            leave.append({'target':t,'doi':doi,'n':len(sub),
                          'raw_mae':float(abs(sub.actual-sub.raw_rf).mean()),
                          'log_mae':float(abs(sub.actual-sub.log_rf).mean()),
                          'delta_log_minus_raw':float(abs(sub.actual-sub.log_rf).mean()-abs(sub.actual-sub.raw_rf).mean())})
        perdoi=pd.DataFrame([r for r in leave if r['target']==t])
        summary[t]={'doi_improved':int((perdoi.delta_log_minus_raw<0).sum()),
                    'doi_worsened':int((perdoi.delta_log_minus_raw>0).sum()),
                    'doi_mean_delta_mae':float(perdoi.delta_log_minus_raw.mean()),
                    'worst_doi_delta_mae':float(perdoi.delta_log_minus_raw.max()),
                    'oof_rows':int(len(part)),
                    'external_dois':int(ep.doi.nunique())}
    return pd.DataFrame(rows),pd.DataFrame(leave),summary


def run(root=ROOT,out=REPORT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grouped,pred,summary=stage2_recommendation_audit()
    strata,doi,s1=stage1_strata_audit(root)
    grouped.to_csv(out/'stage2_strict_comparison_groups.csv',index=False)
    pred.to_csv(out/'stage2_heldout_doi_predictions.csv',index=False)
    strata.to_csv(out/'stage1_error_strata.csv',index=False)
    doi.to_csv(out/'stage1_per_doi_error.csv',index=False)
    grouped.groupby('held_out_doi',as_index=False)['tie_aware_pairwise'].mean().to_csv(out/'stage2_per_doi_pairwise.csv',index=False)
    result={'stage2_ranking':summary,'stage1_robustness':s1}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result

if __name__=='__main__':
    x=run()
    print(json.dumps({'stage2':x['stage2_ranking']['found'],
                      'all_evaluable_groups':x['stage2_ranking']['all_evaluable_groups'],
                      'physically_eligible':x['stage2_ranking']['physical_eligible_subset'],
                      'stage1':x['stage1_robustness']},ensure_ascii=False,indent=2))
