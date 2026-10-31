"""v36 / source-aware, research-only temperature-ranking challenge.

Consumes v35 verified source descriptions plus v36 *single-reader* graph estimates.
NO modification to production model, training core, or active.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DOI_JIA = '10.3390/app8071019'
DOI_SHEN = '10.1016/j.scitotenv.2018.11.282'


def load_panel(v35_file: Path, digitized_file: Path) -> pd.DataFrame:
    old = pd.read_csv(v35_file, encoding='utf-8-sig')
    new = pd.read_csv(digitized_file, encoding='utf-8-sig')
    frames = []
    for _, row in old.iterrows():
        frames.append({
            'group_id': row.group_id, 'doi': row.doi, 'feedstock':row.feedstock,
            'material_id':row.material_id,'temperature_C':float(row.temperature_C),
            'pollutant':row.pollutant, 'C0_mg_L': float(row.C0_mg_L),
            'dose_g_L':float(row.dose_g_L), 'qe_mg_g':float(row.qe_mg_g),
            'uq_mg_g':float(row.qe_uncertainty_reported_mg_g) if pd.notna(row.qe_uncertainty_reported_mg_g) else np.nan,
            'source_type':row.qe_origin, 'source_pdf':row.source_pdf,
            'source_page':str(row.source_page),
            'source_figure':row.source_figure, 'approval':'v35_research_only',
            'analysis_status':'RESEARCH_ONLY_NOT_TRAIN_APPROVED',
            'contact_time_min':float(row.contact_time_min),
            'hold_min':float(row.hold_min),
            'pH_reported':row.initial_pH_reported,
        })
    for _, row in new.iterrows():
        group = 'TG05' if row.feedstock == 'rice husk' else ('TG06' if row.feedstock == 'cotton straw' else 'TG07')
        frames.append({
            'group_id':group, 'doi':row.doi, 'feedstock':row.feedstock,
            'material_id':row.material_id, 'temperature_C':float(row.temperature_C),
            'pollutant':row.pollutant, 'C0_mg_L':float(row.C0_mg_L),
            'dose_g_L':float(row.dose_g_L), 'qe_mg_g':float(row.qe_mg_g_approx),
            'uq_mg_g':float(row.qe_uncertainty_mg_g),
            'source_type':row.source_status,'source_pdf':row.source_pdf,
            'source_page':str(row.source_page_pdf),'source_figure':row.source_figure,
            'approval':row.approval_status,
            'analysis_status':'RESEARCH_ONLY_SINGLE_READER_APPROX_NOT_TRAIN_APPROVED',
            'contact_time_min':float(row.contact_time_min),
            'hold_min':float(row.pyrolysis_hold_min),
            'pH_reported':row.initial_pH,
        })
    p = pd.DataFrame(frames)
    # No DOI or observation leak across split, no qmax or fitted qe allowed.
    if len(p) != 28: raise ValueError(f'Expected 15+13 source rows, got {len(p)}')
    if p.groupby('group_id').doi.nunique().gt(1).any(): raise ValueError('Group contains multiple DOI')
    if p.duplicated(['group_id', 'temperature_C']).any(): raise ValueError('Duplicate temperature within one comparison group')
    if p['source_type'].str.contains('LANGMUIR|QMAX|QE_CAL|FITTED',case=False,regex=True).any():
        raise ValueError('A fitted target is mixed in the measured source panel')
    p['mass_balance_cap_mg_g'] = p.C0_mg_L / p.dose_g_L
    if (p.qe_mg_g < 0).any() or (p.qe_mg_g > p.mass_balance_cap_mg_g + p.uq_mg_g).any():
        raise ValueError('Graph data violate physical mass balance beyond readout uncertainty')
    return p.sort_values(['doi','group_id','temperature_C']).reset_index(drop=True)


def phi(temp):
    x = (np.asarray(temp,dtype=float) - 500.0)/100.0
    return np.stack([x,x*x],axis=-1)


def confident_pairs(panel:pd.DataFrame, include_marginal=False):
    out=[]
    for g, f in panel.groupby('group_id',sort=True):
        f=f.sort_values('temperature_C').reset_index(drop=True)
        for i in range(len(f)):
            for j in range(i+1,len(f)):
                a,b=f.iloc[i],f.iloc[j]
                margin=abs(a.qe_mg_g-b.qe_mg_g)
                bound=a.uq_mg_g+b.uq_mg_g
                qualifies=bool(margin>bound)
                if not qualifies and not include_marginal:continue
                # i<j guaranteed because temperature sort ascending
                out.append(dict(group_id=g,doi=a.doi,T_lower=float(a.temperature_C),T_higher=float(b.temperature_C),
                                qe_lower=float(a.qe_mg_g),qe_higher=float(b.qe_mg_g),
                                margin_mg_g=float(margin),two_readout_uncertainty_sum_mg_g=float(bound),
                                confident=qualifies,preferred_higher=int(b.qe_mg_g>a.qe_mg_g)))
    return pd.DataFrame(out)


def fit_pairwise(panel:pd.DataFrame, leave_out_doi:str):
    data=confident_pairs(panel[panel.doi!=leave_out_doi])
    if data.empty or data.doi.nunique()<2:return None
    X=[];y=[];w=[]
    counts=data.groupby('doi').size().to_dict()
    for r in data.itertuples(index=False):
        diff=phi(r.T_higher)-phi(r.T_lower)
        # Paired augmentation removes global directional bias and intercept.
        for sign in (+1,-1):
            X.append(sign*diff); y.append(r.preferred_higher if sign==1 else 1-r.preferred_higher)
            w.append(1/(2*counts[r.doi]))
    model=LogisticRegression(C=0.5,fit_intercept=False,solver='lbfgs',max_iter=2000)
    model.fit(np.asarray(X),np.asarray(y),sample_weight=np.asarray(w))
    return model


def rank_candidates(f:pd.DataFrame,strategy:str,model=None):
    temp=f.temperature_C.to_numpy(dtype=float)
    if strategy=='highest_temperature':score=temp
    elif strategy=='target_500C':score=-np.abs(temp-500.)-(temp-500.)*1e-6
    elif strategy=='pairwise_quadratic':score=phi(temp)@model.coef_.reshape(-1)
    else:raise KeyError(strategy)
    return list(f.assign(predicted_score=score).sort_values(['predicted_score','temperature_C'],ascending=[False,True]).temperature_C.astype(float))


def evaluate(panel:pd.DataFrame):
    rankings=[]; pair_predictions=[]
    for doi in sorted(panel.doi.unique()):
        train=panel[panel.doi!=doi]; test=panel[panel.doi==doi]
        assert not set(test.doi) & set(train.doi)
        model=fit_pairwise(panel,doi)
        eligible_training_dois=int(confident_pairs(train).doi.nunique())
        eligible_training_pairs=int(len(confident_pairs(train)))
        for g,f in test.groupby('group_id',sort=True):
            f=f.sort_values('temperature_C')
            obs=f.sort_values('qe_mg_g',ascending=False)
            best=obs.iloc[0];second=obs.iloc[1]
            top_robust=bool(best.qe_mg_g-second.qe_mg_g > best.uq_mg_g+second.uq_mg_g)
            for strat in ('highest_temperature','target_500C','pairwise_quadratic'):
                if strat=='pairwise_quadratic' and model is None:continue
                order=rank_candidates(f,strat,model)
                top_t=order[0];topqe=float(f.loc[f.temperature_C==top_t,'qe_mg_g'].iloc[0]);cap=float(f.mass_balance_cap_mg_g.iloc[0])
                rankings.append(dict(heldout_doi=doi,training_doi_count=int(train.doi.nunique()),effective_training_pair_doi_count=eligible_training_dois,effective_training_pairs=eligible_training_pairs,group_id=g,feedstock=f.feedstock.iloc[0],
                     pollutant=f.pollutant.iloc[0],candidate_count=len(f), strategy=strat,predicted_top_C=top_t,
                     observed_top_C=float(best.temperature_C),observed_top_robust_under_readout_uncertainty=top_robust,
                     top1_exact=int(top_t==best.temperature_C),top1_within_readout_margin=int(best.qe_mg_g-topqe <= best.uq_mg_g + float(f.loc[f.temperature_C==top_t,'uq_mg_g'].iloc[0])),
                     top2_contains_point_max=int(best.temperature_C in order[:2]),
                     regret_mg_g=round(float(best.qe_mg_g-topqe),5),
                     regret_pct_of_mass_balance_cap=round(100*(best.qe_mg_g-topqe)/cap,5),
                     source_status='EXPLORATORY_QT24H_AND_REPORTED_UPTAKE_NOT_CERTIFIED_EQUILIBRIUM',
                     training_dois=';'.join(sorted(train.doi.unique()))))
            known=confident_pairs(f)
            if not known.empty:
                for r in known.itertuples(index=False):
                    for strat in ('highest_temperature','target_500C','pairwise_quadratic'):
                        if strat=='pairwise_quadratic' and model is None:continue
                        order=rank_candidates(f,strat,model)
                        predicted_higher=int(order.index(r.T_higher)<order.index(r.T_lower))
                        pair_predictions.append(dict(heldout_doi=doi,group_id=g,strategy=strat,
                                T_lower=r.T_lower,T_higher=r.T_higher,preferred_higher=r.preferred_higher,
                                predicted_higher=predicted_higher,correct=int(predicted_higher==r.preferred_higher),
                                qe_margin_mg_g=r.margin_mg_g,graph_uncertainty_sum_mg_g=r.two_readout_uncertainty_sum_mg_g))
    return pd.DataFrame(rankings),pd.DataFrame(pair_predictions)


def evaluate_frozen_old2_new2(panel:pd.DataFrame):
    """Explicit pre-v36 2-DOI training -> 2-new-DOI temperature challenge."""
    old=panel[~panel.doi.isin([DOI_JIA,DOI_SHEN])]
    novel=panel[panel.doi.isin([DOI_JIA,DOI_SHEN])]
    assert old.doi.nunique()==2 and novel.doi.nunique()==2
    assert not set(old.doi)&set(novel.doi)
    model=fit_pairwise(old,leave_out_doi='__NOT_PRESENT__')
    output=[]
    for group,f in novel.groupby('group_id',sort=True):
        observed=f.sort_values('qe_mg_g',ascending=False).iloc[0]
        for strat in ['highest_temperature','target_500C','pairwise_quadratic']:
            if strat=='pairwise_quadratic' and model is None:
                output.append(dict(group_id=group,doi=f.doi.iloc[0],strategy=strat,
                    train_dois=';'.join(sorted(old.doi.unique())),train_doi_count=old.doi.nunique(),
                    observed_best_C=observed.temperature_C,chosen_C=np.nan,exact_top1=np.nan,regret_mg_g=np.nan,
                    margin_only_point_estimate=True,source='NOT_TRAINABLE_Fewer_than_2_DIFFERENT_DOI_WITH_CONFIDENT_PAIRS'))
                continue
            choice=rank_candidates(f,strat,model)[0]
            q=float(f.loc[f.temperature_C==choice,'qe_mg_g'].iloc[0])
            output.append(dict(group_id=group,doi=f.doi.iloc[0],strategy=strat,
                train_dois=';'.join(sorted(old.doi.unique())),train_doi_count=old.doi.nunique(),
                observed_best_C=observed.temperature_C,chosen_C=choice,
                exact_top1=int(choice==observed.temperature_C),regret_mg_g=float(observed.qe_mg_g-q),
                margin_only_point_estimate=True,source='EXTERNAL_NEW_DOI_RESEARCH_CHALLENGE'))
    return pd.DataFrame(output)


def build_summary(panel, ranks, pairs, prospective):
    output={'version':'v36', 'purpose':'EXPLORATORY_SUBJECT_TO_SOURCE_GRAPH_REVIEW',
            'source_rows':len(panel),'distinct_doi':int(panel.doi.nunique()),
            'panels':int(panel.group_id.nunique()),
            'source_rows_old_v35':15,'new_single_reader_graph_rows':13,
            'target_warning':'Mixed source-reported uptake/24h graph readouts; all research-only, not strict equilibrium qe approved',
            'strategy_summaries':{}, 'prospective_new_two_doi':{}}
    for s,g in ranks.groupby('strategy'):
        pp=pairs[pairs.strategy==s]
        # DOIs, rather than each correlated condition group, are the independent units.
        doi_rates=g.groupby('heldout_doi').top1_exact.mean()
        output['strategy_summaries'][s]={
            'panel_count':len(g), 'heldout_doi_count':int(g.heldout_doi.nunique()),
            'raw_top1_exact':int(g.top1_exact.sum()), 'raw_top1_rate':float(g.top1_exact.mean()),
            'robust_top1_evaluable_panels':int(g.observed_top_robust_under_readout_uncertainty.sum()),
            'robust_top1_correct':int(g.loc[g.observed_top_robust_under_readout_uncertainty,'top1_exact'].sum()),
            'near_optimal_within_readout_margin':int(g.top1_within_readout_margin.sum()),
            'top2_contains_point_max':int(g.top2_contains_point_max.sum()),
            'mean_regret_mg_g':float(g.regret_mg_g.mean()),
            'macro_DOI_mean_regret_mg_g':float(g.groupby('heldout_doi').regret_mg_g.mean().mean()),
            'macro_DOI_raw_top1_rate':float(doi_rates.mean()),
            'confident_pairwise_n':int(len(pp)),
            'confident_pairwise_correct':int(pp.correct.sum()) if len(pp) else 0,
            'confident_pairwise_accuracy':float(pp.correct.mean()) if len(pp) else None,
        }
    for s,g in prospective.groupby('strategy'):
        output['prospective_new_two_doi'][s]={'groups':int(len(g)),'evaluable':int(g.exact_top1.notna().sum()),'top1_exact':int(g.exact_top1.sum()) if g.exact_top1.notna().any() else None,'mean_regret_mg_g':float(g.regret_mg_g.mean()) if g.regret_mg_g.notna().any() else None}
    output['limitations']=['Only four DOI; no valid precision/confidence claim from treating panel pairs as independent.',
        'Single-reader graph readouts require independent graphical verification before training approval.',
        'Pre-v36 two DOI contain only one with numeric uncertainty; a conservative frozen-old2 pairwise model cannot be trained.',
        'Jia and Shen graphs report 24h observations; no automatic conversion to strictly certified equilibrium qe.',
        'Shen initial pH NR; Lv graph initial C0 / experimental duration NR: hold out of ranking ML.',
        'Within-paper adsorption conditions differ; compare only relative ranks within matched panels.',
        'Study was used for model design; results descriptive, not preregistered external performance.']
    return output


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--v35',type=Path,default=Path('data/imports/v36_temperature/v36_reference_v35_source_panel15.csv'))
    parser.add_argument('--v36',type=Path,default=Path('data/imports/v36_temperature/v36_new_graph13_research_only.csv'))
    parser.add_argument('--output',type=Path,default=Path('reports/v36_temperature'))
    a=parser.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=True)
    p=load_panel(a.v35,a.v36)
    ranks,pairs=evaluate(p)
    p.to_csv(a.output/'v36_temperature_panel28_RESEARCH_ONLY.csv',index=False,encoding='utf-8-sig')
    allpairs=confident_pairs(p,include_marginal=True)
    allpairs.to_csv(a.output/'v36_pairwise_source_uncertainty_QA.csv',index=False,encoding='utf-8-sig')
    ranks.to_csv(a.output/'v36_LODO4DOI_ranking.csv',index=False,encoding='utf-8-sig')
    pairs.to_csv(a.output/'v36_LODO4DOI_confident_pairs.csv',index=False,encoding='utf-8-sig')
    prospective=evaluate_frozen_old2_new2(p)
    prospective.to_csv(a.output/'v36_frozen_old2_train_new2_test.csv',index=False,encoding='utf-8-sig')
    summary=build_summary(p,ranks,pairs,prospective)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__': main()
