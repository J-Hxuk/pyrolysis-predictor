"""v35 — reproducible, audit-first temperature ranking panel.

Run from a v34-installed project root:
    python -m scripts.build_v35_temperature_panels
Only standard-library modules; never updates deployment/stage2_models.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
D33 = Path('data/imports/v33_five_papers/stage2_new15_single_metal_measured_or_derived_RESEARCH_STAGING.csv')
MAT33 = Path('data/imports/v33_five_papers/stage1_new30_temperature_series_RESEARCH_STAGING.csv')
P34 = Path('reports/v34_temperature/new_DOI_external_temperature_predictions.csv')
BB34 = Path('data/imports/v34_temperature/zhou_BBD_unique26_QT_240min_NOT_QE.csv')
DOUT = Path('data/imports/v35_temperature')
ROUT = Path('reports/v35_temperature')


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def source_panel(root: Path):
    src = read_csv(root / D33)
    materials = read_csv(root / MAT33)
    assert len(src) == 15, f'Expected 15 v33 real-source observations, got {len(src)}'
    m = {r['material_id']: r for r in materials}
    if len(m) != len(materials): raise ValueError('Duplicate v33 material_id')
    out=[];seen=set()
    for r in src:
        key=r['observation_id']
        if key in seen: raise ValueError(f'Duplicate observation_id: {key}')
        seen.add(key)
        mat = m.get(r['material_id'])
        if not mat or mat['doi'] != r['doi']: raise ValueError(f'Invalid material/DOI link: {key}')
        c0=float(r['C0_mg_L']); dose=float(r['dose_g_L']); q=float(r['qe_mg_g']);cap=c0/dose
        if not 0<=q<=cap+1e-7: raise ValueError(f'Physical invalid source observation: {key}')
        if r['is_fitted_qmax']!='NO' or r['qe_origin'] not in (
            'REPORTED_EXPERIMENTAL_UPTAKE','DERIVED_FROM_REPORTED_EXPERIMENTAL_REMOVAL_PERCENT'):
            raise ValueError(f'Invalid target mixed into qe: {key}')
        # A numeric midpoint of the pH 4.5-5.0 interval is NOT created.
        ph=r['initial_pH']; ph_exact = ph if ph.replace('.','',1).isdigit() else ''
        out.append(dict(observation_id=key,doi=r['doi'],feedstock=mat['feedstock'],material_id=r['material_id'],
            temperature_C=int(float(mat['pyrolysis_temperature_C'])),pollutant=r['pollutant'],
            C0_mg_L=c0,dose_g_L=dose,initial_pH_reported=ph,initial_pH_exact=ph_exact,
            adsorption_T_C=r['adsorption_temperature_C'],contact_time_min=r['contact_time_min'],
            hold_min=mat['pyrolysis_hold_min'],heating_rate_C_min=mat['heating_rate_C_min'],
            atmosphere=mat['pyrolysis_atmosphere'],BET_measured_m2_g=mat['BET_m2_g'],
            pore_volume_reported_cm3_g=mat['pore_volume_cm3_g'],pore_volume_type=mat['pore_volume_type'],
            qe_mg_g=q,physical_cap_mg_g=cap,qe_origin=r['qe_origin'],
            removal_pct=r['removal_pct'],qe_uncertainty_reported_mg_g=r['derived_qe_uncertainty_mg_g'],
            source_pdf=r['source_pdf'],source_page=r['source_pdf_page'],source_figure=r['source_table_or_text'],
            target_type='EXPERIMENTAL_REPORTED_OR_DERIVED_UPTAKE_NOT_QMAX',
            approval='V35_RESEARCH_ONLY_NOT_EQUILIBRIUM_CERTIFIED'))
    # Each group holds constant the known confounders and only temperature changes.
    out.sort(key=lambda r:(r['doi'],r['C0_mg_L'],r['temperature_C']))
    return out


def group_panels(rows: list[dict]):
    groups=defaultdict(list)
    for r in rows:
        key=(r['doi'],r['feedstock'],r['pollutant'],r['C0_mg_L'],r['dose_g_L'],r['initial_pH_reported'],r['contact_time_min'])
        groups[key].append(r)
    if len(groups) != 4: raise ValueError(f'4 temperature groups expected, got {len(groups)}')
    panel=[]
    for group_id,(key,rr) in enumerate(sorted(groups.items()),1):
        temperatures=[r['temperature_C'] for r in rr]
        if len(rr)<3 or len(set(temperatures))!=len(rr):raise ValueError('Insufficient/distorted temperature group')
        if len({r['hold_min'] for r in rr})!=1 or len({r['heating_rate_C_min'] for r in rr})!=1 or len({r['atmosphere'] for r in rr})!=1:
            raise ValueError('Manufacturing condition changed beyond temperature within group')
        # Keep missing values missing and identical (possibly empty).
        sorted_rr=sorted(rr,key=lambda r:r['temperature_C'])
        top=max(sorted_rr,key=lambda r:r['qe_mg_g'])
        runner=sorted(sorted_rr,key=lambda r:r['qe_mg_g'],reverse=True)[1]
        d=top['qe_mg_g']-runner['qe_mg_g']
        sigma_top=float(top['qe_uncertainty_reported_mg_g'] or 'nan')
        sigma_run=float(runner['qe_uncertainty_reported_mg_g'] or 'nan')
        if math.isfinite(sigma_top) and math.isfinite(sigma_run):
            sigma=math.hypot(sigma_top,sigma_run)
            robust='MARGIN_GE_2_COMBINED_REPORTED_UNCERTAINTIES' if d>=2*sigma else 'TOP1_SENSITIVE_TO_REPORTED_UNCERTAINTY'
        else:robust='UNCERTAINTY_NOT_NUMERICALLY_REPORTED'
        gid=f'TG{group_id:02d}'
        for r in sorted_rr:
            panel.append(dict(group_id=gid,group_DOI_count=1,group_label=f"{r['feedstock']} | {r['pollutant']} | C0={r['C0_mg_L']:g}",
                observed_top_temperature_C=top['temperature_C'],top_runnerup_margin_mg_g=d,
                top1_robustness=robust,**r))
    return panel


def pair_score(y:list[float],pred:list[float],tol=1e-9):
    score=0.0; n=0; ties=0
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            if abs(y[i]-y[j])<=tol: continue
            n+=1
            if abs(pred[i]-pred[j])<=tol: score+=.5;ties+=1
            elif (y[i]-y[j])*(pred[i]-pred[j])>0:score+=1
    return score/n if n else None,n,ties


def rank_evaluate(root:Path,panel:list[dict]):
    preds=read_csv(root/P34)
    lookup={}
    for r in preds:
        if r['training']!='old98_only': continue
        model=(r['model'],r['algo']); k=(model,r['observation_id'])
        if k in lookup:raise ValueError(f'Duplicate v34 OOF prediction {k}')
        lookup[k]=float(r['pred_raw'])
    bygroup=defaultdict(list)
    for r in panel:bygroup[r['group_id']].append(r)
    rows=[]; per_candidate=[]
    for gid,rr in sorted(bygroup.items()):
        temps=sorted(r['temperature_C'] for r in rr)
        obs={r['temperature_C']:float(r['qe_mg_g']) for r in rr}
        winner=max(obs,key=obs.get)
        strategies={
            'Baseline|HIGHEST_T': {t:float(t) for t in temps},
            'Baseline|LOWEST_T': {t:-float(t) for t in temps},
            'Baseline|LOWER_MEDIAN_T': {t:-abs(t-temps[(len(temps)-1)//2]) for t in temps},
        }
        for model,algo in [('A','XGBoost'),('B','Ridge'),('B','XGBoost')]:
            key=(model,algo)
            try:strategies[f'{model}|{algo}']={r['temperature_C']:lookup[(key,r['observation_id'])] for r in rr}
            except KeyError as exc:raise ValueError(f'Missing v34 actual held-out DOI predictions {exc}') from exc
        for strategy,pp in strategies.items():
            ordered=sorted(temps,key=lambda t:(-pp[t],t))
            vmax=max(pp.values())
            ties=[t for t in temps if abs(pp[t]-vmax)<=1e-9]
            pred_top=(ties[0] if len(ties)==1 else None)
            # If second and third positions tie, an exact Top-2 set is not identifiable.
            top2_defined = len(ties)==1 and (len(ordered)<=2 or abs(pp[ordered[1]]-pp[ordered[2]])>1e-9)
            top2=ordered[:2] if top2_defined else []
            valid_all=all(0<=pp[t]<=next(float(r['physical_cap_mg_g']) for r in rr if r['temperature_C']==t) for t in temps) if not strategy.startswith('Baseline') else None
            score,n,pred_ties=pair_score([obs[t] for t in temps],[pp[t] for t in temps])
            rows.append(dict(group_id=gid,doi=rr[0]['doi'],feedstock=rr[0]['feedstock'],pollutant=rr[0]['pollutant'],
                  C0_mg_L=rr[0]['C0_mg_L'],candidate_n=len(temps),strategy=strategy,
                  source_training='CORE98_DOI_EXCLUDED' if not strategy.startswith('Baseline') else 'PREDEFINED_RULE_NO_TRAIN',
                  actual_top_C=winner,predicted_top_C=pred_top if pred_top is not None else '',
                  top1_status=('TIE_ABSTAIN' if pred_top is None else ('CORRECT' if pred_top==winner else 'WRONG')),
                  top2_contains_actual=(winner in top2) if top2_defined else '',
                  top2_status=('EVALUABLE' if top2_defined else 'TIE_BOUNDARY_ABSTAIN'),
                  regret_mg_g=(obs[winner]-obs[pred_top]) if pred_top is not None else '',
                  pairwise_score_ties_half=score,pairwise_n=n,predicted_tied_pairs=pred_ties,
                  all_raw_predictions_physically_valid=valid_all if valid_all is not None else '',
                  observed_top1_robustness=rr[0]['top1_robustness']))
            for r in rr:
                per_candidate.append(dict(group_id=gid,doi=r['doi'],model_strategy=strategy,
                    observation_id=r['observation_id'],temperature_C=r['temperature_C'],observed_qe_mg_g=r['qe_mg_g'],
                    prediction_or_rule_score=pp[r['temperature_C']],physical_cap_mg_g=r['physical_cap_mg_g'],
                    raw_violation=(pp[r['temperature_C']]<0 or pp[r['temperature_C']]>r['physical_cap_mg_g']) if not strategy.startswith('Baseline') else '',
                    score_type='PREDICTED_QE_MG_G' if not strategy.startswith('Baseline') else 'ORDINAL_RULE_SCORE_NOT_QE'))
    return rows,per_candidate


def qa_queue():
    # This is a *source audit queue*, not invented graph-digitized data.
    return [
      dict(doi='10.3390/app8071019',feedstock='rice husk',target='Cd(II) at 24 h',temperatures='300,400,500,600,700',source='Jia 2018 Figure 5 page 10',status='GRAPH_MEASURED_REMOVAL_ONLY_REQUIRE_INDEPENDENT_DIGITIZATION',note='Figure 5 is measured removal (%); use C0=100 mg/L, dose=2 g/L only after point-specific experiment mapping; do not use kinetic fitted Table 4 qm'),
      dict(doi='10.3390/app8071019',feedstock='cotton straw',target='Cd(II) at 24 h',temperatures='300,400,500,600,700',source='Jia 2018 Figure 5 page 10',status='GRAPH_MEASURED_REMOVAL_ONLY_REQUIRE_INDEPENDENT_DIGITIZATION',note='Graph plateaus near 100% and may not discriminate high-temperature candidates; microscopy/CHON are char properties'),
      dict(doi='10.1016/j.arabjc.2024.105946',feedstock='soybean straw',target='Cd(II)',temperatures='400,500,600',source='Lv 2024 Figure 2/3 and Table 2 page 3',status='GRAPH_READOUT_REQUIRED_DO_NOT_USE_FITTED_QE',note='Table 2 is kinetic fitted qe.cal; Table 3 is Langmuir Q0; Figure 2 is pH response, verify equivalent conditions'),
      dict(doi='10.3390/plants11233234',feedstock='rice husk',target='Cd(II)',temperatures='300,500,700',source='Li 2022 Figure 2 page 6',status='METHOD_FIGURE_VOLUME_CONFLICT_HOLD',note='Methods: 0.2 g/100 mL (2 g/L), Figure caption: 0.2 g/20 mL (10 g/L); target qe bound depends on actual dose; pH=7'),
      dict(doi='10.3390/plants11233234',feedstock='rice husk',target='Se(IV)',temperatures='300,500,700',source='Li 2022 Figure 3 page 7',status='SE_IV_SEPARATE_TARGET_AND_CONFLICT_HOLD',note='Se(IV) not supported by current core Stage 2; kinetic and isotherm fitted values not observed qe; inconsistent solution setup'),
      dict(doi='10.1016/j.scitotenv.2018.11.282',feedstock='rice straw',target='Pb(II)',temperatures='300,500,700',source='Shen 2019 Figure 4 page 4, Tables 2-3',status='GRAPH_READOUT_REQUIRED_NOT_FITTED',note='Table 2 fitted kinetic qe and Table 3 fitted qmax (some R2<0.8); do not derive 1 h qe from 65/73/94% of total without corresponding total loading'),
      dict(doi='10.3390/molecules30173616',feedstock='rice husk',target='Cd(II)',temperatures='300,400,500',source='Molecules 2025 Figure 9 page 14',status='V30_APPROX_GRAPH_AVAILABLE_NOT_APPROVED',note='Unmodified subset: 3 approximate qe points ±0.5 mg/g, separate from exact source-reported values; modified samples have figure/table conflicts'),
      dict(doi='10.3390/molecules30173616',feedstock='rice husk',target='Pb(II)',temperatures='300,400,500',source='Molecules 2025 Figure 9 page 14',status='V30_APPROX_GRAPH_AVAILABLE_NOT_APPROVED',note='Unmodified subset: 3 approximate qe points ±0.5 mg/g, physically audit before use'),
      dict(doi='10.1038/s41598-022-17901-w',feedstock='water hyacinth',target='Cu(II),Zn(II) qt at 240 min',temperatures='300,400,500',source='Zhou 2022 Table 2 page 5',status='MULTIVARIABLE_BBD_DIFFERENT_TARGET_QT',note='Unique designs 13 per pollutant; hold time and ramp rate vary; not a temperature-only ranking comparison and not certified qe'),
      dict(doi='10.1007/s11356-017-1047-2',feedstock='cedar wood',target='Cu(II),Zn(II) Langmuir Csm',temperatures='250..700',source='Rodríguez-Vila 2018 Table 2',status='FIT_QMAX_SEPARATE_ONLY',note='20 fitted Langmuir capacities, not measured qe; preserve fit R2 warnings'),
    ]


def build(root:Path=ROOT):
    root=Path(root)
    panel=group_panels(source_panel(root))
    if len({r['doi'] for r in panel})!=2:raise ValueError('Expected two independent DOI')
    measured_ids={r['observation_id'] for r in panel}
    rank,candidates=rank_evaluate(root,panel)
    qc=qa_queue()
    bb=read_csv(root/BB34)
    if len(bb)!=26:raise ValueError(f'Zhou design changed: {len(bb)}')
    if any('qt_240min_mg_g' not in d for d in bb):raise ValueError('Zhou missing qt')
    summary={
       'version':'v35','target_contract':'EXPERIMENTAL_REPORTED_OR_DERIVED_UPTAKE_NOT_EQUILIBRIUM_CERTIFIED',
       'dois':sorted({r['doi'] for r in panel}), 'independent_DOI_count':2,
       'real_source_rows':len(panel),'temperature_comparison_groups':len({r['group_id'] for r in panel}),
       'source_accuracy_note':'Wang pH range 4.5-5.0 preserved; computed qe from measured removal and C0/dose; Hu reported qe; no qmax/qt mixing.',
       'all_source_rows_physically_valid':all(0<=r['qe_mg_g']<=r['physical_cap_mg_g'] for r in panel),
       'top1_uncertainty_sensitive_groups':sum('SENSITIVE' in r['top1_robustness'] for r in {r['group_id']:r for r in panel}.values()),
       'zhou_unique_designs_qt_only':len(bb),'digitization_audit_queue':len(qc),'training_approval_rows':0,
       'strategy_summary':{}}
    for strategy in sorted({r['strategy'] for r in rank}):
        x=[r for r in rank if r['strategy']==strategy]
        nonabstain=[r for r in x if r['top1_status']!='TIE_ABSTAIN']
        pred=[r for r in x if r['all_raw_predictions_physically_valid']!='']
        summary['strategy_summary'][strategy]={
           'groups':len(x),'top1_correct':sum(r['top1_status']=='CORRECT' for r in x),
           'top1_evaluable':len(nonabstain),'top2_contains_actual':sum(r['top2_contains_actual'] is True for r in x),
           'top2_evaluable':sum(r['top2_status']=='EVALUABLE' for r in x),
           'mean_regret_mg_g':mean(float(r['regret_mg_g']) for r in nonabstain) if nonabstain else None,
           'mean_pairwise_score_ties_half':mean(r['pairwise_score_ties_half'] for r in x),
           'groups_physically_valid':sum(r['all_raw_predictions_physically_valid'] is True for r in pred),
           'group_count_for_physical_check':len(pred)}
    panel_cols=list(panel[0].keys());rank_cols=list(rank[0].keys());candidate_cols=list(candidates[0].keys())
    write_csv(root/DOUT/'v35_real_source_temperature_panel15_RESEARCH_ONLY.csv',panel,panel_cols)
    write_csv(root/ROUT/'v35_ranking_LODO2_DOI4groups.csv',rank,rank_cols)
    write_csv(root/ROUT/'v35_candidate_prediction_audit.csv',candidates,candidate_cols)
    write_csv(root/DOUT/'v35_graph_and_target_qa_queue.csv',qc,list(qc[0].keys()))
    (root/ROUT).mkdir(parents=True,exist_ok=True)
    (root/ROUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
