"""v32 research-only temperature identifiability, feature ablation, and response audit.

Reuses the 98-row Core cohort, never the v30 graph digitization as training data.
Prespecified fixed Ridge/XGB recipes from v16; no hyperparameter optimization.
Model B feature removal is evaluated in identical LODO DOI folds.
DO NOT interpret predicted temperature curves as causal effects or deploy a ranker.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.compare_phase8_v23 import stage2_targets
from scripts.evaluate_v27_recommendation import extract_comparable_groups, tie_aware_accuracy, expected_top1_and_regret
from scripts.evaluate_v31_rice_temperature_external import (
    HELDOUT_DOI, load_external_unmodified, model_input_rows,
)
from pyrolysis.biochar_stage2.v15_models import build_pipeline
from pyrolysis.biochar_stage2.v15_features import model_inputs
from pyrolysis.biochar_stage2.v16_validate import validation_algorithms

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/v32_temperature_identifiability'
TEMP = 'pyrolysis_temperature_C'
CATEGORICAL = {'pollutant','feedstock_label_normalized','manufacturing_route','pyrolysis_atmosphere','activation_or_modification'}
ALGORITHMS = ('Ridge','XGBoost')
SEED = 20261030


def cleaned_inputs(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x=df.reindex(columns=columns).copy()
    for col in columns:
        if col in CATEGORICAL:
            x[col]=x[col].astype('object')
            x[col]=x[col].where(x[col].notna() & x[col].astype(str).str.strip().ne(''),np.nan)
        else:
            x[col]=pd.to_numeric(x[col],errors='coerce')
    return x


def train_predict(train:pd.DataFrame, test:pd.DataFrame, algorithm:str, features:list[str]) -> np.ndarray:
    if algorithm not in ALGORITHMS:
        raise ValueError('Only declared recipes Ridge/XGBoost')
    if any(f not in train.columns or f not in test.columns for f in features):
        raise ValueError('Required feature is missing')
    estimator=validation_algorithms()[algorithm]()
    pipeline=build_pipeline(estimator,features)
    yy=pd.to_numeric(train.target_value,errors='raise').to_numpy(dtype=float)
    pipeline.fit(cleaned_inputs(train,features),yy)
    return np.asarray(pipeline.predict(cleaned_inputs(test,features)),dtype=float)


def temperature_design_audit(core:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    temp=pd.to_numeric(core[TEMP],errors='coerce')
    a=core.assign(_numeric_temperature=temp)
    table=(a.groupby('doi',dropna=False)
           .agg(n_rows=('doi','size'),n_reported=('_numeric_temperature','count'),
                n_distinct_temperatures=('_numeric_temperature','nunique'),
                min_temperature_C=('_numeric_temperature','min'),
                max_temperature_C=('_numeric_temperature','max'))
           .reset_index())
    same=extract_comparable_groups(core)
    if not same.empty:
        ng=(same.assign(_numeric_temperature=pd.to_numeric(same[TEMP],errors='coerce'))
             .groupby('comparison_id')._numeric_temperature.nunique())
        temp_pair_group_count=int((ng>1).sum())
    else:temp_pair_group_count=0
    audit={'total_observations':int(len(core)),'doi_count':int(core.doi.nunique()),
           'reported_temperature_observations':int(temp.notna().sum()),
           'unreported_temperature_observations':int(temp.isna().sum()),
           'doi_with_more_than_one_reported_temperature':int((table.n_distinct_temperatures>1).sum()),
           'strict_adsorption_identical_groups':int(same.comparison_id.nunique()),
           'strict_temperature_contrast_groups':temp_pair_group_count,
           'note':'DOI-constant temperature makes within-paper temperature effects unidentifiable on Core98.'}
    return table,audit


def feature_ablation_lodo(core:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    full=model_inputs('B')
    reduced=[f for f in full if f!=TEMP]
    if len(full)!=16 or len(reduced)!=15:
        raise ValueError('Unexpected Model B feature contract')
    out=[]
    for doi in sorted(core.doi.unique()):
        train=core.loc[core.doi.ne(doi)]
        test=core.loc[core.doi.eq(doi)]
        assert doi not in set(train.doi)
        for algo in ALGORITHMS:
            for setting,features in [('with_temperature',full),('no_temperature',reduced)]:
                pred=train_predict(train,test,algo,features)
                for i,(_,r) in enumerate(test.iterrows()):
                    cap=float(r.C0_mg_L)/float(r.dose_g_L)
                    out.append(dict(doi=doi,observation_id=r.observation_id,algorithm=algo,setting=setting,
                                    y=float(r.target_value),pred_raw=float(pred[i]),
                                    physical_cap=cap,violation=bool(pred[i]<0 or pred[i]>cap)))
    long=pd.DataFrame(out)
    if long.duplicated(['observation_id','algorithm','setting']).any():
        raise AssertionError('Duplicate OOF score')
    pivot=(long.pivot(index=['doi','observation_id','algorithm','y','physical_cap'],columns='setting',values='pred_raw').reset_index())
    pivot.columns.name=None
    if len(pivot)!=98*len(ALGORITHMS):
        raise AssertionError('Paired OOF model rows incomplete')
    summary={}
    for algo,sub in pivot.groupby('algorithm'):
        y=sub.y.to_numpy()
        for label in ['with_temperature','no_temperature']:
            p=sub[label].to_numpy()
            summary.setdefault(algo,{})[label]={
                'mae_mg_g':float(mean_absolute_error(y,p)),
                'r2':float(r2_score(y,p)),
                'raw_physical_violations':int(((p<0)|(p>sub.physical_cap.to_numpy())).sum())}
        delta=(abs(y-sub.no_temperature.to_numpy())-abs(y-sub.with_temperature.to_numpy()))
        summary[algo]['delta_no_temp_minus_with_temp_mae_mg_g']=float(delta.mean())
        summary[algo]['doi_bootstrap_ci95']=doi_bootstrap_mae_delta(sub)
    return long,pivot,summary


def doi_bootstrap_mae_delta(df:pd.DataFrame, reps:int=4000,seed:int=SEED) -> list[float]:
    """Resample held-out DOI clusters, accounting for sample counts per DOI."""
    keys=sorted(df.doi.unique())
    groups={k:df[df.doi.eq(k)] for k in keys}
    rng=np.random.default_rng(seed)
    draws=[]
    for i in range(reps):
        picked=rng.choice(keys,size=len(keys),replace=True)
        indices=[groups[k] for k in picked]
        block=pd.concat(indices,ignore_index=True)
        y=block.y.to_numpy()
        draws.append(float(np.mean(abs(y-block.no_temperature.to_numpy())-abs(y-block.with_temperature.to_numpy()))))
    return [float(z) for z in np.quantile(draws,[.025,.975])]


def response_curves(core:pd.DataFrame, evidence:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Full Core98 trained test curves -- diagnostic ONLY, not external validation ranker."""
    if HELDOUT_DOI in set(core.doi.astype(str)):
        raise ValueError('External graph source DOI leaked into training')
    if evidence.doi.nunique()!=1 or set(evidence.doi)!={HELDOUT_DOI}:
        raise ValueError('External source evidence provenance changed')
    inp=model_input_rows(evidence)
    temperatures=list(range(300,501,25))
    curve_rows=[]
    for pol,sub in inp.groupby('pollutant',sort=True):
        reference=sub.iloc[[0]].copy()
        test=pd.concat([reference]*len(temperatures),ignore_index=True)
        test[TEMP]=temperatures
        # Add truth placeholder, never fit it -- input contract only.
        for algo in ALGORITHMS:
            pred=train_predict(core.assign(),test.assign(target_value=np.nan),algo,model_inputs('B'))
            cap=float(test.C0_mg_L.iloc[0])/float(test.dose_g_L.iloc[0])
            for t,p in zip(temperatures,pred):
                curve_rows.append({'pollutant':pol,'algorithm':algo,'pyrolysis_temperature_C':t,
                                   'pred_raw_mg_g':float(p),'cap_mg_g':cap,
                                   'raw_physical_violation':bool(p<0 or p>cap)})
    curves=pd.DataFrame(curve_rows)
    observations=evidence[['pollutant','pyrolysis_temp_C','qe_reported_approx_mg_g','digitization_tolerance_mg_g']].copy()
    curve_qa=[]
    for (pol,algo),g in curves.groupby(['pollutant','algorithm'],sort=True):
        p=g.pred_raw_mg_g.to_numpy();t=g.pyrolysis_temperature_C.to_numpy()
        dp=np.diff(p)
        relevant=g[g.pyrolysis_temperature_C.isin([300,400,500])]
        y=(observations.loc[observations.pollutant.eq(pol)].sort_values('pyrolysis_temp_C')
           .qe_reported_approx_mg_g.to_numpy())
        p3=relevant.sort_values('pyrolysis_temperature_C').pred_raw_mg_g.to_numpy()
        top1,regret,_=expected_top1_and_regret(y,p3)
        pairwise, npairs,nties=tie_aware_accuracy(y,p3)
        curve_qa.append({'pollutant':pol,'algorithm':algo,'curve_is_nondecreasing':bool(np.all(dp>=-1e-9)),
                         'curve_is_nonincreasing':bool(np.all(dp<=1e-9)),
                         'curve_has_interior_strict_local_maximum':bool(any(dp[i-1]>1e-9 and dp[i]<-1e-9 for i in range(1,len(dp)))),
                         'n_equal_adjacent_grid_steps':int((abs(dp)<1e-9).sum()),
                         'predicted_best_C_300_400_500':int(relevant.sort_values('pyrolysis_temperature_C').iloc[np.argmax(p3)].pyrolysis_temperature_C) if np.ptp(p3)>1e-9 else None,
                         'observed_best_C':int(observations.loc[observations.pollutant.eq(pol)].sort_values('pyrolysis_temp_C').iloc[y.argmax()].pyrolysis_temp_C),
                         'top1_expected_3_temps':top1,'regret_mg_g':regret,
                         'pairwise':pairwise,'pairwise_pairs':npairs,'pred_ties':nties,
                         'n_grid_raw_violations':int(g.raw_physical_violation.sum())})
    qa=pd.DataFrame(curve_qa)
    return curves,qa,{'grid_min':min(temperatures),'grid_max':max(temperatures),'grid_step':25,
                     'evaluation_points':[300,400,500],
                     'interpretation':'This is a diagnostic temperature response for a fixed external recipe, not learned within-paper temperature causality.'}


def run(root:Path=ROOT,out:Path=REPORT) -> dict:
    root,out=Path(root),Path(out)
    _,core=stage2_targets()
    if len(core)!=98 or core.doi.nunique()!=26:
        raise RuntimeError('Core98 cohort changed; rerun audit before comparing')
    doi_table,design=temperature_design_audit(core)
    loo,pivot,ablation=feature_ablation_lodo(core)
    evidence=load_external_unmodified(root/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv',
                                      root/'reports/v30_graph_evidence/rice_figure9_stage1_linked_18_QA_NOT_TRAINED.csv')
    curves,qa,curve_meta=response_curves(core,evidence)
    out.mkdir(parents=True,exist_ok=True)
    doi_table.to_csv(out/'temperature_reported_by_doi.csv',index=False)
    loo.to_csv(out/'ablation_oof_long.csv',index=False)
    pivot.to_csv(out/'ablation_paired_oof.csv',index=False)
    curves.to_csv(out/'rice_temperature_counterfactual_curve_RESEARCH_ONLY.csv',index=False)
    qa.to_csv(out/'rice_temperature_curve_qa.csv',index=False)
    result={'purpose':'v32 temperature contribution and rank learnability; research-only',
            'source_training':'Core98/26DOI; no v24 external or v30 markers included',
            'external_evidence':'v30 rice graph qt approximations, held out, NOT certified equilibrium qe',
            'design':design,'ablation':ablation,'temperature_curves':curve_meta,
            'ranking_training_feasible':design['strict_temperature_contrast_groups']>=2 and design['doi_with_more_than_one_reported_temperature']>=2,
            'released_model':False,'app_changed':False,
            'hard_limitations':[
                'No within-DOI temperature contrasts in Core98; between-paper temperature feature is confounded with DOI/feedstock/route.',
                'A no-temperature ablation can assess prediction association only; cannot establish causal role of temperature.',
                'No legitimate supervised temperature pairwise ranker can be validated with this cohort.',
                'External one DOI/2 correlated pollutant series cannot estimate generalization accuracy.',
                'DOI cluster bootstrap is exploratory because ablation choices were motivated by earlier failures.',
                'Model B ridge and XGBoost are fixed research recipes refit Core98, not deployed artifacts.',
                'Raw physical constraint violations remain uncorrected in scores.'
            ]}
    result['rice_curve_diagnostics']=qa.to_dict(orient='records')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return result

if __name__=='__main__':
    result=run()
    print(json.dumps({'design':result['design'],'ablation':result['ablation'],
                      'ranking_training_feasible':result['ranking_training_feasible'],
                      'rice_curve_diagnostics':result['rice_curve_diagnostics']},ensure_ascii=False,indent=2))
