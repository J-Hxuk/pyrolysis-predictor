"""v61 unit and local consistency tests; no fixed historical MAE / rank assertions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.run_v61_recommendation_reliability import (
    top_set, top_regret, pairwise_stats, eval_panel, evaluate_chen, run,
)


def test_tie_regret_interval_and_order_independence():
    y=[10.,5.,2.];scores=[6.,6.,1.]
    r=top_regret(y,scores)
    assert r['n_top_tied']==2
    assert r['regret_best_case']==0
    assert r['regret_worst_case']==5
    assert r['regret_mean_if_random_tiebreak']==2.5
    assert top_regret([2.,10.,5.],[1.,6.,6.])['regret_worst_case']==5


def test_predicted_ties_pairwise_get_half_credit():
    res=pairwise_stats([4.,2.,1.],[3.,3.,1.])
    assert res['n_comparable_pairs']==3
    assert res['n_predicted_tied_pairs']==1
    assert res['pairwise_concordance']==pytest.approx(2.5/3)


def test_true_ties_not_counted_as_pairs():
    res=pairwise_stats([3.,3.,1.],[2.,2.,0.])
    assert res['n_comparable_pairs']==2
    assert res['pairwise_concordance']==1.0


def synth(scores,raw=None, y=(5.,4.,3.), n=3):
    df=pd.DataFrame({'panel_id':[1]*n,'doi':['10.1234/example']*n,
         'observation_id':[f'obs_{i}' for i in range(n)],
         'target_value':list(y),'C0_mg_L':[10.]*n,'dose_g_L':[1.]*n,
         '188__B_row':scores, '188__B_row__raw':raw if raw is not None else scores})
    return df


def test_clipped_tie_not_falsely_unique():
    df=synth([10.,10.,2.],[25.,15.,2.])
    r=eval_panel(df,188,'B_row',full=True,strict=True)
    assert r['n_predicted_top_tied']==2
    assert r['n_top_raw']==1
    assert r['new_tied_pairs_after_clipping']==1
    assert r['regret_best_case_mg_g']==0
    assert r['regret_worst_case_mg_g']==1
    assert r['recommendation_gate']=='WITHHOLD_TIED_TOP'
    assert r['raw_cap_exceed_rows']==2


def test_missing_context_blocks_even_unique_top():
    df=synth([8.,7.,2.])
    r=eval_panel(df,188,'B_row',full=False,strict=True)
    assert r['n_predicted_top_tied']==1
    assert r['recommendation_gate']=='WITHHOLD_MISSING_ADSORPTION_CONTEXT'


def test_non_equilibrium_targets_not_recommended():
    df=synth([8.,6.,2.])
    r=eval_panel(df,188,'B_row',full=True,strict=False)
    assert r['recommendation_gate']=='WITHHOLD_TARGET_NOT_STRICT_QE'


def test_cap_saturation_and_margin_are_separate():
    cap_top=eval_panel(synth([10.,5.,2.]),188,'B_row',full=True,strict=True)
    assert cap_top['recommendation_gate']=='WITHHOLD_TOP_AT_MASS_BALANCE_CAP'
    small_gap=eval_panel(synth([8.,7.5,1.]),188,'B_row',full=True,strict=True)
    assert small_gap['recommendation_gate']=='WITHHOLD_SMALL_MARGIN_1MGG_HEURISTIC'


def test_raw_clipped_inconsistency_rejected():
    with pytest.raises(ValueError, match='Clipped predictions do not match'):
        eval_panel(synth([9.,8.,1.],[15.,8.,1.]),188,'B_row',full=True,strict=True)


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):top_set([1.,np.nan])


def test_chen_16_joint_top_kept():
    rows=pd.DataFrame({'material_uid':[f'M{i}' for i in range(16)],
        'reported_removal_pct':list(range(15, -1, -1)),
        'demo_score':[100.]*12+[0.]*4})
    r=evaluate_chen(rows).iloc[0]
    assert r.n_top_tied==12
    assert r.top_regret_best_pp==0
    assert r.top_regret_worst_pp==11
    assert r.recommendation_gate=='WITHHOLD_MISSING_INITIAL_PH_IN_SOURCE'


def test_all_artifacts_derived_from_same_local_oof(tmp_path):
    root=Path(__file__).resolve().parents[1]
    # Synthetic copy of the actual project's source tables, when present.
    # In the deployed v47 project this is automatically available.
    p=root/'reports/v60_separate_effects/all188_doi_oof_predictions.csv'
    if not p.exists():pytest.skip('v60 OOF file not installed')
    manifest,scores,chen=run(root,output=tmp_path)
    assert manifest['n_panels']==15
    assert len(scores)==8
    assert scores['n_tied_top'].ge(0).all()
    assert (scores.n_tied_top+scores.n_unique_top==15).all()
    detailed=pd.read_csv(tmp_path/'panel_recommendation_audit.csv')
    assert len(detailed)==15*8
    assert (detailed['regret_worst_case_mg_g']+1e-9>=detailed['regret_best_case_mg_g']).all()
    assert all(detailed.recommendation_gate.str.startswith(('WITHHOLD_','RESEARCH_')))
    assert len(chen)>0
    assert json.loads((tmp_path/'manifest_v61.json').read_text(encoding='utf-8'))['n_panels']==15
