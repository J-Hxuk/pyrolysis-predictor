"""Project-local read-only v60 regression and leakage guards."""
from pathlib import Path
import numpy as np,pandas as pd
from scripts.run_v60_separate_effects import panel_inventory,concentration_inventory,pairwise,rank_stats
from scripts.run_v60_separate_effects import OLD,PRE,NEW,REPORT
from scripts.audit_v60_effects import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1]

def test_original_datasets_unchanged_and_disjoint():
    old=pd.read_csv(ROOT/OLD);v152=pd.read_csv(ROOT/PRE);new=pd.read_csv(ROOT/NEW)
    assert (len(old),len(v152),len(new))==(140,152,36)
    assert not (set(v152.doi)&set(new.doi))

def test_exact_context_not_conflated_with_missing():
    d=pd.concat([pd.read_csv(ROOT/PRE),pd.read_csv(ROOT/NEW)],ignore_index=True,sort=False)
    pp,p=panel_inventory(d);cc,c=concentration_inventory(d)
    assert len(p)==15 and p.doi.nunique()==9 and len(c)==5 and c.doi.nunique()==2
    assert p.n_missing_ads_context.gt(0).sum()==11
    assert set(pp.loc[pp.panel_id>=0].doi)==set(p.doi)

def test_partial_top_ties_not_fake_success():
    v=np.array([3.,2.,1.]); scores=np.array([10.,10.,7.]);
    r=pd.DataFrame({'panel_id':[0,0,0],'doi':['d']*3,'target_value':v,'s':scores,'feedstock_label_normalized':['x']*3})
    out=rank_stats(r,'s','manufacturing','test')
    assert out[0]['n_tied_top']==2 and not out[0]['unique_top1']
    assert np.isclose(out[0]['top1_regret_worst'],1)

def test_actual_predictions_and_source_folds_disjoint():
    pred=pd.read_csv(ROOT/REPORT/'all188_doi_oof_predictions.csv');fold=pd.read_json(ROOT/REPORT/'doi_folds.json')
    assert len(pred)==188 and len(fold)==5 and pred.observation_id.is_unique
    assert pred.groupby('doi').fold.nunique().eq(1).all()
    for col in ['152__A_row','152__B_doi_material','188__B_doi_material','188__mfg_forest']:
        assert np.isfinite(pred[col]).all()

def test_valid_ranking_scores_and_doigroup_bootstrap():
    scores=pd.read_csv(ROOT/REPORT/'per_panel_rank_results.csv')
    z=scores[(scores.scenario==188)&(scores.panel_type=='manufacturing')]
    ci=paired_bootstrap(z,n_iter=100)
    assert ci['n_panels']==15 and ci['n_doi']==9
    assert -2<=ci['ci95_low']<=ci['ci95_high']<=2
    assert scores.concordance.dropna().between(0,1).all()

def test_cap_never_violated_after_clipping():
    p=pd.read_csv(ROOT/REPORT/'all188_doi_oof_predictions.csv')
    cap=p.C0_mg_L/p.dose_g_L
    for name in ['152__A_row','152__B_row','152__B_doi_material','188__A_row','188__B_row','188__B_doi_material']:
        assert p[name].between(-1e-8,cap+1e-8).all()
