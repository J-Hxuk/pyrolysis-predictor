"""Evidence & reproducibility contracts, not hard-coded environment-specific MAE."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/imports/v55_cumulative_sources'
REPORT=ROOT/'reports/v55_cumulative_source_refit'
DEP=ROOT/'deployment/research_v55_NOT_ACTIVE'


def test_five_original_sources_archived_with_hash():
 import hashlib
 d=pd.read_csv(DATA/'Source_Registry_5.csv')
 assert len(d)==5 and d.doi.is_unique
 for r in d.itertuples(index=False):
  f=ROOT/'data/imports/v55_original_sources/pdfs'/r.source_file
  assert f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==r.sha256


def test_target_typing_and_mass_balance():
 d=pd.read_csv(DATA/'New_Reported_QE_8.csv')
 assert len(d)==8 and d.observation_id.is_unique
 assert (d.target_value>=0).all()
 assert (d.target_value<=d.C0_mg_L/d.dose_g_L+1e-9).all()
 assert (d.source_key=='sugar').sum()==4
 assert (d.source_key=='cocoa').sum()==4
 assert d.loc[d.source_key=='cocoa','C0_condition_status'].str.contains('INFERRED').all()
 assert 'qmax' not in ''.join(d.target_name).lower()


def test_separate_fitted_qmax_and_unit_conflict():
 q=pd.read_csv(DATA/'New_Fitted_Qmax_16.csv')
 assert len(q)==16 and q.observation_id.is_unique
 assert (q.target_name=='Langmuir_qmax_fitted_mg_g').all()
 assert all('FITTED' in s for s in q.fit_origin)
 ambiguous=pd.read_csv(DATA/'Sludge_14_QMAX_UNRESOLVED_UNIT_SOURCE_LEDGER.csv')
 assert len(ambiguous)==14
 assert (ambiguous.target_units_as_printed=='mg kg-1').all()
 assert not set(ambiguous.observation_id)&set(q.observation_id)


def test_staged_properties_all_five_doi():
 p=pd.read_csv(DATA/'New_Stage1_Properties_29.csv')
 assert len(p)==29 and p.material_id.is_unique and p.doi.nunique()==5
 assert int(p.BET_m2_g_training_candidate.notna().sum())==29
 assert int(p.TPV_cm3_g_training_candidate.notna().sum())==10
 assert p.loc[p.source_key=='rice','BET_m2_g_training_candidate'].notna().sum()==8
 assert p.loc[p.source_key=='sludge','BET_m2_g_training_candidate'].notna().sum()==7


def test_stage2_no_overlap_or_source_duplication():
 d=pd.read_csv(DATA/'Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv')
 assert len(d)==152 and d.observation_id.is_unique
 assert d.doi.nunique()==34
 assert (d.doi=='10.1016/j.chemosphere.2013.12.042').sum()==4


def test_metrics_recomputed_from_saved_oof():
 df=pd.read_csv(REPORT/'stage2_same140_group_oof_predictions.csv')
 m=pd.read_csv(REPORT/'stage2_same140_group_oof_metrics.csv')
 assert len(df)==140 and df.observation_id.is_unique
 for r in m.itertuples(index=False):
  predictions=df[f'{r.model}__{r.scenario}'].to_numpy(float)
  actual=df.target_value.to_numpy(float)
  assert np.isfinite(predictions).all()
  assert abs(mean_absolute_error(actual,predictions)-r.mae)<1e-8


def test_manifest_protects_active_deployment():
 m=json.loads((DEP/'manifest_v55.json').read_text(encoding='utf8'))
 assert m['status']=='RESEARCH_ONLY_NOT_ACTIVE'
 assert len(m['models'])==12
 assert (ROOT/'deployment/active.json').exists()
 for a in m['models']:
  assert 'research_v55_NOT_ACTIVE' in a['path']
  assert (ROOT/a['path']).exists()


def test_qmax_not_improved_and_challenge_caution_captured():
 scores=pd.read_csv(REPORT/'qmax_same32_group_oof_metrics.csv')
 assert scores.iloc[0].mae < scores.iloc[1].mae
 dc=pd.read_csv(REPORT/'stage2_new_doi_pre_inclusion_challenge.csv')
 assert set(dc.source_key) if 'source_key' in dc else set(dc.source)=={'sugar','cocoa'}
