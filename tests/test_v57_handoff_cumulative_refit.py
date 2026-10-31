"""v57 release integrity without assuming fixed sklearn-dependent floating-point MAE."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/imports/v57_handoff_sources'
R=ROOT/'reports/v57_handoff_cumulative_refit'
DEP=ROOT/'deployment/research_v57_NOT_ACTIVE'

def test_source36_is_preserved_and_physically_feasible():
 raw=pd.read_csv(D/'Kiro_approved_experimental_qe_rows_SOURCE_ORIGINAL.csv')
 assert len(raw)==36
 assert raw.groupby('source_doi').size().to_dict()=={'10.1007/s11356-017-8847-2':29,'10.1016/j.scitotenv.2019.07.102':7}
 assert (raw.qe_mg_g>=0).all()
 assert (raw.qe_mg_g<=raw.C0_mg_L/raw.dose_g_L+1e-6).all()
 assert not raw.duplicated(['source_doi','material_id','pollutant','C0_mg_L','dose_g_L']).any()

def test_cumulative_scenarios_correct_and_no_duplicate_ids():
 for n in [159,181,188,192,204,208]:
  f=list(D.glob(f'Stage2_v57_{n}_*_RESEARCH_NOT_ACTIVE.csv'))
  assert len(f)==1,n
  d=pd.read_csv(f[0],low_memory=False)
  assert len(d)==n and d.observation_id.is_unique
  assert (d.target_value<=d.C0_mg_L/d.dose_g_L+1e-6).all()
  assert len(d[d.doi.isin(['10.1007/s11356-017-8847-2','10.1016/j.scitotenv.2019.07.102'])])=={159:7,181:29,188:36,192:36,204:36,208:36}[n]

def test_stage1_four_materials_not29_duplicate_BET():
 p=pd.read_csv(D/'Stage1_New4_BET_RESEARCH.csv')
 allp=pd.read_csv(D/'Stage1_450_RESEARCH_NOT_ACTIVE.csv',low_memory=False)
 assert len(p)==4 and p.material_id.is_unique and p.TPV_cm3_g_training_candidate.isna().all()
 assert len(allp)==450
 assert p.BET_m2_g_training_candidate.notna().sum()==4

def test_fixed140_oof_metrics_match_recalculation():
 m=pd.read_csv(R/'stage2_fixed140_sameDOI_metrics.csv')
 p=pd.read_csv(R/'stage2_fixed140_sameDOI_predictions.csv')
 assert len(p)==140 and p.doi.nunique()==32
 for r in m.itertuples(index=False):
  predicted=p[f'{r.model}__{r.scenario}']
  actual=mean_absolute_error(p.target_value,predicted)
  assert abs(actual-r.mae_mg_g)<1e-8
  assert np.isfinite(predicted).all()
  assert ((predicted>=0)&(predicted<=p.C0_mg_L/p.dose_g_L+1e-8)).all()

def test_new_source_predictions_do_not_use_new_source_in_training():
 s=pd.read_csv(R/'stage2_new_DOI_sourceout_metrics.csv')
 assert set(s.source)=={'shen','cao'} and len(s)==8
 assert all(s.protocol=='PRE_INCLUSION_SOURCE_OUT_POSTHOC_NOT_FUTURE_BLIND')
 assert (s.n>0).all()

def test_weighted_study_retains_all_36():
 w=pd.read_csv(R/'stage2_188_doi_weight_sensitivity_metrics.csv')
 assert len(w)==8 and set(w.shen_training_weight)=={1.0,.5,.25,.1}
 assert set(w.training_total_rows)=={188}
 assert set(w.shen_rows_retained)=={29}

def test_artifact_hash_and_active_not_modified():
 j=json.loads((DEP/'manifest_v57.json').read_text(encoding='utf-8'))
 assert j['active'] is False and len(j['artifacts'])==13
 for rec in j['artifacts']:
  fn=ROOT/rec['path']; assert fn.is_file()
  assert hashlib.sha256(fn.read_bytes()).hexdigest()==rec['sha256']
 assert hashlib.sha256((ROOT/'deployment/stage2_models/active.json').read_bytes()).hexdigest()==j['legacy_active_sha256']
 assert hashlib.sha256((D/'Kiro_approved_experimental_qe_rows_SOURCE_ORIGINAL.csv').read_bytes()).hexdigest()==j['source_hash_sha256']

def test_json_report_is_strict_json():
 json.loads((R/'V57_SUMMARY.json').read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
