import importlib.util
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

BASE=Path(__file__).resolve().parents[1]
P=BASE/'scripts'/'build_v36_temperature.py'
spec=importlib.util.spec_from_file_location('v36',P)
v36=importlib.util.module_from_spec(spec);spec.loader.exec_module(v36)
P35=BASE/'data/imports/v36_temperature/v36_reference_v35_source_panel15.csv'
P36=BASE/'data/imports/v36_temperature/v36_new_graph13_research_only.csv'

class V36Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not P35.exists():raise unittest.SkipTest('v35 dependency missing; apply v35 first')
  cls.panel=v36.load_panel(P35,P36)
  cls.ranks,cls.pairs=v36.evaluate(cls.panel)

 def test_counts(self):
  self.assertEqual(len(self.panel),28)
  self.assertEqual(self.panel.doi.nunique(),4)
  self.assertEqual(self.panel.group_id.nunique(),7)

 def test_digitized_new_sources_exactly_13(self):
  q=self.panel[self.panel.approval!='v35_research_only']
  self.assertEqual(len(q),13)
  self.assertEqual(set(q.doi),{v36.DOI_JIA,v36.DOI_SHEN})

 def test_old15_preserved(self):
  old=pd.read_csv(P35,encoding='utf-8-sig')
  actual=self.panel[self.panel.approval=='v35_research_only']
  self.assertEqual(len(actual),15)
  aa=dict(zip(actual.material_id.astype(str)+'/'+actual.C0_mg_L.astype(str),actual.qe_mg_g))
  for r in old.itertuples():
   self.assertAlmostEqual(aa[str(r.material_id)+'/'+str(float(r.C0_mg_L))],float(r.qe_mg_g))

 def test_mass_balance(self):
  for r in self.panel.itertuples():
   self.assertGreaterEqual(r.qe_mg_g,0)
   self.assertLessEqual(r.qe_mg_g,r.mass_balance_cap_mg_g + r.uq_mg_g if not np.isnan(r.uq_mg_g) else r.mass_balance_cap_mg_g)

 def test_no_fitted_qmax(self):
  self.assertFalse(self.panel.source_type.str.contains('FIT|QMAX|CAL',case=False).any())
  self.assertTrue((self.panel.analysis_status.str.contains('NOT_TRAIN_APPROVED')).all())

 def test_non_source_lv_held_out(self):
  hold=pd.read_csv(BASE/'data/imports/v36_temperature/v36_lv_fig2_three_points_HOLD.csv',encoding='utf-8-sig')
  self.assertEqual(len(hold),3)
  self.assertTrue(hold.status.str.contains('DO_NOT_TRAIN').all())
  self.assertTrue(hold.doi.isin(self.panel.doi).sum()==0)

 def test_no_doi_leak(self):
  for r in self.ranks.itertuples():
   self.assertNotIn(r.heldout_doi,r.training_dois.split(';'))
   self.assertGreaterEqual(r.training_doi_count,2)

 def test_old2_not_sufficient_for_strict_pairwise(self):
  old=self.panel[~self.panel.doi.isin([v36.DOI_JIA,v36.DOI_SHEN])]
  self.assertEqual(v36.confident_pairs(old).doi.nunique(),1)
  self.assertIsNone(v36.fit_pairwise(old,leave_out_doi='not_a_DOI'))

 def test_jia_700_600_unresolved(self):
  for g in ['TG05','TG06']:
   ff=self.panel[self.panel.group_id==g].sort_values('qe_mg_g',ascending=False)
   self.assertLessEqual(float(ff.iloc[0].qe_mg_g-ff.iloc[1].qe_mg_g),float(ff.iloc[0].uq_mg_g+ff.iloc[1].uq_mg_g))

 def test_shen_units(self):
  f=self.panel[self.panel.group_id=='TG07']
  self.assertEqual(f.temperature_C.tolist(),[300.,500.,700.])
  self.assertTrue(np.allclose(f.mass_balance_cap_mg_g,207.2))
  self.assertTrue(f.pH_reported.isna().all())

 def test_loo_count(self):
  self.assertEqual(len(self.ranks),7*3)
  self.assertEqual(self.ranks.heldout_doi.nunique(),4)

if __name__=='__main__':unittest.main()
