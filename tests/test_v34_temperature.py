import unittest
from scripts.build_v34_temperature import (DOI,doeraw,dedup_doe,new_bet,cedar_qmax,audit_source_registry)
class V34Contracts(unittest.TestCase):
 def test_DOI_registry_no_repetition(self):
  d=audit_source_registry();self.assertEqual(d.doi.nunique(),5)
  self.assertEqual((d.role=='DUPLICATE_EXCLUDE').sum(),1)
 def test_no_qt_qe_mixing(self):
  d=doeraw();self.assertEqual(len(d),34)
  self.assertEqual(set(d.target_type),{'QT_240MIN_NOT_CERTIFIED_QE'})
  self.assertTrue((d.qt_240min_mg_g<=d.physical_cap_mg_g).all())
 def test_repeated_center_collapse(self):
  d=dedup_doe(doeraw());self.assertEqual(len(d),26)
  self.assertEqual(d.n_replicates.sum(),34)
  self.assertEqual(int((d.n_replicates==5).sum()),2)
 def test_cedar_not_measured_qe(self):
  d=cedar_qmax();self.assertEqual(len(d),20);self.assertTrue(d.target_type.str.contains('NOT_QE').all());self.assertTrue((d.fit_r2<.8).any())
 def test_bet_source_target(self):
  d=new_bet();self.assertEqual(len(d),6)
  self.assertTrue(d.TPV_cm3_g.isna().all())
  self.assertEqual(d.doi.nunique(),2)
if __name__=='__main__':unittest.main()
