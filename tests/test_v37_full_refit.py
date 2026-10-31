import unittest
import numpy as np
import pandas as pd
from scripts.build_v37_full_refit import (
    audit_s2, audit_s1, oof, report_metric, S1_TARGETS)

class TestV37FullRefit(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({'observation_id':['a','b','c','d','e','f'],
            'doi':['a','a','b','b','c','c'],'target_name':['qe_reported_mg_g']*6,
            'target_value':[1.,1.5,2,3,4,5], 'C0_mg_L':[20.]*6,'dose_g_L':[2.]*6,
            'initial_pH':[5]*6,'pollutant':['Cd(II)']*6})
    def test_target_mismatch_refused(self):
        d=self.sample(); d.loc[1,'target_name']='langmuir_qmax'
        with self.assertRaises(ValueError):audit_s2(d)
    def test_duplicate_observation_refused(self):
        d=self.sample(); d.loc[1,'observation_id']='a'
        with self.assertRaises(ValueError):audit_s2(d)
    def test_physical_limit_refused(self):
        d=self.sample(); d.loc[1,'target_value']=11
        with self.assertRaises(ValueError):audit_s2(d)
    def test_group_cross_validation(self):
        d=self.sample(); out=oof(d,inputs_num=['initial_pH','C0_mg_L'],
            inputs_cat=['pollutant'],target_col='target_value',algo='Dummy')
        self.assertEqual(len(out),6)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(report_metric(d.target_value.values,out)['MAE'],0)

if __name__=='__main__':unittest.main()
