"""v38 citable source/target integrity; run with v37 project and v38 staged data."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scripts.build_v38_new_dois import combine, paired_old_oof, S1_NUM,S1_CAT,S1_TARGETS

ROOT=Path(__file__).resolve().parents[1]

class V38SourceQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b1,cls.n1,cls.s1,cls.b2,cls.n2,cls.s2=combine(ROOT)
    def test_input_cardinality(self):
        self.assertEqual((len(self.b1),len(self.n1),len(self.s1)),(333,18,351))
        self.assertEqual((len(self.b2),len(self.n2),len(self.s2)),(113,12,125))
    def test_source_no_duplicate_doi(self):
        self.assertFalse(set(self.b1.doi)&set(self.n1.doi))
        self.assertFalse(set(self.b2.doi)&set(self.n2.doi))
        self.assertTrue(self.n1.material_id.is_unique)
        self.assertTrue(self.n2.observation_id.is_unique)
    def test_bet_tpv_cardinality(self):
        self.assertEqual(self.s1[S1_TARGETS['BET']].notna().sum(),313)
        self.assertEqual(self.s1[S1_TARGETS['TPV']].notna().sum(),205)
    def test_same_qe_definition_only(self):
        self.assertEqual(set(self.n2.target_name),{'qe_reported_mg_g'})
        self.assertFalse(self.n2.source_target_status.str.contains('CERTIFIED',na=False).any())
        self.assertTrue(self.n2.initial_pH.isna().all())
    def test_qe_c0_mass_balance(self):
        self.assertTrue((self.s2.target_value>=0).all())
        self.assertTrue((self.s2.target_value<=self.s2.C0_mg_L/self.s2.dose_g_L+1e-5).all())
    def test_sugarcane_separate_domain(self):
        mix=pd.read_csv(ROOT/'data/imports/v38_new_dois/bagasse_multimetal_6_SEPARATE_DOMAIN.csv')
        self.assertEqual(len(mix),6)
        self.assertTrue((mix.converted_uptake_mg_g<=0.25).all())
        self.assertTrue(set(mix.experiment_matrix)=={'SIX_METAL_SIMULTANEOUS'})
    def test_soybean_duplicate_excluded(self):
        self.assertNotIn('10.1016/j.arabjc.2024.105946',set(self.n1.doi))
        self.assertIn('10.1016/j.arabjc.2024.105946',set(self.b1.doi))
    def test_not_use_postchar_as_feedstock(self):
        for col in ['feedstock_C_pct_original','feedstock_H_pct_original','feedstock_O_pct_original']:
            self.assertTrue(self.n1.reindex(columns=[col])[col].isna().all())
    def test_no_fitted_targets_in_qe(self):
        self.assertTrue(self.n2.target_origin.astype(str).eq('MAIN_TEXT_EQUILIBRIUM_UPTAKE_NOT_KINETIC_FIT').all())
        self.assertEqual(set(self.n2.doi),{'10.1016/j.chemosphere.2022.134051'})
    def test_paired_folds_only_old_doi(self):
        sub,y,p0,p1=paired_old_oof(self.b1,self.n1,S1_TARGETS['BET'],S1_NUM,S1_CAT,'Ridge')
        self.assertEqual(len(sub),299)
        self.assertTrue(np.isfinite(p0).all() and np.isfinite(p1).all())
        self.assertTrue(set(sub.doi).isdisjoint(set(self.n1.doi)))

if __name__=='__main__':unittest.main()
