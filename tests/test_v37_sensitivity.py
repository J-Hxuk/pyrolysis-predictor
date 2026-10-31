import unittest
import math
import numpy as np
import pandas as pd
from pyrolysis.biochar_stage2.v37_sensitivity import (
    VARIABLES, eval_one, sensitivity_frame, make_sweep_values)

class FakePipeline:
    def predict(self, X):
        return np.array([float(X.iloc[0]['C0_mg_L']) / (2 * float(X.iloc[0]['dose_g_L']))])

class InvalidPipeline:
    def predict(self, X):
        return np.array([-10.0])

class TestV37Sensitivity(unittest.TestCase):
    def setUp(self):
        self.A={'metadata':{'input_cols':['C0_mg_L','dose_g_L']},'pipeline':FakePipeline()}
        self.B={'metadata':{'input_cols':['C0_mg_L','dose_g_L']},'pipeline':InvalidPipeline()}
        self.row={'C0_mg_L':100.,'dose_g_L':2.,'initial_pH':5.0}

    def test_models_differ_and_physical_flags_are_visible(self):
        res=sensitivity_frame({'A':self.A,'B':self.B},self.row,
                              variable='C0_mg_L',values=[20,100],show_model_b=True)
        self.assertEqual(len(res),4)
        self.assertEqual(res[res.model=='A'].qe_raw.tolist(),[5,25])
        self.assertTrue(res.loc[res.model=='B','violates'].all())
        self.assertEqual(res.loc[res.model=='B','qe_raw'].tolist(),[-10,-10])
        self.assertTrue((res.loc[res.model=='B','qe_clipped']==0).all())
        self.assertTrue(math.isclose(res.iloc[-1]['cap'],50.0))

    def test_no_unvalidated_model_b_on_unknown_feedstock(self):
        res=sensitivity_frame({'A':self.A,'B':self.B},self.row,
                              variable='initial_pH',values=[3,5],show_model_b=False)
        self.assertEqual(res.model.unique().tolist(),['A'])

    def test_sweep_includes_selected_value_in_range(self):
        vals=make_sweep_values('initial_pH',2,8,5.13,n=7)
        self.assertIn(5.13,vals)
        self.assertEqual(vals,sorted(set(vals)))

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):
            make_sweep_values('bad_name',0,1,0.5)
        with self.assertRaises(ValueError):
            make_sweep_values('initial_pH',5,2,4)

    def test_mass_balance_validation(self):
        with self.assertRaises(ValueError):
            eval_one(self.A,{**self.row,'dose_g_L':0})

if __name__=='__main__':unittest.main()
