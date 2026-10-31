import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_v33_five_papers import stage1_rows,stage2_measured_rows,stage2_leachate_rows,fitted_qmax_rows,build

class V33Tests(unittest.TestCase):
    def test_30_materials_29_bet(self):
        rows=stage1_rows();self.assertEqual(len(rows),30);self.assertEqual(sum(r['BET_m2_g'] is not None for r in rows),29)
    def test_distinct_material_ids(self):
        rows=stage1_rows();self.assertEqual(len(set(r['material_id'] for r in rows)),30)
    def test_plb600_missing(self):
        r=next(x for x in stage1_rows() if x['sample_name']=='PLB600');self.assertIsNone(r['BET_m2_g']);self.assertIsNone(r['pore_volume_cm3_g'])
    def test_jia_micro_not_tpv(self):
        self.assertEqual(sum(r['pore_volume_type']=='MICROPORE_VOLUME_NOT_TPV' for r in stage1_rows()),10)
    def test_product_composition_not_input(self):
        self.assertTrue(all(not r['feedstock_C_pct'] and not r['feedstock_FC_pct'] for r in stage1_rows()))
    def test_measured_15_mass_balance(self):
        rows=stage2_measured_rows();self.assertEqual(len(rows),15)
        for r in rows:self.assertGreaterEqual(r['qe_mg_g'],0);self.assertLessEqual(r['qe_mg_g'],r['qe_physical_cap_mg_g'])
    def test_wang_170mg_max_at_500(self):
        rows=[r for r in stage2_measured_rows() if r['source_id']=='WANG19' and r['C0_mg_L']==170]
        self.assertEqual(max(rows,key=lambda r:r['qe_mg_g'])['sample_name'],'MSB500')
    def test_hu_max_at_400(self):
        rows=[r for r in stage2_measured_rows() if r['source_id']=='HU22'];self.assertEqual(max(rows,key=lambda r:r['qe_mg_g'])['sample_name'],'BC400')
    def test_no_same_qmax(self):
        self.assertEqual(len(fitted_qmax_rows()),23);self.assertFalse(any(r['is_fitted_qmax']=='YES' for r in stage2_measured_rows()))
    def test_leachate_separation(self):
        rows=stage2_leachate_rows();self.assertEqual(len(rows),3)
        self.assertTrue(all(r['quality_status']=='SEPARATE_DOMAIN_NOT_CORE' for r in rows))
    def test_build(self):
        s=build();self.assertEqual(s['stage2_training_approved_rows'],0);self.assertEqual(s['source_papers'],5)

if __name__=='__main__':unittest.main()
