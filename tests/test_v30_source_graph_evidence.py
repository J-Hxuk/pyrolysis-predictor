"""Tests cover source-graph digitization *contract*, not scientific validation of visual estimates."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load_csv(name):
    with (ROOT/'reports/v30_graph_evidence'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


class V30Contract(unittest.TestCase):
    def test_experimental_figure_rows(self):
        self.assertEqual(len(load_csv('rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv')),18)
        self.assertEqual(len(load_csv('urban_waste_fig4_ce_qe_APPROX_NOT_TRAINED.csv')),21)
        self.assertEqual(len(load_csv('sewage_sludge_fig2_removal_APPROX_DOSAGE_CONFLICT.csv')),28)

    def test_no_fitted_qmax_mixed_in_reported_qe(self):
        for name in ('rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv',
                     'urban_waste_fig4_ce_qe_APPROX_NOT_TRAINED.csv',
                     'sewage_sludge_fig2_removal_APPROX_DOSAGE_CONFLICT.csv'):
            d=load_csv(name)
            self.assertTrue(all(r['training_approved']=='False' for r in d))
            self.assertFalse(any('qmax' in r['source_measure_type'].lower() for r in d))

    def test_physical_cap_is_not_masked(self):
        d=load_csv('rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv')
        self.assertEqual(sum(r['physical_cap_pass']=='False' for r in d),5)
        self.assertEqual(sum(r['fig9_vs_table4_5_conflict']=='True' for r in d),4)
        for r in d:
            actual=float(r['qe_reported_approx_mg_g'])
            self.assertEqual(r['physical_cap_pass']=='True',0<=actual<=float(r['qe_physical_cap_mg_g']))

    def test_one_doi_two_qualified_descriptive_temperature_series(self):
        d=load_csv('rice_unmodified_within_paper_ranking_DESCRIPTIVE_ONLY.csv')
        self.assertEqual(len(d),6)
        self.assertEqual(len({r['doi'] for r in d}),1)
        self.assertEqual(len({r['pollutant'] for r in d}),2)
        self.assertTrue(all(r['rank_validation_status'].startswith('DESCRIPTIVE_ONLY') for r in d))
        self.assertTrue(all(r['modification_route']=='UNMODIFIED' for r in d))
        for poll in ['Cd(II)','Pb(II)']:
            vals=[r for r in d if r['pollutant']==poll]
            self.assertEqual({int(r['pyrolysis_temp_C']) for r in vals},{300,400,500})

    def test_sludge_requires_dosage_resolution(self):
        d=load_csv('sewage_sludge_fig2_removal_APPROX_DOSAGE_CONFLICT.csv')
        self.assertTrue(all(float(r['dose_in_figure_caption_g_L'])==1 for r in d))
        self.assertTrue(all(float(r['dose_implied_by_method_20g_in_20mL_g_L'])==1000 for r in d))
        self.assertTrue(all(r['ranking_approved']=='False' for r in d))

    def test_ugw_c0_is_diagnostic_not_observed(self):
        d=load_csv('urban_waste_fig4_ce_qe_APPROX_NOT_TRAINED.csv')
        self.assertTrue(all(r['reported_C0_for_this_point']=='NR' for r in d))
        self.assertTrue(all(r['ranking_approved']=='False' for r in d))
        self.assertEqual({float(r['dose_g_L']) for r in d},{1.5,1.0,0.75})

    def test_exact_stage1_links_exist_and_are_qa_only(self):
        d=load_csv('rice_figure9_stage1_linked_18_QA_NOT_TRAINED.csv')
        self.assertEqual(len(d),18)
        self.assertEqual(len({r['stage1_material_id'] for r in d}),9)
        self.assertTrue(all(r['link_provenance'].startswith('SAME_DOI_EXACT') for r in d))
        self.assertTrue(all(r['training_approved']=='False' for r in d))
        self.assertTrue(all(float(r['stage1_BET_measured_m2_g'])>=0 for r in d))

    def test_script_regeneration_matches_versioned_data(self):
        # script regenerates the exact reviewed CSVs deterministically; no network.
        path=ROOT/'scripts/build_v30_source_graph_evidence.py'
        text=path.read_text(encoding='utf8')
        self.assertIn('RICE_FITTED',text)
        self.assertIn('UGW_POINTS',text)
        self.assertIn('SLUDGE_PCT',text)
        self.assertIn('FIGURE_FITTED_DISCORDANCE',text)

if __name__=='__main__': unittest.main()
