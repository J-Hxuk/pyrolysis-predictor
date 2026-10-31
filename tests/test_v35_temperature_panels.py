"""Audit tests for v35 — from a v34-installed project root.
python -m unittest discover -s tests -p test_v35_temperature_panels.py -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from scripts.build_v35_temperature_panels import (
    build,group_panels,pair_score,qa_queue,rank_evaluate,read_csv,source_panel,
)

ROOT=Path(__file__).resolve().parents[1]


class TestV35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src=source_panel(ROOT)
        cls.panel=group_panels(cls.src)

    def test_01_15_rows_two_independent_doi(self):
        self.assertEqual(len(self.src),15)
        self.assertEqual(len({r['doi'] for r in self.src}),2)

    def test_02_target_and_physical_constraints(self):
        for r in self.src:
            self.assertLessEqual(r['qe_mg_g'],r['physical_cap_mg_g']+1e-9)
            self.assertGreaterEqual(r['qe_mg_g'],0)
            self.assertNotIn('QMAX',r['qe_origin'])
            self.assertNotIn('QT_240',r['target_type'])
            self.assertEqual(r['approval'],'V35_RESEARCH_ONLY_NOT_EQUILIBRIUM_CERTIFIED')

    def test_03_exact_material_link_and_distinct_ids(self):
        self.assertEqual(len({r['observation_id'] for r in self.src}),15)
        self.assertEqual(len({r['material_id'] for r in self.src}),7)
        self.assertTrue(all(r['BET_measured_m2_g'] not in ('',None) for r in self.src))

    def test_04_four_unique_temperature_groups(self):
        groups={r['group_id'] for r in self.panel}
        self.assertEqual(len(groups),4)
        for gid in groups:
            r=[x for x in self.panel if x['group_id']==gid]
            self.assertGreaterEqual(len(r),3)
            self.assertEqual(len({x['temperature_C'] for x in r}),len(r))
            self.assertEqual(len({x['hold_min'] for x in r}),1)

    def test_05_do_not_invent_wang_pH_midpoint(self):
        r=[x for x in self.panel if 'ijerph' in x['doi']]
        self.assertTrue(all(x['initial_pH_reported']=='4.5-5.0' for x in r))
        self.assertTrue(all(x['initial_pH_exact']=='' for x in r))

    def test_06_uncertain_top1_group(self):
        x=[r for r in self.panel if r['C0_mg_L']==80]
        self.assertEqual(len(x),4)
        self.assertTrue(all(r['top1_robustness']=='TOP1_SENSITIVE_TO_REPORTED_UNCERTAINTY' for r in x))

    def test_07_pairwise_ties_half_credit(self):
        score,n,ties=pair_score([1,3,2],[10,10,10])
        self.assertEqual(n,3)
        self.assertEqual(ties,3)
        self.assertAlmostEqual(score,.5)

    def test_08_v34_predictions_are_outside_doi_only(self):
        raw=read_csv(ROOT/'reports/v34_temperature/new_DOI_external_temperature_predictions.csv')
        ids={r['observation_id'] for r in self.src}
        checked=[r for r in raw if r['observation_id'] in ids and r['training']=='old98_only']
        self.assertEqual(len(checked),15*3)
        self.assertEqual({r['doi'] for r in checked},{r['doi'] for r in self.src})

    def test_09_existing_b_model_misses_top1(self):
        rank,_=rank_evaluate(ROOT,self.panel)
        m=[r for r in rank if r['strategy']=='B|XGBoost']
        self.assertEqual(len(m),4)
        self.assertTrue(all(r['top1_status']=='WRONG' for r in m))
        self.assertTrue(all(r['all_raw_predictions_physically_valid'] is False for r in m))

    def test_10_do_not_count_zhou_qt_as_qe(self):
        bb=read_csv(ROOT/'data/imports/v34_temperature/zhou_BBD_unique26_QT_240min_NOT_QE.csv')
        self.assertEqual(len(bb),26)
        self.assertTrue(all('qt_240min_mg_g' in r for r in bb))
        self.assertTrue(all('1038/s41598-022-17901-w' not in r['doi'] for r in self.src))

    def test_11_queue_has_no_numeric_graph_predictions(self):
        q=qa_queue()
        self.assertEqual(len(q),10)
        self.assertTrue(all('qe_mg_g' not in r for r in q))
        self.assertTrue(any('METHOD_FIGURE_VOLUME_CONFLICT' in r['status'] for r in q))

    def test_12_build_only_writes_research_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            dst=Path(t)
            for rel in ['data/imports/v33_five_papers/stage2_new15_single_metal_measured_or_derived_RESEARCH_STAGING.csv',
                        'data/imports/v33_five_papers/stage1_new30_temperature_series_RESEARCH_STAGING.csv',
                        'data/imports/v34_temperature/zhou_BBD_unique26_QT_240min_NOT_QE.csv',
                        'reports/v34_temperature/new_DOI_external_temperature_predictions.csv']:
                target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/rel).read_bytes())
            fake=dst/'deployment/stage2_models/active.json';fake.parent.mkdir(parents=True,exist_ok=True)
            fake.write_text('unchanged model',encoding='utf-8')
            summary=build(dst)
            self.assertEqual(summary['real_source_rows'],15)
            self.assertEqual(summary['training_approval_rows'],0)
            self.assertEqual(fake.read_text(encoding='utf-8'),'unchanged model')
            self.assertTrue((dst/'reports/v35_temperature/summary.json').exists())


if __name__=='__main__':unittest.main()
