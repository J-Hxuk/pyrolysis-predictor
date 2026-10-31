"""v19 restricted candidate screening — no invented experimental recipe."""
import math
from pathlib import Path

import pandas as pd

from pyrolysis.biochar_stage2.candidate_screening import (
    observed_manufacturing_candidates, score_observed_candidates,
)
from pyrolysis.biochar_stage2.v15_loader import load_core_qe
from pyrolysis.biochar_stage2.v15_features import MANUFACTURING_INPUTS


CORE = load_core_qe(Path(__file__).resolve().parents[1]).df


def test_rice_husk_13_adsorption_rows_are_one_manufacturing_recipe():
    q = observed_manufacturing_candidates(CORE, "rice husk", "Cd(II)")
    assert q.count_observations == 13
    assert q.count_unique == 1 and q.status == "single"


def test_peanut_shell_has_two_distinct_routes_for_pb():
    q = observed_manufacturing_candidates(CORE, "peanut shell", "Pb(II)")
    assert q.count_unique == 2 and q.status == "comparable"
    assert set(q.candidates.manufacturing_route) == {"CLAY_POST_MODIFICATION", "SINGLE_STAGE_PYROLYSIS"}


def test_no_unseen_feedstock_candidate_and_no_pollutant_crossfill():
    assert observed_manufacturing_candidates(CORE, None, "Cd(II)").status == "unknown_feedstock"
    assert observed_manufacturing_candidates(CORE, "rice husk", "Pb(II)").status == "no_pair"


def test_duplicate_adsorption_rows_not_counted_as_manufacture_options():
    q = observed_manufacturing_candidates(CORE, "mango peel", "Cd(II)")
    assert q.count_observations > q.count_unique


def test_score_ranks_by_model_not_reported_target_and_excludes_physical_violations():
    q = observed_manufacturing_candidates(CORE, "peanut shell", "Pb(II)")

    class FakeModel:
        def predict(self, X):
            return [15.0 if X.manufacturing_route.iloc[0] == "SINGLE_STAGE_PYROLYSIS" else 120.0]

    model = {"metadata": {"input_cols": ["manufacturing_route"]}, "pipeline": FakeModel()}
    scores = score_observed_candidates(q, model, {"C0_mg_L": 100, "dose_g_L": 2})
    assert len(scores) == 2
    assert scores.eligible_for_ranking.sum() == 1
    assert scores.loc[0, "manufacturing_route"] == "SINGLE_STAGE_PYROLYSIS"
    assert scores.loc[0, "rank"] == 1
    assert pd.isna(scores.loc[1, "rank"])
    assert scores.loc[1, "physical_violation"]
    assert "target_value" not in scores.columns


def test_unrankable_single_candidate_never_scores():
    class FakeModel:
        def predict(self, X):
            raise AssertionError("should not run")
    q = observed_manufacturing_candidates(CORE, "rice husk", "Cd(II)")
    assert score_observed_candidates(q, {"metadata": {"input_cols": []}, "pipeline": FakeModel()},
                                     {"C0_mg_L": 100, "dose_g_L": 2}).empty


def test_identical_scores_are_ties_not_arbitrary_ranked():
    q = observed_manufacturing_candidates(CORE, "peanut shell", "Pb(II)")
    class ConstantModel:
        def predict(self, X):
            return [20.0]
    m = {"metadata": {"input_cols": ["manufacturing_route"]}, "pipeline": ConstantModel()}
    scores = score_observed_candidates(q, m, {"C0_mg_L": 100, "dose_g_L": 2})
    assert scores["rank"].tolist() == [1, 1]
