"""Contract tests for inactive v146 candidate. No training or deployment changes."""
from pathlib import Path

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2.research_blend_v146 import (
    RESEARCH_MODEL_VERSION, ResearchFractionBlend75KNN25ET,
)
from pyrolysis.biochar_stage2.stage2_blend_v139 import FractionBlend, FEATURES


class _Predictor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, frame):
        assert list(frame.columns) == list(FEATURES)
        return np.asarray(self.prediction, dtype=float)


def test_v146_is_research_only_and_does_not_mutate_v139():
    assert RESEARCH_MODEL_VERSION.endswith("NOT_ACTIVE")
    x = pd.DataFrame([{col: np.nan for col in FEATURES} for _ in range(2)])
    cand = ResearchFractionBlend75KNN25ET()
    cand.knn = _Predictor([0.2, 1.5])
    cand.et = _Predictor([0.8, -0.2])
    np.testing.assert_allclose(cand.predict(x), [0.35, 0.75])
    baseline = FractionBlend()
    baseline.knn = cand.knn
    baseline.et = cand.et
    np.testing.assert_allclose(baseline.predict(x), [0.5, 0.5])


def test_v146_candidate_not_imported_by_active_engine():
    src = (Path(__file__).resolve().parents[1] /
           "pyrolysis/biochar_stage2/v112_single_engine.py").read_text("utf-8")
    assert "research_blend_v146" not in src
    assert "ResearchFractionBlend75KNN25ET" not in src
