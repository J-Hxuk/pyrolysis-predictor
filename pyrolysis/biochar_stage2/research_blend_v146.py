"""Research-only Stage 2 candidate: 75% KNN + 25% weighted Extra Trees.

This must NEVER be substituted into v139 deployment without an independent
source/DOI holdout and documented model-selection validation.
Identical v139 training fit; only mixture weights differ.
"""
from __future__ import annotations

import numpy as np
from pyrolysis.biochar_stage2.stage2_blend_v139 import FEATURES, FractionBlend

RESEARCH_MODEL_VERSION = "research_v146_knn75_et25_NOT_ACTIVE"


class ResearchFractionBlend75KNN25ET(FractionBlend):
    """Same model fits and preprocessing as v139; research-only inference mix."""

    def predict(self, frame):
        knn_fraction = np.clip(self.knn.predict(frame[FEATURES]), 0, 1)
        et_fraction = np.clip(self.et.predict(frame[FEATURES]), 0, 1)
        return 0.75 * knn_fraction + 0.25 * et_fraction
