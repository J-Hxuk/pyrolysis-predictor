"""Restricted Stage 2 manufacturing-condition screening, not global optimization.

Input candidate configurations are drawn from the *recorded input features* of
matching feedstock/pollutant records in Core 97. Reported qe is NEVER used to
rank candidates. A frozen Model B scores them under the user's fixed adsorption
conditions. Predictions are exploratory and unvalidated as prescriptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import math
import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2.input_support import predict_from_contract


@dataclass(frozen=True)
class CandidateScreen:
    status: str
    candidates: pd.DataFrame
    message: str
    count_observations: int
    count_unique: int


def observed_manufacturing_candidates(core: pd.DataFrame, matched: str | None, pollutant: str) -> CandidateScreen:
    """Use only exact matched precursor and pollutant to avoid invented recipes."""
    blank = pd.DataFrame(columns=fc.MANUFACTURING_INPUTS)
    if not matched:
        return CandidateScreen("unknown_feedstock", blank, "미학습 원료: 제조조건 비교를 지원하지 않습니다.", 0, 0)
    sub = core.loc[(core[fc.GROUP_FEEDSTOCK] == matched) & (core["pollutant"] == pollutant)]
    if sub.empty:
        return CandidateScreen("no_pair", blank, "원료·오염물질 조합에 해당하는 제조조건 기록이 없습니다.", 0, 0)
    # Explicitly retain missing values as missing. Do not silently substitute a
    # process temperature, gas atmosphere, impregnation agent, etc.
    cols = list(fc.MANUFACTURING_INPUTS)
    distinct = sub.loc[:, cols].drop_duplicates().reset_index(drop=True)
    n = len(distinct)
    status = "comparable" if n >= 2 else "single"
    msg = (f"문헌 제조조건 {n}개를 비교할 수 있습니다(학습 관측 {len(sub)}행)." if n >= 2 else
           f"관측 {len(sub)}행이지만 서로 다른 제조조건은 1개뿐이라 최적조건을 비교할 수 없습니다.")
    return CandidateScreen(status, distinct, msg, len(sub), n)


def score_observed_candidates(screen: CandidateScreen, model_obj: Mapping, ads: Mapping) -> pd.DataFrame:
    """Rank only distinct candidates; no fabricated sample, no target leakage.

    If ANY raw prediction violates the physical mass balance, record violation
    and do not recommend it. A clipped prediction must not win artificially.
    """
    if screen.status != "comparable":
        return pd.DataFrame()
    rows = []
    for i, (_, cand) in enumerate(screen.candidates.iterrows(), start=1):
        row = {**ads, **cand.to_dict()}
        raw, clipped, cap = predict_from_contract(model_obj, row)
        violation = (not math.isfinite(raw)) or (raw < 0) or (raw > cap)
        rows.append({"candidate_id": i, **cand.to_dict(), "qe_raw_mg_g": raw,
                     "qe_clipped_mg_g": clipped, "cap_mg_g": cap,
                     "physical_violation": bool(violation),
                     "estimated_removal_pct": np.nan if violation else 100.0 * raw / cap,
                     "eligible_for_ranking": not violation})
    result = pd.DataFrame(rows)
    # Sort valid predictions first; invalid ones are displayed without rank.
    result = result.sort_values(["eligible_for_ranking", "qe_raw_mg_g"],
                                ascending=[False, False], kind="stable").reset_index(drop=True)
    result["rank"] = pd.NA
    valid = result["eligible_for_ranking"]
    # Tied tree-model outputs must never receive arbitrary distinct ranks.
    result.loc[valid, "rank"] = result.loc[valid, "qe_raw_mg_g"].rank(ascending=False, method="min").astype(int).tolist()
    return result
