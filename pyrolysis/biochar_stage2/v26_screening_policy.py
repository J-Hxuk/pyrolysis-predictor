"""Conservative, non-deploying v26 precheck for exploratory manufacture screening.

This does NOT validate the accuracy of Model B; it only prevents obvious OOD
rankings from being portrayed as supported recommendations. No model is run here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2.candidate_screening import observed_manufacturing_candidates

@dataclass(frozen=True)
class ScreeningScope:
    eligible_for_exploratory_ranking: bool
    unique_candidates: int
    reasons: tuple[str, ...]
    disclaimer: str = (
        '후보군 내부 탐색적 순위일 뿐, 신규 원료 최적 제조조건이나 '\
        '검증된 성능 개선·예측구간을 의미하지 않습니다.'
    )


def assess_scope(core:pd.DataFrame, feedstock:str|None, pollutant:str, ads:Mapping) -> ScreeningScope:
    """Hold the model to its *recorded data support*, not just an arbitrary global box.

    Missing numeric values (e.g. contact time) are not imputed for an unseen
    request. Values are checked against training-cohort empirical min/max;
    satisfying these limits is necessary but not sufficient for valid modeling.
    """
    candidates=observed_manufacturing_candidates(core,feedstock,pollutant)
    reasons=[]
    if candidates.status!='comparable':
        reasons.append('candidate_status:'+candidates.status)
    if pollutant not in set(core.pollutant.dropna()):
        reasons.append('unseen_pollutant')
    if feedstock is None or feedstock not in set(core[fc.GROUP_FEEDSTOCK].dropna()):
        reasons.append('unseen_feedstock')
    for field in ['C0_mg_L','dose_g_L','initial_pH','adsorption_T_C','contact_time_min']:
        series=pd.to_numeric(core[field],errors='coerce').dropna()
        value=ads.get(field,None)
        try:
            number=float(value) if value not in (None,'') else np.nan
        except (ValueError,TypeError):
            number=np.nan
        if not np.isfinite(number):
            reasons.append('missing_or_invalid:'+field)
            continue
        if field in ('C0_mg_L','dose_g_L') and number<=0:
            reasons.append('nonpositive:'+field)
            continue
        if not series.empty and not (series.min() <= number <= series.max()):
            reasons.append('outside_training_range:'+field)
    # Screening previously picked among at least two distinct observed recipes,
    # but not necessarily under identical adsorption experiment conditions.
    # None of these checks certifies out-of-DOI optimality.
    return ScreeningScope(
        eligible_for_exploratory_ranking=not reasons,
        unique_candidates=candidates.count_unique,
        reasons=tuple(dict.fromkeys(reasons)),
    )
