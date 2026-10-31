"""Conservative applicability screening for the *frozen, active* Stage 2 A/B models.

Only the active model's Core-97 empirical training cohort is an admissible
reference. Research v40 data are NOT evidence of deployed-model coverage.
Passing range checks is never a calibrated prediction interval or endorsement.
This module does not modify inputs, models, or datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2 import v15_features as features

CATEGORICAL = frozenset(("pollutant", "feedstock_label_normalized", "manufacturing_route", "pyrolysis_atmosphere", "activation_or_modification"))
REQUIRED_ADS = ("pollutant", "C0_mg_L", "dose_g_L")
REQUIRED_FOR_RANKING = ("feedstock_label_normalized", "manufacturing_route", "pyrolysis_temperature_C")


@dataclass(frozen=True)
class ScopeReport:
    model: str
    status: str  # INVALID / OUTSIDE_RANGE / LIMITED_EVIDENCE / RANGE_ONLY
    n_reference_rows: int
    n_reference_doi: int
    n_cohort_rows: int
    n_cohort_doi: int
    n_controlled_temperature_panels: int
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    can_rank: bool
    coverage: tuple[tuple[str, int, int], ...]  # field, reported, total rows
    note: str = ("한 변수씩 범위를 확인한 결과일 뿐, 변수의 동시 조합 지원 여부나 "
                 "예측 신뢰구간·최적 제조조건을 보장하지 않습니다.")


def _absent(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip()
    try:
        value = pd.isna(x)
        return bool(value) if np.ndim(value) == 0 else False
    except (TypeError, ValueError):
        return False


def _number(x: object) -> float | None:
    if _absent(x):
        return None
    try:
        val = float(x)
    except (ValueError, TypeError, OverflowError):
        return None
    return val if math.isfinite(val) else None


def _features(model: str) -> tuple[str, ...]:
    if model not in ("A", "B"):
        raise ValueError("활성 Stage 2 A/B만 지원합니다")
    return tuple(features.model_inputs(model))


def _controlled_temperature_panels(subset: pd.DataFrame) -> int:
    """Count distinct temperature groups with identical *reported* adsorption controls.

    Missing controls cannot establish a matched experiment and are excluded.
    Merely having multiple temperature values in one DOI is not sufficient.
    """
    controls = ("doi", "C0_mg_L", "dose_g_L", "initial_pH",
                "adsorption_T_C", "contact_time_min", "pyrolysis_temperature_C")
    if any(col not in subset.columns for col in controls):
        return 0
    known = subset.dropna(subset=list(controls))
    if known.empty:
        return 0
    grouped = known.groupby(list(controls[:-1]), dropna=False)["pyrolysis_temperature_C"].nunique()
    return int((grouped >= 2).sum())


def inspect_active_scope(core: pd.DataFrame, row: Mapping[str, object], model: str,
                         *, min_doi_for_ranking: int = 2) -> ScopeReport:
    """Compute advisory warnings from the ACTIVE Core-97, not an experimental refit.

    For each feature, global empirical min/max are checked. The matched
    pollutant (A) or same pollutant/feedstock (B) is additionally counted.
    A small/missing cohort is a warning, not filled with related raw feedstocks.
    Ranking needs >=2 *different* DOI, but still isn't certified if it passes.
    """
    cols = _features(model)
    if min_doi_for_ranking < 2:
        raise ValueError("서로 다른 DOI 2개 이상 요구")
    absent_cols = [c for c in ("doi", "pollutant", *cols) if c not in core.columns]
    if absent_cols:
        raise ValueError("Core 데이터에 필수 열이 없습니다: " + ", ".join(sorted(set(absent_cols))))
    blockers: list[str] = []
    warnings: list[str] = []
    invalid = False
    outside = False
    for name in REQUIRED_ADS:
        if _absent(row.get(name)):
            invalid = True
            blockers.append(f"필수 흡착조건 누락: {name}")
    for name in ("C0_mg_L", "dose_g_L"):
        value = _number(row.get(name))
        if value is None or value <= 0:
            invalid = True
            blockers.append(f"양수 입력 필요: {name}")
    for field in cols:
        given = row.get(field)
        if _absent(given):
            if model == "B" and field in REQUIRED_FOR_RANKING:
                warnings.append(f"제조조건 미보고: {field}")
            continue
        series = core[field].dropna()
        if field in CATEGORICAL:
            known = set(series.astype(str).tolist())
            if str(given) not in known:
                outside = True
                warnings.append(f"학습에서 관측되지 않은 범주: {field}={given}")
        else:
            val = _number(given)
            if val is None:
                invalid = True
                blockers.append(f"유효하지 않은 수치: {field}")
                continue
            vals = pd.to_numeric(series, errors="coerce").dropna()
            if not len(vals):
                warnings.append(f"학습 입력이 전부 미보고: {field}")
            elif val < float(vals.min()) or val > float(vals.max()):
                outside = True
                warnings.append(f"학습 전체 범위 밖: {field}={val:g} (기록 {float(vals.min()):g}–{float(vals.max()):g})")

    # No approximate matching: a new biomass must not be assigned a known one.
    pol = row.get("pollutant")
    matched = core.loc[core["pollutant"].astype(str) == str(pol)]
    if model == "B":
        fs = row.get("feedstock_label_normalized")
        matched = matched.loc[matched["feedstock_label_normalized"].astype(str) == str(fs)]
    n_cohort_doi = int(matched["doi"].nunique())
    n_cohort_rows = int(len(matched))
    n_panels = _controlled_temperature_panels(matched) if model == "B" else 0
    if n_cohort_rows == 0:
        warnings.append("해당 오염물질·원료 조합의 활성 학습 관측이 없습니다" if model == "B" else "해당 오염물질의 활성 학습 관측이 없습니다")
    elif n_cohort_doi < min_doi_for_ranking:
        warnings.append(f"해당 조합의 독립 DOI {n_cohort_doi}편: 논문 간 일반화 근거 부족")

    if model == "B" and n_panels == 0:
        warnings.append("같은 DOI·동일 흡착조건에서 제조온도가 달라지는 확인된 비교군 0개")

    # Coverage is descriptive; missing inputs are NEVER replaced with a hard-coded zero.
    coverage = tuple((col, int(core[col].notna().sum()), int(len(core))) for col in cols)
    low_coverage = [col for col, n, total in coverage
                    if total and 0 < n < 0.30 * total and _absent(row.get(col))]
    if low_coverage:
        warnings.append("활성 학습 자료에서 보고율이 낮은 입력: " + ", ".join(low_coverage))

    rank_ready = (model == "B" and not invalid and not outside and
                  n_cohort_doi >= min_doi_for_ranking and n_panels >= 1 and
                  all(not _absent(row.get(k)) for k in REQUIRED_FOR_RANKING))
    if invalid:
        status = "INVALID"
    elif outside:
        status = "OUTSIDE_RANGE"
    elif n_cohort_doi < 2 or (model == "B" and not rank_ready):
        status = "LIMITED_EVIDENCE"
    else:
        status = "RANGE_ONLY"
    return ScopeReport(model, status, int(len(core)), int(core["doi"].nunique()),
                       n_cohort_rows, n_cohort_doi, n_panels,
                       tuple(dict.fromkeys(blockers)), tuple(dict.fromkeys(warnings)),
                       rank_ready, coverage)


def scope_to_table(scope: ScopeReport) -> pd.DataFrame:
    return pd.DataFrame([
        {"모델 입력": col, "활성 Core 보고 행": reported,
         "전체 행": total, "보고율 (%)": round(100 * reported / total, 1) if total else 0.0}
        for col, reported, total in scope.coverage
    ])
