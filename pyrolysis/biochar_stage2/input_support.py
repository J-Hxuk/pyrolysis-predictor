"""Stage2 UI input validation and feedstock lookup (no Streamlit, no training).

Names are descriptors, not a way to infer biomass chemistry. Unknown names are
never substituted by the nearest known biomass. The selected model's stored
feature contract determines which optional fields actually affect predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Mapping

import pandas as pd

# Only unambiguous *name aliases*. Never treat related plant residues as synonyms.
_ALIASES = {
    "왕겨": "rice husk", "벼껍질": "rice husk", "rice hull": "rice husk",
    "rice hulls": "rice husk", "rice husks": "rice husk",
    "볏짚": "rice straw", "rice straws": "rice straw",
    "땅콩껍질": "peanut shell", "땅콩 껍질": "peanut shell",
    "peanut shells": "peanut shell",
    "대나무": "bamboo", "bamboo culm": "bamboo",
    "망고껍질": "mango peel", "망고 껍질": "mango peel",
    "아몬드껍질": "almond shell", "아몬드 껍질": "almond shell",
    "음식물쓰레기": "food waste", "음식물 쓰레기": "food waste",
    "하수슬러지": "municipal sewage sludge", "하수 슬러지": "municipal sewage sludge",
}


def normalize_name(name: str) -> str:
    """Normalize spelling only: does not merge distinct biomass types."""
    name = str(name or "").casefold().replace("_", " ")
    name = re.sub(r"[\-\s]+", " ", name).strip()
    return name


@dataclass(frozen=True)
class FeedstockMatch:
    original: str
    normalized: str
    matched: str | None
    status: str  # empty / exact / alias / suggestion / unknown
    suggestions: tuple[str, ...]


def match_feedstock(text: str, known: list[str], limit: int = 6) -> FeedstockMatch:
    original = str(text or "").strip()
    norm = normalize_name(original)
    if not norm:
        return FeedstockMatch(original, "", None, "empty", ())
    canonical = {normalize_name(k): k for k in known if str(k).strip()}
    if norm in canonical:
        return FeedstockMatch(original, norm, canonical[norm], "exact", (canonical[norm],))
    alias = _ALIASES.get(norm)
    if alias and alias in canonical:
        return FeedstockMatch(original, norm, canonical[alias], "alias", (canonical[alias],))
    # Fuzzy similarity means suggestions *only*. Never assign matched based on score.
    ranked = []
    search_term = alias or norm
    for norm_key, display in canonical.items():
        score = SequenceMatcher(None, search_term, norm_key).ratio()
        if norm_key.startswith(search_term) or search_term in norm_key or norm_key in search_term:
            score += 0.22
        if score >= 0.49:
            ranked.append((score, display))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    candidates = tuple(k for _, k in ranked[:limit])
    return FeedstockMatch(original, norm, None, "suggestion" if candidates else "unknown", candidates)


# Optional inputs for a *future, separately validated* composition-assisted model.
# Raw biomass chemistry, not char composition after pyrolysis.
OPTIONAL_COMPOSITION = {
    "raw_C_wt_pct": "C", "raw_H_wt_pct": "H", "raw_O_wt_pct": "O",
    "raw_N_wt_pct": "N", "raw_S_wt_pct": "S",
    "raw_ash_wt_pct": "Ash", "raw_VM_wt_pct": "VM", "raw_FC_wt_pct": "FC",
    "raw_moisture_wt_pct": "Moisture",
    "raw_cellulose_pct": "Cellulose", "raw_hemicellulose_pct": "Hemicellulose",
    "raw_lignin_pct": "Lignin",
}


def validate_composition(raw: Mapping[str, str | float | int | None]) -> tuple[dict[str, float], list[str]]:
    """Blank = genuinely unreported; zero is distinct. Do not impute or normalize."""
    out, errors = {}, []
    for key in OPTIONAL_COMPOSITION:
        value = raw.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            errors.append(f"{OPTIONAL_COMPOSITION[key]}: 숫자로 입력해야 합니다.")
            continue
        if not math.isfinite(val) or val < 0 or val > 100:
            errors.append(f"{OPTIONAL_COMPOSITION[key]}: 0–100 wt% 범위를 확인하세요.")
        else:
            out[key] = val
    # Only a warning when full dry proximate analysis is available.
    trio = ("raw_ash_wt_pct", "raw_VM_wt_pct", "raw_FC_wt_pct")
    if all(k in out for k in trio) and sum(out[k] for k in trio) > 101:
        errors.append("Ash + VM + FC가 100 wt%를 초과합니다. 분석 기준과 수분 포함 여부를 확인하세요.")
    return out, errors


def model_used_fields(model_meta: Mapping) -> set[str]:
    return set(model_meta.get("input_cols", []))


def applied_optional_fields(composition: Mapping, model_meta: Mapping) -> tuple[list[str], list[str]]:
    """Make the *real* model-input/ignored-input distinction explicit."""
    used = model_used_fields(model_meta)
    return sorted(set(composition) & used), sorted(set(composition) - used)


def related_records(df: pd.DataFrame, matched: str | None) -> pd.DataFrame:
    if not matched:
        return df.iloc[:0].copy()
    return df[df["feedstock_label_normalized"].astype(str) == matched].copy()


def predict_from_contract(model_obj: Mapping, row: Mapping) -> tuple[float, float, float]:
    """Predict using *only* input_cols persisted with the trained model.

    Returns raw qe, clipped qe and mass-balance cap in mg/g. This function does
    not present a clipped value as a physically constrained trained estimator.
    """
    import numpy as np
    cols = list(model_obj["metadata"]["input_cols"])
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    raw = float(model_obj["pipeline"].predict(X)[0])
    c0 = float(row["C0_mg_L"])
    dose = float(row["dose_g_L"])
    if not math.isfinite(c0) or not math.isfinite(dose) or c0 <= 0 or dose <= 0:
        raise ValueError("C0와 dose는 유한한 양수여야 합니다.")
    cap = c0 / dose
    return raw, max(0.0, min(raw, cap)), cap
