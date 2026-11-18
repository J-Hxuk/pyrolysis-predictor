"""Pure UI state utilities for v141; no model training or inference changes."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


def input_signature(inputs: Mapping[str, Any]) -> str:
    """Stable fingerprint of values used for one prediction, not widget state.

    Store this with prediction results. If current inputs differ, UI must not
    present the stored prediction as the result of the new input.
    """
    serialized = json.dumps(
        dict(inputs), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False, default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def safe_export_stem(feedstock: str, pollutant: str, temperature: object) -> str:
    """Safe, deterministic filename stem; never use unchecked user input as path."""
    def part(raw: object) -> str:
        clean = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", str(raw)).strip("_")
        return (clean or "unknown")[:36]

    return "biochar_v141_" + "_".join((part(feedstock), part(pollutant), part(temperature)))


def comparison_delta(case_a: Mapping[str, float], case_b: Mapping[str, float]) -> dict[str, float]:
    """B-A for the same three already-computed model outputs."""
    return {key: float(case_b[key]) - float(case_a[key]) for key in
            ("BET_m2_g", "TPV_cm3_g", "reported_capacity_mg_g")}
