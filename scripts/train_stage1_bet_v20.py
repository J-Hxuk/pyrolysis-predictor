"""Regenerate portable, *experimental* composition BET models from eval37/base.csv.

Command: python -m scripts.train_stage1_bet_v20
No modification of v60 or Stage2 deployed models. Does NOT train TPV.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from pyrolysis.biochar_stage1.data_loader import load_bet_base, file_sha256
from pyrolysis.biochar_stage1.extract import resolve_eval37_dir
from pyrolysis.biochar_stage1.comp_ablation_v18 import make_model, validate_data, TARGET
from pyrolysis.biochar_stage1.composition_inference_v20 import FEATURES, default_model_path, forest_predict_one


def export_forest(model, columns):
    """Pass only finite, fully observed training values; do not need imputer at inference."""
    trees = []
    rf = model.named_steps["model"]
    for estim in rf.estimators_:
        t = estim.tree_
        trees.append({
            "left": t.children_left.tolist(), "right": t.children_right.tolist(),
            "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
            "value": t.value[:, 0, 0].tolist(),
        })
    return {"features": list(columns), "trees": trees}


def build(root: Path, *, destination: Path | None = None) -> dict:
    root = Path(root)
    df = load_bet_base(root)
    info = validate_data(df)
    models = {}
    for name in ("proximate", "full"):
        cols = FEATURES[name]
        pipeline = make_model("RandomForest")
        pipeline.fit(df[cols], df[TARGET].to_numpy(float))
        spec = export_forest(pipeline, cols)
        spec["feature_range"] = {k: [float(df[k].min()), float(df[k].max())] for k in cols}
        # Portable tree export must reproduce sklearn on held training rows (not validation).
        original = pipeline.predict(df[cols].head(12))
        portable = np.array([forest_predict_one(spec, row.to_dict()) for _, row in df[cols].head(12).iterrows()])
        if not np.allclose(original, portable, atol=1e-9, rtol=1e-9):
            raise RuntimeError(f"Portable model failed parity test: {name}")
        models[name] = spec

    data_path = resolve_eval37_dir(root) / "base.csv"
    result = {
        "schema_version": 1,
        "research_status": "EXPLORATORY_NOT_VALIDATED_FOR_DEPLOYMENT",
        "target": "BET_m2_g",
        "prediction_type": "NON_ACTIVATED_SINGLE_STAGE_RAW_BIOMASS_ONLY",
        "train_dataset": "eval37/base.csv",
        "train_data_sha256": file_sha256(data_path),
        "train_rows": info["rows"], "train_doi_groups": info["doi_groups"],
        "trainer_sklearn_version": sklearn.__version__,
        "validation": "Preexisting v18 leave-one-DOI-out results; post-selection exploratory, not independent external test.",
        "validation_results": {
            "proximate": {"r2_oof": 0.25282621543966766, "mae_m2_g": 69.52078758792392},
            "full": {"r2_oof": 0.256990278243987, "mae_m2_g": 66.42461527371756},
        },
        "warning": "Experimental research-only. No TPV prediction, no qe, no production optimizer, no external novelty validation.",
        "models": models,
    }
    dest = destination or default_model_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"path": str(dest), "bytes": dest.stat().st_size, "data_sha256": result["train_data_sha256"], "rows": info["rows"]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build(root), ensure_ascii=False, indent=2))
