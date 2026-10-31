"""Run after `python -m scripts.rebuild_v46_research_models_local --root .`."""
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "deployment" / "research_v46_NOT_ACTIVE"


def test_v46_local_rebuild_manifest_and_models():
    meta_file = MODEL_DIR / "V46_LOCAL_REBUILD_MANIFEST.json"
    assert meta_file.exists(), (
        "로컬 호환 모델 미생성: "
        "python -m scripts.rebuild_v46_research_models_local --root ."
    )
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["active_model_changed"] is False
    assert meta["data_changed"] is False
    assert len(meta["models"]) == 4
    for record in meta["models"]:
        model = joblib.load(MODEL_DIR / record["file"])
        assert hasattr(model, "predict")
        assert record["rows"] > 0
