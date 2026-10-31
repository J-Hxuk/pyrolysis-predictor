"""
test_artifacts.py - Task 16 완료 기준 검증.

합성 자료로만 테스트한다. 저장소를 오염시키지 않도록 tmp_path 를 output_dir 로 쓴다.

확인 항목:
1. 저장 전후 예측이 허용 오차 안에서 일치하고 원본 학습 파일 없이 실행된다.
2. demo 산출물의 연구용 내보내기를 거부한다.
3. 미평가 대상의 내보내기를 거부한다.
4. 기존 묶음을 일부만 덮어쓰지 않는다 (중복 거부).
5. manifest 파일 해시로 훼손·누락·경로 이탈을 검사한다.
6. 비활성 대상은 모델 없이 사유만 포함한다.
7. active.json 이 묶음 ID 와 manifest 해시를 가리킨다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyrolysis.artifacts import (
    DEMO_BUNDLE_MARKER,
    BundleError,
    ExportError,
    export_bundle,
    load_research_bundle,
    verify_bundle_integrity,
    write_active_pointer,
)
from pyrolysis.contracts import InputSpec
from pyrolysis.models import get_regression_candidates
from pyrolysis.preprocessing import build_regression_pipeline


def _specs():
    return [
        InputSpec(id="x1", label_ko="x1", description="x1", role="feedstock", dtype="float", required=True),
        InputSpec(id="x2", label_ko="x2", description="x2", role="feedstock", dtype="float", required=True),
    ]


def _fitted_model(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=30), "x2": rng.normal(size=30)})
    y = X["x1"] * 2.0 - X["x2"]
    pipe = build_regression_pipeline(_specs(), get_regression_candidates()["linear_regression"])
    pipe.fit(X, y)
    return pipe, X


def _contract(bundle_type="research"):
    return {
        "bundle_type": bundle_type,
        "input_ids": ["x1", "x2"],
        "input_dtypes": {"x1": "float", "x2": "float"},
        "target_id": "y1",
        "target_task": "regression",
    }


def _valid_summary(evaluated=True, tids=("y1",)):
    return {tid: {"evaluated": evaluated, "mae": 0.1, "method": "holdout"} for tid in tids}


def _training_domain(tids=("y1",)):
    return {tid: {"numeric_ranges": {"x1": [-3, 3], "x2": [-3, 3]}} for tid in tids}


class TestExportRoundtrip:
    def test_export_and_reload_predictions_match(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(
            out, "research_v1", _contract(), {"y1": pipe},
            _valid_summary(), _training_domain(),
            verify_frame=X, tolerance=1e-6,
        )
        assert bundle_dir.exists()
        # 원본 학습 파일 없이 로드·예측
        loaded = load_research_bundle(bundle_dir)
        pred = loaded["models"]["y1"].predict(X[["x1", "x2"]])
        np.testing.assert_allclose(pred, pipe.predict(X), rtol=0, atol=1e-6)

    def test_manifest_has_file_hashes_excluding_itself(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "manifest.json" not in manifest["file_hashes"]  # 자기 자신 제외
        assert "contract.json" in manifest["file_hashes"]
        assert "models/y1.joblib" in manifest["file_hashes"]

    def test_required_files_present(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        for name in ("manifest.json", "contract.json", "validation_summary.json",
                     "training_domain.json", "environment.json"):
            assert (bundle_dir / name).exists(), name
        assert (bundle_dir / "models" / "y1.joblib").exists()


class TestExportRejections:
    def test_demo_bundle_rejected_for_research(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        with pytest.raises(ExportError):
            export_bundle(out, "demo_v1", _contract(bundle_type=DEMO_BUNDLE_MARKER),
                          {"y1": pipe}, _valid_summary(), _training_domain(), verify_frame=X)

    def test_unevaluated_target_rejected(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        with pytest.raises(ExportError):
            export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                          _valid_summary(evaluated=False), _training_domain(), verify_frame=X)

    def test_no_verify_frame_rejected(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        with pytest.raises(ExportError):
            export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                          _valid_summary(), _training_domain(), verify_frame=None)

    def test_duplicate_bundle_rejected(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                      _valid_summary(), _training_domain(), verify_frame=X)
        with pytest.raises(ExportError):
            export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                          _valid_summary(), _training_domain(), verify_frame=X)

    def test_no_active_models_rejected(self, tmp_path):
        out = tmp_path / "bundles"
        _, X = _fitted_model()
        with pytest.raises(ExportError):
            export_bundle(out, "research_v1", _contract(), {},
                          _valid_summary(), _training_domain(), verify_frame=X)


class TestInactiveTargets:
    def test_inactive_target_has_reason_no_model(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(
            out, "research_v1", _contract(), {"y1": pipe},
            _valid_summary(), _training_domain(),
            inactive_targets={"y2": "데이터 부족"}, verify_frame=X,
        )
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["inactive_targets"] == {"y2": "데이터 부족"}
        # 비활성 대상 모델 파일은 없다
        assert not (bundle_dir / "models" / "y2.joblib").exists()


class TestIntegrityChecks:
    def test_verify_passes_for_intact_bundle(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        verify_bundle_integrity(bundle_dir)  # no raise

    def test_tampered_file_detected(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        # contract.json 변조
        (bundle_dir / "contract.json").write_text("{\"tampered\": true}", encoding="utf-8")
        with pytest.raises(BundleError):
            verify_bundle_integrity(bundle_dir)

    def test_missing_file_detected(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        (bundle_dir / "environment.json").unlink()
        with pytest.raises(BundleError):
            verify_bundle_integrity(bundle_dir)

    def test_load_fails_on_tampered_bundle(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        (bundle_dir / "models" / "y1.joblib").write_bytes(b"corrupt")
        with pytest.raises(BundleError):
            load_research_bundle(bundle_dir)


class TestActivePointer:
    def test_active_pointer_references_bundle_and_hash(self, tmp_path):
        pipe, X = _fitted_model()
        out = tmp_path / "bundles"
        bundle_dir = export_bundle(out, "research_v1", _contract(), {"y1": pipe},
                                   _valid_summary(), _training_domain(), verify_frame=X)
        active_path = tmp_path / "active.json"
        pointer = write_active_pointer(active_path, bundle_dir)
        assert active_path.exists()
        saved = json.loads(active_path.read_text(encoding="utf-8"))
        assert saved["bundle_id"] == "research_v1"
        assert saved["manifest_hash"] == pointer["manifest_hash"]
        assert len(saved["manifest_hash"]) == 64  # SHA-256 hex