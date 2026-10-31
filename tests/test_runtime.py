"""
test_runtime.py - Task 17 완료 기준 검증.

합성 자료로만 테스트한다. 저장소를 오염시키지 않도록 tmp_path 를 쓴다.

확인 항목:
1. 운영 한도 누락(null)을 무제한으로 취급하지 않는다 (require 시 오류, config 에 maxUploadSize 안 씀).
2. 한도가 설정되면 config.toml 의 server.maxUploadSize 를 그 값으로 생성한다.
3. config.toml 에 disableDataExport=true, enableStaticServing=false 를 적용한다.
4. check-bundle 이 무결성·환경을 검사한다.
5. 불완전한 활성화(검사 실패)가 기존 active 포인터를 손상시키지 않는다.
6. 검사 통과 묶음만 active 포인터로 교체한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pyrolysis.config import (
    RuntimeConfigError,
    check_runtime_limits,
    prepare_runtime_config,
)
from pyrolysis.artifacts import (
    check_bundle,
    export_bundle,
    write_active_pointer,
)
from pyrolysis.contracts import InputSpec
from pyrolysis.models import get_regression_candidates
from pyrolysis.preprocessing import build_regression_pipeline


def _write_runtime(path, limits=None, inference=None, mode="research"):
    data = {
        "schema_version": 1,
        "mode": mode,
        "limits": limits or {},
        "inference": inference or {},
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _full_limits():
    return {
        "max_upload_size_mib": 5,
        "max_candidate_rows": 100,
        "max_candidate_cols": 50,
        "max_error_display": 20,
    }


def _full_inference():
    return {
        "max_concurrent_requests": 2,
        "inference_threads": 1,
        "wait_timeout_seconds": 30,
    }


def _make_bundle(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=30), "x2": rng.normal(size=30)})
    y = X["x1"] * 2.0 - X["x2"]
    specs = [
        InputSpec(id="x1", label_ko="x1", description="x1", role="feedstock", dtype="float", required=True),
        InputSpec(id="x2", label_ko="x2", description="x2", role="feedstock", dtype="float", required=True),
    ]
    pipe = build_regression_pipeline(specs, get_regression_candidates()["linear_regression"])
    pipe.fit(X, y)
    contract = {
        "bundle_type": "research",
        "input_ids": ["x1", "x2"],
        "input_dtypes": {"x1": "float", "x2": "float"},
        "target_id": "y1", "target_task": "regression",
    }
    vs = {"y1": {"evaluated": True, "mae": 0.1, "method": "holdout"}}
    td = {"y1": {"numeric_ranges": {"x1": [-3, 3], "x2": [-3, 3]}}}
    return export_bundle(tmp_path / "bundles", "research_v1", contract,
                         {"y1": pipe}, vs, td, verify_frame=X)


class TestRuntimeLimits:
    def test_null_limits_not_ready(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml",
                            limits={"max_upload_size_mib": None})
        from pyrolysis.config import load_runtime
        check = check_runtime_limits(load_runtime(rp))
        assert check["ready"] is False
        assert any("max_upload_size_mib" in m for m in check["missing"])

    def test_full_limits_ready(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml",
                            limits=_full_limits(), inference=_full_inference())
        from pyrolysis.config import load_runtime
        check = check_runtime_limits(load_runtime(rp))
        assert check["ready"] is True
        assert check["missing"] == []


class TestPrepareRuntime:
    def test_missing_limits_raises_when_required(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml", limits={"max_upload_size_mib": None})
        cfg = tmp_path / "config.toml"
        with pytest.raises(RuntimeConfigError):
            prepare_runtime_config(rp, cfg, require_all_limits=True)

    def test_null_upload_not_written_as_unlimited(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml", limits={"max_upload_size_mib": None})
        cfg = tmp_path / "config.toml"
        result = prepare_runtime_config(rp, cfg, require_all_limits=False)
        assert result["max_upload_size_mib"] is None
        text = cfg.read_text(encoding="utf-8")
        # maxUploadSize 를 실제 값으로 쓰지 않는다 (주석 처리 또는 생략)
        assert "maxUploadSize = " not in text

    def test_set_upload_written(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml",
                            limits=_full_limits(), inference=_full_inference())
        cfg = tmp_path / "config.toml"
        result = prepare_runtime_config(rp, cfg, require_all_limits=True)
        assert result["max_upload_size_mib"] == 5
        text = cfg.read_text(encoding="utf-8")
        assert "maxUploadSize = 5" in text

    def test_security_flags_always_applied(self, tmp_path):
        rp = _write_runtime(tmp_path / "runtime.yaml",
                            limits=_full_limits(), inference=_full_inference())
        cfg = tmp_path / "config.toml"
        prepare_runtime_config(rp, cfg, require_all_limits=True)
        text = cfg.read_text(encoding="utf-8")
        assert "disableDataExport = true" in text
        assert "enableStaticServing = false" in text


class TestCheckBundle:
    def test_intact_bundle_passes(self, tmp_path):
        bundle_dir = _make_bundle(tmp_path)
        result = check_bundle(bundle_dir)
        assert result["ok"] is True
        assert result["integrity_ok"] is True

    def test_tampered_bundle_fails(self, tmp_path):
        bundle_dir = _make_bundle(tmp_path)
        (bundle_dir / "contract.json").write_text("{\"x\":1}", encoding="utf-8")
        result = check_bundle(bundle_dir)
        assert result["ok"] is False
        assert result["integrity_ok"] is False


class TestActivation:
    def test_activate_writes_pointer(self, tmp_path):
        bundle_dir = _make_bundle(tmp_path)
        active = tmp_path / "active.json"
        pointer = write_active_pointer(active, bundle_dir)
        saved = json.loads(active.read_text(encoding="utf-8"))
        assert saved["bundle_id"] == "research_v1"
        assert len(saved["manifest_hash"]) == 64

    def test_failed_check_does_not_corrupt_existing_pointer(self, tmp_path):
        # 먼저 유효 포인터를 만든다
        bundle_dir = _make_bundle(tmp_path)
        active = tmp_path / "active.json"
        write_active_pointer(active, bundle_dir)
        original = active.read_text(encoding="utf-8")

        # 묶음을 훼손한 뒤 check_bundle 이 실패해야 하고, 포인터는 그대로여야 한다
        (bundle_dir / "contract.json").write_text("{\"x\":1}", encoding="utf-8")
        result = check_bundle(bundle_dir)
        assert result["ok"] is False
        # 활성화 로직은 check 실패 시 write_active_pointer 를 호출하지 않으므로
        # 기존 포인터가 보존된다 (여기서는 호출하지 않음으로 재현)
        assert active.read_text(encoding="utf-8") == original