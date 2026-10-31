"""
test_demo_roundtrip.py - Task 4 완료 기준 검증.

확인 항목:
1. 저장 전후 동일 후보의 결과가 허용 오차(1e-6) 안에서 일치한다.
2. 재학습 없이 별도 프로세스에서 예측 가능 (load_bundle 후 predict_all).
3. 모델·규격 누락은 BundleError, 유효하지 않은 출력은 completed=False.
4. demo 묶음이 DEMO_BUNDLE_MARKER 를 포함한다.
5. 단일 후보를 CandidateTable 한 행으로 전달해 predict_all 을 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrolysis.artifacts import (
    DEMO_BUNDLE_MARKER,
    BundleError,
    load_bundle,
    save_demo_bundle,
)
from pyrolysis.config import load_problem
from pyrolysis.inference import make_single_candidate_table, predict_all
from pyrolysis.training import make_demo_data, run_demo_training

_REPO_ROOT = Path(__file__).parent.parent
_DEMO_YAML = _REPO_ROOT / "configs" / "demo.yaml"


@pytest.fixture
def demo_spec():
    return load_problem(_DEMO_YAML)


@pytest.fixture
def trained_result(demo_spec, tmp_path):
    """demo 학습 결과 딕셔너리."""
    return run_demo_training(demo_spec, tmp_path / "runs")


@pytest.fixture
def demo_bundle_dir(demo_spec, trained_result, tmp_path):
    """저장된 demo 묶음 디렉터리."""
    bundle_dir = save_demo_bundle(
        pipeline=trained_result["pipeline"],
        input_specs=demo_spec.inputs,
        target_spec=demo_spec.targets[0],
        output_dir=tmp_path / "bundles",
        bundle_id="demo_bundle_test",
        disclaimer=trained_result["disclaimer"],
    )
    return bundle_dir


# ── 1. 저장 전후 예측 일치 ────────────────────────────────────────────────────


class TestRoundtrip:
    def test_predictions_match_before_and_after_save(
        self, demo_spec, trained_result, demo_bundle_dir
    ) -> None:
        """저장 전후 동일 후보 예측값이 1e-6 허용 오차 안에서 일치해야 한다."""
        # 평가 자료로 저장 전 예측
        X, y = make_demo_data(demo_spec.inputs, demo_spec.targets)
        X_eval = X.iloc[:5]  # 5행만 사용
        pred_before = trained_result["pipeline"].predict(X_eval)

        # 묶음 로드 후 predict_all 로 예측
        bundle = load_bundle(demo_bundle_dir)
        candidates = make_single_candidate_table(
            X_eval.iloc[0].to_dict(), "demo_schema"
        )
        # 전체 5행 테이블로 예측
        from pyrolysis.contracts import CandidateTable
        full_table = CandidateTable(
            input_schema_id="demo_schema",
            source_rows=list(range(1, 6)),
            candidate_ids=[f"c{i}" for i in range(5)],
            data=X_eval.reset_index(drop=True),
        )
        batch = predict_all(bundle, full_table)

        assert batch.completed, f"예측 실패: {batch.error}"
        pred_after = np.array(batch.predictions[demo_spec.targets[0].id].values)
        np.testing.assert_allclose(
            pred_before, pred_after, rtol=0, atol=1e-6,
            err_msg="저장 전후 예측값이 허용 오차(1e-6)를 초과합니다."
        )

    def test_load_and_predict_without_retraining(
        self, demo_spec, demo_bundle_dir
    ) -> None:
        """로드 후 재학습 없이 예측이 성공해야 한다."""
        bundle = load_bundle(demo_bundle_dir)
        assert "pipeline" in bundle

        X, _ = make_demo_data(demo_spec.inputs, demo_spec.targets)
        from pyrolysis.contracts import CandidateTable
        table = CandidateTable(
            input_schema_id="demo_schema",
            source_rows=[1],
            candidate_ids=["c0"],
            data=X.iloc[:1].reset_index(drop=True),
        )
        batch = predict_all(bundle, table)
        assert batch.completed

    def test_single_candidate_via_make_single_table(
        self, demo_spec, trained_result, demo_bundle_dir
    ) -> None:
        """단일 입력 -> make_single_candidate_table -> predict_all 경로가 성공해야 한다."""
        X, _ = make_demo_data(demo_spec.inputs, demo_spec.targets)
        single_row = X.iloc[0].to_dict()

        bundle = load_bundle(demo_bundle_dir)
        candidates = make_single_candidate_table(single_row, "demo_schema")
        assert len(candidates.data) == 1

        batch = predict_all(bundle, candidates)
        assert batch.completed
        target_id = demo_spec.targets[0].id
        assert target_id in batch.predictions
        assert len(batch.predictions[target_id].values) == 1


# ── 2. 묶음 누락·손상 시 BundleError ────────────────────────────────────────


class TestBundleErrors:
    def test_missing_model_file_raises(self, demo_bundle_dir) -> None:
        """model.joblib 이 없으면 BundleError 가 발생해야 한다."""
        (demo_bundle_dir / "model.joblib").unlink()
        with pytest.raises(BundleError, match="model.joblib"):
            load_bundle(demo_bundle_dir)

    def test_missing_contract_file_raises(self, demo_bundle_dir) -> None:
        """contract.json 이 없으면 BundleError 가 발생해야 한다."""
        (demo_bundle_dir / "contract.json").unlink()
        with pytest.raises(BundleError, match="contract.json"):
            load_bundle(demo_bundle_dir)

    def test_nonexistent_dir_raises(self, tmp_path) -> None:
        """존재하지 않는 디렉터리이면 BundleError 가 발생해야 한다."""
        with pytest.raises(BundleError):
            load_bundle(tmp_path / "no_such_bundle")

    def test_missing_input_column_gives_failed_batch(
        self, demo_bundle_dir
    ) -> None:
        """필수 입력 열이 없으면 completed=False 인 배치를 반환해야 한다."""
        bundle = load_bundle(demo_bundle_dir)
        from pyrolysis.contracts import CandidateTable
        bad_table = CandidateTable(
            input_schema_id="demo_schema",
            source_rows=[1],
            candidate_ids=["c0"],
            data=pd.DataFrame([{"wrong_col": 1.0}]),
        )
        batch = predict_all(bundle, bad_table)
        assert not batch.completed
        assert batch.error is not None
        assert batch.error.code == "SCHEMA_MISMATCH"

    def test_no_partial_result_on_failure(self, demo_bundle_dir) -> None:
        """예측 실패 시 predictions 딕셔너리가 비어 있어야 한다."""
        bundle = load_bundle(demo_bundle_dir)
        from pyrolysis.contracts import CandidateTable
        bad_table = CandidateTable(
            input_schema_id="demo_schema",
            source_rows=[1],
            candidate_ids=["c0"],
            data=pd.DataFrame([{"wrong_col": 1.0}]),
        )
        batch = predict_all(bundle, bad_table)
        assert not batch.completed
        assert len(batch.predictions) == 0


# ── 3. demo 묶음 표지 확인 ───────────────────────────────────────────────────


class TestDemoBundleMarker:
    def test_contract_contains_demo_marker(self, demo_bundle_dir) -> None:
        """contract.json 의 bundle_type 이 DEMO_BUNDLE_MARKER 여야 한다."""
        bundle = load_bundle(demo_bundle_dir)
        assert bundle["contract"]["bundle_type"] == DEMO_BUNDLE_MARKER

    def test_demo_bundle_has_disclaimer(self, demo_bundle_dir) -> None:
        """contract.json 에 disclaimer 가 있어야 한다."""
        bundle = load_bundle(demo_bundle_dir)
        assert "disclaimer" in bundle["contract"]
        assert len(bundle["contract"]["disclaimer"]) > 0

    def test_environment_json_exists(self, demo_bundle_dir) -> None:
        """environment.json 이 저장되어야 한다."""
        assert (demo_bundle_dir / "environment.json").exists()