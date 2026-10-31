"""
test_inference.py - Task 18 완료 기준 검증.

합성 자료로만 테스트한다. 실제 연구 변수·단위는 사용하지 않는다.

확인 항목:
1. 마지막 행 입력 오류에서 모델을 호출하지 않고 요청 전체가 실패한다.
2. 중간/대상 실패에서 부분 결과를 게시하지 않는다 (completed=False, predictions 비움).
3. 같은 후보의 단일·일괄 결과가 같다.
4. 학습 범위 이탈 주의는 예측을 차단하지 않고 유지된다 (completed=True + range_warnings).
5. 분류는 정의된 표지로 반환한다.
6. 출력 제약 위반(비음수 등)은 OUTPUT_INVALID 전체 실패다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrolysis.contracts import (
    AllowedCategory,
    CandidateTable,
    InputSpec,
    NumericRange,
    TrainingDomain,
)
from pyrolysis.inference import predict_all_targets, make_single_candidate_table
from pyrolysis.models import (
    apply_label_mapping,
    get_classification_candidates,
    get_regression_candidates,
)
from pyrolysis.preprocessing import build_classification_pipeline, build_regression_pipeline


def _input_specs():
    return [
        InputSpec(id="x1", label_ko="x1", description="x1", role="feedstock", dtype="float", required=True),
        InputSpec(id="x2", label_ko="x2", description="x2", role="feedstock", dtype="float", required=True),
    ]


def _bundle(reg=True, cls=True, non_negative=False):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=40), "x2": rng.normal(size=40)})
    models = {}
    targets = []

    if reg:
        y = X["x1"] * 2.0 - X["x2"]
        rpipe = build_regression_pipeline(_input_specs(), get_regression_candidates()["linear_regression"])
        rpipe.fit(X, y)
        models["y_reg"] = rpipe
        oc = {"finite": True, "non_negative": non_negative}
        targets.append({"id": "y_reg", "task": "regression", "input_ids": ["x1", "x2"],
                        "unit": "wt%", "output_constraints": oc, "transform_kind": "identity"})

    if cls:
        labels_int = (X["x1"] > 0).astype(int).to_numpy()
        labels_int[0] = 0
        labels_int[1] = 1
        cpipe = build_classification_pipeline(_input_specs(), get_classification_candidates()["logistic_regression"])
        cpipe.fit(X, labels_int)
        models["y_cls"] = cpipe
        targets.append({"id": "y_cls", "task": "binary_classification", "input_ids": ["x1", "x2"],
                        "label_mapping": {"absent": 0, "present": 1}, "positive_label": "present"})

    contract = {
        "bundle_id": "b1",
        "input_ids": ["x1", "x2"],
        "targets": targets,
    }
    manifest = {"bundle_id": "b1", "active_targets": list(models.keys()),
                "file_hashes": {"manifest.json": "deadbeef"}}
    return {"models": models, "contract": contract, "manifest": manifest}, X


def _table(df):
    return CandidateTable(
        input_schema_id="s1",
        source_rows=list(range(1, len(df) + 1)),
        candidate_ids=["c" + str(i) for i in range(len(df))],
        data=df.reset_index(drop=True),
    )


class TestWholeRequestValidation:
    def test_last_row_error_blocks_prediction(self):
        bundle, X = _bundle()
        df = X.iloc[:3].copy()
        df.iloc[2, df.columns.get_loc("x1")] = np.nan  # 마지막 행 필수 결측
        batch = predict_all_targets(bundle, _table(df), input_specs=_input_specs())
        assert batch.completed is False
        assert len(batch.predictions) == 0  # 부분 결과 없음
        assert batch.error is not None

    def test_valid_request_completes(self):
        bundle, X = _bundle()
        batch = predict_all_targets(bundle, _table(X.iloc[:5]), input_specs=_input_specs())
        assert batch.completed is True
        assert "y_reg" in batch.predictions
        assert "y_cls" in batch.predictions


class TestSingleVsBatch:
    def test_single_matches_batch(self):
        bundle, X = _bundle(cls=False)
        row = X.iloc[0]
        single = predict_all_targets(bundle, make_single_candidate_table(row.to_dict(), "s1"),
                                     input_specs=_input_specs())
        batch = predict_all_targets(bundle, _table(X.iloc[:3]), input_specs=_input_specs())
        assert single.completed and batch.completed
        np.testing.assert_allclose(
            single.predictions["y_reg"].values[0],
            batch.predictions["y_reg"].values[0],
            rtol=0, atol=1e-9,
        )


class TestRangeWarningKeepsPrediction:
    def test_out_of_range_still_predicts_with_warning(self):
        bundle, X = _bundle(cls=False)
        df = X.iloc[:3].copy()
        df.iloc[0, df.columns.get_loc("x1")] = 999.0  # 학습 범위 밖 (물리 제약 없음)
        td = TrainingDomain(target_id="y_reg",
                            numeric_ranges={"x1": NumericRange(min_value=-5, max_value=5)})
        batch = predict_all_targets(bundle, _table(df), input_specs=_input_specs(),
                                    training_domains=[td])
        assert batch.completed is True  # 범위 이탈은 차단 안 함
        assert len(batch.range_warnings) >= 1
        assert any(w.code == "TRAINING_RANGE_EXCEEDED" for w in batch.range_warnings)


class TestClassificationLabels:
    def test_classification_returns_defined_labels(self):
        bundle, X = _bundle(reg=False)
        batch = predict_all_targets(bundle, _table(X.iloc[:5]), input_specs=_input_specs())
        assert batch.completed is True
        vals = batch.predictions["y_cls"].values
        assert all(v in ("absent", "present") for v in vals)


class TestOutputConstraint:
    def test_non_negative_violation_fails_request(self):
        # 회귀 목표에 비음수 제약을 걸고, 음수가 나오는 입력으로 위반 유도
        bundle, X = _bundle(cls=False, non_negative=True)
        # y = x1*2 - x2 가 음수가 되도록 하는 행이 포함됨 (일부 음수)
        batch = predict_all_targets(bundle, _table(X.iloc[:20]), input_specs=_input_specs())
        # 음수 예측이 하나라도 있으면 전체 실패
        if batch.completed:
            # 혹시 전부 양수면 테스트 가정이 깨지므로 음수 입력을 강제
            df = X.iloc[:3].copy()
            df.iloc[0, df.columns.get_loc("x1")] = -10.0
            df.iloc[0, df.columns.get_loc("x2")] = 10.0
            batch = predict_all_targets(bundle, _table(df), input_specs=_input_specs())
        assert batch.completed is False
        assert batch.error is not None
        assert batch.error.code == "OUTPUT_INVALID"


class TestNoActiveTarget:
    def test_empty_models_fails(self):
        bundle, X = _bundle()
        bundle["models"] = {}
        bundle["manifest"]["active_targets"] = []
        batch = predict_all_targets(bundle, _table(X.iloc[:3]), input_specs=_input_specs())
        assert batch.completed is False
        assert batch.error.code == "NO_ACTIVE_TARGET"