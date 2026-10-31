"""
test_model_pipelines.py - Task 12 완료 기준 검증 (새로 추가한 부분만).

기존 test_demo_training.py 가 이미 검증하는 항목(후보 키 존재, 전처리 fit 누출 방지,
Pipeline 구조)은 중복하지 않는다. 여기서는 작업 12에서 추가한
목표 변환/역변환, 라벨 매핑, 후보 적용 가능 여부, 회귀·분류 실제 실행만 검증한다.

합성 자료로만 테스트한다. 실제 연구 변수·변환은 사용하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrolysis.models import (
    CandidateStatus,
    LabelMappingError,
    apply_label_mapping,
    check_classification_candidates,
    get_classification_candidates,
    get_regression_candidates,
    invert_label_mapping,
)
from pyrolysis.preprocessing import (
    TargetTransform,
    TargetTransformError,
    build_classification_pipeline,
    build_regression_pipeline,
)
from pyrolysis.contracts import AllowedCategory, InputSpec


def _specs():
    return [
        InputSpec(id="x1", label_ko="x1", description="x1", role="feedstock", dtype="float", required=True),
        InputSpec(id="x2", label_ko="x2", description="x2", role="feedstock", dtype="float", required=True),
        InputSpec(id="x3", label_ko="x3", description="x3", role="feedstock", dtype="category",
                  required=True, allowed_categories=[AllowedCategory(value=c, label_ko=c) for c in ("a", "b")]),
    ]


def _X(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.choice(["a", "b"], size=n),
    })


class TestTargetTransform:
    def test_identity_roundtrip(self):
        t = TargetTransform("identity")
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.inverse(t.forward(y)), y)

    def test_log_roundtrip(self):
        t = TargetTransform("log")
        y = np.array([1.0, 2.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.forward(y)), y, rtol=1e-9)

    def test_unsupported_kind_rejected(self):
        with pytest.raises(TargetTransformError):
            TargetTransform("sqrt")  # 자동 선택·미지원 변환 거부

    def test_log_rejects_nonpositive_without_adding_constant(self):
        t = TargetTransform("log")
        with pytest.raises(TargetTransformError):
            t.forward(np.array([1.0, 0.0, 2.0]))  # 0 이하 -> 상수 안 더하고 중단

    def test_forward_preserves_nan(self):
        t = TargetTransform("identity")
        out = t.forward(np.array([1.0, np.nan, 3.0]))
        assert np.isnan(out[1])  # 목표 결측을 학습값으로 채우지 않음

    def test_default_is_identity(self):
        # 기본값이 변환 없음(identity) 이어야 자동 변환을 하지 않는다
        t = TargetTransform()
        assert t.kind == "identity"


class TestLabelMapping:
    def test_apply_and_invert(self):
        m = {"low": 0, "high": 1}
        ints = apply_label_mapping(["low", "high", "high"], m)
        assert ints == [0, 1, 1]
        back = invert_label_mapping(ints, m)
        assert back == ["low", "high", "high"]

    def test_unknown_label_rejected_not_substituted(self):
        m = {"low": 0, "high": 1}
        with pytest.raises(LabelMappingError):
            apply_label_mapping(["low", "medium"], m)  # 미정의 표지 치환 안 함

    def test_independent_binary_targets_allow_multiple_positives(self):
        # 작용기 2개를 독립 이진 대상으로 두면 한 시료가 둘 다 양성일 수 있다
        m = {"absent": 0, "present": 1}
        fg1 = apply_label_mapping(["present", "absent", "present"], m)
        fg2 = apply_label_mapping(["present", "present", "absent"], m)
        # 행 0 은 두 작용기 모두 양성 -> 동시 검출 표현 가능
        assert fg1[0] == 1 and fg2[0] == 1


class TestCandidateAvailability:
    def test_both_classes_present_all_available(self):
        st = check_classification_candidates(n_positive=5, n_negative=5)
        assert all(s.available for s in st.values())

    def test_one_class_missing_excludes_non_baseline_with_reason(self):
        st = check_classification_candidates(n_positive=10, n_negative=0)
        assert st["majority_baseline"].available is True
        for name in ("logistic_regression", "random_forest", "xgboost"):
            assert st[name].available is False
            assert st[name].reason  # 사유가 비어 있지 않음

    def test_no_samples_baseline_unavailable(self):
        st = check_classification_candidates(n_positive=0, n_negative=0)
        assert st["majority_baseline"].available is False


class TestCandidatesRunOnSyntheticData:
    def test_regression_candidates_fit_predict(self):
        specs = _specs()
        X = _X()
        y = X["x1"] * 2.0 + 1.0  # 합성 선형 관계
        for name, model in get_regression_candidates().items():
            pipe = build_regression_pipeline(specs, model)
            pipe.fit(X, y)
            pred = pipe.predict(X)
            assert len(pred) == len(X), name

    def test_classification_candidates_fit_predict(self):
        specs = _specs()
        X = _X()
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=len(X))  # 합성 이진 라벨 (두 클래스)
        # 두 클래스가 모두 존재하도록 보장
        y[0] = 0
        y[1] = 1
        for name, model in get_classification_candidates().items():
            pipe = build_classification_pipeline(specs, model)
            pipe.fit(X, y)
            pred = pipe.predict(X)
            assert len(pred) == len(X), name

    def test_regression_transform_integration(self):
        # 목표 변환을 적용해 학습하고 역변환으로 원래 단위 예측을 얻는다
        specs = _specs()
        X = _X()
        y = np.abs(X["x1"]) + 1.0  # 양수 목표
        t = TargetTransform("log")
        y_fit = t.forward(y.to_numpy())
        pipe = build_regression_pipeline(specs, get_regression_candidates()["linear_regression"])
        pipe.fit(X, y_fit)
        pred_log = pipe.predict(X)
        pred_orig = t.inverse(pred_log)
        assert len(pred_orig) == len(X)
        assert np.all(np.isfinite(pred_orig))