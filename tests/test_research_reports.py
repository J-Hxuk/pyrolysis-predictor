"""
test_research_reports.py - Task 15 완료 기준 검증.

합성 자료로만 테스트한다. 실제 분석 방법은 작업 27에서 확정한다.

확인 항목:
1. 미지원 분석 방법은 오류다 (자동 대체 안 함).
2. 변수 기여도를 명시적 방법으로 계산하고 JSON+PNG 로 저장한다.
3. 변환된 변수 -> 원래 변수 대응을 기록한다.
4. 후보 비교표를 대상별로 분리 저장한다 (단위 다른 대상과 안 섞음).
5. 예측·실측 비교 그림을 저장한다.
6. 산출물에 인과관계 비설명 주의 문구가 포함된다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyrolysis.analysis import (
    DISCLAIMER,
    AnalysisError,
    compute_variable_importance,
    save_candidate_comparison,
    save_pred_vs_actual,
    save_variable_importance,
)
from pyrolysis.models import get_regression_candidates
from pyrolysis.preprocessing import build_regression_pipeline
from pyrolysis.contracts import InputSpec


def _specs():
    return [
        InputSpec(id="x1", label_ko="x1", description="x1", role="feedstock", dtype="float", required=True),
        InputSpec(id="x2", label_ko="x2", description="x2", role="feedstock", dtype="float", required=True),
    ]


def _fitted(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=40), "x2": rng.normal(size=40)})
    y = X["x1"] * 3.0 - X["x2"]  # x1 이 더 중요한 합성 관계
    pipe = build_regression_pipeline(_specs(), get_regression_candidates()["linear_regression"])
    pipe.fit(X, y)
    return pipe, X, y


class TestVariableImportance:
    def test_unsupported_method_rejected(self):
        pipe, X, y = _fitted()
        with pytest.raises(AnalysisError):
            compute_variable_importance(pipe, X, y, method="shap")  # 미지원 -> 오류

    def test_permutation_runs(self):
        pipe, X, y = _fitted()
        imp = compute_variable_importance(pipe, X, y, method="permutation",
                                          n_repeats=5, random_seed=1)
        assert imp["method"] == "permutation"
        assert set(imp["importances_mean"].keys()) == {"x1", "x2"}
        assert "disclaimer" in imp

    def test_save_importance_creates_json_and_png(self, tmp_path):
        pipe, X, y = _fitted()
        imp = compute_variable_importance(pipe, X, y, method="permutation",
                                          n_repeats=5, random_seed=1)
        jpath, ppath = save_variable_importance(
            imp, tmp_path, "y1",
            feature_origin={"x1": "x1_raw", "x2": "x2_raw"},
        )
        assert jpath.exists()
        assert ppath.exists()
        rec = json.loads(jpath.read_text(encoding="utf-8"))
        assert rec["target_id"] == "y1"
        assert rec["feature_origin"] == {"x1": "x1_raw", "x2": "x2_raw"}
        # 인과관계 비설명 주의 문구가 기록된다
        assert "인과관계" in rec["disclaimer"]


class TestCandidateComparison:
    def test_saved_per_target_with_unit(self, tmp_path):
        rows = [
            {"candidate": "linear_regression", "score": 0.5, "ran": True, "reason": ""},
            {"candidate": "xgboost", "score": None, "ran": False, "reason": "제외"},
        ]
        path = save_candidate_comparison(rows, tmp_path, "y1",
                                         metric_name="mae", metric_unit="wt%")
        assert path.exists()
        df = pd.read_csv(path)
        assert (df["target_id"] == "y1").all()
        assert (df["metric_name"] == "mae").all()
        assert (df["metric_unit"] == "wt%").all()

    def test_separate_files_for_different_targets(self, tmp_path):
        rows = [{"candidate": "a", "score": 0.1, "ran": True, "reason": ""}]
        p1 = save_candidate_comparison(rows, tmp_path, "y1", metric_name="mae", metric_unit="wt%")
        p2 = save_candidate_comparison(rows, tmp_path, "y2", metric_name="f1", metric_unit=None)
        # 단위가 다른 대상은 서로 다른 파일로 분리된다 (한 파일에 합산 안 함)
        assert p1 != p2
        assert p1.exists() and p2.exists()


class TestPredVsActual:
    def test_saves_png(self, tmp_path):
        pipe, X, y = _fitted()
        y_pred = pipe.predict(X)
        path = save_pred_vs_actual(y, y_pred, tmp_path, "y1", unit="wt%")
        assert path.exists()
        assert path.suffix == ".png"


class TestDisclaimer:
    def test_disclaimer_mentions_no_causation(self):
        assert "인과관계" in DISCLAIMER
        assert "합산" in DISCLAIMER  # 서로 다른 단위 합산 안 함 명시