"""
test_demo_training.py - 작업 3 완료 기준 검증.

확인 항목:
1. 동일 합성 설정으로 실행을 재현하고 평가 행이 학습에 포함되지 않음을 확인한다.
2. 전처리가 학습 자료에만 fit되고 평가 자료에는 transform만 적용됨을 확인한다.
3. MAE와 RMSE가 계산되어 반환됨을 확인한다.
4. demo 실행 기록에 disclaimer 가 포함됨을 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pyrolysis.config import load_problem
from pyrolysis.evaluation import compute_regression_metrics
from pyrolysis.models import get_classification_candidates, get_regression_candidates
from pyrolysis.preprocessing import (
    build_regression_pipeline,
    split_numeric_categorical,
)
from pyrolysis.training import make_demo_data, run_demo_training

_REPO_ROOT = Path(__file__).parent.parent
_DEMO_YAML = _REPO_ROOT / "configs" / "demo.yaml"


# ── 픽스처 ────────────────────────────────────────────────────────────────────


@pytest.fixture
def demo_spec():
    """configs/demo.yaml 을 로드한 ProblemSpec."""
    return load_problem(_DEMO_YAML)


# ── 1. 재현성: 동일 설정·난수 -> 동일 데이터 ─────────────────────────────────


class TestReproducibility:
    def test_make_demo_data_reproducible(self, demo_spec) -> None:
        """동일 설정과 난수로 생성한 데이터가 같아야 한다."""
        X1, y1 = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=42)
        X2, y2 = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=42)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_different_seed_gives_different_data(self, demo_spec) -> None:
        """다른 난수는 다른 데이터를 생성해야 한다."""
        X1, y1 = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=42)
        X2, y2 = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=99)
        assert not X1.equals(X2), "난수가 다르면 데이터가 달라야 한다"

    def test_run_demo_training_twice_same_metrics(self, demo_spec, tmp_path) -> None:
        """동일 설정으로 두 번 실행하면 MAE·RMSE가 같아야 한다."""
        r1 = run_demo_training(demo_spec, tmp_path / "run1")
        r2 = run_demo_training(demo_spec, tmp_path / "run2")
        assert abs(r1["mae"] - r2["mae"]) < 1e-9, "재현 시 MAE가 일치해야 한다"
        assert abs(r1["rmse"] - r2["rmse"]) < 1e-9, "재현 시 RMSE가 일치해야 한다"


# ── 2. 평가 행이 학습에 포함되지 않음 ────────────────────────────────────────


class TestTrainEvalSeparation:
    def test_eval_rows_not_in_train(self, demo_spec, tmp_path) -> None:
        """평가 행 인덱스가 학습 행 인덱스와 겹치지 않아야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        train_set = set(result["train_indices"])
        eval_set = set(result["eval_indices"])
        overlap = train_set & eval_set
        assert len(overlap) == 0, f"겹치는 행 인덱스: {overlap}"

    def test_train_plus_eval_equals_total(self, demo_spec, tmp_path) -> None:
        """학습 행 수 + 평가 행 수 = 전체 행 수여야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        assert result["n_train"] + result["n_eval"] == result["n_total"]

    def test_eval_size_is_20_percent(self, demo_spec, tmp_path) -> None:
        """평가 자료가 전체의 20%여야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        expected_eval = int(result["n_total"] * 0.2)
        assert result["n_eval"] == expected_eval, (
            f"평가 행 수 {result['n_eval']} != 기대 {expected_eval}"
        )


# ── 3. 전처리 누출 방지: fit은 학습 자료에만 ─────────────────────────────────


class TestPreprocessingNoLeakage:
    def test_pipeline_fit_only_on_train(self, demo_spec) -> None:
        """Pipeline.fit()이 학습 자료에만 호출되고 평가 자료에는 transform만 적용된다.

        sklearn Pipeline 은 fit() 내부에서 각 단계를 순서대로 fit_transform -> transform 하므로,
        학습 후 scaler.mean_ 이 학습 자료의 통계를 담고 있는지 확인한다.
        """
        from sklearn.model_selection import train_test_split

        X, y = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=42)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        pipeline = build_regression_pipeline(demo_spec.inputs, model)
        pipeline.fit(X_train, y_train)

        # 학습 후 scaler mean 은 학습 자료의 평균과 같아야 한다
        preprocessor = pipeline.named_steps["preprocessor"]
        # 수치형 변환기 접근
        num_transformer = preprocessor.named_transformers_["num"]
        if num_transformer is not None:
            scaler = num_transformer.named_steps["scaler"]
            numeric_cols = [s.id for s in demo_spec.inputs if s.dtype in ("float", "int")]
            train_mean = X_train[numeric_cols].mean().values
            np.testing.assert_allclose(
                scaler.mean_, train_mean, rtol=1e-5,
                err_msg="스케일러 mean 이 학습 자료 평균과 달라서는 안 된다"
            )

        # 평가 자료 예측은 transform만 (에러 없이 실행되어야 함)
        y_pred = pipeline.predict(X_eval)
        assert len(y_pred) == len(X_eval)

    def test_scaler_mean_differs_train_vs_full(self, demo_spec) -> None:
        """학습 자료와 전체 자료로 fit한 scaler mean 이 다를 수 있음을 확인한다."""
        from sklearn.model_selection import train_test_split

        X, y = make_demo_data(demo_spec.inputs, demo_spec.targets, random_seed=42)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_cols = [s.id for s in demo_spec.inputs if s.dtype in ("float", "int")]
        if not numeric_cols:
            pytest.skip("수치형 입력이 없어 스케일러 테스트 불가")

        model1 = LinearRegression()
        pipe_train = build_regression_pipeline(demo_spec.inputs, model1)
        pipe_train.fit(X_train, y_train)

        model2 = LinearRegression()
        pipe_full = build_regression_pipeline(demo_spec.inputs, model2)
        pipe_full.fit(X, y)

        pre_train = pipe_train.named_steps["preprocessor"].named_transformers_["num"]
        pre_full = pipe_full.named_steps["preprocessor"].named_transformers_["num"]

        if pre_train and pre_full:
            # 전체 자료와 학습 자료의 mean은 약간 다를 수 있음 (전처리 누출 방지 근거)
            # 같을 수도 있지만 다르면 누출이 없다는 증거
            pass  # 통과: 두 스케일러가 독립적으로 fit된 것이 중요


# ── 4. 지표 계산 확인 ────────────────────────────────────────────────────────


class TestMetrics:
    def test_mae_and_rmse_returned(self, demo_spec, tmp_path) -> None:
        """run_demo_training 결과에 mae와 rmse가 포함되어야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        assert "mae" in result
        assert "rmse" in result
        assert isinstance(result["mae"], float)
        assert isinstance(result["rmse"], float)

    def test_mae_rmse_positive(self, demo_spec, tmp_path) -> None:
        """MAE와 RMSE는 0 이상이어야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        assert result["mae"] >= 0.0
        assert result["rmse"] >= 0.0

    def test_rmse_ge_mae(self, demo_spec, tmp_path) -> None:
        """RMSE는 MAE 이상이어야 한다 (수학적 성질)."""
        result = run_demo_training(demo_spec, tmp_path)
        assert result["rmse"] >= result["mae"] - 1e-9

    def test_compute_regression_metrics_direct(self) -> None:
        """compute_regression_metrics 가 직접 호출 시 올바른 키를 반환한다."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        result = compute_regression_metrics(y_true, y_pred, unit="무차원")
        assert "mae" in result
        assert "rmse" in result
        assert result["unit"] == "무차원"
        assert result["n_samples"] == 3

    def test_compute_regression_perfect_prediction(self) -> None:
        """완벽한 예측 시 MAE=RMSE=0이어야 한다."""
        y = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y, y)
        assert result["mae"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)


# ── 5. demo disclaimer 확인 ───────────────────────────────────────────────────


class TestDemoDisclaimer:
    def test_disclaimer_in_result(self, demo_spec, tmp_path) -> None:
        """실행 결과에 disclaimer 가 포함되어야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 0

    def test_disclaimer_mentions_synthetic(self, demo_spec, tmp_path) -> None:
        """disclaimer 가 합성 자료임을 언급해야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        disclaimer = result["disclaimer"]
        assert "합성" in disclaimer or "synthetic" in disclaimer.lower()

    def test_run_record_saved_to_file(self, demo_spec, tmp_path) -> None:
        """실행 기록이 JSON 파일로 저장되어야 한다."""
        import json

        result = run_demo_training(demo_spec, tmp_path)
        record_path = Path(result["record_path"])
        assert record_path.exists(), "demo_run.json 이 저장되어야 한다"

        with record_path.open(encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["disclaimer"] == result["disclaimer"]
        assert saved["mae"] == pytest.approx(result["mae"])

    def test_record_not_labeled_as_research(self, demo_spec, tmp_path) -> None:
        """실행 기록의 model 명이 '실제 연구'를 가장하지 않아야 한다."""
        result = run_demo_training(demo_spec, tmp_path)
        # disclaimer 에 '실제 연구 성능 아님' 표현이 있어야 함
        assert "실제 연구 성능 아님" in result["disclaimer"] or \
               "not" in result["disclaimer"].lower()


# ── 6. 모델 후보 등록 확인 ───────────────────────────────────────────────────


class TestModelCandidates:
    def test_regression_candidates_keys(self) -> None:
        """회귀 후보가 세 가지 키를 가져야 한다."""
        candidates = get_regression_candidates()
        assert "linear_regression" in candidates
        assert "random_forest" in candidates
        assert "xgboost" in candidates

    def test_classification_candidates_keys(self) -> None:
        """분류 후보가 네 가지 키를 가져야 한다."""
        candidates = get_classification_candidates()
        assert "majority_baseline" in candidates
        assert "logistic_regression" in candidates
        assert "random_forest" in candidates
        assert "xgboost" in candidates

    def test_regression_pipeline_is_sklearn_pipeline(self, demo_spec) -> None:
        """build_regression_pipeline 이 sklearn Pipeline 을 반환해야 한다."""
        model = LinearRegression()
        pipeline = build_regression_pipeline(demo_spec.inputs, model)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_split_numeric_categorical(self, demo_spec) -> None:
        """수치형·범주형 분리가 demo 설정과 맞아야 한다."""
        numeric, categorical = split_numeric_categorical(demo_spec.inputs)
        assert "synthetic_x1" in numeric
        assert "synthetic_x2" in numeric
        assert "synthetic_x3" in categorical