"""
test_dataset_audit.py - Task 7 완료 기준 검증.

합성 자료로만 테스트한다. 실제 연구 변수·측정값은 사용하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import os
from pyrolysis.config import load_problem
from pyrolysis.io import (
    DatasetAuditReport,
    DatasetReadError,
    TargetAuditResult,
    TrainingData,
    audit_dataset,
    read_training_file,
)

_REPO_ROOT = Path(__file__).parent.parent
_DEMO_YAML = _REPO_ROOT / "configs" / "demo.yaml"


@pytest.fixture
def demo_spec():
    return load_problem(_DEMO_YAML)


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """
    합성 학습 파일 (demo.yaml 규격 일치).

    row0: x1=1.0, x2=2.0, x3=cat_a, x4=0.5 | y1=10.0, y2=high
    row1: x1=2.0, x2=3.0, x3=cat_b, x4=0.6 | y1=NaN,  y2=low
    row2: x1=3.0, x2=4.0, x3=cat_c, x4=0.7 | y1=30.0, y2=high
    row3: x1=1.0, x2=2.0, x3=cat_a, x4=0.5 | y1=15.0, y2=NaN   (row0과 입력 동일)
    row4: x1=5.0, x2=6.0, x3=cat_a, x4=0.9 | y1=0.0,  y2=low
    """
    content = (
        "synthetic_x1,synthetic_x2,synthetic_x3,synthetic_x4,synthetic_y1,synthetic_y2" + chr(10) +
        "1.0,2.0,cat_a,0.5,10.0,high" + chr(10) +
        "2.0,3.0,cat_b,0.6,,low" + chr(10) +
        "3.0,4.0,cat_c,0.7,30.0,high" + chr(10) +
        "1.0,2.0,cat_a,0.5,15.0," + chr(10) +
        "5.0,6.0,cat_a,0.9,0.0,low" + chr(10)
    )
    p = tmp_path / "synthetic_train.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def training_data(synthetic_csv: Path, demo_spec) -> TrainingData:
    return read_training_file(synthetic_csv, demo_spec)


@pytest.fixture
def audit_report(training_data: TrainingData, demo_spec) -> DatasetAuditReport:
    return audit_dataset(training_data, demo_spec)


class TestReadTrainingFile:
    def test_read_csv_returns_training_data(self, synthetic_csv, demo_spec):
        data = read_training_file(synthetic_csv, demo_spec)
        assert isinstance(data, TrainingData)

    def test_total_rows_preserved(self, training_data):
        assert len(training_data.inputs) == 5
        assert len(training_data.targets) == 5

    def test_input_columns_separated(self, training_data):
        assert "synthetic_x1" in training_data.inputs.columns
        assert "synthetic_x4" in training_data.inputs.columns
        assert "synthetic_y1" not in training_data.inputs.columns

    def test_target_columns_separated(self, training_data):
        assert "synthetic_y1" in training_data.targets.columns
        assert "synthetic_y2" in training_data.targets.columns
        assert "synthetic_x1" not in training_data.targets.columns

    def test_values_not_modified(self, training_data):
        assert training_data.inputs["synthetic_x1"].iloc[0] == 1.0
        assert training_data.inputs["synthetic_x1"].iloc[4] == 5.0

    def test_extra_column_excluded_with_note(self, tmp_path, demo_spec):
        content = (
            "synthetic_x1,synthetic_x2,synthetic_x3,synthetic_x4,"
            "synthetic_y1,synthetic_y2,unknown_extra_col" + chr(10) +
            "1.0,2.0,cat_a,0.5,10.0,high,extra_val" + chr(10)
        )
        p = tmp_path / "extra.csv"
        p.write_text(content, encoding="utf-8")
        data = read_training_file(p, demo_spec)
        assert "unknown_extra_col" in data.extra_columns
        assert "unknown_extra_col" not in data.inputs.columns
        assert any("알 수 없는" in note for note in data.notes)

    def test_id_columns_separated(self, tmp_path, demo_spec):
        content = (
            "sample_id,synthetic_x1,synthetic_x2,synthetic_x3,synthetic_x4,synthetic_y1,synthetic_y2" + chr(10) +
            "S001,1.0,2.0,cat_a,0.5,10.0,high" + chr(10)
        )
        p = tmp_path / "with_id.csv"
        p.write_text(content, encoding="utf-8")
        data = read_training_file(p, demo_spec, id_columns=["sample_id"])
        assert "sample_id" in data.identifiers.columns
        assert "sample_id" not in data.inputs.columns
        assert "sample_id" not in data.targets.columns

    def test_column_map_applied(self, tmp_path, demo_spec):
        content = (
            "x1_custom,synthetic_x2,synthetic_x3,synthetic_x4,synthetic_y1,synthetic_y2" + chr(10) +
            "1.0,2.0,cat_a,0.5,10.0,high" + chr(10)
        )
        p = tmp_path / "mapped.csv"
        p.write_text(content, encoding="utf-8")
        data = read_training_file(p, demo_spec, column_map={"x1_custom": "synthetic_x1"})
        assert "synthetic_x1" in data.inputs.columns

    def test_nonexistent_file_raises(self, tmp_path, demo_spec):
        with pytest.raises(DatasetReadError):
            read_training_file(tmp_path / "no_such_file.csv", demo_spec)

    def test_empty_file_raises(self, tmp_path, demo_spec):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(DatasetReadError):
            read_training_file(p, demo_spec)

    def test_unsupported_format_raises(self, tmp_path, demo_spec):
        p = tmp_path / "data.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(DatasetReadError):
            read_training_file(p, demo_spec)

    def test_missing_target_column_noted_not_error(self, tmp_path, demo_spec):
        content = (
            "synthetic_x1,synthetic_x2,synthetic_x3,synthetic_x4,synthetic_y1" + chr(10) +
            "1.0,2.0,cat_a,0.5,10.0" + chr(10)
        )
        p = tmp_path / "partial.csv"
        p.write_text(content, encoding="utf-8")
        data = read_training_file(p, demo_spec)
        assert any("synthetic_y2" in note for note in data.notes)
        assert "synthetic_y2" in data.targets.columns
        assert data.targets["synthetic_y2"].isna().all()


class TestPartialTargets:
    def test_y1_available_count(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        assert y1.n_total == 5
        assert y1.n_available == 4
        assert y1.n_missing == 1

    def test_y2_available_count(self, audit_report):
        y2 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y2")
        assert y2.n_total == 5
        assert y2.n_available == 4
        assert y2.n_missing == 1

    def test_y1_missing_rows_not_removed(self, training_data):
        assert len(training_data.targets) == 5
        assert pd.isna(training_data.targets["synthetic_y1"].iloc[1])

    def test_y2_missing_rows_not_removed(self, training_data):
        assert len(training_data.targets) == 5
        assert pd.isna(training_data.targets["synthetic_y2"].iloc[3])

    def test_each_target_independent_count(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        y2 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y2")
        assert y1.n_available == 4
        assert y2.n_available == 4
        # 결측 개수는 같을 수 있으나(각 1행), 결측된 행 위치가 서로 다르다.
        # row1은 y1 결측, row3은 y2 결측 -> 대상별로 독립적으로 가용 행이 계산된다.
        assert y1.n_missing == 1
        assert y2.n_missing == 1


class TestMissingNotReplaced:
    def test_nan_stays_nan_in_targets(self, training_data):
        assert pd.isna(training_data.targets["synthetic_y1"].iloc[1])

    def test_nan_count_matches_expected(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        assert y1.n_missing == 1

    def test_zero_and_nan_reported_separately(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        assert y1.n_zero == 1
        assert y1.n_missing == 1
        assert y1.n_available == 4

    def test_zero_counted_in_available_not_missing(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        assert y1.n_available + y1.n_missing == y1.n_total


class TestReplicatedMeasurements:
    def test_replicated_rows_detected(self, audit_report):
        assert audit_report.n_replicated_rows == 2
        assert audit_report.replicated_input_groups == 1

    def test_replicated_rows_not_removed(self, training_data):
        assert len(training_data.inputs) == 5

    def test_replicated_noted(self, audit_report):
        assert any("반복" in note or "중복" in note for note in audit_report.notes)

    def test_no_replicated_when_all_unique(self, tmp_path, demo_spec):
        content = (
            "synthetic_x1,synthetic_x2,synthetic_x3,synthetic_x4,synthetic_y1,synthetic_y2" + chr(10) +
            "1.0,2.0,cat_a,0.1,10.0,high" + chr(10) +
            "2.0,3.0,cat_b,0.2,20.0,low" + chr(10) +
            "3.0,4.0,cat_c,0.3,30.0,high" + chr(10)
        )
        p = tmp_path / "unique.csv"
        p.write_text(content, encoding="utf-8")
        data = read_training_file(p, demo_spec)
        report = audit_dataset(data, demo_spec)
        assert report.n_replicated_rows == 0
        assert report.replicated_input_groups == 0


class TestClassificationCounts:
    def test_class_counts_for_classification_target(self, audit_report):
        y2 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y2")
        assert y2.class_counts is not None
        assert "high" in y2.class_counts
        assert "low" in y2.class_counts
        assert y2.class_counts["high"] == 2
        assert y2.class_counts["low"] == 2

    def test_class_counts_exclude_nan(self, audit_report):
        y2 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y2")
        total_classified = sum(y2.class_counts.values())
        assert total_classified == y2.n_available

    def test_regression_has_no_class_counts(self, audit_report):
        y1 = next(t for t in audit_report.target_results if t.target_id == "synthetic_y1")
        assert y1.class_counts is None


class TestAuditCommand:
    def test_audit_command_runs(self, synthetic_csv):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pyrolysis.cli", "audit",
             "--problem", str(_DEMO_YAML),
             "--dataset", str(synthetic_csv)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(_REPO_ROOT),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )
        assert result.returncode == 0, "audit 명령 실패:" + chr(10) + result.stderr
        assert "5" in result.stdout

    def test_audit_command_without_dataset_fails(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pyrolysis.cli", "audit",
             "--problem", str(_DEMO_YAML)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(_REPO_ROOT),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )
        assert result.returncode == 1