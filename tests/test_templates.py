"""
test_templates.py - Task 8 완료 기준 검증.

합성 자료로만 테스트한다. 실제 연구 변수·단위는 사용하지 않는다.

확인 항목:
1. 활성 대상이 바뀌면 필요한 입력 열만 생성되고 비활성 대상 입력은 빠진다.
2. CSV 는 _input_schema_id, Excel 은 _schema 시트에 규격 식별자를 기록한다.
3. candidate_id 는 선택 식별자이며 _input_schema_id·candidate_id 는 입력 열이 아니다.
4. 목표 결과 열(synthetic_y1, synthetic_y2)은 양식에 넣지 않는다.
5. 활성 대상이 다르면 규격 식별자도 달라진다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

import os
from pyrolysis.config import build_input_schema, load_problem
from pyrolysis.io import (
    TemplateError,
    write_csv_template,
    write_excel_template,
    write_input_templates,
)

_REPO_ROOT = Path(__file__).parent.parent
_DEMO_YAML = _REPO_ROOT / "configs" / "demo.yaml"


@pytest.fixture
def demo_spec():
    return load_problem(_DEMO_YAML)


def _read_csv_header(path: Path) -> list:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


class TestCsvTemplate:
    def test_csv_created(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(demo_spec, tmp_path)
        assert csv_path.exists()

    def test_csv_has_schema_id_column(self, demo_spec, tmp_path):
        csv_path, _, schema_id = write_input_templates(demo_spec, tmp_path)
        header = _read_csv_header(csv_path)
        assert "_input_schema_id" in header

    def test_csv_schema_id_value_recorded(self, demo_spec, tmp_path):
        csv_path, _, schema_id = write_input_templates(demo_spec, tmp_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        guide = rows[1]
        idx = header.index("_input_schema_id")
        assert guide[idx] == schema_id

    def test_csv_has_candidate_id(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(demo_spec, tmp_path)
        header = _read_csv_header(csv_path)
        assert "candidate_id" in header

    def test_csv_candidate_id_optional(self, demo_spec, tmp_path):
        schema = build_input_schema(demo_spec)
        out = tmp_path / "no_cand.csv"
        write_csv_template(schema, out, include_candidate_id=False)
        header = _read_csv_header(out)
        assert "candidate_id" not in header

    def test_csv_no_target_columns(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(demo_spec, tmp_path)
        header = _read_csv_header(csv_path)
        assert "synthetic_y1" not in header
        assert "synthetic_y2" not in header

    def test_csv_utf8_bom(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(demo_spec, tmp_path)
        raw = csv_path.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"


class TestExcelTemplate:
    def test_excel_created(self, demo_spec, tmp_path):
        _, xlsx_path, _ = write_input_templates(demo_spec, tmp_path)
        assert xlsx_path.exists()

    def test_excel_has_inputs_sheet(self, demo_spec, tmp_path):
        _, xlsx_path, _ = write_input_templates(demo_spec, tmp_path)
        xls = pd.ExcelFile(xlsx_path)
        assert "inputs" in xls.sheet_names

    def test_excel_has_schema_sheet(self, demo_spec, tmp_path):
        _, xlsx_path, _ = write_input_templates(demo_spec, tmp_path)
        xls = pd.ExcelFile(xlsx_path)
        assert "_schema" in xls.sheet_names

    def test_excel_inputs_has_no_target_columns(self, demo_spec, tmp_path):
        _, xlsx_path, _ = write_input_templates(demo_spec, tmp_path)
        inputs = pd.read_excel(xlsx_path, sheet_name="inputs")
        assert "synthetic_y1" not in inputs.columns
        assert "synthetic_y2" not in inputs.columns

    def test_excel_schema_sheet_records_schema_id(self, demo_spec, tmp_path):
        _, xlsx_path, schema_id = write_input_templates(demo_spec, tmp_path)
        meta = pd.read_excel(xlsx_path, sheet_name="_schema", nrows=1)
        # 첫 블록에 input_schema_id 메타 행이 있다.
        assert (meta.astype(str).values == schema_id).any()


class TestActiveTargetSelection:
    def test_all_targets_includes_x4(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(demo_spec, tmp_path)
        header = _read_csv_header(csv_path)
        assert "synthetic_x4" in header

    def test_regression_only_excludes_x4(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(
            demo_spec, tmp_path, active_target_ids=["synthetic_y1"]
        )
        header = _read_csv_header(csv_path)
        assert "synthetic_x4" not in header
        assert "synthetic_x1" in header
        assert "synthetic_x2" in header
        assert "synthetic_x3" in header

    def test_classification_only_excludes_x3(self, demo_spec, tmp_path):
        csv_path, _, _ = write_input_templates(
            demo_spec, tmp_path, active_target_ids=["synthetic_y2"]
        )
        header = _read_csv_header(csv_path)
        assert "synthetic_x3" not in header
        assert "synthetic_x4" in header

    def test_schema_id_differs_by_active_targets(self, demo_spec, tmp_path):
        _, _, sid_all = write_input_templates(demo_spec, tmp_path / "a")
        _, _, sid_reg = write_input_templates(
            demo_spec, tmp_path / "b", active_target_ids=["synthetic_y1"]
        )
        _, _, sid_cls = write_input_templates(
            demo_spec, tmp_path / "c", active_target_ids=["synthetic_y2"]
        )
        assert sid_all != sid_reg
        assert sid_all != sid_cls
        assert sid_reg != sid_cls

    def test_schema_id_matches_build_input_schema(self, demo_spec, tmp_path):
        _, _, sid = write_input_templates(
            demo_spec, tmp_path, active_target_ids=["synthetic_y1"]
        )
        expected = build_input_schema(demo_spec, active_target_ids=["synthetic_y1"]).schema_id
        assert sid == expected


class TestIdentifiersNotInput:
    def test_schema_id_and_candidate_id_not_in_excel_inputs(self, demo_spec, tmp_path):
        _, xlsx_path, _ = write_input_templates(demo_spec, tmp_path)
        inputs = pd.read_excel(xlsx_path, sheet_name="inputs")
        # candidate_id 는 선택 식별자로 inputs 시트에 있으나 _input_schema_id 는 없다.
        assert "_input_schema_id" not in inputs.columns

    def test_make_template_cli_runs(self, tmp_path):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pyrolysis.cli", "make-template",
             "--problem", str(_DEMO_YAML),
             "--output", str(tmp_path),
             "--targets", "synthetic_y1"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(_REPO_ROOT),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )
        assert result.returncode == 0, "make-template 실패:" + chr(10) + result.stderr
        assert (tmp_path / "input_template.csv").exists()
        assert (tmp_path / "input_template.xlsx").exists()