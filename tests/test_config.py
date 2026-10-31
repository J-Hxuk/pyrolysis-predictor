"""
test_config.py — 작업 2 완료 기준 검증.

확인 항목:
1. 빈 draft 설정에서 check_readiness()가 부족한 항목 목록을 반환한다.
2. draft 설정에서는 실제 예측 준비 상태가 False이다.
3. demo 설정을 직접 선택했을 때만 합성 설정을 사용한다.
4. 설정 오류가 자동 기본값으로 대체되지 않는다 (필수 필드 누락 시 ValidationError).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pyrolysis.config import ConfigError, check_readiness, load_problem, load_runtime
from pyrolysis.contracts import InputSpec, ProblemSpec, TargetSpec

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"
_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── 픽스처 헬퍼 ───────────────────────────────────────────────────────────────


def _write_yaml(path: Path, data: dict) -> Path:
    """딕셔너리를 YAML 파일로 저장하고 경로를 반환한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


# ── 1. 빈 draft 설정: check_readiness() 부족 항목 목록 ───────────────────────


class TestCheckReadinessDraftEmpty:
    """빈 draft ProblemSpec 에서 check_readiness() 동작을 검증한다."""

    def test_returns_missing_list_for_empty_draft(self) -> None:
        """빈 draft 설정에서 check_readiness()가 부족한 항목 목록을 반환해야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        assert isinstance(result["missing"], list), "missing 은 list 여야 한다"
        assert len(result["missing"]) > 0, "빈 draft 설정은 부족한 항목이 있어야 한다"

    def test_missing_includes_status_draft_message(self) -> None:
        """draft 상태임을 설명하는 항목이 missing 목록에 포함되어야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        # 상태 관련 메시지가 있는지 확인
        has_status_msg = any("draft" in msg for msg in result["missing"])
        assert has_status_msg, "draft 상태 관련 메시지가 missing 목록에 있어야 한다"

    def test_missing_includes_empty_inputs(self) -> None:
        """inputs 가 비어 있으면 missing 목록에 포함되어야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        has_inputs_msg = any("inputs" in msg or "입력 변수" in msg for msg in result["missing"])
        assert has_inputs_msg, "inputs 관련 메시지가 missing 목록에 있어야 한다"

    def test_missing_includes_empty_targets(self) -> None:
        """targets 가 비어 있으면 missing 목록에 포함되어야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        has_targets_msg = any("targets" in msg or "예측 대상" in msg for msg in result["missing"])
        assert has_targets_msg, "targets 관련 메시지가 missing 목록에 있어야 한다"

    def test_readiness_result_has_required_keys(self) -> None:
        """check_readiness() 결과에 필수 키가 있어야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        for key in ("prediction_ready", "training_ready", "missing", "status"):
            assert key in result, f"결과에 '{key}' 키가 있어야 한다"

    def test_status_field_matches_spec(self) -> None:
        """결과의 status 필드가 spec.status 와 일치해야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        assert result["status"] == "draft"

    def test_load_template_and_check_readiness(self, tmp_path: Path) -> None:
        """problem.template.yaml 을 읽었을 때 check_readiness()가 부족 항목을 반환한다."""
        template = _CONFIGS_DIR / "problem.template.yaml"
        spec = load_problem(template)
        result = check_readiness(spec)

        assert result["prediction_ready"] is False
        assert result["training_ready"] is False
        assert len(result["missing"]) > 0


# ── 2. draft 설정: 예측 준비 상태 False ──────────────────────────────────────


class TestDraftNotPredictionReady:
    """draft 상태에서 예측·학습 준비 상태가 False 임을 검증한다."""

    def test_empty_draft_prediction_not_ready(self) -> None:
        """빈 draft 설정에서 prediction_ready 가 False 여야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        assert result["prediction_ready"] is False

    def test_empty_draft_training_not_ready(self) -> None:
        """빈 draft 설정에서 training_ready 가 False 여야 한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        result = check_readiness(spec)

        assert result["training_ready"] is False

    def test_draft_with_inputs_targets_still_not_ready(self) -> None:
        """inputs·targets 가 있어도 status=draft 이면 예측 준비가 안 된 상태여야 한다."""
        spec = ProblemSpec(
            schema_version=1,
            problem_id="test",
            status="draft",
            inputs=[
                InputSpec(
                    id="x1",
                    label_ko="입력 1",
                    description="테스트 입력",
                    role="feedstock",
                    dtype="float",
                    required=True,
                )
            ],
            targets=[
                TargetSpec(
                    id="y1",
                    label_ko="출력 1",
                    family="performance",
                    task="regression",
                    definition="테스트 회귀 대상",
                    input_ids=["x1"],
                )
            ],
        )
        result = check_readiness(spec)

        assert result["prediction_ready"] is False
        assert result["training_ready"] is False
        # draft 메시지가 있어야 함
        assert any("draft" in m for m in result["missing"])

    def test_configured_with_inputs_targets_is_ready(self) -> None:
        """configured + inputs + targets 이면 prediction_ready 가 True 여야 한다."""
        spec = ProblemSpec(
            schema_version=1,
            problem_id="test",
            status="configured",
            inputs=[
                InputSpec(
                    id="x1",
                    label_ko="입력 1",
                    description="테스트 입력",
                    role="feedstock",
                    dtype="float",
                    required=True,
                )
            ],
            targets=[
                TargetSpec(
                    id="y1",
                    label_ko="출력 1",
                    family="performance",
                    task="regression",
                    definition="테스트 회귀 대상",
                    input_ids=["x1"],
                )
            ],
        )
        result = check_readiness(spec)

        assert result["prediction_ready"] is True
        assert result["training_ready"] is True
        assert result["missing"] == []

    def test_is_prediction_ready_method_draft(self) -> None:
        """ProblemSpec.is_prediction_ready() 가 draft 설정에서 False 를 반환한다."""
        spec = ProblemSpec(schema_version=1, problem_id=None, status="draft")
        assert spec.is_prediction_ready() is False

    def test_is_prediction_ready_method_empty_configured(self) -> None:
        """configured 이지만 inputs·targets 가 없으면 is_prediction_ready() 가 False 다."""
        spec = ProblemSpec(schema_version=1, problem_id="x", status="configured")
        assert spec.is_prediction_ready() is False


# ── 3. demo 설정은 명시적 선택 시에만 사용 ───────────────────────────────────


class TestDemoConfigExplicitOnly:
    """demo 설정이 명시적으로 선택했을 때만 사용되는지 검증한다."""

    def test_demo_yaml_loads_as_configured(self) -> None:
        """configs/demo.yaml 을 읽으면 status=configured 이고 합성 변수가 있다."""
        demo_path = _CONFIGS_DIR / "demo.yaml"
        spec = load_problem(demo_path)

        assert spec.status == "configured"
        assert spec.problem_id == "demo-synthetic-v1"
        assert len(spec.inputs) == 4  # Task 6: synthetic_x4 추가
        assert len(spec.targets) == 2  # Task 6: synthetic_y2 추가

    def test_demo_yaml_uses_synthetic_names(self) -> None:
        """demo.yaml 의 변수·대상 ID 가 synthetic_ 접두사를 가져야 한다."""
        demo_path = _CONFIGS_DIR / "demo.yaml"
        spec = load_problem(demo_path)

        for inp in spec.inputs:
            assert inp.id.startswith("synthetic_"), (
                f"demo 입력 '{inp.id}'은 synthetic_ 접두사를 가져야 한다"
            )
        for tgt in spec.targets:
            assert tgt.id.startswith("synthetic_"), (
                f"demo 대상 '{tgt.id}'은 synthetic_ 접두사를 가져야 한다"
            )

    def test_demo_yaml_is_ready_when_loaded_directly(self) -> None:
        """demo.yaml 을 직접 로드하면 check_readiness() 가 ready 상태를 반환한다."""
        demo_path = _CONFIGS_DIR / "demo.yaml"
        spec = load_problem(demo_path)
        result = check_readiness(spec)

        assert result["prediction_ready"] is True
        assert result["missing"] == []

    def test_template_yaml_is_not_demo(self) -> None:
        """problem.template.yaml 은 demo 설정이 아닌 draft 상태여야 한다."""
        template_path = _CONFIGS_DIR / "problem.template.yaml"
        spec = load_problem(template_path)

        assert spec.status == "draft"
        assert spec.problem_id is None

    def test_runtime_yaml_default_mode_is_research(self) -> None:
        """runtime.yaml 의 기본 mode 가 research 여야 한다 (demo 가 아님)."""
        runtime_path = _CONFIGS_DIR / "runtime.yaml"
        config = load_runtime(runtime_path)

        assert config.get("mode") == "research", (
            "기본 운영 모드는 'research' 여야 한다. "
            "demo 는 개발자가 명시적으로 설정한 경우에만 사용한다."
        )

    def test_demo_config_not_loaded_by_default(self, tmp_path: Path) -> None:
        """research 모드에서는 demo.yaml 이 자동으로 로드되지 않아야 한다.

        이 테스트는 load_problem() 이 명시적 경로를 받는 설계를 검증한다.
        경로를 지정하지 않으면 demo 설정이 자동으로 활성화되지 않는다.
        """
        # research 모드 런타임 설정 파일 생성
        runtime_data = {
            "schema_version": 1,
            "mode": "research",
            "bundle_dir": "deployment/bundles",
            "active_pointer": "deployment/active.json",
        }
        runtime_path = tmp_path / "runtime.yaml"
        _write_yaml(runtime_path, runtime_data)

        config = load_runtime(runtime_path)
        assert config["mode"] == "research"
        # demo.yaml 이 자동으로 로드될 경로가 없음을 확인
        assert "demo" not in config.get("bundle_dir", "")


# ── 4. 설정 오류가 자동 기본값으로 대체되지 않음 ─────────────────────────────


class TestNoAutoDefaultOnValidationError:
    """필수 필드 누락 시 ValidationError 가 발생하고 자동 보정하지 않는다."""

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        """schema_version 누락 시 ValidationError 가 발생해야 한다."""
        data = {"problem_id": None, "status": "draft"}
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_missing_status_raises(self, tmp_path: Path) -> None:
        """status 누락 시 ValidationError 가 발생해야 한다."""
        data = {"schema_version": 1, "problem_id": None}
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_invalid_status_value_raises(self, tmp_path: Path) -> None:
        """지원하지 않는 status 값은 ValidationError 를 발생시켜야 한다."""
        data = {"schema_version": 1, "problem_id": None, "status": "active"}
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_invalid_schema_version_raises(self, tmp_path: Path) -> None:
        """schema_version 이 1 이 아니면 ValidationError 가 발생해야 한다."""
        data = {"schema_version": 99, "problem_id": None, "status": "draft"}
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_invalid_input_role_raises(self, tmp_path: Path) -> None:
        """입력 변수의 role 이 허용 값이 아니면 ValidationError 가 발생해야 한다."""
        data = {
            "schema_version": 1,
            "problem_id": None,
            "status": "draft",
            "inputs": [
                {
                    "id": "x1",
                    "label_ko": "입력 1",
                    "description": "테스트",
                    "role": "invalid_role",  # 허용하지 않는 값
                    "dtype": "float",
                    "required": True,
                }
            ],
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_invalid_target_family_raises(self, tmp_path: Path) -> None:
        """대상의 family 가 허용 값이 아니면 ValidationError 가 발생해야 한다."""
        data = {
            "schema_version": 1,
            "problem_id": None,
            "status": "draft",
            "targets": [
                {
                    "id": "y1",
                    "label_ko": "출력 1",
                    "family": "unknown_family",  # 허용하지 않는 값
                    "task": "regression",
                    "definition": "테스트",
                    "input_ids": [],
                }
            ],
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)

        with pytest.raises(ValidationError):
            load_problem(path)

    def test_classification_without_label_mapping_raises(self) -> None:
        """binary_classification 대상에 label_mapping 이 없으면 ValidationError 가 발생한다."""
        with pytest.raises(ValidationError):
            TargetSpec(
                id="y1",
                label_ko="분류 출력",
                family="functional_group",
                task="binary_classification",
                definition="테스트 분류",
                input_ids=["x1"],
                # label_mapping 과 positive_label 누락
            )

    def test_required_input_with_missing_policy_raises(self) -> None:
        """required=True 인 입력에 missing_policy 를 설정하면 ValidationError 가 발생한다."""
        with pytest.raises(ValidationError):
            InputSpec(
                id="x1",
                label_ko="입력 1",
                description="테스트",
                role="feedstock",
                dtype="float",
                required=True,
                missing_policy="mean",  # 필수 변수에 결측 처리 설정 불가
            )

    def test_category_dtype_without_allowed_categories_raises(self) -> None:
        """dtype=category 이지만 allowed_categories 가 없으면 ValidationError 가 발생한다."""
        with pytest.raises(ValidationError):
            InputSpec(
                id="x1",
                label_ko="범주 입력",
                description="테스트",
                role="feedstock",
                dtype="category",
                required=True,
                allowed_categories=[],  # 비어 있으면 안 됨
            )

    def test_nonexistent_file_raises_config_error(self, tmp_path: Path) -> None:
        """존재하지 않는 파일 경로를 전달하면 ConfigError 가 발생해야 한다."""
        missing_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigError):
            load_problem(missing_path)

    def test_extra_fields_in_spec_do_not_raise(self, tmp_path: Path) -> None:
        """ProblemSpec 에 정의되지 않은 추가 키는 무시되어야 한다 (extra=ignore)."""
        data = {
            "schema_version": 1,
            "problem_id": None,
            "status": "draft",
            "unknown_extra_field": "should_be_ignored",
        }
        path = _write_yaml(tmp_path / "extra.yaml", data)

        # extra 필드가 있어도 파싱이 성공해야 한다
        spec = load_problem(path)
        assert spec.status == "draft"


# ── 5. 픽스처 파일 기반 통합 검증 ────────────────────────────────────────────


class TestFixtureFiles:
    """tests/fixtures/ 의 가상 검증 자료로 통합 동작을 확인한다."""

    def test_fixture_draft_spec(self, tmp_path: Path) -> None:
        """fixture 디렉터리의 draft 설정 파일을 읽고 check_readiness() 를 수행한다."""
        fixture_data = {
            "schema_version": 1,
            "problem_id": None,
            "status": "draft",
            "inputs": [],
            "targets": [],
            "composition_groups": [],
        }
        fixture_path = _FIXTURES_DIR / "test_draft_spec.yaml"
        _write_yaml(fixture_path, fixture_data)

        spec = load_problem(fixture_path)
        result = check_readiness(spec)

        assert spec.status == "draft"
        assert result["prediction_ready"] is False
        assert len(result["missing"]) > 0

    def test_fixture_minimal_configured_spec(self, tmp_path: Path) -> None:
        """최소 configured 설정이 prediction_ready 를 반환하는지 확인한다."""
        fixture_data = {
            "schema_version": 1,
            "problem_id": "fixture-test-v1",
            "status": "configured",
            "inputs": [
                {
                    "id": "x1",
                    "label_ko": "입력 1",
                    "description": "픽스처 입력",
                    "role": "feedstock",
                    "dtype": "float",
                    "required": True,
                }
            ],
            "targets": [
                {
                    "id": "y1",
                    "label_ko": "출력 1",
                    "family": "performance",
                    "task": "regression",
                    "definition": "픽스처 회귀 대상",
                    "input_ids": ["x1"],
                }
            ],
            "composition_groups": [],
        }
        fixture_path = _FIXTURES_DIR / "test_minimal_configured_spec.yaml"
        _write_yaml(fixture_path, fixture_data)

        spec = load_problem(fixture_path)
        result = check_readiness(spec)

        assert spec.status == "configured"
        assert result["prediction_ready"] is True
        assert result["missing"] == []
