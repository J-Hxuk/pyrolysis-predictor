from __future__ import annotations
from pathlib import Path
import pytest
from pydantic import ValidationError
from pyrolysis.config import ConfigError, build_input_schema, load_problem
from pyrolysis.contracts import AllowedCategory, InputSchema, InputSpec, ProblemSpec, TargetSpec

_DEMO = Path(__file__).parent.parent / "configs" / "demo.yaml"


def _float(id, unit=None, basis=None):
    return InputSpec(id=id, label_ko=id, description=id, role="feedstock",
                     dtype="float", unit=unit, basis=basis, required=True)

def _cat(id, cats):
    return InputSpec(id=id, label_ko=id, description=id, role="feedstock",
                     dtype="category", required=True,
                     allowed_categories=[AllowedCategory(value=v, label_ko=v) for v in cats])

def _reg(id, input_ids):
    return TargetSpec(id=id, label_ko=id, family="performance", task="regression",
                      definition=id, input_ids=input_ids)

def _cls(id, input_ids):
    return TargetSpec(id=id, label_ko=id, family="performance", task="binary_classification",
                      definition=id, input_ids=input_ids,
                      label_mapping={"low": 0, "high": 1}, positive_label="high")


class TestInputSchemaBuild:
    def test_single_spec(self):
        s = InputSchema.build([_float("x1")])
        assert s.input_ids == ["x1"]

    def test_sorted_by_id(self):
        s = InputSchema.build([_float("x3"), _float("x1"), _float("x2")])
        assert s.input_ids == ["x1", "x2", "x3"]

    def test_schema_id_12_hex(self):
        s = InputSchema.build([_float("x1")])
        assert len(s.schema_id) == 12
        assert all(c in "0123456789abcdef" for c in s.schema_id)

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            InputSchema.build([_float("x1"), _float("x1")])


class TestSchemaIdRules:
    def test_identical_same_id(self):
        assert InputSchema.build([_float("x1")]).schema_id == InputSchema.build([_float("x1")]).schema_id

    def test_label_change_keeps_id(self):
        a = InputSpec(id="x1", label_ko="A", description="d", role="feedstock", dtype="float", required=True)
        b = InputSpec(id="x1", label_ko="B", description="d2", role="feedstock", dtype="float", required=True)
        assert InputSchema.build([a]).schema_id == InputSchema.build([b]).schema_id

    def test_display_order_keeps_id(self):
        a = InputSpec(id="x1", label_ko="n", description="d", role="feedstock", dtype="float", required=True, display_order=1)
        b = InputSpec(id="x1", label_ko="n", description="d", role="feedstock", dtype="float", required=True, display_order=99)
        assert InputSchema.build([a]).schema_id == InputSchema.build([b]).schema_id

    def test_unit_change_changes_id(self):
        assert InputSchema.build([_float("x1", unit="kg")]).schema_id != InputSchema.build([_float("x1", unit="g")]).schema_id

    def test_dtype_change_changes_id(self):
        f = _float("x1")
        i = InputSpec(id="x1", label_ko="n", description="d", role="feedstock", dtype="int", required=True)
        assert InputSchema.build([f]).schema_id != InputSchema.build([i]).schema_id

    def test_category_values_change_id(self):
        assert InputSchema.build([_cat("x1", ["a", "b"])]).schema_id != InputSchema.build([_cat("x1", ["a", "c"])]).schema_id

    def test_category_order_irrelevant(self):
        assert InputSchema.build([_cat("x1", ["a", "b", "c"])]).schema_id == InputSchema.build([_cat("x1", ["c", "a", "b"])]).schema_id

    def test_add_input_changes_id(self):
        assert InputSchema.build([_float("x1")]).schema_id != InputSchema.build([_float("x1"), _float("x2")]).schema_id

    def test_basis_change_changes_id(self):
        assert InputSchema.build([_float("x1", basis="dry")]).schema_id != InputSchema.build([_float("x1", basis="wet")]).schema_id


class TestBuildInputSchema:
    def _spec(self):
        return ProblemSpec(schema_version=1, problem_id="t", status="configured",
                           inputs=[_float("shared"), _float("reg_only"), _float("cls_only")],
                           targets=[_reg("ry", ["shared", "reg_only"]), _cls("cy", ["shared", "cls_only"])])

    def test_all_active_includes_all(self):
        s = build_input_schema(self._spec())
        assert set(s.input_ids) == {"shared", "reg_only", "cls_only"}

    def test_reg_only_excludes_cls_input(self):
        s = build_input_schema(self._spec(), active_target_ids=["ry"])
        assert "cls_only" not in s.input_ids
        assert "reg_only" in s.input_ids

    def test_cls_only_excludes_reg_input(self):
        s = build_input_schema(self._spec(), active_target_ids=["cy"])
        assert "reg_only" not in s.input_ids
        assert "cls_only" in s.input_ids

    def test_shared_always_included(self):
        sp = self._spec()
        assert "shared" in build_input_schema(sp, ["ry"]).input_ids
        assert "shared" in build_input_schema(sp, ["cy"]).input_ids

    def test_no_active_raises(self):
        with pytest.raises(ConfigError, match="No active"):
            build_input_schema(self._spec(), active_target_ids=[])

    def test_unknown_active_id_raises(self):
        with pytest.raises(ConfigError, match="No active"):
            build_input_schema(self._spec(), active_target_ids=["no_such"])

    def test_different_active_different_schema_id(self):
        sp = self._spec()
        assert build_input_schema(sp).schema_id != build_input_schema(sp, ["ry"]).schema_id

    def test_same_active_same_schema_id(self):
        sp = self._spec()
        assert build_input_schema(sp, ["ry"]).schema_id == build_input_schema(sp, ["ry"]).schema_id


class TestConflictRejection:
    def test_duplicate_input_ids_raises(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            ProblemSpec(schema_version=1, problem_id="t", status="configured",
                        inputs=[_float("x1"), _float("x1")],
                        targets=[_reg("y1", ["x1"])])

    def test_duplicate_target_ids_raises(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            ProblemSpec(schema_version=1, problem_id="t", status="configured",
                        inputs=[_float("x1")],
                        targets=[_reg("y1", ["x1"]), _reg("y1", ["x1"])])

    def test_unknown_input_ref_raises_when_configured(self):
        with pytest.raises(ValidationError, match="unknown inputs"):
            ProblemSpec(schema_version=1, problem_id="t", status="configured",
                        inputs=[_float("x1")],
                        targets=[_reg("y1", ["x1", "x_missing"])])

    def test_unknown_input_ref_ok_in_draft(self):
        sp = ProblemSpec(schema_version=1, problem_id=None, status="draft",
                         inputs=[], targets=[_reg("y1", ["x_missing"])])
        assert sp.status == "draft"

    def test_classification_without_label_mapping_raises(self):
        with pytest.raises(ValidationError):
            TargetSpec(id="y1", label_ko="c", family="performance", task="binary_classification",
                       definition="d", input_ids=["x1"])


class TestDemoYaml:
    def test_has_regression_and_classification(self):
        spec = load_problem(_DEMO)
        tasks = {t.task for t in spec.targets}
        assert "regression" in tasks
        assert "binary_classification" in tasks

    def test_classification_has_label_mapping(self):
        spec = load_problem(_DEMO)
        for t in spec.targets:
            if t.task == "binary_classification":
                assert t.label_mapping is not None
                assert t.positive_label is not None

    def test_all_active_includes_x4(self):
        spec = load_problem(_DEMO)
        s = build_input_schema(spec)
        assert "synthetic_x4" in s.input_ids

    def test_regression_only_excludes_x4(self):
        spec = load_problem(_DEMO)
        s = build_input_schema(spec, active_target_ids=["synthetic_y1"])
        assert "synthetic_x4" not in s.input_ids

    def test_classification_only_excludes_x3(self):
        spec = load_problem(_DEMO)
        s = build_input_schema(spec, active_target_ids=["synthetic_y2"])
        assert "synthetic_x3" not in s.input_ids

    def test_schema_id_differs_by_active_set(self):
        spec = load_problem(_DEMO)
        s_all = build_input_schema(spec)
        s_reg = build_input_schema(spec, ["synthetic_y1"])
        s_cls = build_input_schema(spec, ["synthetic_y2"])
        assert s_all.schema_id != s_reg.schema_id
        assert s_all.schema_id != s_cls.schema_id
        assert s_reg.schema_id != s_cls.schema_id

    def test_same_active_same_schema_id(self):
        spec = load_problem(_DEMO)
        assert build_input_schema(spec, ["synthetic_y1"]).schema_id == build_input_schema(spec, ["synthetic_y1"]).schema_id

    def test_passes_check_readiness(self):
        from pyrolysis.config import check_readiness
        spec = load_problem(_DEMO)
        r = check_readiness(spec)
        assert r["prediction_ready"] is True
        assert r["missing"] == []
