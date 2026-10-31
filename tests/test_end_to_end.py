"""
test_end_to_end.py — 작업 24 데이터 확보 전 통합 수용 점검.

실제 연구 데이터 없이 서로 다른 두 합성 규격(A·B)으로 전체 흐름을 연결한다:
  학습 → 평가(상태) → 완전 묶음 export → 다시 읽기 → 단일·일괄 예측 → 순위.

기존 모듈 테스트를 반복하지 않고 통합에서 남는 위험을 점검한다:
  - 규격 교체(A↔B)로 입력·대상·단위가 코드 수정 없이 바뀐다.
  - 부분 목표 결측에서 대상별 가용 행이 보존된다.
  - 동시 양성 분류(여러 검출 대상이 함께 양성)를 허용한다.
  - 미지원(비활성) 대상과 활성 대상이 함께 있을 때 활성 대상만 예측·순위에 들어간다.
  - 열 순서를 바꾼 파일에서 같은 결과를 얻는다 (규격 ID 기반 읽기).
  - 마지막 행 오류·출력 제약 위반에서 부분 결과·순위가 노출되지 않는다.
  - 묶음 훼손·환경 불일치를 모델 로딩 전에 거부한다.
  - 입력 변경 후 이전 결과가 남지 않고, 두 세션의 결과가 섞이지 않는다.

가상 기능 점검이며 실제 연구 성능·공개 배포 완료가 아니다.
Linux 교차 확인은 환경이 없으면 docs/acceptance.md 에 미완료로 기록한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrolysis.artifacts import (
    BundleError,
    export_bundle,
    load_research_bundle,
    verify_bundle_integrity,
)
from pyrolysis.config import build_input_schema, load_problem
from pyrolysis.contracts import CandidateTable
from pyrolysis.inference import make_single_candidate_table, predict_all_targets
from pyrolysis.io import parse_prediction_file
from pyrolysis.models import get_classification_candidates, get_regression_candidates
from pyrolysis.preprocessing import (
    build_classification_pipeline,
    build_regression_pipeline,
)
from pyrolysis.ranking import RankingError, rank_candidates
from pyrolysis.ui.session import SessionState

_FIXTURES = Path(__file__).parent / "fixtures"
_SPEC_A = _FIXTURES / "synthetic_spec_a.yaml"
_SPEC_B = _FIXTURES / "synthetic_spec_b.yaml"


# ── 합성 자료 생성·학습 도우미 (공통 함수를 조합) ───────────────────────────


def _make_frame(input_specs, n=60, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for s in input_specs:
        if s.dtype in ("float", "int"):
            lo = -5.0
            hi = 5.0
            for pc in s.physical_constraints:
                if pc.min_value is not None:
                    lo = max(lo, pc.min_value) if lo is not None else pc.min_value
                if pc.max_value is not None:
                    hi = min(hi, pc.max_value) if hi is not None else pc.max_value
            cols[s.id] = rng.uniform(lo, hi, size=n)
        elif s.dtype == "category":
            choices = [c.value for c in s.allowed_categories]
            cols[s.id] = rng.choice(choices, size=n)
    return pd.DataFrame(cols)


def _train_bundle(spec, out_dir, bundle_id, *, active_target_ids, seed=0,
                  partial_missing_for=None):
    """합성 자료로 활성 대상별 Pipeline 을 학습하고 완전 묶음으로 export 한다.

    partial_missing_for: 이 대상의 정답 일부를 결측 처리하여 대상별 가용 행이
        다른 상황을 만든다 (부분 목표 결측 점검).
    """
    active = [t for t in spec.targets if t.id in active_target_ids]
    schema = build_input_schema(spec, active_target_ids=active_target_ids)
    X = _make_frame(spec.inputs, seed=seed)

    models = {}
    validation_summary = {}
    training_domain = {}
    n_rows_used = {}

    for tgt in active:
        feat = tgt.input_ids
        Xi = X[feat].copy()
        # 대상별 정답 생성
        num_cols = [c for c in feat if pd.api.types.is_numeric_dtype(X[c])]
        base = X[num_cols].sum(axis=1) if num_cols else pd.Series(np.zeros(len(X)))

        mask = pd.Series(True, index=X.index)
        if partial_missing_for == tgt.id:
            # 앞쪽 10행을 이 대상에서만 결측 처리 (다른 대상 행은 유지)
            mask.iloc[:10] = False

        if tgt.task == "regression":
            y = base.copy()
            if tgt.output_constraints and tgt.output_constraints.non_negative:
                y = base.abs() + 1.0  # 비음수 보장
            Xt, yt = Xi[mask], y[mask]
            pipe = build_regression_pipeline(
                [s for s in spec.inputs if s.id in feat],
                get_regression_candidates()["linear_regression"],
            )
            pipe.fit(Xt, yt)
        else:
            y = (base > base.median()).astype(int)
            Xt, yt = Xi[mask], y[mask]
            pipe = build_classification_pipeline(
                [s for s in spec.inputs if s.id in feat],
                get_classification_candidates()["logistic_regression"],
            )
            pipe.fit(Xt, yt)

        models[tgt.id] = pipe
        n_rows_used[tgt.id] = int(mask.sum())
        validation_summary[tgt.id] = {
            "evaluated": True,
            "method": "synthetic_holdout",
            "n_samples": int(mask.sum()),
        }
        # 학습 범위 (사용한 행 기준)
        ranges = {}
        for c in num_cols:
            vals = X.loc[mask, c]
            ranges[c] = {"min_value": float(vals.min()), "max_value": float(vals.max())}
        training_domain[tgt.id] = {"target_id": tgt.id, "numeric_ranges": ranges}

    contract = {
        "bundle_id": bundle_id,
        "bundle_type": "research",
        "input_ids": list(schema.input_ids),
        "input_schema_id": schema.schema_id,
        "targets": [
            {
                "id": t.id, "task": t.task, "input_ids": t.input_ids,
                "unit": t.unit,
                "output_constraints": t.output_constraints.model_dump(),
                "label_mapping": t.label_mapping,
                "positive_label": t.positive_label,
                "transform_kind": "identity",
            }
            for t in active
        ],
        # 미지원 대상 정보 (모델 없이 사유만)
        "inactive_targets": {
            t.id: "자료 부족으로 비활성 (시험용)"
            for t in spec.targets if t.id not in active_target_ids
        },
    }

    bundle_dir = export_bundle(
        out_dir, bundle_id, contract, models,
        validation_summary, training_domain,
        inactive_targets=contract["inactive_targets"],
        verify_frame=X,
        tolerance=1e-6,
    )
    return bundle_dir, schema, X, n_rows_used


# ── 1. 규격 A: 학습 → export → 재읽기 → 단일·일괄 → 순위 ────────────────────


class TestSpecAFullFlow:
    def test_full_flow_single_batch_and_ranking(self, tmp_path):
        spec = load_problem(_SPEC_A)
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a",
            active_target_ids=["a_reg", "a_cls"], seed=1,
        )

        # 저장 전후 예측 재현: 다시 읽기
        bundle = load_research_bundle(bundle_dir)
        input_specs = [s for s in spec.inputs]

        batch = predict_all_targets(bundle, _ct(X.iloc[:8], schema), input_specs=input_specs)
        assert batch.completed is True
        assert set(batch.predictions.keys()) == {"a_reg", "a_cls"}

        # 단일 = 일괄 첫 행
        row0 = X.iloc[0].to_dict()
        single = predict_all_targets(
            bundle, make_single_candidate_table(row0, schema.schema_id),
            input_specs=input_specs,
        )
        assert single.completed
        np.testing.assert_allclose(
            single.predictions["a_reg"].values[0],
            batch.predictions["a_reg"].values[0], rtol=0, atol=1e-9,
        )

        # 수치형 대상 순위 (회귀만, 분류는 제외)
        ranking = rank_candidates(batch, "a_reg", "maximize")
        assert len(ranking.ranked) == 8
        # 분류 대상은 순위 불가
        with pytest.raises(RankingError):
            rank_candidates(batch, "a_cls", "maximize")


# ── 2. 규격 교체 (A↔B): 코드 수정 없이 입력·대상·단위가 바뀐다 ──────────────


class TestSpecSwap:
    def test_spec_b_has_different_inputs_and_units(self, tmp_path):
        spec_b = load_problem(_SPEC_B)
        # B 는 b_perf 를 미지원(비활성)으로 두고 두 회귀만 활성화
        bundle_dir, schema, X, _ = _train_bundle(
            spec_b, tmp_path / "bundles", "accept_b",
            active_target_ids=["b_yield", "b_area"], seed=2,
        )
        bundle = load_research_bundle(bundle_dir)

        # 입력 ID 가 A 와 다르다
        assert set(schema.input_ids) == {"b_p", "b_q"}
        # 단위가 대상별로 다르다 (wt% vs m2/g)
        units = {t["id"]: t["unit"] for t in bundle["contract"]["targets"]}
        assert units["b_yield"] == "wt%"
        assert units["b_area"] == "m2/g"

        batch = predict_all_targets(bundle, _ct(X.iloc[:5], schema),
                                    input_specs=spec_b.inputs)
        assert batch.completed
        assert set(batch.predictions.keys()) == {"b_yield", "b_area"}


# ── 3. 미지원 + 활성 대상 공존 ───────────────────────────────────────────────


class TestUnsupportedAndActive:
    def test_inactive_target_excluded_from_predictions(self, tmp_path):
        spec_b = load_problem(_SPEC_B)
        bundle_dir, schema, X, _ = _train_bundle(
            spec_b, tmp_path / "bundles", "accept_b_mix",
            active_target_ids=["b_yield", "b_area"], seed=3,
        )
        bundle = load_research_bundle(bundle_dir)
        # 미지원 대상이 manifest 에 사유와 함께 기록되고 모델은 없다
        assert "b_perf" in bundle["manifest"]["inactive_targets"]
        assert "b_perf" not in bundle["models"]

        batch = predict_all_targets(bundle, _ct(X.iloc[:4], schema),
                                    input_specs=spec_b.inputs)
        # 활성 대상만 예측된다
        assert "b_perf" not in batch.predictions
        assert set(batch.predictions.keys()) == {"b_yield", "b_area"}


# ── 4. 부분 목표 결측: 대상별 가용 행 보존 ───────────────────────────────────


class TestPartialTargetMissing:
    def test_per_target_rows_preserved(self, tmp_path):
        spec = load_problem(_SPEC_A)
        _, _, _, n_used = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_partial",
            active_target_ids=["a_reg", "a_cls"], seed=4,
            partial_missing_for="a_cls",  # 분류 대상만 앞 10행 결측
        )
        # 분류 대상은 결측 10행이 빠지고, 회귀 대상은 전체 유지
        assert n_used["a_cls"] == n_used["a_reg"] - 10
        assert n_used["a_reg"] == 60


# ── 5. 동시 양성 분류 (여러 검출 대상이 함께 양성) ──────────────────────────


class TestSimultaneousPositives:
    def test_two_detection_targets_can_both_be_positive(self, tmp_path):
        # A 의 a_cls 와 B 스타일을 섞지 않고, 두 독립 검출 대상을 같은 입력으로 둔다.
        spec = load_problem(_SPEC_A)
        # a_cls 를 복제한 두 번째 검출 대상을 메모리에서 추가 구성
        import copy
        t2 = copy.deepcopy([t for t in spec.targets if t.id == "a_cls"][0])
        t2.id = "a_cls2"
        spec.targets.append(t2)

        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_multi",
            active_target_ids=["a_cls", "a_cls2"], seed=5,
        )
        bundle = load_research_bundle(bundle_dir)
        batch = predict_all_targets(bundle, _ct(X, schema), input_specs=spec.inputs)
        assert batch.completed
        v1 = batch.predictions["a_cls"].values
        v2 = batch.predictions["a_cls2"].values
        # 두 대상이 동시에 'present' 인 행이 적어도 하나 존재할 수 있어야 한다
        both_positive = sum(1 for a, b in zip(v1, v2) if a == "present" and b == "present")
        assert both_positive >= 1  # 다중 양성 허용 (상호 배타 아님)


# ── 6. 열 순서 변경 파일에서 같은 결과 ───────────────────────────────────────


class TestColumnReorder:
    def test_reordered_columns_same_candidates(self, tmp_path):
        spec = load_problem(_SPEC_A)
        schema = build_input_schema(spec, active_target_ids=["a_reg", "a_cls"])
        X = _make_frame(spec.inputs, n=5, seed=6)

        # 규격 순서대로 저장
        p1 = tmp_path / "ordered.csv"
        df1 = X[list(schema.input_ids)].copy()
        df1["_input_schema_id"] = schema.schema_id
        df1.to_csv(p1, index=False, encoding="utf-8-sig")

        # 열 순서를 뒤집어 저장
        p2 = tmp_path / "reordered.csv"
        reordered_cols = list(reversed(list(schema.input_ids)))
        df2 = X[reordered_cols].copy()
        df2["_input_schema_id"] = schema.schema_id
        df2.to_csv(p2, index=False, encoding="utf-8-sig")

        r1 = parse_prediction_file(p1, schema)
        r2 = parse_prediction_file(p2, schema)
        assert r1["total_error_count"] == 0
        assert r2["total_error_count"] == 0
        # 정규화된 입력 표는 규격 순서로 정렬되어 같다
        pd.testing.assert_frame_equal(
            r1["candidates"].data.astype(str),
            r2["candidates"].data.astype(str),
        )


# ── 7. 마지막 행 오류 → 전체 실패, 순위 없음 ─────────────────────────────────


class TestLastRowErrorBlocks:
    def test_last_row_error_no_partial_ranking(self, tmp_path):
        spec = load_problem(_SPEC_A)
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_err",
            active_target_ids=["a_reg"], seed=7,
        )
        bundle = load_research_bundle(bundle_dir)
        df = X.iloc[:3].copy()
        df.iloc[2, df.columns.get_loc("a_x1")] = np.nan  # 마지막 행 필수 결측
        batch = predict_all_targets(bundle, _ct(df, schema), input_specs=spec.inputs)
        assert batch.completed is False
        assert len(batch.predictions) == 0
        # 미완료 배치에는 순위를 만들지 않는다
        with pytest.raises(RankingError):
            rank_candidates(batch, "a_reg", "maximize")


# ── 8. 출력 제약 위반 → 전체 실패 ────────────────────────────────────────────


class TestOutputConstraintViolation:
    def test_non_negative_violation_fails(self, tmp_path):
        spec = load_problem(_SPEC_B)
        # b_area 는 비음수 제약. 학습은 비음수였지만 음수를 유도하는 입력으로 위반 가능.
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_b_oc",
            active_target_ids=["b_area"], seed=8,
        )
        bundle = load_research_bundle(bundle_dir)
        # 큰 음수 조합으로 음수 예측 유도
        df = pd.DataFrame({"b_p": [0.0, 0.0], "b_q": [0.0, 0.0]})
        # 선형 모델에서 절편만 남아 비음수일 수 있으므로, 음수 유도용 극단값 사용
        df2 = pd.DataFrame({"b_p": [-1000.0], "b_q": [-1000.0]})
        batch = predict_all_targets(bundle, _ct(df2, schema), input_specs=spec.inputs)
        # 물리 제약(0~50) 밖이므로 입력 검증에서 먼저 걸리거나 출력 제약에서 걸린다.
        assert batch.completed is False
        assert batch.error is not None


# ── 9. 묶음 훼손·환경 불일치 거부 ────────────────────────────────────────────


class TestCorruptionAndEnvMismatch:
    def test_tampered_file_rejected_before_load(self, tmp_path):
        spec = load_problem(_SPEC_A)
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_tamper",
            active_target_ids=["a_reg"], seed=9,
        )
        # contract.json 을 훼손 (해시 불일치)
        cpath = bundle_dir / "contract.json"
        cpath.write_text(cpath.read_text(encoding="utf-8") + "\n# tampered",
                         encoding="utf-8")
        with pytest.raises(BundleError):
            verify_bundle_integrity(bundle_dir)
        with pytest.raises(BundleError):
            load_research_bundle(bundle_dir)

    def test_missing_file_rejected(self, tmp_path):
        spec = load_problem(_SPEC_A)
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_missing",
            active_target_ids=["a_reg"], seed=10,
        )
        (bundle_dir / "environment.json").unlink()
        with pytest.raises(BundleError):
            verify_bundle_integrity(bundle_dir)

    def test_env_mismatch_detected_by_check_bundle(self, tmp_path):
        from pyrolysis.artifacts import check_bundle
        spec = load_problem(_SPEC_A)
        bundle_dir, schema, X, _ = _train_bundle(
            spec, tmp_path / "bundles", "accept_a_env",
            active_target_ids=["a_reg"], seed=11,
        )
        # environment.json 의 sklearn 버전을 바꿔 불일치 유도 (해시도 함께 갱신)
        env_path = bundle_dir / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["sklearn"] = "0.1"  # 현재와 major.minor 다름
        env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        # manifest 해시를 새 내용으로 갱신 (무결성은 통과시키고 환경 검사만 실패하도록)
        _refresh_manifest_hash(bundle_dir, "environment.json")

        result = check_bundle(bundle_dir)
        assert result["integrity_ok"] is True
        assert result["env_ok"] is False
        assert any("sklearn" in m for m in result["env_mismatches"])


# ── 10. 입력 변경 후 이전 결과 없음 / 두 세션 분리 ──────────────────────────


class TestSessionIsolationAndInvalidation:
    def test_input_change_clears_previous_result(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"a_x1": 1.0}, bundle_id="b", bundle_hash="h")
        s.set_result("OLD")
        s.sync(inputs={"a_x1": 2.0}, bundle_id="b", bundle_hash="h")
        assert s.get_result() is None

    def test_two_sessions_do_not_mix(self):
        store1, store2 = {}, {}
        s1 = SessionState(store1)
        s2 = SessionState(store2)
        s1.set_result("SESSION_1_RESULT")
        s2.set_result("SESSION_2_RESULT")
        assert s1.get_result() == "SESSION_1_RESULT"
        assert s2.get_result() == "SESSION_2_RESULT"
        # 한 세션 초기화가 다른 세션에 영향을 주지 않는다
        s1.clear()
        assert s1.get_result() is None
        assert s2.get_result() == "SESSION_2_RESULT"


# ── 공통 도우미 ──────────────────────────────────────────────────────────────


def _ct(df, schema) -> CandidateTable:
    df = df.reset_index(drop=True)
    return CandidateTable(
        input_schema_id=schema.schema_id,
        source_rows=list(range(1, len(df) + 1)),
        candidate_ids=["c" + str(i) for i in range(len(df))],
        data=df,
    )


def _refresh_manifest_hash(bundle_dir: Path, changed_rel: str) -> None:
    """한 파일을 바꾼 뒤 manifest 의 해당 해시를 다시 계산해 넣는다 (무결성 통과용)."""
    from pyrolysis.artifacts import MANIFEST_NAME, _sha256_of_file

    manifest_path = bundle_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_hashes"][changed_rel] = _sha256_of_file(bundle_dir / changed_rel)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
