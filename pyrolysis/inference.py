"""
inference.py - 공통 예측 인터페이스 (Task 4 최소 구현).

단일 후보도 한 행의 CandidateTable 로 받아 predict_all 을 호출한다.
전체 행이 성공한 경우에만 완료된 PredictionBatch 를 반환한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np
import pandas as pd

from pyrolysis.contracts import (
    CandidateTable,
    Issue,
    PredictionBatch,
    TargetPrediction,
)

__all__ = [
    "predict_all",
    "make_single_candidate_table",
    "InferenceError",
    "InferenceBusyError",
    "predict_all_targets",
    "run_with_inference_lock",
]


class InferenceError(Exception):
    """예측 실행 실패."""


class InferenceBusyError(Exception):
    """대기 한도 안에 묶음별 추론 잠금을 얻지 못함 (재시도 안내)."""


# ── 묶음별 추론 잠금과 대기 한도 (작업 22) ───────────────────────────────────
#
# 설계 10.3:
#   공유 모델에 요청별 값을 써 넣거나 fit 을 호출하지 않는다.
#   묶음별 추론 잠금으로 동시 모델 접근을 직렬화하고, 입력 검증·화면 상태는 세션별로 둔다.
#   처리 중 다른 요청이 오면 정한 대기 한도를 적용하고 초과 시 재시도로 응답한다.
#   요청 데이터 자체를 공통 작업 큐에 보관하지 않는다 (잠금만 공유, 데이터는 세션).

import threading as _threading

# 묶음 ID -> Lock. 잠금 객체만 공유하며 사용자 데이터는 담지 않는다.
_BUNDLE_LOCKS: dict[str, _threading.Lock] = {}
_LOCKS_GUARD = _threading.Lock()


def _get_bundle_lock(bundle_id: str) -> _threading.Lock:
    """묶음 ID 에 해당하는 추론 잠금을 반환한다 (없으면 생성)."""
    with _LOCKS_GUARD:
        lock = _BUNDLE_LOCKS.get(bundle_id)
        if lock is None:
            lock = _threading.Lock()
            _BUNDLE_LOCKS[bundle_id] = lock
        return lock


def run_with_inference_lock(bundle_id: str, fn, *, wait_timeout_seconds):
    """묶음별 추론 잠금을 얻어 fn() 을 실행한다.

    - 대기 한도(wait_timeout_seconds) 안에 잠금을 얻지 못하면 InferenceBusyError.
      None(미설정)은 무제한 대기로 취급하지 않는다 (설정 누락은 호출자가 거부해야 함).
    - fn 이 예외를 던져도 잠금을 반드시 해제한다 (finally).
    - 한 요청 실패가 다음 요청을 막지 않도록, 실패해도 잠금은 풀린다.

    Args:
        bundle_id: 묶음 식별자.
        fn: 인자 없이 호출 가능한 함수 (예측 수행).
        wait_timeout_seconds: 대기 한도(초). None 이면 ValueError.

    Returns:
        fn() 의 반환값.

    Raises:
        ValueError: wait_timeout_seconds 가 None(미설정).
        InferenceBusyError: 대기 한도 초과.
    """
    if wait_timeout_seconds is None:
        raise ValueError(
            "대기 한도(wait_timeout_seconds)가 설정되지 않았습니다. "
            "미설정 한도를 무제한 대기로 취급하지 않습니다."
        )

    lock = _get_bundle_lock(bundle_id)
    acquired = lock.acquire(timeout=float(wait_timeout_seconds))
    if not acquired:
        raise InferenceBusyError(
            "다른 요청을 처리 중입니다. 대기 한도("
            + str(wait_timeout_seconds) + "초)를 초과하여 재시도가 필요합니다."
        )
    try:
        return fn()
    finally:
        lock.release()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """DataFrame 내용의 SHA-256 해시를 반환한다 (재현성 확인용)."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def make_single_candidate_table(
    input_dict: dict[str, Any],
    input_schema_id: str,
) -> CandidateTable:
    """단일 입력 딕셔너리를 한 행짜리 CandidateTable 로 변환한다.

    Args:
        input_dict: {입력 ID: 값} 딕셔너리.
        input_schema_id: 입력 규격 식별자.

    Returns:
        CandidateTable (행 1개).
    """
    df = pd.DataFrame([input_dict])
    return CandidateTable(
        input_schema_id=input_schema_id,
        source_rows=[1],
        candidate_ids=["single_0"],
        data=df,
    )


def predict_all(
    bundle: dict[str, Any],
    candidates: CandidateTable,
) -> PredictionBatch:
    """묶음의 Pipeline으로 모든 후보를 예측하고 PredictionBatch 를 반환한다.

    - 입력 열이 contract.input_ids 와 일치하는지 확인한다.
    - 예측 실패 시 completed=False 인 PredictionBatch 를 반환한다.
      부분 결과를 정상 결과로 반환하지 않는다.
    - 유효하지 않은 출력(NaN·Inf)은 오류로 처리한다.

    Args:
        bundle: load_bundle() 반환값. "pipeline"·"contract" 키 필요.
        candidates: CandidateTable.

    Returns:
        PredictionBatch (completed=True/False).
    """
    request_id = uuid.uuid4().hex[:12]
    contract = bundle["contract"]
    pipeline = bundle["pipeline"]
    bundle_id = contract["bundle_id"]
    bundle_hash = contract.get("bundle_id", "unknown")  # Task 16에서 SHA-256으로 교체
    input_snapshot_hash = _hash_dataframe(candidates.data)

    # ── 입력 열 존재 확인 ─────────────────────────────────────────────────────
    required_cols = contract["input_ids"]
    missing_cols = [c for c in required_cols if c not in candidates.data.columns]
    if missing_cols:
        err = Issue(
            code="SCHEMA_MISMATCH",
            message_ko=f"필수 입력 열이 없습니다: {missing_cols}",
            severity="error",
        )
        return PredictionBatch(
            request_id=request_id,
            bundle_id=bundle_id,
            bundle_hash=bundle_hash,
            input_snapshot_hash=input_snapshot_hash,
            completed=False,
            error=err,
        )

    # ── 예측 실행 ────────────────────────────────────────────────────────────
    try:
        X = candidates.data[required_cols]
        y_pred_raw = pipeline.predict(X)
    except Exception as exc:
        err = Issue(
            code="PREDICTION_FAILED",
            message_ko=f"모델 예측 중 오류 발생: {exc}",
            severity="error",
        )
        return PredictionBatch(
            request_id=request_id,
            bundle_id=bundle_id,
            bundle_hash=bundle_hash,
            input_snapshot_hash=input_snapshot_hash,
            completed=False,
            error=err,
        )

    # ── 출력 유효성 검사 (NaN·Inf 거부) ──────────────────────────────────────
    if np.any(~np.isfinite(y_pred_raw)):
        err = Issue(
            code="OUTPUT_INVALID",
            message_ko="예측값에 NaN 또는 Inf 가 포함되어 있습니다.",
            severity="error",
        )
        return PredictionBatch(
            request_id=request_id,
            bundle_id=bundle_id,
            bundle_hash=bundle_hash,
            input_snapshot_hash=input_snapshot_hash,
            completed=False,
            error=err,
        )

    # ── 결과 구성 ─────────────────────────────────────────────────────────────
    target_id = contract["target_id"]
    target_unit = contract.get("target_unit")
    values = [float(v) for v in y_pred_raw]

    target_prediction = TargetPrediction(
        target_id=target_id,
        values=values,
        unit=target_unit,
    )

    return PredictionBatch(
        request_id=request_id,
        bundle_id=bundle_id,
        bundle_hash=bundle_hash,
        input_snapshot_hash=input_snapshot_hash,
        predictions={target_id: target_prediction},
        completed=True,
    )


# ── 모든 활성 대상의 요청 단위 예측 (작업 18) ───────────────────────────────
#
# 설계 8.1:
#   요청 시작 시 묶음·해시를 고정하고 모든 행을 먼저 검증한다.
#   오류가 있으면 모델을 호출하지 않는다. 각 대상에는 정의된 입력만 전달한다.
#   역변환 후 원래 단위로 유한성·출력 범위를 검사한다. 전체 성공만 게시한다.
#   학습 범위 이탈은 주의로 분리하며 이 정보만으로 차단·보정하지 않는다.

from pyrolysis.contracts import TargetPrediction as _TP


class _TargetSpecView:
    """contract 의 대상 정의를 가벼운 뷰로 감싼다 (출력 제약·역변환 참조용)."""

    def __init__(self, target_id, task, input_ids, unit=None,
                 output_constraints=None, label_mapping=None, transform_kind="identity"):
        self.id = target_id
        self.task = task
        self.input_ids = list(input_ids)
        self.unit = unit
        self.output_constraints = output_constraints or {}
        self.label_mapping = label_mapping
        self.transform_kind = transform_kind


def _target_views_from_manifest(bundle):
    """load_research_bundle 결과에서 대상별 뷰를 만든다.

    manifest.active_targets 와 contract 의 대상 정의(targets 목록 또는 단일 target_id)를 쓴다.
    """
    contract = bundle["contract"]
    manifest = bundle.get("manifest", {})
    active = manifest.get("active_targets")
    views = {}

    # contract 에 targets 목록이 있으면 사용, 없으면 단일 target_id (demo/최소 계약 호환)
    target_defs = {}
    if isinstance(contract.get("targets"), list):
        for t in contract["targets"]:
            target_defs[t["id"]] = t
    elif contract.get("target_id"):
        target_defs[contract["target_id"]] = {
            "id": contract["target_id"],
            "task": contract.get("target_task", "regression"),
            "input_ids": contract.get("input_ids", []),
            "unit": contract.get("target_unit"),
        }

    if active is None:
        active = list(bundle.get("models", {}).keys())

    for tid in active:
        td = target_defs.get(tid, {})
        views[tid] = _TargetSpecView(
            target_id=tid,
            task=td.get("task", "regression"),
            input_ids=td.get("input_ids", contract.get("input_ids", [])),
            unit=td.get("unit", contract.get("target_unit")),
            output_constraints=td.get("output_constraints", {}),
            label_mapping=td.get("label_mapping"),
            transform_kind=td.get("transform_kind", "identity"),
        )
    return views


def _check_output_constraints(values, oc, target_id):
    """역변환 후 원래 단위 값에 출력 제약(유한성·비음수·범위)을 검사한다.

    제약 위반을 잘라내거나 숨기지 않고 Issue 로 보고한다.
    """
    issues = []
    arr = np.asarray(values, dtype=float)
    finite_required = oc.get("finite", True)
    if finite_required and np.any(~np.isfinite(arr)):
        issues.append(Issue(
            code="OUTPUT_INVALID",
            message_ko="예측값에 NaN 또는 Inf 가 있습니다.",
            severity="error", target_id=target_id,
        ))
        return issues  # 유한하지 않으면 범위 검사 불가
    if oc.get("non_negative", False) and np.any(arr < 0):
        issues.append(Issue(
            code="OUTPUT_INVALID",
            message_ko="비음수 제약을 위반한 예측값이 있습니다.",
            severity="error", target_id=target_id,
        ))
    mn = oc.get("min_value")
    mx = oc.get("max_value")
    if mn is not None and np.any(arr < mn):
        issues.append(Issue(
            code="OUTPUT_INVALID",
            message_ko="출력 최솟값(" + str(mn) + ") 미만 예측값이 있습니다.",
            severity="error", target_id=target_id,
        ))
    if mx is not None and np.any(arr > mx):
        issues.append(Issue(
            code="OUTPUT_INVALID",
            message_ko="출력 최댓값(" + str(mx) + ") 초과 예측값이 있습니다.",
            severity="error", target_id=target_id,
        ))
    return issues


def predict_all_targets(
    bundle,
    candidates,
    *,
    input_specs=None,
    training_domains=None,
):
    """모든 활성 대상에 대해 요청 단위 예측을 수행한다 (작업 18).

    bundle 은 load_research_bundle 결과({"models", "contract", "manifest", ...})를 기대한다.
    요청 시작 시 묶음을 그대로 사용하며(이미 확보한 한 묶음), 중간에 바뀌어도 교체하지 않는다.

    처리 순서:
      1. 모든 행의 입력·물리 제약을 먼저 검증. 오류가 있으면 모델을 호출하지 않는다.
      2. 각 대상에 정의된 입력만 전달해 예측.
      3. 목표 역변환 후 원래 단위로 유한성·출력 범위를 검사.
      4. 학습 범위 이탈은 주의로 분리 (차단·보정하지 않음).
      5. 전체 대상·행이 성공한 경우에만 completed=True.

    Returns:
        PredictionBatch.
    """
    from pyrolysis.preprocessing import TargetTransform
    from pyrolysis.models import invert_label_mapping
    from pyrolysis.validation import validate_candidates

    request_id = uuid.uuid4().hex[:12]
    contract = bundle["contract"]
    manifest = bundle.get("manifest", {})
    bundle_id = contract.get("bundle_id", manifest.get("bundle_id", "unknown"))
    bundle_hash = manifest.get("file_hashes", {}).get("manifest.json", "") or "unknown"
    input_snapshot_hash = _hash_dataframe(candidates.data)
    models = bundle.get("models", {})

    views = _target_views_from_manifest(bundle)

    def _fail(code, msg):
        return PredictionBatch(
            request_id=request_id, bundle_id=bundle_id, bundle_hash=bundle_hash,
            input_snapshot_hash=input_snapshot_hash, completed=False,
            error=Issue(code=code, message_ko=msg, severity="error"),
        )

    if not models:
        return _fail("NO_ACTIVE_TARGET", "활성 대상 모델이 없습니다.")

    # 1) 전체 행 입력 검증 (물리 제약 포함). 오류가 있으면 모델 호출 안 함.
    all_errors = []
    range_warnings = []
    if input_specs is not None:
        vres = validate_candidates(candidates.data, input_specs, training_domains=training_domains)
        all_errors.extend(vres["errors"])
        range_warnings.extend(vres["range_warnings"])
    if all_errors:
        first = all_errors[0]
        return PredictionBatch(
            request_id=request_id, bundle_id=bundle_id, bundle_hash=bundle_hash,
            input_snapshot_hash=input_snapshot_hash, completed=False,
            error=Issue(code="INPUT_INVALID",
                        message_ko="입력 검증 오류가 있어 예측을 실행하지 않습니다 ("
                        + str(len(all_errors)) + "건).",
                        severity="error", row=first.row, col=first.col),
        )

    # 2-3) 대상별 예측 + 역변환 + 출력 제약
    predictions = {}
    for tid, view in views.items():
        model = models.get(tid)
        if model is None:
            return _fail("PREDICTION_FAILED", "대상 '" + tid + "' 의 모델이 없습니다.")

        missing = [c for c in view.input_ids if c not in candidates.data.columns]
        if missing:
            return _fail("SCHEMA_MISMATCH", "대상 '" + tid + "' 필수 입력 열이 없습니다: " + str(missing))

        X = candidates.data[view.input_ids]
        try:
            raw = model.predict(X)
        except Exception as exc:
            return _fail("PREDICTION_FAILED", "대상 '" + tid + "' 예측 실패: " + str(exc))

        if view.task == "regression":
            inv = TargetTransform(view.transform_kind).inverse(raw)
            oc_issues = _check_output_constraints(inv, view.output_constraints, tid)
            if oc_issues:
                return PredictionBatch(
                    request_id=request_id, bundle_id=bundle_id, bundle_hash=bundle_hash,
                    input_snapshot_hash=input_snapshot_hash, completed=False,
                    error=oc_issues[0],
                )
            values = [float(v) for v in inv]
        else:
            # 분류: 정의된 표지로 반환
            if view.label_mapping:
                try:
                    values = invert_label_mapping([int(v) for v in raw], view.label_mapping)
                except Exception as exc:
                    return _fail("OUTPUT_INVALID", "대상 '" + tid + "' 표지 역매핑 실패: " + str(exc))
            else:
                values = [str(v) for v in raw]

        tgt_warnings = [w for w in range_warnings if w.target_id == tid]
        predictions[tid] = _TP(
            target_id=tid, values=values, unit=view.unit, warnings=tgt_warnings,
        )

    # 5) 전체 성공 -> 완료
    return PredictionBatch(
        request_id=request_id, bundle_id=bundle_id, bundle_hash=bundle_hash,
        input_snapshot_hash=input_snapshot_hash,
        predictions=predictions, range_warnings=range_warnings, completed=True,
    )
