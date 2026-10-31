"""
config.py — YAML 설정 읽기, 준비 상태 확인, 입력 규격 생성.

yaml.safe_load 를 사용하여 임의 Python 객체 생성을 방지한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pyrolysis.contracts import InputSchema, ProblemSpec

__all__ = [
    "load_problem",
    "check_readiness",
    "build_input_schema",
    "load_runtime",
    "ConfigError",
]

# ── 예외 ───────────────────────────────────────────────────────────────────────


class ConfigError(Exception):
    """설정 파일 읽기·검증 실패."""


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    """yaml.safe_load 로 YAML 파일을 읽는다.

    Args:
        path: YAML 파일 경로.

    Returns:
        파싱된 딕셔너리. 빈 파일이면 빈 딕셔너리.

    Raises:
        ConfigError: 파일이 없거나 YAML 파싱에 실패한 경우.
    """
    if not path.exists():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 파싱 오류 ({path}): {exc}") from exc
    return data if isinstance(data, dict) else {}


# ── 공개 함수 ─────────────────────────────────────────────────────────────────


def load_problem(path: Path) -> ProblemSpec:
    """YAML 파일을 안전한 로더로 읽어 ProblemSpec 으로 파싱한다.

    - yaml.safe_load 를 사용하여 임의 Python 객체 생성을 방지한다.
    - Pydantic 검증 실패 시 ValidationError 를 그대로 전파한다.
      설정 오류를 자동 기본값으로 대체하지 않는다.
    - draft 상태의 설정은 구조 검사만 통과하며 실제 예측·학습을
      실행하지 않는다. 실행 가능 여부는 check_readiness() 로 확인한다.

    Args:
        path: problem.yaml (또는 원형 파일) 경로.

    Returns:
        파싱된 ProblemSpec 객체.

    Raises:
        ConfigError: 파일이 없거나 YAML 파싱에 실패한 경우.
        pydantic.ValidationError: 설정 구조가 계약과 맞지 않는 경우.
    """
    raw = _safe_load_yaml(path)
    # ValidationError 는 호출자에게 그대로 전파 (자동 보정 금지)
    return ProblemSpec.model_validate(raw)


def check_readiness(spec: ProblemSpec) -> dict[str, Any]:
    """설정의 준비 상태를 확인하고 부족한 항목 목록을 반환한다.

    draft 상태와 실제 예측 준비 상태를 구분한다.
    아직 정의되지 않은 항목을 임의 기본값으로 채우지 않는다.

    Args:
        spec: 파싱된 ProblemSpec 객체.

    Returns:
        다음 키를 포함하는 딕셔너리::

            {
                "prediction_ready": bool,   # 웹 예측 가능 여부
                "training_ready": bool,     # 학습 실행 가능 여부
                "missing": list[str],       # 부족한 항목 설명 목록
                "status": str,             # 현재 status 값
            }
    """
    missing: list[str] = []

    if spec.status == "draft":
        missing.append("status가 'draft'입니다. 학습·예측을 실행하려면 'configured'로 변경해야 합니다.")

    if not spec.problem_id:
        missing.append("problem_id가 설정되지 않았습니다.")

    if not spec.inputs:
        missing.append("inputs(입력 변수 목록)가 비어 있습니다. 데이터 확보 후 정의해야 합니다.")

    if not spec.targets:
        missing.append("targets(예측 대상 목록)가 비어 있습니다. 데이터 확보 후 정의해야 합니다.")

    # 입력 변수 개별 점검
    for inp in spec.inputs:
        if inp.dtype == "category" and not inp.allowed_categories:
            missing.append(
                f"입력 '{inp.id}': dtype=category 이지만 allowed_categories가 비어 있습니다."
            )

    # 예측 대상 개별 점검
    for tgt in spec.targets:
        if tgt.task == "binary_classification":
            if not tgt.label_mapping:
                missing.append(
                    f"대상 '{tgt.id}': binary_classification 이지만 label_mapping이 없습니다."
                )
            if not tgt.positive_label:
                missing.append(
                    f"대상 '{tgt.id}': binary_classification 이지만 positive_label이 없습니다."
                )
        if not tgt.input_ids:
            missing.append(
                f"대상 '{tgt.id}': input_ids(사용 입력 목록)가 비어 있습니다."
            )

    prediction_ready = spec.is_prediction_ready() and len(missing) == 0
    training_ready = spec.is_training_ready() and len(missing) == 0

    return {
        "prediction_ready": prediction_ready,
        "training_ready": training_ready,
        "missing": missing,
        "status": spec.status,
    }



def build_input_schema(
    spec: ProblemSpec,
    active_target_ids: list | None = None,
) -> InputSchema:
    """Build InputSchema from union of active target inputs.

    - active_target_ids=None activates all targets.
    - Inputs of inactive targets are excluded.

    Raises:
        ConfigError: no active targets, or target refs unknown input ID.
    """
    if active_target_ids is None:
        active = spec.targets
    else:
        active = [t for t in spec.targets if t.id in active_target_ids]

    if not active:
        raise ConfigError("No active targets. Cannot build InputSchema.")

    needed: set = set()
    for tgt in active:
        needed.update(tgt.input_ids)

    by_id = {s.id: s for s in spec.inputs}
    missing = needed - set(by_id)
    if missing:
        raise ConfigError("Target requires undefined input IDs: " + str(sorted(missing)))

    return InputSchema.build([by_id[id_] for id_ in needed])


def load_runtime(path: Path) -> dict[str, Any]:
    """runtime.yaml 파일을 안전한 로더로 읽는다.

    - yaml.safe_load 를 사용한다.
    - null 값을 무제한으로 해석하지 않는다. 호출자가 None 여부를 확인해야 한다.

    Args:
        path: runtime.yaml 경로.

    Returns:
        파싱된 설정 딕셔너리.

    Raises:
        ConfigError: 파일이 없거나 YAML 파싱에 실패한 경우.
    """
    return _safe_load_yaml(path)


# ── 운영 설정·시작 검사 (작업 17) ────────────────────────────────────────────
#
# 설계: 업로드 MiB 한도에서 server.maxUploadSize 를 생성하고 앱 바이트 검사와 일치시킨다.
# 실제 운영 한도 누락(null)을 무제한으로 취급하지 않는다.


class RuntimeConfigError(Exception):
    """운영 설정 검사·생성 실패."""


# research 모드에서 반드시 값이 있어야 하는 한도 (null 이면 미설정으로 처리)
_REQUIRED_LIMIT_KEYS = (
    "max_upload_size_mib",
    "max_candidate_rows",
    "max_candidate_cols",
    "max_error_display",
)
_REQUIRED_INFERENCE_KEYS = (
    "max_concurrent_requests",
    "inference_threads",
    "wait_timeout_seconds",
)


def check_runtime_limits(runtime: dict) -> dict:
    """운영 한도의 설정 여부를 점검한다. null(미설정)을 무제한으로 해석하지 않는다.

    Args:
        runtime: load_runtime() 결과.

    Returns:
        {
            "mode": str,
            "ready": bool,            # research 모드에서 모든 필수 한도가 설정됐는지
            "missing": list[str],     # 미설정(null) 한도 키 목록
        }
    """
    mode = runtime.get("mode", "research")
    limits = runtime.get("limits", {}) or {}
    inference = runtime.get("inference", {}) or {}

    missing: list = []
    for k in _REQUIRED_LIMIT_KEYS:
        if limits.get(k) is None:
            missing.append("limits." + k)
    for k in _REQUIRED_INFERENCE_KEYS:
        if inference.get(k) is None:
            missing.append("inference." + k)

    return {
        "mode": mode,
        "ready": len(missing) == 0,
        "missing": missing,
    }


def prepare_runtime_config(
    runtime_path: Path,
    streamlit_config_path: Path,
    *,
    require_all_limits: bool = True,
) -> dict:
    """runtime.yaml 에서 Streamlit 설정(.streamlit/config.toml)을 생성·갱신한다.

    - limits.max_upload_size_mib 가 설정되면 server.maxUploadSize 를 그 값으로 쓴다.
      미설정(null)이면 maxUploadSize 를 쓰지 않고(무제한으로 두지 않음) 보고에 남긴다.
    - client.disableDataExport=true, server.enableStaticServing=false 를 항상 적용한다.
    - require_all_limits=True 인데 필수 한도가 누락되면 RuntimeConfigError.

    Args:
        runtime_path: runtime.yaml 경로.
        streamlit_config_path: .streamlit/config.toml 경로.
        require_all_limits: 모든 필수 한도를 요구할지 여부.

    Returns:
        {"max_upload_size_mib": int|None, "missing": list, "written": bool}

    Raises:
        RuntimeConfigError: require_all_limits=True 이고 한도가 누락된 경우.
    """
    runtime = load_runtime(runtime_path)
    check = check_runtime_limits(runtime)

    if require_all_limits and not check["ready"]:
        raise RuntimeConfigError(
            "운영 한도가 설정되지 않았습니다 (작업 29에서 확정). 누락: "
            + str(check["missing"])
            + ". 미설정 한도를 무제한으로 취급하지 않습니다."
        )

    limits = runtime.get("limits", {}) or {}
    max_mib = limits.get("max_upload_size_mib")

    lines = []
    lines.append("# 자동 생성: prepare-runtime (작업 17). runtime.yaml 과 동기화됨.")
    lines.append("")
    lines.append("[client]")
    lines.append("disableDataExport = true")
    lines.append("")
    lines.append("[server]")
    lines.append("enableStaticServing = false")
    if max_mib is not None:
        lines.append("maxUploadSize = " + str(int(max_mib)))
    else:
        lines.append("# maxUploadSize 미설정: runtime.yaml limits.max_upload_size_mib 가 null 입니다.")
        lines.append("# 무제한으로 두지 않습니다. 작업 29에서 실측 후 채웁니다.")
    lines.append("")
    lines.append("[theme]")
    lines.append('base = "light"')
    lines.append("")

    streamlit_config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = streamlit_config_path.with_suffix(".toml.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(streamlit_config_path)

    return {
        "max_upload_size_mib": int(max_mib) if max_mib is not None else None,
        "missing": check["missing"],
        "written": True,
    }
