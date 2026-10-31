"""
artifacts.py - 배포 묶음 생성·검증·읽기 (Task 4 최소 구현).

demo 전용 묶음을 저장하고 다시 읽는 경로를 제공한다.
실제 연구 배포용 전체 묶음(manifest·SHA-256·환경 검사)은 Task 16에서 완성한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib

__all__ = [
    "save_demo_bundle",
    "load_bundle",
    "BundleError",
    "DEMO_BUNDLE_MARKER",
    "ExportError",
    "export_bundle",
    "verify_bundle_integrity",
    "load_research_bundle",
    "write_active_pointer",
    "check_bundle",
    "RESEARCH_BUNDLE_MARKER",
]

DEMO_BUNDLE_MARKER = "demo-synthetic"  # 실제 연구 묶음과 구분하는 표지


class BundleError(Exception):
    """묶음 로드·검증 실패."""


def save_demo_bundle(
    pipeline: Any,
    input_specs: list[Any],
    target_spec: Any,
    output_dir: Path,
    bundle_id: str,
    disclaimer: str,
) -> Path:
    """demo Pipeline을 묶음 디렉터리에 저장한다.

    저장 내용:
    - model.joblib       : 전처리+모델 Pipeline
    - contract.json      : 입력 규격·대상 정의·demo 표지
    - environment.json   : Python·패키지 버전

    demo 묶음에는 DEMO_BUNDLE_MARKER 가 포함되어 연구용 활성화 경로가
    이 묶음을 거부할 수 있도록 한다 (Task 17에서 강제).

    Args:
        pipeline: fit된 sklearn Pipeline.
        input_specs: InputSpec 목록.
        target_spec: TargetSpec (첫 번째 대상).
        output_dir: 묶음 상위 디렉터리 (예: deployment/bundles/).
        bundle_id: 묶음 식별자.
        disclaimer: demo 고지 문구.

    Returns:
        묶음 디렉터리 경로.
    """
    bundle_dir = output_dir / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # ── model.joblib ──────────────────────────────────────────────────────────
    model_path = bundle_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # ── contract.json ─────────────────────────────────────────────────────────
    contract = {
        "bundle_id": bundle_id,
        "bundle_type": DEMO_BUNDLE_MARKER,
        "disclaimer": disclaimer,
        "input_ids": [s.id for s in input_specs],
        "input_dtypes": {s.id: s.dtype for s in input_specs},
        "target_id": target_spec.id,
        "target_task": target_spec.task,
        "target_unit": target_spec.unit,
    }
    contract_path = bundle_dir / "contract.json"
    with contract_path.open("w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)

    # ── environment.json ──────────────────────────────────────────────────────
    import sklearn
    try:
        import xgboost as xgb
        xgb_ver = xgb.__version__
    except Exception:
        xgb_ver = "unavailable"

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgb_ver,
    }
    env_path = bundle_dir / "environment.json"
    with env_path.open("w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    return bundle_dir


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    """묶음 디렉터리에서 Pipeline과 계약 정보를 읽는다.

    필수 파일이 없거나 contract.json 형식이 잘못된 경우 BundleError를 발생시킨다.
    이용자 업로드 파일·URL·경로를 모델 로딩에 사용하지 않는다.

    Args:
        bundle_dir: 묶음 디렉터리 경로.

    Returns:
        {
            "pipeline": Pipeline,
            "contract": dict,
            "bundle_dir": Path,
        }

    Raises:
        BundleError: 필수 파일 누락, 형식 오류, 역직렬화 실패.
    """
    if not bundle_dir.is_dir():
        raise BundleError(f"묶음 디렉터리가 없습니다: {bundle_dir}")

    # 필수 파일 존재 확인
    model_path = bundle_dir / "model.joblib"
    contract_path = bundle_dir / "contract.json"

    for required in (model_path, contract_path):
        if not required.exists():
            raise BundleError(
                f"묶음 필수 파일이 없습니다: {required.name} (위치: {bundle_dir})"
            )

    # contract.json 읽기
    try:
        with contract_path.open(encoding="utf-8") as f:
            contract = json.load(f)
    except Exception as exc:
        raise BundleError(f"contract.json 읽기 실패: {exc}") from exc

    # 필수 키 확인
    for key in ("bundle_id", "bundle_type", "input_ids", "target_id", "target_task"):
        if key not in contract:
            raise BundleError(
                f"contract.json에 필수 키 '{key}'가 없습니다."
            )

    # Pipeline 역직렬화
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise BundleError(
            f"model.joblib 역직렬화 실패 (환경 호환성 확인 필요): {exc}"
        ) from exc

    return {
        "pipeline": pipeline,
        "contract": contract,
        "bundle_dir": bundle_dir,
    }


# ── 완전한 배포 묶음 내보내기와 무결성 검사 (작업 16) ────────────────────────
#
# 설계 7.1-7.2:
#   manifest.json(파일 SHA-256, 자기 자신 제외), contract.json, models/<id>.joblib,
#   validation_summary.json, training_domain.json, environment.json
#   임시 경로 쓰기 -> 다시 읽어 저장 전후 예측 일치 검사 -> 고유 ID 확정.
#   demo 산출물·미평가 대상의 연구용 내보내기 거부. 기존 묶음 일부 덮어쓰기 금지.

import hashlib
import shutil
import tempfile

import numpy as np
import pandas as pd

MANIFEST_NAME = "manifest.json"
RESEARCH_BUNDLE_MARKER = "research"


class ExportError(Exception):
    """배포 묶음 내보내기 실패 (검증 위반·중복 등)."""


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment_info() -> dict:
    import sys
    import platform
    import sklearn
    import joblib as _joblib
    try:
        import xgboost as _xgb
        xgb_ver = _xgb.__version__
    except Exception:
        xgb_ver = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "joblib": _joblib.__version__,
        "xgboost": xgb_ver,
    }


def export_bundle(
    output_dir: Path,
    bundle_id: str,
    contract: dict,
    target_models: dict,
    validation_summary: dict,
    training_domain: dict,
    *,
    inactive_targets: dict | None = None,
    verify_frame: pd.DataFrame | None = None,
    tolerance: float = 1e-6,
    allow_demo: bool = False,
) -> Path:
    """완전한 배포 묶음을 생성한다.

    순서: 임시 디렉터리에 전체 쓰기 -> 다시 읽어 저장 전후 예측 일치 검사
          -> 모든 검사 통과 시 고유 ID 디렉터리로 확정.

    Args:
        output_dir: 묶음 상위 디렉터리 (예: deployment/bundles).
        bundle_id: 고유 묶음 ID.
        contract: 입력·출력 정의 (bundle_type 포함).
        target_models: {target_id: fitted Pipeline} (활성 대상만).
        validation_summary: 대상별 평가 지표·방법·범위.
        training_domain: 대상별 학습 입력 범위·지원 범주.
        inactive_targets: {target_id: 사유} (모델 없이 사유만 포함).
        verify_frame: 저장 전후 예측 일치 검사용 입력 표. None 이면 검사 생략하지 않고
            대상별로 예측 재현을 확인할 수 없으므로 ExportError.
        tolerance: 저장 전후 예측 허용 오차 (명시적).
        allow_demo: demo 묶음 내보내기 허용 여부. 기본 False (연구용 거부).

    Returns:
        확정된 묶음 디렉터리 경로.

    Raises:
        ExportError: demo 거부, 미평가 대상, 중복 묶음, 예측 불일치 등.
    """
    inactive_targets = dict(inactive_targets or {})

    # demo 산출물의 연구용 내보내기 거부
    btype = contract.get("bundle_type", RESEARCH_BUNDLE_MARKER)
    if btype == DEMO_BUNDLE_MARKER and not allow_demo:
        raise ExportError(
            "demo 산출물은 연구용으로 내보낼 수 없습니다. "
            "allow_demo=True 로 명시해야 하며 연구 활성화 경로는 이를 거부합니다."
        )

    # 활성 대상이 하나도 없으면 거부
    if not target_models:
        raise ExportError("활성 대상 모델이 없습니다. 내보낼 수 없습니다.")

    # 미평가 대상 거부: validation_summary 에 각 활성 대상의 평가가 있어야 한다
    for tid in target_models:
        vs = validation_summary.get(tid)
        if not vs or not vs.get("evaluated", False):
            raise ExportError(
                "대상 '" + tid + "' 의 독립 평가가 완료되지 않아 내보낼 수 없습니다."
            )

    # 예측 재현 검사용 입력 필수
    if verify_frame is None:
        raise ExportError("저장 전후 예측 일치 검사를 위한 verify_frame 이 필요합니다.")

    final_dir = output_dir / bundle_id
    if final_dir.exists():
        raise ExportError(
            "이미 존재하는 묶음입니다: " + str(final_dir)
            + ". 기존 묶음을 일부만 덮어쓰지 않습니다."
        )

    input_ids = contract["input_ids"]
    X = verify_frame[input_ids] if all(c in verify_frame.columns for c in input_ids) else verify_frame

    # 저장 전 예측 (메모리)
    pred_before = {tid: np.asarray(model.predict(X), dtype=float)
                   for tid, model in target_models.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bundle_tmp_", dir=str(output_dir)))
    try:
        models_dir = tmp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # 대상별 모델 저장
        model_files = {}
        for tid, model in target_models.items():
            mp = models_dir / (tid + ".joblib")
            joblib.dump(model, mp)
            model_files[tid] = "models/" + tid + ".joblib"

        # contract.json
        contract_out = dict(contract)
        contract_out["bundle_id"] = bundle_id
        contract_out.setdefault("bundle_type", RESEARCH_BUNDLE_MARKER)
        (tmp_dir / "contract.json").write_text(
            json.dumps(contract_out, ensure_ascii=False, indent=2), encoding="utf-8")

        # validation_summary.json / training_domain.json / environment.json
        (tmp_dir / "validation_summary.json").write_text(
            json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "training_domain.json").write_text(
            json.dumps(training_domain, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "environment.json").write_text(
            json.dumps(_environment_info(), ensure_ascii=False, indent=2), encoding="utf-8")

        # manifest.json: 자기 자신 제외한 모든 파일의 SHA-256
        file_list = [p for p in tmp_dir.rglob("*") if p.is_file()]
        file_hashes = {}
        for p in sorted(file_list):
            rel = p.relative_to(tmp_dir).as_posix()
            file_hashes[rel] = _sha256_of_file(p)

        manifest = {
            "bundle_id": bundle_id,
            "bundle_type": contract_out["bundle_type"],
            "active_targets": sorted(target_models.keys()),
            "inactive_targets": inactive_targets,
            "model_files": model_files,
            "file_hashes": file_hashes,
            "tolerance": tolerance,
        }
        (tmp_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # 다시 읽어 저장 전후 예측 일치 검사
        for tid in target_models:
            reloaded = joblib.load(tmp_dir / "models" / (tid + ".joblib"))
            pred_after = np.asarray(reloaded.predict(X), dtype=float)
            if not np.allclose(pred_before[tid], pred_after, rtol=0, atol=tolerance):
                raise ExportError(
                    "대상 '" + tid + "' 의 저장 전후 예측이 허용 오차("
                    + str(tolerance) + ")를 벗어났습니다."
                )

        # 모든 검사 통과 -> 고유 ID 로 확정 (원자적 이동)
        tmp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return final_dir


def verify_bundle_integrity(bundle_dir: Path) -> None:
    """manifest 의 파일 해시·상대 경로를 모델 로딩 전에 검사한다.

    Raises:
        BundleError: 누락·훼손·경로 이탈·manifest 불일치.
    """
    if not bundle_dir.is_dir():
        raise BundleError("묶음 디렉터리가 없습니다: " + str(bundle_dir))
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleError("manifest.json 이 없습니다: " + str(bundle_dir))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleError("manifest.json 읽기 실패: " + str(exc)) from exc

    file_hashes = manifest.get("file_hashes", {})
    if not file_hashes:
        raise BundleError("manifest 에 파일 해시가 없습니다.")

    for rel, expected in file_hashes.items():
        # 상대 경로 이탈 거부
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise BundleError("묶음 밖을 가리키는 경로입니다: " + rel)
        target = bundle_dir / rel
        if not target.exists():
            raise BundleError("묶음 파일이 누락되었습니다: " + rel)
        actual = _sha256_of_file(target)
        if actual != expected:
            raise BundleError("파일 해시가 일치하지 않습니다 (훼손 의심): " + rel)


def load_research_bundle(bundle_dir: Path) -> dict:
    """완전 묶음을 무결성·환경 검사 후 로드한다.

    manifest·파일 해시·상대 경로를 확인한 뒤 모델을 읽는다 (설계 7.2-4).
    이용자 업로드 경로를 모델 로딩에 사용하지 않는다.

    Returns:
        {"manifest": dict, "contract": dict, "models": {tid: Pipeline}, "bundle_dir": Path}
    """
    verify_bundle_integrity(bundle_dir)
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    contract = json.loads((bundle_dir / "contract.json").read_text(encoding="utf-8"))

    models = {}
    for tid, rel in manifest.get("model_files", {}).items():
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise BundleError("모델 경로가 묶음 밖을 가리킵니다: " + rel)
        try:
            models[tid] = joblib.load(bundle_dir / rel)
        except Exception as exc:
            raise BundleError(
                "모델 역직렬화 실패 (환경 호환성 확인 필요): " + rel + " / " + str(exc)
            ) from exc

    return {"manifest": manifest, "contract": contract, "models": models, "bundle_dir": bundle_dir}


def write_active_pointer(active_path: Path, bundle_dir: Path) -> dict:
    """active.json 에 묶음 ID 와 manifest.json 의 해시를 기록한다.

    불완전한 활성화가 기존 포인터를 손상시키지 않도록 임시 파일에 쓴 뒤 교체한다.

    Returns:
        기록한 포인터 딕셔너리.
    """
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleError("manifest.json 이 없어 활성화할 수 없습니다: " + str(bundle_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer = {
        "bundle_id": manifest["bundle_id"],
        "manifest_hash": _sha256_of_file(manifest_path),
    }
    active_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = active_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(active_path)
    return pointer


# ── 묶음·환경 종합 검사 (작업 17) ────────────────────────────────────────────


def check_bundle(bundle_dir: Path) -> dict:
    """활성화 전 묶음 무결성과 환경 호환성을 종합 점검한다.

    - verify_bundle_integrity 로 파일 해시·경로·누락을 검사한다.
    - environment.json 의 저장 당시 라이브러리 버전과 현재 환경을 비교한다.
      핵심 라이브러리(sklearn, xgboost, joblib)의 major.minor 가 다르면 오류로 본다.
      경고를 무시하고 예측을 계속하지 않도록 호출자가 ok 를 확인해야 한다.

    Args:
        bundle_dir: 묶음 디렉터리.

    Returns:
        {
            "ok": bool,
            "integrity_ok": bool,
            "env_ok": bool,
            "env_mismatches": list[str],
            "errors": list[str],
        }
    """
    errors: list = []
    integrity_ok = True
    try:
        verify_bundle_integrity(bundle_dir)
    except BundleError as exc:
        integrity_ok = False
        errors.append("무결성 검사 실패: " + str(exc))

    env_mismatches: list = []
    env_ok = True
    env_path = bundle_dir / "environment.json"
    if not env_path.exists():
        env_ok = False
        errors.append("environment.json 이 없습니다.")
    else:
        try:
            saved_env = json.loads(env_path.read_text(encoding="utf-8"))
        except Exception as exc:
            env_ok = False
            errors.append("environment.json 읽기 실패: " + str(exc))
            saved_env = None

        if saved_env is not None:
            current = _environment_info()
            for lib in ("sklearn", "joblib", "xgboost"):
                saved_v = str(saved_env.get(lib, ""))
                cur_v = str(current.get(lib, ""))
                if saved_v and cur_v and saved_v != "unavailable" and cur_v != "unavailable":
                    saved_mm = ".".join(saved_v.split(".")[:2])
                    cur_mm = ".".join(cur_v.split(".")[:2])
                    if saved_mm != cur_mm:
                        env_mismatches.append(
                            lib + " 버전 불일치 (저장: " + saved_v + ", 현재: " + cur_v + ")"
                        )
            if env_mismatches:
                env_ok = False
                errors.append("환경 호환성 경고를 무시하지 않습니다: " + "; ".join(env_mismatches))

    return {
        "ok": integrity_ok and env_ok,
        "integrity_ok": integrity_ok,
        "env_ok": env_ok,
        "env_mismatches": env_mismatches,
        "errors": errors,
    }
