"""
predictor.py — Stage 1 v60 예측 진입점(패키지 예측기에 위임 + 호환성 게이트).

사용:
    from pyrolysis.biochar_stage1_v60 import predictor
    handle = predictor.load_v60()           # 추출 + 예측기 로드 + 호환성 검사
    result = predictor.predict(handle, inp) # 호환 시 BET·TPV 예측, 아니면 차단 결과

이 모듈은 v60 패키지의 `predict_stage1_v60.predict` 를 그대로 호출한다(재구현 금지).
호환성 게이트가 통과하지 못하면 수치 예측을 반환하지 않고 차단 사유를 담은 결과를 준다.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyrolysis.biochar_stage1_v60 import compat, features
from pyrolysis.biochar_stage1_v60.extract import resolve_v60_final_dir

__all__ = ["V60Handle", "load_v60", "predict", "V60PredictionBlocked"]


class V60PredictionBlocked(Exception):
    """환경 비호환으로 v60 수치 예측이 차단됨."""


@dataclass
class V60Handle:
    """로드된 v60 예측기와 호환성 판정."""

    final_dir: Path
    predict_module: Any
    compat: compat.CompatResult
    final_status: dict[str, Any]

    @property
    def usable(self) -> bool:
        return self.compat.ok


def _load_predict_module(final_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "predict_stage1_v60", final_dir / "predict_stage1_v60.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_stage1_v60"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_v60(base: Path | None = None) -> V60Handle:
    """v60 패키지를 추출·로드하고 호환성을 검사한다(예측 수행 없음)."""
    import json

    final_dir = resolve_v60_final_dir(base)
    result = compat.check_compatibility(final_dir)
    # 예측기 모듈은 compat 자가검증에서 이미 로드됐을 수 있으나 명시적으로 확보한다.
    mod = _load_predict_module(final_dir)
    status_p = final_dir / "FINAL_STATUS.json"
    final_status = json.loads(status_p.read_text(encoding="utf-8")) if status_p.is_file() else {}
    return V60Handle(
        final_dir=final_dir, predict_module=mod, compat=result, final_status=final_status,
    )


def predict(handle: V60Handle, inp: dict[str, Any], *, allow_incompatible: bool = False) -> dict[str, Any]:
    """제조조건 JSON 으로 BET·TPV 를 예측한다(패키지 예측기에 위임).

    호환성 게이트가 실패하면 기본적으로 수치 예측을 **차단**하고 사유를 담은 결과를 반환한다.
    allow_incompatible=True 를 명시해야만 손상 가능 예측을 강제로 받을 수 있다(권장하지 않음).
    """
    if not handle.usable and not allow_incompatible:
        return {
            "data_version": "v60",
            "blocked": True,
            "status": handle.compat.status,
            "reason": handle.compat.messages,
            "compatibility": handle.compat.to_dict(),
            "note": (
                "환경 비호환으로 v60 수치 예측을 차단했습니다. support_grade·경고는 참고용으로만 "
                "제공될 수 있으나 BET·TPV 수치는 신뢰할 수 없습니다."
            ),
        }

    result = handle.predict_module.predict(inp)
    result["blocked"] = False
    result["compatibility_ok"] = handle.usable
    if not handle.usable:
        result["warning_env"] = (
            "allow_incompatible=True 로 강제 실행됨: 환경 비호환 상태이므로 BET·TPV 수치가 "
            "손상되었을 수 있습니다. 정상 값으로 간주하지 마세요."
        )
    return result


def model_info(handle: V60Handle) -> dict[str, Any]:
    """모델 메타·적용 범위·검증 상태 요약(표시용)."""
    return {
        "data_version": "v60",
        "status": features.V60_APPLICABILITY["status"],
        "usable_in_current_env": handle.usable,
        "compatibility_status": handle.compat.status,
        "required_libraries": handle.compat.version_info["required"],
        "current_libraries": handle.compat.version_info["current"],
        "targets": features.V60_TARGETS,
        "applicability": features.V60_APPLICABILITY,
        "final_status": handle.final_status,
        "composition_inputs_dropped": features.V60_COMPOSITION_FIELDS_DROPPED,
    }
