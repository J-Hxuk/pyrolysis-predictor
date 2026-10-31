"""
compat.py — v60 모델 환경 호환성 게이트.

v60 모델은 scikit-learn 1.8.0 / xgboost 3.4.1 로 저장되었다. 버전이 다르면 전처리
역직렬화가 어긋나 **예측값이 조용히 손상**될 수 있다(검증됨: 현재 venv sklearn 1.5.2 /
xgboost 2.1.3 에서 BET 849 → 5). 손상된 값을 정상으로 반환하지 않기 위해, 다음 두 단계를
확인한다.

1. 버전 비교: 저장 환경(sklearn 1.8.x, xgboost 3.4.x)과 현재 환경의 major.minor 비교.
2. 자가 검증: 패키지에 포함된 example/out_of_domain 예제 입력으로 예측을 재현하여, 동봉된
   example_output 의 BET·TPV 값과 허용 오차 안에서 일치하는지 확인(가장 신뢰할 수 있는 판정).

자가 검증(2)이 통과하면 버전이 달라도 사용 가능으로 본다. 자가 검증이 실패하면
ENVIRONMENT_INCOMPATIBLE 로 판정하고 수치 예측을 거부한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "REQUIRED_SKLEARN",
    "REQUIRED_XGBOOST",
    "CompatResult",
    "library_versions",
    "check_versions",
    "verify_example_reproduction",
    "check_compatibility",
]

# v60 저장 환경(패키지 versions·booster config 기준)
REQUIRED_SKLEARN = "1.8"
REQUIRED_XGBOOST = "3.4"

# 예제 예측 재현 허용 상대 오차 (0 에 가까워야 함; 손상 시 수백 배 벗어남)
REPRO_RTOL = 1e-4


def library_versions() -> dict[str, str]:
    import sklearn
    try:
        import xgboost
        xgb = xgboost.__version__
    except Exception:
        xgb = "unavailable"
    return {"sklearn": sklearn.__version__, "xgboost": xgb}


def _major_minor(v: str) -> str:
    return ".".join(str(v).split(".")[:2])


def check_versions() -> dict[str, Any]:
    """저장 환경과 현재 환경의 major.minor 일치 여부(정보용). 판정은 자가 검증이 우선."""
    cur = library_versions()
    sk_ok = _major_minor(cur["sklearn"]) == REQUIRED_SKLEARN
    xgb_ok = _major_minor(cur["xgboost"]) == REQUIRED_XGBOOST
    return {
        "current": cur,
        "required": {"sklearn": REQUIRED_SKLEARN + ".x", "xgboost": REQUIRED_XGBOOST + ".x"},
        "sklearn_match": sk_ok,
        "xgboost_match": xgb_ok,
        "all_match": sk_ok and xgb_ok,
    }


@dataclass
class CompatResult:
    """호환성 판정 결과."""

    ok: bool
    status: str                       # "OK" | "ENVIRONMENT_INCOMPATIBLE" | "PACKAGE_ERROR"
    version_info: dict[str, Any]
    reproduction: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "version_info": self.version_info,
            "reproduction": self.reproduction,
            "messages": self.messages,
        }


def verify_example_reproduction(final_dir: Path, *, rtol: float = REPRO_RTOL) -> dict[str, Any]:
    """패키지 예제 입력으로 예측을 재현해 동봉 출력과 비교한다.

    예측 로직은 패키지의 predict_stage1_v60.predict 에 위임한다(재구현 금지).
    Returns: {"reproduced": bool, "details": [...], "error": str|None}
    """
    import importlib.util
    import sys

    final_dir = Path(final_dir)
    details: list[dict[str, Any]] = []
    try:
        spec = importlib.util.spec_from_file_location(
            "predict_stage1_v60", final_dir / "predict_stage1_v60.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["predict_stage1_v60"] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {"reproduced": False, "details": [], "error": f"예측기 로드 실패: {exc}"}

    reproduced = True
    for name in ("example", "out_of_domain_test"):
        in_p = final_dir / f"{name}_input.json"
        out_p = final_dir / f"{name}_output.json"
        if not in_p.is_file() or not out_p.is_file():
            continue
        try:
            inp = json.loads(in_p.read_text(encoding="utf-8"))
            exp = json.loads(out_p.read_text(encoding="utf-8"))
            got = mod.predict(inp)
        except Exception as exc:
            return {"reproduced": False, "details": details, "error": f"{name} 예측 실패: {exc}"}
        for tgt in ("BET", "TPV"):
            ep = float(exp[tgt]["prediction"])
            gp = float(got[tgt]["prediction"])
            rel = abs(ep - gp) / max(1.0, abs(ep))
            match = rel <= rtol
            reproduced = reproduced and match
            details.append({
                "case": name, "target": tgt,
                "expected": ep, "got": gp, "rel_diff": rel, "match": match,
            })
    return {"reproduced": reproduced, "details": details, "error": None}


def check_compatibility(final_dir: Path) -> CompatResult:
    """버전 확인 + 예제 재현 자가 검증으로 호환성을 판정한다.

    판정 기준: 예제 재현이 성공해야 ok=True. 재현 실패 시 ENVIRONMENT_INCOMPATIBLE.
    """
    vinfo = check_versions()
    repro = verify_example_reproduction(final_dir)
    messages: list[str] = []

    if repro.get("error"):
        return CompatResult(
            ok=False, status="PACKAGE_ERROR", version_info=vinfo,
            reproduction=repro, messages=[repro["error"]],
        )

    if not vinfo["all_match"]:
        messages.append(
            "라이브러리 버전이 저장 환경과 다릅니다 "
            f"(현재 sklearn {vinfo['current']['sklearn']} / xgboost {vinfo['current']['xgboost']}, "
            f"요구 sklearn {vinfo['required']['sklearn']} / xgboost {vinfo['required']['xgboost']}). "
            "버전 불일치에서 예측값이 손상될 수 있습니다."
        )

    if not repro["reproduced"]:
        messages.append(
            "동봉 예제 예측 재현에 실패했습니다. 현재 환경에서 v60 수치 예측을 신뢰할 수 "
            "없으므로 예측을 차단합니다. sklearn 1.8.0 / xgboost 3.4.1 환경에서 실행하세요."
        )
        return CompatResult(
            ok=False, status="ENVIRONMENT_INCOMPATIBLE", version_info=vinfo,
            reproduction=repro, messages=messages,
        )

    messages.append("예제 예측 재현 통과. v60 수치 예측을 사용할 수 있습니다.")
    return CompatResult(
        ok=True, status="OK", version_info=vinfo, reproduction=repro, messages=messages,
    )
