"""
model_registry.py — Stage1 BET 모델·버전 관리.

학습(training)과 분리된 레지스트리. 저장된 모델 버전을 나열하고 활성 버전을
가리키는 포인터(active.json)를 관리한다. 앱은 레지스트리를 통해 활성 모델만 읽고,
앱 실행 시 재학습하지 않는다.

기본 저장 위치: deployment/stage1_models/
  deployment/stage1_models/<model_version>/{model.joblib, metadata.json}
  deployment/stage1_models/active.json  -> {"active_version": "..."}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_REGISTRY_DIR",
    "Stage1Registry",
]

DEFAULT_REGISTRY_DIR = Path("deployment/stage1_models")
ACTIVE_POINTER = "active.json"
MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"


@dataclass
class Stage1Registry:
    """Stage1 BET 모델 버전 레지스트리 (파일 시스템 기반)."""

    root: Path

    @classmethod
    def default(cls, base: Path | None = None) -> "Stage1Registry":
        base = Path(base) if base is not None else Path.cwd()
        return cls(root=base / DEFAULT_REGISTRY_DIR)

    def list_versions(self) -> list[str]:
        """저장된 모델 버전 목록 (model.joblib + metadata.json 가 있는 디렉터리)."""
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / MODEL_FILE).is_file() and (d / META_FILE).is_file():
                out.append(d.name)
        return out

    def version_dir(self, version: str) -> Path:
        return self.root / version

    def read_metadata(self, version: str) -> dict[str, Any]:
        p = self.version_dir(version) / META_FILE
        if not p.is_file():
            raise FileNotFoundError(f"metadata.json 이 없습니다: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def set_active(self, version: str) -> dict[str, Any]:
        """활성 버전 포인터를 교체한다 (임시 파일 후 교체로 손상 방지)."""
        if version not in self.list_versions():
            raise ValueError(f"등록되지 않은 버전입니다: {version}")
        self.root.mkdir(parents=True, exist_ok=True)
        pointer = {"active_version": version}
        active_path = self.root / ACTIVE_POINTER
        tmp = active_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(active_path)
        return pointer

    def get_active_version(self) -> str | None:
        """현재 활성 버전. 포인터가 없거나 가리키는 버전이 없으면 None."""
        active_path = self.root / ACTIVE_POINTER
        if not active_path.is_file():
            # 포인터가 없으면 유일한 버전이 있을 때 그것을 쓰지 않고 None 을 반환한다
            # (명시적 활성화를 강제). 단, 편의상 버전이 하나뿐이면 그것을 반환.
            versions = self.list_versions()
            return versions[-1] if len(versions) == 1 else None
        try:
            pointer = json.loads(active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        v = pointer.get("active_version")
        if v and v in self.list_versions():
            return v
        return None

    def get_active_dir(self) -> Path | None:
        """활성 버전 디렉터리. 없으면 None."""
        v = self.get_active_version()
        return self.version_dir(v) if v else None
