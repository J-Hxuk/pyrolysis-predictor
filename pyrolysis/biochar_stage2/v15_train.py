"""
v15_train.py — 최종 Stage2 모델 학습·버전 저장 (앱과 분리된 학습 단계).

지시: 학습과 앱 실행을 분리한다. 앱은 저장된 모델만 로드하고 재학습하지 않는다.

이 모듈은 Core 97 전체로 Model A/B/C 최종 모델을 학습해 저장한다. 저장 모델에는
교차검증(그룹 단위) 결과·입력 계약·적용 범위·데이터 해시를 메타데이터로 함께 기록한다.
그룹 CV 결과가 '무작위 분할 성능'이 아님을, 그리고 Model B 가 A 대비 개선 여부를 명시한다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2 import v15_models as M
from pyrolysis.biochar_stage2.v15_loader import Stage2CoreDataset, load_core_qe

__all__ = [
    "DEFAULT_REGISTRY_DIR",
    "train_stage2_model",
    "save_stage2_model",
    "Stage2Registry",
]

DEFAULT_REGISTRY_DIR = Path("deployment/stage2_models")


def _dataset_hash(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df.fillna("NA"), index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _lib_versions() -> dict[str, str]:
    import sklearn
    try:
        import xgboost
        xgb = xgboost.__version__
    except Exception:
        xgb = "unavailable"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__, "xgboost": xgb}


def train_stage2_model(
    model_name: str, algo_name: str, *, ds: Stage2CoreDataset | None = None,
    base: Path | None = None, include_raw_composition: bool = False,
    run_cv: bool = True,
) -> dict[str, Any]:
    """Core 97 전체로 지정 모델(A/B/C)×알고리즘을 학습하고 메타데이터를 구성한다.

    Returns: {"pipeline": fitted Pipeline, "metadata": {...}}.
    """
    if ds is None:
        ds = load_core_qe(base)
    df = ds.df
    algos = M.available_algorithms()
    if algo_name not in algos:
        raise ValueError(f"사용 불가 알고리즘: {algo_name}. 가능: {list(algos)}")

    input_cols = fc.model_inputs(model_name, include_raw_composition=include_raw_composition)
    input_cols = [c for c in input_cols if c in df.columns]
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)

    pipeline = M.build_pipeline(algos[algo_name](), input_cols)
    pipeline.fit(df[input_cols], y)

    cv_block: dict[str, Any] | None = None
    if run_cv:
        cv_block = {}
        for scheme in ("LODO", "GroupKFold_DOI", "LOFO"):
            res = M.evaluate_model(df, model_name, algo_name, algos[algo_name], scheme,
                                   include_raw_composition=include_raw_composition)
            cv_block[scheme] = res.to_dict()

    meta = {
        "stage": "stage2",
        "model_name": model_name,
        "model_label": fc.MODEL_DEFS.get(model_name, {}).get("name"),
        "algorithm": algo_name,
        "target": {"column": fc.TARGET_COLUMN, "unit": fc.TARGET_UNIT,
                   "name": "qe_reported_mg_g",
                   "qualification": "REPORTED_EXPERIMENTAL_UPTAKE_EQUILIBRIUM_NOT_CERTIFIED"},
        "input_cols": input_cols,
        "n_inputs": len(input_cols),
        "include_raw_composition": include_raw_composition,
        "training_data": {
            "source": "03_Stage2_QE_Core_97.csv",
            "n_rows": int(len(df)), "n_doi": int(df[fc.GROUP_DOI].nunique()),
            "n_feedstock": int(df[fc.GROUP_FEEDSTOCK].nunique()),
            "dataset_hash": _dataset_hash(df),
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_validation": cv_block,
        "library_versions": _lib_versions(),
        "applicability": {
            "scope": "탐색적 reported-uptake qe (엄격한 평형 미인증). 실측 문헌 적용범위 내 보조 도구.",
            "not_for": "실험 측정 대체·검증 없는 전역 최적조건 정량 추천.",
            "feedstock_policy": "미학습 원료는 적용범위 밖으로 판정. 이름 유사 원료로 자동 대체 금지. 이름에서 조성 생성 금지.",
            "stage1_link": "Stage1 예측 BET/TPV 연계(Model D)는 이번 작업에서 비활성.",
        },
        "honest_notes": [
            "그룹 단위(LODO/GroupKFold_DOI/LOFO) 교차검증 결과만 성능으로 본다(무작위 행 분할 아님).",
            "Model B(제조조건 추가)가 Model A(흡착조건만)보다 반드시 개선되지는 않는다 — CV 결과로 판단.",
            "물리제약(0≤qe≤C0/dose) 위반율과 제약 후 성능을 함께 본다.",
        ],
    }
    return {"pipeline": pipeline, "metadata": meta}


def save_stage2_model(trained: dict[str, Any], out_dir: Path) -> Path:
    """학습 모델을 버전 디렉터리에 저장한다 (model.joblib + metadata.json)."""
    meta = trained["metadata"]
    version = f"stage2_{meta['model_name']}_{meta['algorithm']}_{meta['training_data']['dataset_hash']}"
    vdir = Path(out_dir) / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained["pipeline"], vdir / "model.joblib")
    meta = dict(meta); meta["model_version"] = version
    (vdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return vdir


class Stage2Registry:
    """저장된 Stage2 모델 버전 관리(앱은 활성 모델만 로드, 재학습 없음)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def default(cls, base: Path | None = None) -> "Stage2Registry":
        base = Path(base) if base is not None else Path.cwd()
        return cls(root=base / DEFAULT_REGISTRY_DIR)

    def list_versions(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and (d / "model.joblib").is_file())

    def load(self, version: str) -> dict[str, Any]:
        vdir = self.root / version
        pipe = joblib.load(vdir / "model.joblib")
        meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
        return {"pipeline": pipe, "metadata": meta, "version_dir": vdir}

    def set_active(self, version: str) -> None:
        if version not in self.list_versions():
            raise ValueError(f"등록되지 않은 버전: {version}")
        tmp = self.root / "active.json.tmp"
        tmp.write_text(json.dumps({"active_version": version}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.root / "active.json")

    def get_active(self) -> dict[str, Any] | None:
        ap = self.root / "active.json"
        if not ap.is_file():
            return None
        v = json.loads(ap.read_text(encoding="utf-8")).get("active_version")
        if v and v in self.list_versions():
            return self.load(v)
        return None
