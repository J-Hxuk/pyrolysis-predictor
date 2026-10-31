"""
training.py — 최종 Stage1 BET 모델 학습·버전 저장.

지시서 §2:
  - 재현 가능한 RandomForest 기준모델 (n_estimators=300, min_samples_leaf=2,
    random_state=42, n_jobs=1).
  - 학습 파이프라인 내부에 SimpleImputer(strategy='median', add_indicator=True).
  - 최종 학습 모델은 별도 버전 파일로 저장하며 메타데이터를 함께 기록한다.

데이터 불러오기(data_loader)와 학습을 분리한다. 앱 실행 시 재학습하지 않는다.

기존 보고서(eval37)의 검증 점수를 새 학습 모델의 '독립 검증 결과'처럼 표현하지
않는다. GroupKFold 결과는 탐색적 일반화 검증으로만 메타데이터에 기록한다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pyrolysis.biochar_stage1 import evaluation as stage1_eval
from pyrolysis.biochar_stage1.data_loader import Stage1Dataset, load_stage1_dataset
from pyrolysis.biochar_stage1.evaluation import (
    RF_FIXED_PARAMS,
    make_bet_pipeline,
)
from pyrolysis.biochar_stage1.features import (
    APPLICABILITY_SCOPE,
    BET_GROUP_COL,
    BET_INPUT_ORDER,
    BET_INPUT_VARIABLES,
    BET_TARGET,
    BET_TARGET_COL,
)

__all__ = [
    "MODEL_NAME",
    "build_training_frame",
    "train_bet_model",
    "save_bet_model",
    "TrainedBetModel",
]

MODEL_NAME = "stage1_bet_rf"
MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"


@dataclass
class TrainedBetModel:
    """학습된 BET 모델과 메타데이터."""

    pipeline: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]


def build_training_frame(ds: Stage1Dataset) -> tuple[pd.DataFrame, pd.Series, dict]:
    """최종 학습용 X, y 를 만든다 (base + 전체 extra 후보).

    - 학습 대상은 base(182) + extra(99). 최종 학습은 평가와 달리 전체 적격 자료를 쓴다.
    - extra 의 결측 입력은 파이프라인 내부 중앙값 대체로 처리한다 (여기서 채우지 않음).
    - value(BET) 가 결측인 행만 학습에서 제외한다 (타깃 없는 행은 학습 불가).

    Returns: (X, y, provenance 정보)
    """
    frames = []
    for name, df in (("base", ds.base), ("extra", ds.extra)):
        sub = df.copy()
        sub["_source_split"] = name
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)

    y_all = pd.to_numeric(combined[BET_TARGET_COL], errors="coerce")
    target_ok = y_all.notna()
    combined = combined[target_ok].reset_index(drop=True)
    y = pd.to_numeric(combined[BET_TARGET_COL], errors="coerce")

    X = combined[BET_INPUT_ORDER].apply(pd.to_numeric, errors="coerce")

    provenance = {
        "n_total_with_target": int(len(combined)),
        "n_base": int((combined["_source_split"] == "base").sum()),
        "n_extra": int((combined["_source_split"] == "extra").sum()),
        "provenance_groups": sorted(
            str(g) for g in combined[BET_GROUP_COL].dropna().unique()
        ),
        "n_provenance_groups": int(combined[BET_GROUP_COL].dropna().nunique()),
    }
    return X, y, provenance


def _library_versions() -> dict[str, str]:
    import sklearn
    try:
        import xgboost
        xgb_ver = xgboost.__version__
    except Exception:
        xgb_ver = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgb_ver,
    }


def train_bet_model(
    ds: Stage1Dataset | None = None,
    *,
    base: Path | None = None,
    run_cv: bool = True,
) -> TrainedBetModel:
    """최종 BET RF 모델을 학습하고 메타데이터를 구성한다.

    Args:
        ds: 미리 로드한 데이터셋. None 이면 load_stage1_dataset 로 읽는다.
        base: 데이터 경로 기준 (ds 가 None 일 때 사용).
        run_cv: GroupKFold 탐색적 평가를 메타데이터에 포함할지 여부.

    Returns:
        TrainedBetModel (pipeline + metadata).
    """
    if ds is None:
        ds = load_stage1_dataset(base)

    X, y, provenance = build_training_frame(ds)

    # 학습 자료 변수별 범위 (적용 범위 판정용). base+extra 의 관측 최소·최대.
    training_numeric_ranges: dict[str, dict[str, float]] = {}
    for c in BET_INPUT_ORDER:
        col = X[c].dropna()
        if len(col):
            training_numeric_ranges[c] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "n": int(len(col)),
            }

    model = RandomForestRegressor(**RF_FIXED_PARAMS)
    pipeline = make_bet_pipeline(model)
    pipeline.fit(X, y)

    # 모델 버전: 데이터셋 해시 + 하이퍼파라미터 + 입력 순서 지문
    fingerprint = {
        "dataset_version_hash": ds.dataset_version_hash,
        "rf_params": RF_FIXED_PARAMS,
        "inputs": BET_INPUT_ORDER,
    }
    fp_hash = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()[:12]
    model_version = f"{MODEL_NAME}-{ds.dataset_version_hash}-{fp_hash}"

    cv_block: dict[str, Any] | None = None
    if run_cv:
        res = stage1_eval.evaluate_groupkfold(ds.base, ds.extra, model_name="RF")
        cmp = stage1_eval.compare_with_reference(res, base)
        cv_block = {
            "evaluation_type": (
                "탐색적 일반화 검증 (GroupKFold, 새 원료·새 논문). "
                "이 최종 학습 모델의 독립 검증 결과가 아니다."
            ),
            "config": "RF fixed (n_estimators=300, min_samples_leaf=2, random_state=42)",
            "outer_seeds": list(stage1_eval.OUTER_SEEDS),
            "outer_folds": stage1_eval.OUTER_FOLDS,
            "group_column": BET_GROUP_COL,
            "result": res.to_dict(),
            "comparison_with_eval37_reference": cmp,
        }

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "stage": "stage1",
        "target": {
            "id": BET_TARGET.id,
            "label_ko": BET_TARGET.label_ko,
            "unit": BET_TARGET.unit,
            "description": BET_TARGET.description,
        },
        "inputs": [
            {
                "id": vid,
                "order": i,
                "unit": BET_INPUT_VARIABLES[vid].unit,
                "basis": BET_INPUT_VARIABLES[vid].basis,
                "group": BET_INPUT_VARIABLES[vid].group,
            }
            for i, vid in enumerate(BET_INPUT_ORDER)
        ],
        "input_order": BET_INPUT_ORDER,
        "analysis_basis": {
            "elemental_CHNO": "dry ash-free (daf), 원료 기준",
            "proximate_VM_Ash_FC": "dry basis, 원료 기준",
            "note": "원료 조성 기준. 생성 바이오차 조성이 아니다. 합을 100 으로 정규화하지 않음.",
        },
        "model_type": "RandomForestRegressor",
        "hyperparameters": dict(RF_FIXED_PARAMS),
        "preprocessing": "SimpleImputer(strategy='median', add_indicator=True) (파이프라인 내부, 학습 자료로만 fit)",
        "training_data": {
            **ds.version_info(),
            **provenance,
        },
        "training_numeric_ranges": training_numeric_ranges,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_validation": cv_block,
        "library_versions": _library_versions(),
        "applicability": APPLICABILITY_SCOPE,
        "unvalidated_notes": [
            "입력에 없는 인자(활성화·함침·세척·반응기·분위기 등)의 영향은 검증하지 않았다.",
            "GroupKFold 점수는 탐색적 일반화 검증이며 이 배포 모델의 독립 검증이 아니다.",
            "Leng2022 기반 비활성화 바이오매스 열분해 범위 밖은 외삽이다.",
            "예측 신뢰도 백분율을 제공하지 않는다.",
        ],
    }
    return TrainedBetModel(pipeline=pipeline, metadata=metadata)


def save_bet_model(model: TrainedBetModel, out_dir: Path) -> Path:
    """학습 모델을 버전 디렉터리에 저장한다 (model.joblib + metadata.json).

    out_dir/<model_version>/ 아래에 저장한다. 기존 버전을 덮어쓰지 않는다 (있으면 재사용).

    Returns: 저장된 버전 디렉터리.
    """
    out_dir = Path(out_dir)
    version_dir = out_dir / model.version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.pipeline, version_dir / MODEL_FILE)
    (version_dir / META_FILE).write_text(
        json.dumps(model.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return version_dir
