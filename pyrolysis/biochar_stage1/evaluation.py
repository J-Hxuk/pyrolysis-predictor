"""
evaluation.py — eval37 GroupKFold 평가 재현 (탐색적 일반화 검증).

지시서 §2: eval37/run.py 의 GroupKFold 평가를 우선 재현한다.
  - 출처 그룹(provenance_group)별 분할
  - 5-fold, seed 0/1/2
  - R²/MAE/RMSE
  - 학습/검증 출처 중복(누설) 검사
  - 추가 후보(extra_leng)의 동일·유사 관측값 혼입 검사

이 모듈은 '고정(fixed)' RF 설정의 외부 OOF 평가만 재현한다. 내부 GridSearch
튜닝은 재현 범위에서 제외하고 지시서가 요구한 고정 RF 기준모델을 평가한다.
테스트 fold 로 설정을 선택하지 않는다.

주의: 이 평가 점수는 '탐색적 일반화 검증'이며, 최종 학습 모델의 독립 검증 결과가
아니다. 또한 실행 환경(sklearn/xgboost 버전) 차이로 eval37 보고 수치와 소수점이
다를 수 있으며, 그 차이를 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from pyrolysis.biochar_stage1.features import (
    BET_GROUP_COL,
    BET_INPUT_ORDER,
    BET_TARGET_COL,
)

__all__ = [
    "RF_FIXED_PARAMS",
    "make_bet_pipeline",
    "filter_leaking_candidates",
    "shuffled_group_kfold",
    "evaluate_groupkfold",
    "GroupKFoldResult",
    "load_reference_summary",
    "compare_with_reference",
]


def shuffled_group_kfold(groups, n_folds: int, seed: int):
    """출처 그룹 단위 K-fold 분할 (seed 로 그룹 순서를 섞는다).

    sklearn 1.8 의 GroupKFold(shuffle=True, random_state=seed) 동작을 버전 독립적으로
    재현한다 (로컬 sklearn 이 shuffle 인자를 지원하지 않을 수 있음). 그룹을 seed 로
    섞은 뒤 라운드로빈 방식으로 fold 에 배정해 그룹이 fold 를 가로질러 섞이지 않게 한다.

    Yields: (train_index, test_index) 배열 쌍. 인덱스는 위치 기반(iloc).
    """
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    rng = np.random.RandomState(seed)
    order = unique_groups.copy()
    rng.shuffle(order)

    # 각 그룹을 크기 내림차순 근사 없이, 섞인 순서로 라운드로빈 배정 (그룹 단위 유지).
    fold_of_group: dict = {}
    for i, g in enumerate(order):
        fold_of_group[g] = i % n_folds

    fold_assignment = np.array([fold_of_group[g] for g in groups])
    all_idx = np.arange(len(groups))
    for f in range(n_folds):
        test_mask = fold_assignment == f
        if not test_mask.any():
            continue
        yield all_idx[~test_mask], all_idx[test_mask]

# 지시서 §2 가 명시한 고정 RF 설정. eval37/run.py 의 RF 'fixed' 와 동일.
RF_FIXED_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1,
)

# 외부 평가 설정 (eval37/plan.json)
OUTER_SEEDS = (0, 1, 2)
OUTER_FOLDS = 5


def make_bet_pipeline(model: Any | None = None) -> Pipeline:
    """BET 학습 파이프라인. 결측 대체를 파이프라인 안에 포함한다.

    eval37/run.py 와 동일하게 SimpleImputer(strategy='median', add_indicator=True)
    를 쓴다. 전처리는 각 학습 fold 안에서만 fit 된다.
    """
    if model is None:
        model = RandomForestRegressor(**RF_FIXED_PARAMS)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", model),
    ])


def filter_leaking_candidates(
    extra: pd.DataFrame, test_block: pd.DataFrame
) -> pd.DataFrame:
    """추가 후보 중 테스트 블록과 동일·유사 관측값을 제거한다 (eval37 누설 방지).

    eval37/run.py 로직 재현:
      - 테스트 출처 그룹과 같은 provenance_group 의 후보는 제외.
      - 입력 10개 중 8개 이상이 유한하고 그 값들이 테스트 어떤 행과 거의 같으면 제외.
    """
    cols = BET_INPUT_ORDER
    # 1) 같은 출처 그룹 제외
    z = extra[~extra[BET_GROUP_COL].isin(test_block[BET_GROUP_COL])].copy()
    if len(z) == 0:
        return z
    test = test_block[cols].to_numpy(dtype=float)
    keep = []
    for _, row in z.iterrows():
        v = row[cols].to_numpy(dtype=float)
        known = np.isfinite(v)
        if known.sum() >= 8:
            # 알려진 입력이 테스트 어떤 행과 거의 동일하면 유사행 -> 제외
            close = np.isclose(
                test[:, known], v[known], rtol=1e-6, atol=1e-7
            ).all(axis=1).any()
            keep.append(not close)
        else:
            keep.append(True)
    return z[np.array(keep, dtype=bool)]


@dataclass
class GroupKFoldResult:
    """고정 RF 의 외부 GroupKFold OOF 평가 결과."""

    model_name: str
    config: str
    per_seed: list[dict[str, Any]]
    mean_r2: float
    mean_mae: float
    mean_rmse: float
    r2_min: float
    r2_max: float
    n_base_rows: int
    n_groups: int
    leakage_checks_passed: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config,
            "per_seed": self.per_seed,
            "mean_r2": self.mean_r2,
            "mean_mae": self.mean_mae,
            "mean_rmse": self.mean_rmse,
            "r2_min": self.r2_min,
            "r2_max": self.r2_max,
            "n_base_rows": self.n_base_rows,
            "n_groups": self.n_groups,
            "leakage_checks_passed": self.leakage_checks_passed,
            "note": self.note,
        }


def evaluate_groupkfold(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    model_factory=None,
    model_name: str = "RF",
    seeds: tuple[int, ...] = OUTER_SEEDS,
    n_folds: int = OUTER_FOLDS,
    merge_extra: bool = True,
) -> GroupKFoldResult:
    """고정 설정의 외부 GroupKFold OOF 평가를 수행한다 (eval37 재현).

    Args:
        base: 182행 기준 평가 데이터.
        extra: 99행 추가 후보.
        model_factory: 인자 없이 새 모델을 반환하는 호출 가능 객체. None 이면 고정 RF.
        seeds: 외부 반복 seed (기본 0,1,2).
        n_folds: 외부 GroupKFold fold 수 (기본 5).
        merge_extra: 누설 제거 후 추가 후보를 학습에 병합할지 여부.

    Returns:
        GroupKFoldResult. 학습/검증 출처 중복이 발생하면 AssertionError 가 아니라
        leakage_checks_passed=False 로 기록한다 (여기서는 검사를 강제 수행).
    """
    if model_factory is None:
        model_factory = lambda: RandomForestRegressor(**RF_FIXED_PARAMS)

    cols = BET_INPUT_ORDER
    y_all = pd.to_numeric(base[BET_TARGET_COL], errors="coerce").to_numpy(dtype=float)
    groups = base[BET_GROUP_COL].to_numpy()

    per_seed: list[dict[str, Any]] = []
    leakage_ok = True

    for seed in seeds:
        oof = np.zeros(len(base), dtype=float)
        for tr_idx, te_idx in shuffled_group_kfold(groups, n_folds, seed):
            test_block = base.iloc[te_idx]
            train_block = base.iloc[tr_idx]

            if merge_extra:
                z = filter_leaking_candidates(extra, test_block)
                train_df = pd.concat([train_block, z], ignore_index=True)
            else:
                train_df = train_block.copy()

            # 누설 검사: 학습/테스트 출처 그룹 중복 없어야 함
            if not set(train_df[BET_GROUP_COL]).isdisjoint(set(test_block[BET_GROUP_COL])):
                leakage_ok = False

            pipe = make_bet_pipeline(model_factory())
            pipe.fit(train_df[cols], pd.to_numeric(train_df[BET_TARGET_COL], errors="coerce"))
            oof[te_idx] = pipe.predict(test_block[cols])

        r2 = float(r2_score(y_all, oof))
        mae = float(mean_absolute_error(y_all, oof))
        rmse = float(np.sqrt(mean_squared_error(y_all, oof)))
        per_seed.append({"seed": int(seed), "r2": r2, "mae": mae, "rmse": rmse})

    r2s = [p["r2"] for p in per_seed]
    maes = [p["mae"] for p in per_seed]
    rmses = [p["rmse"] for p in per_seed]

    return GroupKFoldResult(
        model_name=model_name,
        config="fixed",
        per_seed=per_seed,
        mean_r2=float(np.mean(r2s)),
        mean_mae=float(np.mean(maes)),
        mean_rmse=float(np.mean(rmses)),
        r2_min=float(np.min(r2s)),
        r2_max=float(np.max(r2s)),
        n_base_rows=int(len(base)),
        n_groups=int(base[BET_GROUP_COL].nunique()),
        leakage_checks_passed=leakage_ok,
        note=(
            "탐색적 일반화 검증 (새 원료·새 논문). 최종 학습 모델의 독립 검증이 아니다. "
            "eval37/run.py 의 고정 RF 설정을 재현한다 (내부 GridSearch 튜닝 제외). "
            "로컬 sklearn 이 GroupKFold(shuffle=) 를 지원하지 않을 수 있어 그룹 셔플 "
            "분할을 자체 구현(shuffled_group_kfold)으로 재현했으며, eval37 분할과 "
            "완전히 동일하지 않을 수 있다."
        ),
    )


def load_reference_summary(base_path=None) -> dict[str, dict[str, float]]:
    """eval37/summary.csv 의 보고 수치를 읽어 {model_config: metrics} 로 반환한다."""
    from pyrolysis.biochar_stage1.extract import resolve_eval37_dir

    d = resolve_eval37_dir(base_path)
    df = pd.read_csv(d / "summary.csv")
    out: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        key = f"{r['model']}_{r['config']}"
        out[key] = {
            "r2": float(r["r2"]), "mae": float(r["mae"]), "rmse": float(r["rmse"]),
            "r2_min": float(r["r2_min"]), "r2_max": float(r["r2_max"]),
        }
    return out


def compare_with_reference(
    result: GroupKFoldResult, base_path=None
) -> dict[str, Any]:
    """재현 결과를 eval37/summary.csv 의 RF_fixed 와 비교한다.

    실행 환경(라이브러리 버전)·분할 구현 차이로 생기는 수치 차이를 보고한다.
    """
    ref = load_reference_summary(base_path)
    key = f"{result.model_name}_fixed"
    reference = ref.get(key)
    if reference is None:
        return {"reference_found": False, "key": key}
    return {
        "reference_found": True,
        "reference_key": key,
        "reference": reference,
        "reproduced": {
            "r2": result.mean_r2, "mae": result.mean_mae, "rmse": result.mean_rmse,
            "r2_min": result.r2_min, "r2_max": result.r2_max,
        },
        "delta": {
            "r2": result.mean_r2 - reference["r2"],
            "mae": result.mean_mae - reference["mae"],
            "rmse": result.mean_rmse - reference["rmse"],
        },
        "note": (
            "로컬 sklearn/xgboost 버전이 eval37(versions.json: sklearn 1.8.0, xgboost 3.4.1)과 "
            "다르면 OOF 예측과 지표에 소수점 차이가 생길 수 있다. GroupKFold 구현·난수도 "
            "버전에 따라 분할이 달라질 수 있어 차이를 그대로 보고한다."
        ),
    }
