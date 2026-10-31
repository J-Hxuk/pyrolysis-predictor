"""
v15_models.py — Stage2 Model A/B/C 학습·검증 파이프라인.

지시 사항:
  - 알고리즘: DummyRegressor, Ridge, RandomForest, ExtraTrees, XGBoost, (환경 허용 시) CatBoost.
  - 전처리(결측 대체·인코딩·스케일링)는 **학습 fold 안에서만 fit**.
  - 무작위 행 분할 성능을 최종 성능으로 보고하지 않는다. 검증은 그룹 단위:
      1) Leave-One-DOI-Out, 2) GroupKFold(DOI), 3) Leave-One-Feedstock-Out.
  - 지표: R², MAE, RMSE, DOI별, 원료별, 물리제약 위반율.
  - 물리 제약: 0 ≤ qe ≤ C0/dose. 예측 원시값과 제약 보정값을 구분해 보고.
  - 과적합 통제를 우선(데이터 97행으로 매우 작음).

Streamlit 을 import 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pyrolysis.biochar_stage2 import v15_features as fc

__all__ = [
    "RANDOM_STATE",
    "available_algorithms",
    "build_pipeline",
    "make_splits",
    "evaluate_model",
    "compare_models",
    "physical_cap",
]

RANDOM_STATE = 42

# Model A/B/C 에서 범주형으로 취급할 입력 (나머지는 수치)
_CATEGORICAL = {
    "pollutant", "feedstock_label_normalized", "manufacturing_route",
    "pyrolysis_atmosphere", "activation_or_modification",
}


def available_algorithms() -> dict[str, Callable[[], Any]]:
    """사용 가능한 회귀 알고리즘 팩토리. 과적합 통제를 우선한 보수적 설정.

    데이터가 97행으로 매우 작으므로 트리 깊이·리프를 제한하고 정규화를 둔다.
    CatBoost 는 설치된 경우에만 포함한다.
    """
    algos: dict[str, Callable[[], Any]] = {
        "Dummy": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
    }
    try:
        from xgboost import XGBRegressor
        algos["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            min_child_weight=5, random_state=RANDOM_STATE, n_jobs=1,
            tree_method="hist", verbosity=0)
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        algos["CatBoost"] = lambda: CatBoostRegressor(
            iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=6.0,
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
    except Exception:
        pass  # 환경에 CatBoost 없음 → 생략 (지시: 환경 허용 시에만)
    return algos


def build_pipeline(model, input_cols: list[str]) -> Pipeline:
    """전처리 + 모델 Pipeline. 전처리는 fit(=학습 fold)에서만 학습된다.

    - 수치: 중앙값 대치 + 표준화.
    - 범주: 최빈값 대치 + one-hot(handle_unknown='ignore' 로 미학습 범주 허용).
    범주 미보고(결측)를 특정 값으로 임의 보충하지 않는다 — 대치는 fold 내부 통계로만.
    """
    num_cols = [c for c in input_cols if c not in _CATEGORICAL]
    cat_cols = [c for c in input_cols if c in _CATEGORICAL]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("model", model)])


def physical_cap(pred: np.ndarray, c0: np.ndarray, dose: np.ndarray) -> np.ndarray:
    """0 ≤ qe ≤ C0/dose 로 보정한 예측값(제약 후)."""
    cap = np.divide(c0, dose, out=np.full_like(c0, np.inf, dtype=float),
                    where=(dose != 0) & np.isfinite(dose))
    capped = np.clip(pred, 0.0, cap)
    return capped


def make_splits(df: pd.DataFrame, scheme: str, *, n_folds: int = 5):
    """그룹 단위 분할 인덱스 쌍을 생성한다 (위치 기반 iloc).

    scheme:
      - "LODO"             : Leave-One-DOI-Out (DOI 하나씩 테스트)
      - "GroupKFold_DOI"   : DOI 그룹 K-fold
      - "LOFO"             : Leave-One-Feedstock-Out
    동일 논문/원료가 학습·검증에 동시에 들어가지 않도록 그룹을 유지한다.
    """
    idx = np.arange(len(df))
    if scheme == "LODO":
        groups = df[fc.GROUP_DOI].to_numpy()
        for g in pd.unique(groups):
            te = idx[groups == g]
            tr = idx[groups != g]
            if len(tr) and len(te):
                yield str(g), tr, te
    elif scheme == "LOFO":
        groups = df[fc.GROUP_FEEDSTOCK].to_numpy()
        for g in pd.unique(groups):
            te = idx[groups == g]
            tr = idx[groups != g]
            if len(tr) and len(te):
                yield str(g), tr, te
    elif scheme == "GroupKFold_DOI":
        from sklearn.model_selection import GroupKFold
        groups = df[fc.GROUP_DOI].to_numpy()
        y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy()
        n = min(n_folds, len(pd.unique(groups)))
        gkf = GroupKFold(n_splits=n)
        for i, (tr, te) in enumerate(gkf.split(idx, y, groups)):
            yield f"fold{i}", tr, te
    else:
        raise ValueError(f"알 수 없는 분할: {scheme}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "r2": None, "mae": None, "rmse": None}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # R² 는 테스트 샘플이 2개 미만이거나 분산 0이면 정의 불가
    if len(y_true) >= 2 and np.var(y_true) > 0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None
    return {"n": int(len(y_true)), "r2": r2, "mae": mae, "rmse": rmse}


@dataclass
class ModelEvalResult:
    model_name: str
    algo: str
    scheme: str
    input_cols: list[str]
    overall_raw: dict[str, Any]
    overall_capped: dict[str, Any]
    per_group: dict[str, dict[str, Any]]
    n_constraint_violations: int
    n_total: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name, "algo": self.algo, "scheme": self.scheme,
            "n_inputs": len(self.input_cols),
            "overall_raw": self.overall_raw, "overall_capped": self.overall_capped,
            "per_group_summary": {
                "n_groups": len(self.per_group),
                "groups": self.per_group,
            },
            "constraint_violation_rate": (
                round(self.n_constraint_violations / self.n_total, 4) if self.n_total else None
            ),
            "n_constraint_violations": self.n_constraint_violations,
            "n_total": self.n_total,
            "note": self.note,
        }


def evaluate_model(
    df: pd.DataFrame, model_name: str, algo_name: str, algo_factory,
    scheme: str, *, include_raw_composition: bool = False,
) -> ModelEvalResult:
    """하나의 (모델 A/B/C, 알고리즘, 분할)에 대해 OOF 평가를 수행한다.

    전처리·모델은 각 학습 fold 로만 fit. OOF 예측을 모아 전체·그룹별 지표와 물리제약
    위반율을 계산한다. 원시 예측과 제약 보정 예측을 각각 보고한다.
    """
    from sklearn.base import clone

    input_cols = fc.model_inputs(model_name, include_raw_composition=include_raw_composition)
    input_cols = [c for c in input_cols if c in df.columns]
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    c0 = pd.to_numeric(df[fc.C0_COL], errors="coerce").to_numpy(dtype=float)
    dose = pd.to_numeric(df[fc.DOSE_COL], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    oof_raw = np.full(n, np.nan)
    oof_group = np.empty(n, dtype=object)

    for gname, tr, te in make_splits(df, scheme):
        pipe = build_pipeline(algo_factory(), input_cols)
        pipe.fit(df.iloc[tr][input_cols], y[tr])
        pred = np.asarray(pipe.predict(df.iloc[te][input_cols]), dtype=float)
        oof_raw[te] = pred
        oof_group[te] = gname

    mask = ~np.isnan(oof_raw) & ~np.isnan(y)
    yt = y[mask]
    yr = oof_raw[mask]
    yc = physical_cap(yr, c0[mask], dose[mask])

    # 물리제약 위반(원시 예측 기준): qe<0 또는 qe>C0/dose
    cap_limit = np.divide(c0[mask], dose[mask],
                          out=np.full(mask.sum(), np.inf), where=(dose[mask] != 0))
    violations = int(np.sum((yr < -1e-9) | (yr > cap_limit * (1 + 1e-9))))

    # 그룹별 지표(원시 기준)
    per_group: dict[str, dict[str, Any]] = {}
    groups_masked = oof_group[mask]
    for g in pd.unique(groups_masked):
        gm = groups_masked == g
        per_group[str(g)] = _metrics(yt[gm], yr[gm])

    return ModelEvalResult(
        model_name=model_name, algo=algo_name, scheme=scheme, input_cols=input_cols,
        overall_raw=_metrics(yt, yr), overall_capped=_metrics(yt, yc),
        per_group=per_group, n_constraint_violations=violations, n_total=int(mask.sum()),
        note=(
            "그룹 단위 OOF 평가. 전처리는 학습 fold 로만 fit. 무작위 행 분할 아님. "
            "R² 는 소표본 그룹에서 None 일 수 있음."
        ),
    )


def compare_models(
    df: pd.DataFrame, *, schemes: tuple[str, ...] = ("LODO", "GroupKFold_DOI", "LOFO"),
    models: tuple[str, ...] = ("A", "B", "C"),
    include_raw_composition: bool = False,
) -> dict[str, Any]:
    """Model A/B/C × 알고리즘 × 분할 전수 비교. 동일 데이터·동일 분할에서 공정 비교.

    Returns: 중첩 dict {scheme: {model: {algo: result_dict}}} + 요약.
    """
    algos = available_algorithms()
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_doi": int(df[fc.GROUP_DOI].nunique()),
        "n_feedstock": int(df[fc.GROUP_FEEDSTOCK].nunique()),
        "algorithms": list(algos.keys()),
        "catboost_available": "CatBoost" in algos,
        "schemes": list(schemes),
        "results": {},
        "note": (
            "동일 데이터·동일 그룹 분할에서 A/B/C 비교. Model B 가 A 대비 실제 개선되는지 확인. "
            "개선되지 않아도 결과를 그대로 보고한다."
        ),
    }
    for scheme in schemes:
        out["results"][scheme] = {}
        for m in models:
            out["results"][scheme][m] = {}
            for aname, afac in algos.items():
                res = evaluate_model(df, m, aname, afac, scheme,
                                     include_raw_composition=include_raw_composition)
                out["results"][scheme][m][aname] = res.to_dict()
    return out
