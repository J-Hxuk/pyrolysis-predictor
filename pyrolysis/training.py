"""
training.py - demo 학습 실행 (작업 3).

실제 연구 설정이 없을 때 합성 자료로 학습-평가 경로가 연결되는지 확인한다.
이 파일에서 생성하는 결과는 '시험용 합성 자료'임을 명시한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from pyrolysis.evaluation import compute_regression_metrics
from pyrolysis.preprocessing import build_regression_pipeline

__all__ = [
    "make_demo_data",
    "run_demo_training",
    "SelectionConfig",
    "SelectionError",
    "CandidateScore",
    "SelectionResult",
    "select_candidate",
    "config_hash",
]

_DEMO_DISCLAIMER = (
    "시험용 합성 자료 - 실제 연구 성능 아님. "
    "이 결과를 실제 촉매 성능 예측의 검증 결과로 사용하지 않습니다."
)


def make_demo_data(
    input_specs: list[Any],
    target_specs: list[Any],
    n_samples: int = 100,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """합성 회귀 데이터를 생성한다.

    sklearn.datasets.make_regression 으로 수치형 열을 만들고,
    범주형 열은 allowed_categories 에서 무작위로 채운다.
    합성 변수는 실제 열분해 지표처럼 표현하지 않는다.

    Args:
        input_specs: InputSpec 목록.
        target_specs: TargetSpec 목록 (첫 번째 regression 대상만 사용).
        n_samples: 생성할 행 수.
        random_seed: 재현성을 위한 난수 시드.

    Returns:
        (X DataFrame, y Series) 튜플.
    """
    rng = np.random.default_rng(random_seed)

    # 수치형·범주형 분리
    numeric_specs = [s for s in input_specs if s.dtype in ("float", "int")]
    cat_specs = [s for s in input_specs if s.dtype in ("category", "bool")]

    n_numeric = len(numeric_specs)
    if n_numeric == 0:
        n_numeric = 1  # 최소 1개

    X_numeric, y_raw = make_regression(
        n_samples=n_samples,
        n_features=n_numeric,
        noise=10.0,
        random_state=random_seed,
    )

    # 수치형 열 DataFrame 구성
    col_names = [s.id for s in numeric_specs] if numeric_specs else ["synthetic_num_0"]
    X = pd.DataFrame(X_numeric, columns=col_names)

    # 범주형 열 추가
    for spec in cat_specs:
        choices = [c.value for c in spec.allowed_categories]
        X[spec.id] = rng.choice(choices, size=n_samples)

    # 대상 이름 결정
    regression_targets = [t for t in target_specs if t.task == "regression"]
    y_name = regression_targets[0].id if regression_targets else "synthetic_y"
    y = pd.Series(y_raw, name=y_name)

    return X, y


def run_demo_training(
    demo_spec: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """합성 설정으로 LinearRegression을 학습·평가하고 기록을 저장한다.

    학습·평가 자료를 명시적으로 분리하고, 전처리는 학습 자료에만 fit한다.
    평가 행이 학습에 포함되지 않음을 train_test_split 의 random_state 로 보장한다.

    Args:
        demo_spec: ProblemSpec (configs/demo.yaml 로드 결과).
        output_dir: 실행 기록을 저장할 상위 디렉터리 (예: runs/).

    Returns:
        실행 기록 딕셔너리 (run_id, 파일 경로, MAE, RMSE, 행 수 포함).
    """
    # ── 합성 데이터 생성 ──────────────────────────────────────────────────────
    X, y = make_demo_data(
        input_specs=demo_spec.inputs,
        target_specs=demo_spec.targets,
        n_samples=100,
        random_seed=42,
    )

    # ── 학습·평가 분리 (train_test_split, random_state=42) ────────────────────
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    train_indices = list(X_train.index)
    eval_indices = list(X_eval.index)

    # 평가 행이 학습 행과 겹치지 않는지 확인
    assert len(set(train_indices) & set(eval_indices)) == 0, (
        "평가 행이 학습 행과 겹쳐서는 안 됩니다."
    )

    # ── Pipeline 구성·학습 (전처리는 학습 자료에만 fit) ──────────────────────
    model = LinearRegression()
    pipeline = build_regression_pipeline(demo_spec.inputs, model)
    pipeline.fit(X_train, y_train)  # fit: 학습 자료만 사용

    # ── 평가 (transform만 적용) ───────────────────────────────────────────────
    y_pred = pipeline.predict(X_eval)

    target_spec = demo_spec.targets[0]
    metrics = compute_regression_metrics(
        y_true=y_eval,
        y_pred=y_pred,
        unit=target_spec.unit,
    )

    # ── 설정 해시 ─────────────────────────────────────────────────────────────
    spec_str = json.dumps(demo_spec.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
    config_hash = hashlib.sha256(spec_str.encode()).hexdigest()[:16]

    # ── 실행 기록 구성 ────────────────────────────────────────────────────────
    run_id = f"demo_{uuid.uuid4().hex[:8]}"
    record: dict[str, Any] = {
        "run_id": run_id,
        "disclaimer": _DEMO_DISCLAIMER,
        "model": "LinearRegression",
        "config_hash": config_hash,
        "random_seed": 42,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "target_id": target_spec.id,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "unit": metrics["unit"],
        "note": metrics["note"],
    }

    # ── 파일 저장 ─────────────────────────────────────────────────────────────
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "demo_run.json"
    with record_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    record["run_dir"] = str(run_dir)
    record["record_path"] = str(record_path)
    record["pipeline"] = pipeline  # 호출자가 저장 작업에 사용할 수 있도록 반환

    return record


# ── 대상별 내부 비교·모델 선정 (작업 13) ─────────────────────────────────────
#
# 설계 6.4: 임계값 조정·모델 선택·목표 변환은 학습·내부 평가 자료에서만 수행한다.
# 선정 지표·방향·동점 처리 기준이 설정되지 않으면 모델 선택을 시작하지 않는다.
# 외부 평가 결과로 모델·임계값을 재선정하지 않는다.

import hashlib as _hashlib
import json as _json
from dataclasses import dataclass as _dataclass, field as _field


class SelectionError(Exception):
    """모델 선정 설정 오류 (지표·방향·동점 규칙 누락 등)."""


@_dataclass
class SelectionConfig:
    """후보 선정 설정. 데이터 확인 전 기본값으로 확정하지 않는다.

    metric:    선정 지표 이름 (예: "mae", "f1").
    direction: "minimize" 또는 "maximize".
    tie_breaker: 동점 시 우선할 후보 이름 순서. 비면 사전순으로 결정적 처리.
    random_seed: 내부 분할·학습 난수.
    """

    metric: str
    direction: str
    tie_breaker: list = _field(default_factory=list)
    random_seed: int | None = None

    def validate(self) -> None:
        if not self.metric:
            raise SelectionError("선정 지표(metric)가 설정되지 않았습니다.")
        if self.direction not in ("minimize", "maximize"):
            raise SelectionError(
                "선정 방향(direction)은 'minimize' 또는 'maximize' 여야 합니다: "
                + repr(self.direction)
            )


@_dataclass
class CandidateScore:
    """후보 하나의 내부 평가 점수 또는 실행 실패 사유."""

    name: str
    score: float | None
    ran: bool
    reason: str = ""


@_dataclass
class SelectionResult:
    """대상 하나의 내부 선정 결과."""

    target_id: str
    metric: str
    direction: str
    candidate_scores: list
    selected: str | None
    reason: str = ""


def _better(a: float, b: float, direction: str) -> bool:
    """a 가 b 보다 더 나은 점수인지 (direction 기준)."""
    if direction == "minimize":
        return a < b
    return a > b


def select_candidate(
    target_id: str,
    candidate_scores: list,
    config: SelectionConfig,
) -> SelectionResult:
    """내부 평가 점수에서 최적 후보를 결정론적으로 선정한다.

    - 실행하지 않은(ran=False) 후보에는 점수를 만들지 않는다.
    - 동점은 tie_breaker 순서, 없으면 후보 이름 사전순으로 처리한다 (결정적).
    - 유효 점수가 하나도 없으면 selected=None 과 사유를 남긴다.

    Args:
        target_id: 대상 ID.
        candidate_scores: CandidateScore 목록 (내부 평가 결과).
        config: SelectionConfig (validate 된 상태).

    Returns:
        SelectionResult.
    """
    config.validate()
    scored = [c for c in candidate_scores if c.ran and c.score is not None]
    if not scored:
        return SelectionResult(
            target_id=target_id, metric=config.metric, direction=config.direction,
            candidate_scores=candidate_scores, selected=None,
            reason="유효한 점수를 가진 후보가 없습니다 (실행된 후보 없음).",
        )

    def sort_key(c):
        if config.tie_breaker and c.name in config.tie_breaker:
            tie_rank = config.tie_breaker.index(c.name)
        else:
            tie_rank = len(config.tie_breaker)
        # 1차: 점수(방향 반영), 2차: tie_breaker 순위, 3차: 이름 사전순
        primary = c.score if config.direction == "minimize" else -c.score
        return (primary, tie_rank, c.name)

    best = sorted(scored, key=sort_key)[0]
    return SelectionResult(
        target_id=target_id, metric=config.metric, direction=config.direction,
        candidate_scores=candidate_scores, selected=best.name,
    )


def config_hash(obj) -> str:
    """설정·자료 식별용 결정적 해시(16자)."""
    raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return _hashlib.sha256(raw.encode()).hexdigest()[:16]
