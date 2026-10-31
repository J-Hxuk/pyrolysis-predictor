"""
v15_recommend.py — Chen 2022 16개 제조조건 추천 검증 (09_...Chen16.csv).

지시·계약 요지:
  - Chen16 = 옥수숫대(corn stover), Cd(II), 고정 C0=100/dose=10/T=25/t=1440.
    varying: pyrolysis temperature(400~700), hold_min, heating_rate. 타깃: reported_removal_pct(%).
  - 타깃은 '제거율(%)'이며 qe 와 혼합하지 않는다. 제거율에서 유도한 qe 를 독립값으로 추가하지 않는다.
  - 초기 pH 원문 미확인 → 임의로 채우지 않는다(입력 제외).
  - **조건 홀드아웃** 평가를 수행한다. 16개 전체로 학습한 뒤 같은 16개 순위를 맞혔다고
    독립 검증으로 주장하지 않는다.
  - 16개는 직교배열(orthogonal array)이며 완전요인 64조합이 아니다.
  - **동일 원료·동일 논문 내 조건 보간 검증**(여기서 수행)과 **새 원료 일반화 검증**을 구분한다.
    Chen16 은 in-paper 보간 검증 범위다.

지표: 제거율 예측 MAE(%p), Spearman 순위상관, Top-1/Top-3 적중, 최적 관측 대비 regret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = [
    "CHEN_TARGET",
    "CHEN_FIXED_NOTE",
    "chen_inputs",
    "recommend_benchmark",
    "recommend_single_fit_holdout",
    "RecommendationResult",
]

CHEN_TARGET = "reported_removal_pct"
# Chen16 에서 varying 하는 제조조건만 입력으로 쓴다(고정 흡착조건·pH 제외).
CHEN_INPUT_COLS = ["temperature_C", "hold_min", "heating_rate_C_min"]
CHEN_FIXED_NOTE = (
    "고정: C0=100 mg/L, dose=10 g/L, T=25°C, t=1440 min, 원료=corn stover, 오염물질=Cd(II). "
    "초기 pH 원문 미확인으로 입력에 넣지 않는다. 16개는 직교배열(완전요인 64 아님)."
)


def chen_inputs(df: pd.DataFrame) -> list[str]:
    return [c for c in CHEN_INPUT_COLS if c in df.columns]


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        return None if np.isnan(rho) else float(rho)
    except Exception:
        # 순위 pearson fallback
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if np.std(ra) == 0 or np.std(rb) == 0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class RecommendationResult:
    scheme: str
    algo: str
    n_conditions: int
    removal_mae_pp: float | None
    spearman: float | None
    top1_hit: bool | None
    top3_overlap: int | None
    regret_pp: float | None
    note: str = ""
    per_condition: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme, "algo": self.algo, "n_conditions": self.n_conditions,
            "removal_mae_pp": self.removal_mae_pp, "spearman": self.spearman,
            "top1_hit": self.top1_hit, "top3_overlap_of_3": self.top3_overlap,
            "regret_pp": self.regret_pp, "note": self.note,
        }


def _fit_predict_loo(df: pd.DataFrame, input_cols: list[str], algo_factory) -> np.ndarray:
    """조건 Leave-One-Out: 각 조건을 하나씩 빼고 나머지로 학습해 그 조건을 예측(OOF).

    전처리는 학습 fold 로만 fit. 16개 전체 학습 후 자기 순위를 맞히는 방식이 아니다.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = pd.to_numeric(df[CHEN_TARGET], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    oof = np.full(n, np.nan)
    X = df[input_cols].apply(pd.to_numeric, errors="coerce")
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", algo_factory()),
        ])
        pipe.fit(X.iloc[tr], y[tr])
        oof[i] = float(pipe.predict(X.iloc[[i]])[0])
    return oof


def recommend_benchmark(
    chen: pd.DataFrame, *, algos: dict[str, Callable[[], Any]] | None = None, top_k: int = 3,
) -> dict[str, Any]:
    """Chen16 조건 홀드아웃(Leave-One-Condition-Out) 추천 검증.

    각 알고리즘에 대해 OOF 제거율 예측 → MAE(%p), Spearman, Top-1 적중, Top-3 겹침, regret.
    """
    if algos is None:
        from pyrolysis.biochar_stage2.v15_models import available_algorithms
        algos = available_algorithms()

    input_cols = chen_inputs(chen)
    y = pd.to_numeric(chen[CHEN_TARGET], errors="coerce").to_numpy(dtype=float)
    n = len(chen)

    out: dict[str, Any] = {
        "benchmark": "Chen2022 corn stover Cd(II) 16 conditions",
        "validation_type": "IN_PAPER_CONDITION_INTERPOLATION (not new-feedstock generalization)",
        "diagnostic_type": "CROSS_FOLD_POOLED_DIAGNOSTIC (not single-fit ranking)",
        "n_conditions": n,
        "input_cols": input_cols,
        "fixed_conditions_note": CHEN_FIXED_NOTE,
        "orthogonal_array_note": "16개는 직교배열이며 완전요인 64조합이 아니다.",
        "ph_note": "초기 pH 원문 미확인 → 입력에서 제외(임의 보충 금지).",
        "double_count_note": "제거율에서 유도한 qe 를 독립 실험값으로 추가하지 않는다.",
        "pooled_loo_caveat": (
            "LOO OOF 예측은 조건마다 서로 다른 fit model 의 출력을 모은 pooled diagnostic 이다. "
            "하나의 학습된 모델로 16개를 동시에 추천하는 상황의 평가가 아니다. "
            "단일 모델 추천 성능은 recommend_single_fit_holdout 의 반복 12/4 holdout 으로 평가한다."),
        "dummy_anti_correlation_note": (
            "평균예측 Dummy 를 LOO 로 적용하면 yhat_i=(Σy−y_i)/(n−1) 이므로 비상수 y 에서 "
            "Spearman(y,yhat)=−1 이 수학적으로 강제된다. Dummy 의 Spearman=−1·top3=0 은 "
            "구조적 산물이며 학습 모델 대비 순위성능의 타당한 비교가 아니다."),
        "results": {},
    }

    best_obs_idx = int(np.argmax(y))
    best_obs_val = float(y[best_obs_idx])
    obs_rank = pd.Series(-y).rank(method="min").to_numpy()  # 1 = 최고 제거율

    for aname, afac in algos.items():
        oof = _fit_predict_loo(chen, input_cols, afac)
        valid = ~np.isnan(oof) & ~np.isnan(y)
        if valid.sum() < 3:
            continue
        mae = float(np.mean(np.abs(oof[valid] - y[valid])))
        rho = _spearman(y[valid], oof[valid])

        # Top-1: 예측 최고 조건이 실제 최고 조건과 같은가
        pred_best_idx = int(np.argmax(np.where(valid, oof, -np.inf)))
        top1 = bool(pred_best_idx == best_obs_idx)

        # Top-3 겹침: 예측 상위 3 ∩ 실측 상위 3
        pred_top3 = set(np.argsort(-np.where(valid, oof, -np.inf))[:top_k].tolist())
        obs_top3 = set(np.argsort(-y)[:top_k].tolist())
        top3_overlap = int(len(pred_top3 & obs_top3))

        # regret: 모델이 추천한(예측 최고) 조건의 실제 제거율과 실측 최고의 차
        regret = float(best_obs_val - y[pred_best_idx])

        per_cond = []
        for i in range(n):
            if not valid[i]:
                continue
            per_cond.append({
                "material_uid": str(chen.iloc[i].get("material_uid", i)),
                "temperature_C": float(chen.iloc[i].get("temperature_C", np.nan)),
                "hold_min": float(chen.iloc[i].get("hold_min", np.nan)),
                "heating_rate_C_min": float(chen.iloc[i].get("heating_rate_C_min", np.nan)),
                "observed_removal_pct": float(y[i]),
                "predicted_removal_pct": round(float(oof[i]), 3),
                "observed_rank": int(obs_rank[i]),
            })

        res = RecommendationResult(
            scheme="leave_one_condition_out", algo=aname, n_conditions=int(valid.sum()),
            removal_mae_pp=round(mae, 3), spearman=(round(rho, 4) if rho is not None else None),
            top1_hit=top1, top3_overlap=top3_overlap, regret_pp=round(regret, 3),
            note=(
                "조건 홀드아웃(LOO) OOF. 16개 전체 학습 후 자기 순위 맞춤이 아님. "
                "in-paper 조건 보간 범위이며 새 원료 일반화 검증이 아니다."
            ),
            per_condition=per_cond,
        )
        d = res.to_dict()
        d["per_condition"] = per_cond
        out["results"][aname] = d

    out["best_observed"] = {
        "material_uid": str(chen.iloc[best_obs_idx].get("material_uid", best_obs_idx)),
        "temperature_C": float(chen.iloc[best_obs_idx].get("temperature_C", np.nan)),
        "hold_min": float(chen.iloc[best_obs_idx].get("hold_min", np.nan)),
        "heating_rate_C_min": float(chen.iloc[best_obs_idx].get("heating_rate_C_min", np.nan)),
        "removal_pct": best_obs_val,
    }
    return out


def _pairwise_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """동일 test set 내 모든 쌍 (i,j) 에 대해 예측 순서가 실제 순서와 일치하는 비율.

    실제값 동률 쌍은 분모에서 제외한다. 예측값 동률은 '불일치'(0.5 도 아님)로 보수적으로 센다.
    평가 가능한 쌍이 없으면 None.
    """
    n = len(y_true)
    conc = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue  # 실제 동률 쌍 제외
            total += 1
            # 예측이 동률이면 순서 판정 불가 → 불일치로 처리(보수적)
            if y_pred[i] == y_pred[j]:
                continue
            same_order = (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j])
            if same_order:
                conc += 1
    if total == 0:
        return None
    return conc / total


def recommend_single_fit_holdout(
    chen: pd.DataFrame, *, algos: dict[str, Callable[[], Any]] | None = None,
    n_test: int = 4, n_repeats: int = 50, base_seed: int = 42, top_k: int = 3,
) -> dict[str, Any]:
    """단일 fit 반복 holdout 추천 검증 (Chen16, train 12 / test 4 기본).

    각 반복에서 고정 시드로 train/test 를 나누고, **하나의 모델**을 train 으로 학습한 뒤 test 의
    여러 조건을 '동시에' 예측·순위화한다. pooled LOO 와 달리 '단일 모델이 미관측 조건을
    동시에 추천'하는 상황에 맞는 평가다.

    지표(반복 평균): pairwise ranking accuracy, held-out top-1 hit, Top-3 overlap, regret(%p),
    제거율 MAE(%p). Dummy 등 상수 예측의 ranking/top-1 은 동률 tie-break 기대값으로 처리한다.

    주의: 12/4 holdout 은 '새 원료 일반화' 가 아니라 동일 논문·동일 설계 내 미관측 조건의
    제한적 순위 예측 검증이다. 반복 split 간 test 조건은 중복될 수 있어 상호 의존적이다.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import ShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if algos is None:
        from pyrolysis.biochar_stage2.v15_models import available_algorithms
        algos = available_algorithms()

    input_cols = chen_inputs(chen)
    y = pd.to_numeric(chen[CHEN_TARGET], errors="coerce").to_numpy(dtype=float)
    X = chen[input_cols].apply(pd.to_numeric, errors="coerce")
    n = len(chen)
    n_test = min(max(1, n_test), n - 1)

    ss = ShuffleSplit(n_splits=n_repeats, test_size=n_test, random_state=base_seed)
    splits = list(ss.split(np.arange(n)))
    # 조건별 test 출현 횟수 기록(반복 split 간 의존성 가시화)
    test_counts = np.zeros(n, dtype=int)
    for _, te in splits:
        test_counts[te] += 1

    out: dict[str, Any] = {
        "scheme": "single_fit_repeated_holdout",
        "n_train": n - n_test, "n_test": n_test, "n_repeats": n_repeats,
        "base_seed": base_seed, "input_cols": input_cols,
        "test_appearance_counts": test_counts.tolist(),
        "generalization_caveat": (
            "동일 논문·동일 설계 내 미관측 조건의 제한적 순위 검증. 새 원료/64 전체조합 일반화 아님. "
            "반복 split 의 test 조건은 중복될 수 있어 지표는 상호 의존적이다."),
        "tie_handling": (
            "pairwise: 실제 동률 쌍 제외, 예측 동률 쌍은 불일치로 보수적 처리. "
            "top-1: 예측 최댓값 동률이면 균등 tie-break 기대 적중확률(1/동률수)로 집계."),
        "primary_metric": "pairwise_ranking_accuracy (Top-3 은 test 4개 중 3개라 우연적중 높아 보조지표)",
        "results": {},
    }

    rng = np.random.RandomState(base_seed)
    for aname, afac in algos.items():
        pw_list, top1_list, top3_list, regret_list, mae_list = [], [], [], [], []
        for tr, te in splits:
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            if np.isnan(yte).any() or np.isnan(ytr).any():
                continue
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", afac()),
            ])
            pipe.fit(Xtr, ytr)
            pred = np.asarray(pipe.predict(Xte), dtype=float)

            mae_list.append(float(np.mean(np.abs(pred - yte))))
            pw = _pairwise_ranking_accuracy(yte, pred)
            if pw is not None:
                pw_list.append(pw)

            # top-1: 예측 최댓값(동률이면 균등 tie-break 기대 적중확률)
            best_obs_local = int(np.argmax(yte))
            pmax = pred.max()
            tied = np.where(pred >= pmax - 1e-12)[0]
            top1_list.append(float(1.0 / len(tied)) if best_obs_local in tied else 0.0)

            # Top-3 overlap (test 내 상위 min(top_k, n_test))
            k = min(top_k, len(te))
            pred_top = set(np.argsort(-pred)[:k].tolist())
            obs_top = set(np.argsort(-yte)[:k].tolist())
            top3_list.append(len(pred_top & obs_top))

            # regret: 추천(예측 최고) 조건의 실제값과 test 내 실제 최고값의 차
            rec_idx = int(tied[rng.randint(len(tied))]) if len(tied) > 1 else int(tied[0])
            regret_list.append(float(yte[best_obs_local] - yte[rec_idx]))

        def _mean(xs):
            return round(float(np.mean(xs)), 4) if xs else None

        out["results"][aname] = {
            "pairwise_ranking_accuracy": _mean(pw_list),
            "top1_hit_rate": _mean(top1_list),
            "top3_overlap_mean_of_%d" % min(top_k, n_test): _mean(top3_list),
            "regret_pp_mean": _mean(regret_list),
            "removal_mae_pp_mean": _mean(mae_list),
            "n_evaluated_splits": len(mae_list),
        }
    return out
