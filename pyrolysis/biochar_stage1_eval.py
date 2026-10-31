"""
biochar_stage1_eval.py — Stage1 탐색적 평가 (LOGO, 새 원료·새 논문 일반화 검증).

이 평가는 **'탐색적 평가 — 새 원료·새 논문에 대한 일반화 검증'** 이다.
Stage2 성능이나 전체 연계 성능이 아니다. 최종 배포 모델 선정이 아니다.

지시 준수:
  - 원본 후보 파일의 승인 플래그(eligible_for_training)를 보존한다. 읽기만 한다.
  - 분위기와 장치 정보를 구분한다. hypoxia·oxygen-limited 는 산소제한으로 정규화하되
    원문 표기를 유지한다. muffle 만으로 분위기를 추정하지 않는다.
  - 최소/확장 구성을 비교한다. 각 구성의 전체 적격 데이터 결과와, 확장 구성의 동일 시료
    에서 두 구성을 비교한 결과를 구분한다.
  - 각 구성에서 실제 남은 그룹으로 LOGO 를 다시 만든다 (전체 후보의 fold 수를 재사용 안 함).
  - 평균 예측 기준모델·RandomForest·XGBoost 를 고정된 보수적 설정으로 비교한다.
    테스트 fold 로 설정을 고르지 않는다.
  - 전처리는 각 학습 fold 안에서만 fit 한다. 전체 OOF 예측의 MAE·RMSE·R² 와 논문별 오차
    를 저장한다. 작은 테스트 그룹에서 R² 가 정의되지 않으면 NA 로 둔다.

Streamlit 을 import 하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyrolysis.biochar_data import (
    MIN_CONFIG_INPUTS,
    EXTENDED_CONFIG_INPUTS,
    load_stage1_candidates,
    normalize_modification,
    resolve_biochar_root,
)

__all__ = [
    "normalize_atmosphere",
    "prepare_stage1_frame",
    "eligible_subset",
    "exclusion_table",
    "fixed_models",
    "logo_oof_evaluate",
    "run_stage1_exploratory_evaluation",
    "save_evaluation_manifests",
]


# ── 분위기·장치 정규화 (원문 표기 보존) ─────────────────────────────────────

# 원문 atmosphere -> (정규화 분위기, 장치, 원문표기 유지)
# muffle 만으로 분위기를 추정하지 않는다: 'oxygen-limited muffle furnace' 는 원문에
# 'oxygen-limited' 가 명시돼 있으므로 산소제한으로 본다. 장치는 muffle 로 분리.
def normalize_atmosphere(value: Any) -> dict[str, Any]:
    """atmosphere 원문값을 (분위기 정규화, 장치, 원문표기)로 분해한다.

    Returns: {"atm_norm": str, "device": str|None, "atm_raw": str|None}
      atm_norm: "inert_N2" | "oxygen_limited" | "unreported"
      device:   "muffle_furnace" | None (명시된 장치만)
    """
    if value is None:
        return {"atm_norm": "unreported", "device": None, "atm_raw": None}
    try:
        if pd.isna(value):
            return {"atm_norm": "unreported", "device": None, "atm_raw": None}
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    low = raw.lower()
    if raw == "":
        return {"atm_norm": "unreported", "device": None, "atm_raw": None}

    device = "muffle_furnace" if "muffle" in low else None

    if low in ("n2", "n₂", "nitrogen"):
        atm = "inert_N2"
    elif "oxygen-limited" in low or "oxygen limited" in low or "hypoxia" in low:
        atm = "oxygen_limited"
    else:
        # 미지 표기는 추정하지 않고 미보고로 둔다 (값 생성 금지).
        atm = "unreported"
    return {"atm_norm": atm, "device": device, "atm_raw": raw}


# ── 평가용 프레임 준비 (승인 플래그 보존, 읽기 전용 파생) ────────────────────


def prepare_stage1_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """후보 DataFrame 에서 평가용 파생 열을 만든다. 원본 열·플래그를 보존한다.

    - modification_method -> mod_class (none/modified/unreported)
    - atmosphere -> atm_norm / device / atm_raw (원문 표기 유지)
    - 수치 입력을 수치형으로 강제 변환(비수치는 NaN).
    - target_col 수치 변환.
    원본 eligible_for_training 등은 그대로 둔다 (변경·삭제 없음).
    """
    out = df.copy()
    out["mod_class"] = out["modification_method"].map(normalize_modification)
    atm = out["atmosphere"].map(normalize_atmosphere)
    out["atm_norm"] = atm.map(lambda d: d["atm_norm"])
    out["device"] = atm.map(lambda d: d["device"])
    out["atm_raw"] = atm.map(lambda d: d["atm_raw"])
    for c in ("pyrolysis_temp_C", "holding_time_min", "heating_rate_C_min"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["_target"] = pd.to_numeric(out[target_col], errors="coerce")
    return out


def exclusion_table(frame: pd.DataFrame, config: str) -> pd.DataFrame:
    """각 행이 구성에서 제외되는지와 제외 사유를 표로 만든다 (승인 플래그 보존).

    제외는 '삭제'가 아니라 해당 구성에서 사용 불가 표시일 뿐이다. 원본 플래그는 그대로.
    Returns: 시료별 included(bool)·exclusion_reasons(세미콜론 구분) 표.
    """
    feats = _config_features(config)
    rows = []
    for idx, r in frame.iterrows():
        reasons = []
        if pd.isna(r["_target"]):
            reasons.append("target_missing")
        for c in feats["numeric"]:
            if pd.isna(r[c]):
                reasons.append(f"{c}_missing")
        for c in feats["categorical"]:
            if c == "mod_class":
                if r[c] not in ("none", "modified"):
                    reasons.append("modification_unreported")
            elif c == "atm_norm":
                if r[c] not in ("inert_N2", "oxygen_limited"):
                    reasons.append("atmosphere_unreported")
            else:
                if pd.isna(r[c]):
                    reasons.append(f"{c}_missing")
        rows.append({
            "config": config,
            "canonical_material_id": r.get("canonical_material_id"),
            "sample_id": r.get("sample_id"),
            "leakage_group": r.get("leakage_group"),
            "feedstock": r.get("feedstock"),
            "eligible_for_training_original": r.get("eligible_for_training"),
            "included": len(reasons) == 0,
            "exclusion_reasons": ";".join(reasons),
        })
    return pd.DataFrame(rows)


# 평가 구성별 사용할 feature 목록 (정규화 열 사용)
_MIN_FEATURES = {
    "categorical": ["feedstock", "mod_class"],
    "numeric": ["pyrolysis_temp_C"],
}
_EXT_FEATURES = {
    "categorical": ["feedstock", "mod_class", "atm_norm"],
    "numeric": ["pyrolysis_temp_C", "holding_time_min", "heating_rate_C_min"],
}


def _config_features(config: str) -> dict[str, list[str]]:
    return _MIN_FEATURES if config == "minimal" else _EXT_FEATURES


def eligible_subset(frame: pd.DataFrame, config: str) -> pd.DataFrame:
    """구성의 필수 feature 가 사용 가능한 행만 남긴다 (집계상 제외, 원본 미변경).

    - 범주: mod_class 는 none/modified 만 사용 가능(미보고 제외). feedstock/atm_norm 은
      값이 있어야 함(미보고 제외).
    - 수치: 해당 수치 feature 가 모두 수치여야 함.
    - target 이 수치여야 함.
    """
    feats = _config_features(config)
    mask = frame["_target"].notna()
    for c in feats["numeric"]:
        mask &= frame[c].notna()
    for c in feats["categorical"]:
        if c == "mod_class":
            mask &= frame[c].isin(["none", "modified"])
        elif c == "atm_norm":
            mask &= frame[c].isin(["inert_N2", "oxygen_limited"])
        else:
            mask &= frame[c].notna()
    return frame[mask].copy()


# ── 고정 보수적 모델 (테스트 fold 로 튜닝하지 않음) ──────────────────────────


def fixed_models(random_state: int = 20261023) -> dict[str, Any]:
    """고정된 보수적 설정의 모델 3종을 반환한다.

    - mean_baseline: 평균 예측 (DummyRegressor).
    - random_forest: 소규모 데이터용 보수적 RF.
    - xgboost: 보수적 설정. 미설치면 생략한다.
    하이퍼파라미터는 고정값이며 테스트 fold 로 선택하지 않는다.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor

    models: dict[str, Any] = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=3,
            max_features="sqrt", random_state=random_state, n_jobs=1,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_weight=3, random_state=random_state, n_jobs=1,
            tree_method="hist", verbosity=0,
        )
    except Exception:
        pass
    return models


def _build_pipeline(model, feats: dict[str, list[str]]):
    """전처리+모델 Pipeline. 전처리는 fit 시점(=학습 fold)에서만 학습된다.

    - 수치: 중앙값 대치 + 표준화 (fold 내 통계).
    - 범주: 최빈값 대치 + one-hot(handle_unknown='ignore' 로 미학습 범주 허용).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, feats["numeric"]),
        ("cat", cat_pipe, feats["categorical"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


# ── LOGO OOF 평가 ────────────────────────────────────────────────────────────


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R². 분산이 0(샘플 1개 또는 상수)이면 정의 불가 -> None(NA)."""
    if len(y_true) < 2:
        return None
    denom = float(((y_true - y_true.mean()) ** 2).sum())
    if denom == 0.0:
        return None
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / denom


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = _safe_r2(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": (None if r2 is None else float(r2)), "n": int(len(y_true))}


@dataclass
class LOGOResult:
    config: str
    which: str
    model: str
    n_samples: int
    n_groups: int
    group_col: str
    overall: dict[str, Any]
    per_group: dict[str, dict[str, Any]]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config, "which": self.which, "model": self.model,
            "n_samples": self.n_samples, "n_groups": self.n_groups,
            "group_col": self.group_col, "overall": self.overall,
            "per_group": self.per_group, "note": self.note,
        }


def logo_oof_evaluate(
    data: pd.DataFrame, config: str, which: str, model_name: str, model,
    *, group_col: str = "leakage_group",
) -> LOGOResult:
    """실제 남은 그룹으로 LOGO 를 만들어 OOF 예측을 모으고 지표를 낸다.

    각 fold: 한 그룹을 테스트로 빼고 나머지로 Pipeline 을 fit(전처리도 fold 내에서만).
    OOF 예측을 모두 모아 전체 MAE/RMSE/R² 와 논문(그룹)별 오차를 계산한다.
    """
    feats = _config_features(config)
    all_cols = feats["numeric"] + feats["categorical"]
    groups = list(pd.unique(data[group_col]))
    oof_true: list[float] = []
    oof_pred: list[float] = []
    oof_group: list[str] = []

    for g in groups:
        te = data[data[group_col] == g]
        tr = data[data[group_col] != g]
        if len(tr) == 0 or len(te) == 0:
            continue
        pipe = _build_pipeline(_clone(model), feats)
        pipe.fit(tr[all_cols], tr["_target"].to_numpy())
        pred = pipe.predict(te[all_cols])
        oof_true.extend(te["_target"].tolist())
        oof_pred.extend([float(v) for v in pred])
        oof_group.extend([str(g)] * len(te))

    yt = np.asarray(oof_true, dtype=float)
    yp = np.asarray(oof_pred, dtype=float)
    overall = _metrics(yt, yp) if len(yt) else {"mae": None, "rmse": None, "r2": None, "n": 0}

    per_group: dict[str, dict[str, Any]] = {}
    gg = np.asarray(oof_group)
    for g in pd.unique(gg) if len(gg) else []:
        m = gg == g
        per_group[str(g)] = _metrics(yt[m], yp[m])

    return LOGOResult(
        config=config, which=which, model=model_name,
        n_samples=len(yt), n_groups=len(set(oof_group)),
        group_col=group_col, overall=overall, per_group=per_group,
        note="탐색적 평가 — 새 원료·새 논문 일반화 검증. Stage2·전체 연계 성능 아님.",
    )


def _clone(model):
    from sklearn.base import clone
    return clone(model)


# ── 전체 실행 ────────────────────────────────────────────────────────────────


def run_stage1_exploratory_evaluation(base: Path | None = None) -> dict[str, Any]:
    """BET·Pore 각각, 최소/확장 구성, 3개 모델로 LOGO 탐색적 평가를 수행한다.

    두 가지 비교를 구분해 저장한다:
      A) 각 구성의 '전체 적격 데이터' 결과 (구성마다 사용 가능한 시료가 다름).
      B) '확장 구성의 동일 시료'에서 최소 vs 확장 구성 비교 (시료를 통제한 비교).

    승인 플래그를 바꾸지 않고 최종 모델을 선정하지 않는다.
    """
    paths = resolve_biochar_root(base)
    models = fixed_models()

    report: dict[str, Any] = {
        "evaluation_type": "탐색적 평가 — 새 원료·새 논문에 대한 일반화 검증",
        "disclaimer": (
            "LOGO(leakage_group)로 새 논문·새 원료 일반화를 본다. "
            "Stage2 흡착 성능이나 전체 연계 성능이 아니다. 최종 배포 모델 선정이 아니다. "
            "feedstock 범주가 논문 종속이라 held-out 그룹의 원료는 대부분 미학습 범주다."
        ),
        "models": list(models.keys()),
        "fixed_settings_note": "하이퍼파라미터는 고정 보수적 설정. 테스트 fold 로 선택하지 않음.",
        "flags_preserved": "원본 후보 파일의 eligible_for_training 을 변경하지 않았다.",
        "results": {},
    }

    for which, target_col in (("bet", "BET_m2_g"), ("pore", "pore_volume_cm3_g")):
        df = load_stage1_candidates(paths, which)
        frame = prepare_stage1_frame(df, target_col)

        sub_min = eligible_subset(frame, "minimal")
        sub_ext = eligible_subset(frame, "extended")

        block: dict[str, Any] = {
            "target_col": target_col,
            "n_eligible_minimal": int(len(sub_min)),
            "n_eligible_extended": int(len(sub_ext)),
            "groups_minimal": sorted(set(sub_min["leakage_group"].astype(str))),
            "groups_extended": sorted(set(sub_ext["leakage_group"].astype(str))),
            # A) 각 구성 전체 적격 데이터
            "A_full_eligible": {"minimal": {}, "extended": {}},
            # B) 확장 구성의 동일 시료에서 최소 vs 확장 비교
            "B_same_samples_extended_subset": {"minimal": {}, "extended": {}},
        }

        # A) 전체 적격 데이터로 각 구성 평가 (실제 남은 그룹으로 LOGO 재생성)
        for cfg, sub in (("minimal", sub_min), ("extended", sub_ext)):
            for mname, model in models.items():
                res = logo_oof_evaluate(sub, cfg, which, mname, model)
                block["A_full_eligible"][cfg][mname] = res.to_dict()

        # B) 동일 시료 통제 비교: 확장 구성 적격 시료(sub_ext)에서 두 구성 평가
        #    같은 행·같은 LOGO 그룹에서 feature 구성만 바꿔 공정 비교.
        for cfg in ("minimal", "extended"):
            for mname, model in models.items():
                res = logo_oof_evaluate(sub_ext, cfg, which, mname, model)
                block["B_same_samples_extended_subset"][cfg][mname] = res.to_dict()

        report["results"][which] = block

    return report


def save_evaluation_manifests(out_dir: Path, base: Path | None = None) -> dict[str, str]:
    """평가용 데이터 목록과 제외 사유를 별도 CSV 로 저장한다 (지시 1).

    원본 후보 파일을 수정하지 않는다. 승인 플래그(eligible_for_training)는 '원본 값'을
    그대로 열에 담아 보존한다. 평가 포함/제외는 구성별 별도 표로 저장한다.

    Returns: {name: path} 저장한 파일 경로.
    """
    paths = resolve_biochar_root(base)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for which, target_col in (("bet", "BET_m2_g"), ("pore", "pore_volume_cm3_g")):
        df = load_stage1_candidates(paths, which)
        frame = prepare_stage1_frame(df, target_col)
        for config in ("minimal", "extended"):
            tbl = exclusion_table(frame, config)
            fname = f"eval_manifest_{which}_{config}.csv"
            p = out_dir / fname
            tbl.to_csv(p, index=False, encoding="utf-8-sig")
            saved[f"{which}_{config}"] = str(p)
    return saved
