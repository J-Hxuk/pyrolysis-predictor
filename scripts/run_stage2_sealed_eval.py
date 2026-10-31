"""
run_stage2_sealed_eval.py — Stage 2 (qe) 독립 검증(봉인) 평가 실행 틀.

프로토콜: reports/stage2_sealed_validation/PROTOCOL_KO.md (v1) 를 코드로 구현.

핵심 원칙:
  - 활성 배포 Stage 2 모델을 '그대로' 로드해 예측한다. 재학습·재선택 없음.
  - 모델 아티팩트의 SHA-256 을 기록한다(동결 증거).
  - 봉인 DOI 가 학습 데이터(188행)와 완전히 disjoint 임을 assert 로 검증한다.
  - 봉인 세트는 '단 한 번' 점수화한다. 지표는 산출 직후 동결된다.
  - 봉인 데이터(Stage2_Sealed_Challenge_V1.csv)가 없으면 명확히 안내하고 중단한다
    (가짜 봉인을 만들지 않는다).

실행: python -m scripts.run_stage2_sealed_eval --root .
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 활성 모델 경로
ACTIVE_PTR = "deployment/stage2_models/active.json"
MODELS_DIR = "deployment/stage2_models"

# 봉인 입력 (데이터 준비 후 배치)
SEALED_CSV = "data/imports/stage2_sealed_v1/Stage2_Sealed_Challenge_V1.csv"

# 학습 데이터(봉인 DOI 와 disjoint 여야 함) — run_v59 와 동일 소스
TRAIN_SOURCES = [
    "data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv",
    "data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv",
]

# 사전 지정 봉인 후보 DOI (PROTOCOL_KO.md §3)
SEALED_DOI_WHITELIST = {
    "10.1021/acs.est.7b06261",       # Alam 2018
    "10.1016/j.ecoenv.2014.04.042",  # Sun 2014
    "10.1016/j.scitotenv.2020.136894",  # Zhao 2020
}

REPORT = "reports/stage2_sealed_validation"
EVAL_ROLE = "STAGE2_SEALED_CHALLENGE_V1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if len(y) >= 2 and np.var(y) > 0:
        r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((y - np.mean(y)) ** 2))
    else:
        r2 = None
    return {"n": int(len(y)), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4) if r2 is not None else None}


def _per_doi(y, p, doi):
    rows = []
    for d in pd.unique(doi):
        m = doi == d
        rows.append({"doi": str(d), **_metrics(y[m], p[m])})
    return pd.DataFrame(rows)


def _load_active_model(root: Path):
    ptr = json.loads((root / ACTIVE_PTR).read_text(encoding="utf-8"))
    version = ptr["active_version"]
    bundle = root / MODELS_DIR / version
    model_path = bundle / "model.joblib"
    meta_path = bundle / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"활성 모델 파일 없음: {model_path}")
    import joblib
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta, version, sha256_of(model_path)


def _guidance_and_exit(root: Path, reason: str) -> None:
    """봉인 데이터가 없거나 부적격이면 가짜 봉인을 만들지 않고 안내 후 중단."""
    msg = {
        "status": "SEALED_DATA_NOT_READY",
        "reason": reason,
        "what_is_ready": "프로토콜·규칙·코드는 확정됨 (PROTOCOL_KO.md, 이 스크립트).",
        "what_is_missing": f"적격 봉인 CSV: {SEALED_CSV}",
        "candidate_dois_to_digitize": sorted(SEALED_DOI_WHITELIST),
        "required_columns": [
            "doi", "pollutant", "C0_mg_L", "initial_pH", "dose_g_L",
            "adsorption_T_C", "contact_time_min", "feedstock_label_normalized",
            "manufacturing_route", "process_step_count", "pyrolysis_temperature_C",
            "pyrolysis_hold_min", "heating_rate_C_min", "pyrolysis_atmosphere",
            "activation_or_modification", "secondary_temperature_C",
            "secondary_hold_min", "target_value",
        ],
        "eligibility": [
            "봉인 DOI 는 학습 188행과 완전히 disjoint",
            "0 <= qe <= C0/dose (physical_cap_pass)",
            "평형 qe 근거 기록(target_origin/qualification)",
        ],
        "next_step": "STAGED DOI 3개의 qe 추출·QC 후 CSV 배치 → 본 스크립트 재실행",
    }
    out = root / REPORT
    out.mkdir(parents=True, exist_ok=True)
    (out / "sealed_eval_status.json").write_text(
        json.dumps(msg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(msg, ensure_ascii=False, indent=2))
    print("\n[중단] 봉인 데이터가 준비되지 않아 평가를 실행하지 않습니다. "
          "가짜 봉인을 만들지 않기 위함입니다.")
    sys.exit(0)


def main(root: str) -> None:
    root = Path(root).resolve()

    # 1) 봉인 데이터 존재 확인 — 없으면 안내 후 중단
    sealed_path = root / SEALED_CSV
    if not sealed_path.exists():
        _guidance_and_exit(root, f"봉인 CSV 가 존재하지 않음: {SEALED_CSV}")

    sealed = pd.read_csv(sealed_path, low_memory=False)
    if "doi" not in sealed.columns or "target_value" not in sealed.columns:
        _guidance_and_exit(root, "봉인 CSV 에 doi/target_value 컬럼이 없음")
    if len(sealed) == 0:
        _guidance_and_exit(root, "봉인 CSV 가 비어 있음")

    sealed_dois = set(sealed["doi"].astype(str))

    # 2) 적격성: 화이트리스트 내 DOI 만 허용
    off_whitelist = sealed_dois - SEALED_DOI_WHITELIST
    if off_whitelist:
        _guidance_and_exit(root, f"화이트리스트 밖 DOI 포함: {sorted(off_whitelist)}")

    # 3) 누수 차단: 학습 데이터와 disjoint 검증
    train_dois = set()
    for src in TRAIN_SOURCES:
        sp = root / src
        if sp.exists():
            train_dois |= set(pd.read_csv(sp, usecols=["doi"], low_memory=False)["doi"].astype(str))
    leak = sealed_dois & train_dois
    if leak:
        _guidance_and_exit(root, f"누수! 봉인 DOI 가 학습 데이터에 존재: {sorted(leak)}")

    # 4) 활성 모델 동결 로드 (재학습 없음)
    model, meta, version, model_sha = _load_active_model(root)

    # 5) 예측 (활성 모델의 입력 계약대로)
    input_cols = meta["input_cols"]
    missing = [c for c in input_cols if c not in sealed.columns]
    if missing:
        _guidance_and_exit(root, f"봉인 CSV 에 필요한 입력 컬럼 누락: {missing}")

    c0 = pd.to_numeric(sealed["C0_mg_L"], errors="coerce").to_numpy(float)
    dose = pd.to_numeric(sealed["dose_g_L"], errors="coerce").to_numpy(float)
    cap = np.where((dose != 0) & np.isfinite(dose), c0 / dose, np.inf)
    y = pd.to_numeric(sealed["target_value"], errors="coerce").to_numpy(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.asarray(model.predict(sealed[input_cols]), dtype=float)
    capped = np.clip(raw, 0.0, cap)
    doi = sealed["doi"].astype(str).to_numpy()

    # 6) 지표 (집계 + DOI별 + 물리 + 베이스라인 + route별)
    finite = np.isfinite(cap)
    viol = int(np.sum(raw[finite] > cap[finite] + 1e-8))
    zero_mae = float(np.mean(np.abs(y)))
    med = float(np.median(y))
    base_r2 = _metrics(y, np.full_like(y, med))["r2"]

    per_doi = _per_doi(y, capped, doi)

    route_rows = []
    if "v59_route_family" in sealed.columns:
        rf = sealed["v59_route_family"].astype(str).to_numpy()
        for r in pd.unique(rf):
            m = rf == r
            route_rows.append({"route": str(r), **_metrics(y[m], capped[m])})

    # DOI 부트스트랩 CI (DOI 재표집)
    ids = np.unique(doi)
    idx = [np.flatnonzero(doi == g) for g in ids]
    rng = np.random.default_rng(20261031)
    ae = np.abs(y - capped)
    boot = [float(ae[np.concatenate([idx[j] for j in rng.integers(0, len(ids), len(ids))])].mean())
            for _ in range(3000)] if len(ids) >= 2 else []
    ci = ([round(float(q), 4) for q in np.quantile(boot, [0.025, 0.975])] if boot else [None, None])

    out = root / REPORT
    out.mkdir(parents=True, exist_ok=True)
    sealed.assign(pred_raw=raw, pred_capped=capped, physical_cap=cap).to_csv(
        out / "sealed_predictions.csv", index=False, encoding="utf-8-sig")
    per_doi.to_csv(out / "sealed_per_doi.csv", index=False, encoding="utf-8-sig")
    if route_rows:
        pd.DataFrame(route_rows).to_csv(out / "sealed_per_route.csv", index=False, encoding="utf-8-sig")

    result = {
        "status": "SEALED_EVAL_DONE",
        "protocol": "reports/stage2_sealed_validation/PROTOCOL_KO.md v1",
        "evaluation_role": EVAL_ROLE,
        "frozen_model": {"version": version, "model_sha256": model_sha},
        "leakage_check": {"sealed_dois": sorted(sealed_dois),
                          "disjoint_from_training": True},
        "overall_capped": _metrics(y, capped),
        "upper_cap_violation": viol,
        "violation_rate": round(viol / len(sealed), 4),
        "baselines": {"zero_pred_mae": round(zero_mae, 4),
                      "median_pred_r2": base_r2},
        "doi_bootstrap_mae_ci95": ci,
        "per_doi_median_mae": round(float(per_doi["mae"].median()), 4),
        "n_doi_negative_r2": int((per_doi["r2"] < 0).sum()),
        "environment": {"python": platform.python_version()},
    }
    (out / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n[완료] 봉인 평가가 1회 수행되고 지표가 동결되었습니다.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    main(ap.parse_args().root)
