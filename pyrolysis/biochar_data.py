"""
biochar_data.py — Biochar Stage1 Expansion v5 데이터 버전 로더 (읽기 전용 집계).

이 모듈은 첨부 ZIP(`data/imports/Biochar_Stage1_Expansion_v5/`)을 **별도 데이터 버전**으로
읽어 변수·결측·중복·출처 그룹을 집계한다.

원칙 (지시 사항):
  - 기존 프로젝트 파일·데이터·모델을 수정하지 않는다. 이 모듈은 읽기만 한다.
  - 원본(v2 `biochar/data`)·검증 이력(verified_v3·v4)·최신 통합 후보(verified_v5)를 구분한다.
  - 여러 폴더의 CSV 를 전부 합치지 않는다 (같은 자료의 원본·검증본·통합본 중복).
  - Stage1 최신 통합 후보는 BET 47 / 총 공극부피 42 를 확인한다. 최종 학습 승인 아님.
  - Stage2 는 원본·검증 결과를 대조한다. 미검증 흡착량을 실측 평형 qe 로 취급하지 않는다.
  - eligible_for_training 을 변경하지 않는다. 모델 학습·재학습을 하지 않는다.

Streamlit 을 import 하지 않는다. scikit-learn 등 모델 라이브러리도 쓰지 않는다 (집계 전용).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "BiocharPaths",
    "resolve_biochar_root",
    "Stage1CandidateSummary",
    "Stage2Summary",
    "load_stage1_candidates",
    "summarize_stage1_candidates",
    "summarize_stage2",
    "build_biochar_report",
    # Stage1 학습 데이터 선정 준비 (작업: 상세 점검)
    "normalize_modification",
    "MIN_CONFIG_INPUTS",
    "EXTENDED_CONFIG_INPUTS",
    "detailed_manufacturing_check",
    "usable_samples_by_config",
    "category_paper_dependency",
    "logo_feasibility",
    "build_stage1_selection_report",
    "summarize_stage2_versions",
]


# 기본 import 위치. 지시는 data/imports 이지만 실제 ZIP 이 data/ 에 있던 경우도 허용한다.
_DEFAULT_IMPORT_DIRS = (
    Path("data/imports/Biochar_Stage1_Expansion_v5"),
    Path("data/Biochar_Stage1_Expansion_v5"),
)

# Stage1 통합 후보의 입력(제조조건) / 물성 타깃 열
STAGE1_MANUFACTURING_INPUTS = [
    "feedstock",
    "pyrolysis_temp_C",
    "holding_time_min",
    "heating_rate_C_min",
    "atmosphere",
    "gas_flow_mL_min",
    "modification_method",
]
STAGE1_FEEDSTOCK_COMPOSITION = [
    "feedstock_C_wt_pct",
    "feedstock_H_wt_pct",
    "feedstock_O_wt_pct",
    "feedstock_N_wt_pct",
    "feedstock_Ash_wt_pct",
    "feedstock_VM_wt_pct",
    "feedstock_FC_wt_pct",
    "feedstock_Moisture_wt_pct",
]
STAGE1_BET_TARGET = "BET_m2_g"
STAGE1_PORE_TARGET = "pore_volume_cm3_g"

# Stage2 흡착 조건 입력 + 물성(Stage1 예측 결합 지점) + qe 타깃
STAGE2_ADSORPTION_INPUTS = [
    "pollutant",
    "initial_conc_mg_L",
    "initial_pH",
    "adsorbent_dose_g_L",
    "adsorption_temp_C",
    "contact_time_min",
]
STAGE2_PROPERTY_LINK = ["BET_m2_g", "pore_volume_cm3_g"]
STAGE2_QE_TARGET = "measured_qe_mg_g"  # 검증된 실측 qe (미검증이면 비어 있음)

# 출처 그룹 후보 열 (논문/누수 그룹/원실험 DOI)
SOURCE_GROUP_COLS = ["leakage_group", "original_experiment_doi", "source_file"]


class BiocharDataError(Exception):
    """Biochar 데이터 버전 로드 실패."""


@dataclass
class BiocharPaths:
    """Biochar 데이터 버전별 폴더·파일 경로 (읽기 전용)."""

    root: Path                 # .../Biochar_Stage1_Expansion_v5
    data_v2: Path              # biochar/data (원본/빌드 산출물)
    verified_v3: Path          # 검증 이력 v3
    verified_v4: Path          # 검증 이력 v4
    verified_v5: Path          # 최신 Stage1 통합 후보
    stage1_bet_v5: Path        # Stage1_BET_47_source_checked_candidates.csv
    stage1_pore_v5: Path       # Stage1_Pore_42_source_checked_candidates.csv
    stage2_qe: Path            # data/Stage2_qe.csv

    def exists_report(self) -> dict[str, bool]:
        return {
            "root": self.root.is_dir(),
            "data_v2": self.data_v2.is_dir(),
            "verified_v3": self.verified_v3.is_dir(),
            "verified_v4": self.verified_v4.is_dir(),
            "verified_v5": self.verified_v5.is_dir(),
            "stage1_bet_v5": self.stage1_bet_v5.is_file(),
            "stage1_pore_v5": self.stage1_pore_v5.is_file(),
            "stage2_qe": self.stage2_qe.is_file(),
        }


def resolve_biochar_root(base: Path | None = None) -> BiocharPaths:
    """import 디렉터리를 찾아 버전별 경로를 구성한다.

    Args:
        base: 워크스페이스 루트. None 이면 현재 작업 디렉터리 기준.

    Raises:
        BiocharDataError: 어느 후보 경로에도 root 가 없으면.
    """
    base = Path(base) if base is not None else Path.cwd()
    root = None
    for cand in _DEFAULT_IMPORT_DIRS:
        p = base / cand
        if p.is_dir():
            root = p
            break
    if root is None:
        raise BiocharDataError(
            "Biochar import 폴더를 찾을 수 없습니다. 다음 중 하나가 필요합니다: "
            + ", ".join(str(base / c) for c in _DEFAULT_IMPORT_DIRS)
        )

    biochar = root / "biochar"
    return BiocharPaths(
        root=root,
        data_v2=biochar / "data",
        verified_v3=biochar / "verified_v3",
        verified_v4=biochar / "verified_v4",
        verified_v5=biochar / "verified_v5",
        stage1_bet_v5=biochar / "verified_v5" / "Stage1_BET_47_source_checked_candidates.csv",
        stage1_pore_v5=biochar / "verified_v5" / "Stage1_Pore_42_source_checked_candidates.csv",
        stage2_qe=biochar / "data" / "Stage2_qe.csv",
    )


def load_stage1_candidates(paths: BiocharPaths, which: str) -> pd.DataFrame:
    """최신 Stage1 통합 후보(v5)를 읽는다. which 는 'bet' 또는 'pore'.

    원본·검증본과 합치지 않는다. 이 파일 하나만 읽는다. 플래그를 바꾸지 않는다.
    """
    if which == "bet":
        p = paths.stage1_bet_v5
    elif which == "pore":
        p = paths.stage1_pore_v5
    else:
        raise BiocharDataError("which 는 'bet' 또는 'pore' 여야 합니다: " + repr(which))
    if not p.is_file():
        raise BiocharDataError("Stage1 후보 파일이 없습니다: " + str(p))
    return pd.read_csv(p)


def _column_availability(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    """열별 (available, missing) 집계. 열이 없으면 전부 missing 으로 본다."""
    n = len(df)
    out: dict[str, dict[str, int]] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"available": 0, "missing": n, "present_column": 0}
        else:
            avail = int(df[c].notna().sum())
            out[c] = {"available": avail, "missing": n - avail, "present_column": 1}
    return out


def _group_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    """출처 그룹 열의 고유 그룹 수와 결측 수."""
    if col not in df.columns:
        return {"present_column": 0, "n_groups": 0, "n_missing": len(df)}
    s = df[col]
    return {
        "present_column": 1,
        "n_groups": int(s.dropna().nunique()),
        "n_missing": int(s.isna().sum()),
    }


@dataclass
class Stage1CandidateSummary:
    """Stage1 통합 후보 하나(BET 또는 Pore)의 집계."""

    which: str
    n_rows: int
    target_col: str
    target_available: int
    eligible_for_training_counts: dict[str, int]
    stage1_candidate_counts: dict[str, int]
    qa_status_counts: dict[str, int]
    manufacturing_input_availability: dict[str, dict[str, int]]
    feedstock_composition_availability: dict[str, dict[str, int]]
    source_group_counts: dict[str, dict[str, int]]
    duplicate_rows: int
    feedstock_categories: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "which": self.which,
            "n_rows": self.n_rows,
            "target_col": self.target_col,
            "target_available": self.target_available,
            "eligible_for_training_counts": self.eligible_for_training_counts,
            "stage1_candidate_counts": self.stage1_candidate_counts,
            "qa_status_counts": self.qa_status_counts,
            "manufacturing_input_availability": self.manufacturing_input_availability,
            "feedstock_composition_availability": self.feedstock_composition_availability,
            "source_group_counts": self.source_group_counts,
            "duplicate_rows": self.duplicate_rows,
            "feedstock_categories": self.feedstock_categories,
            "notes": self.notes,
        }


def _value_counts(df: pd.DataFrame, col: str, limit: int = 50) -> dict[str, int]:
    if col not in df.columns:
        return {}
    vc = df[col].value_counts(dropna=False).head(limit)
    return {str(k): int(v) for k, v in vc.items()}


def summarize_stage1_candidates(df: pd.DataFrame, which: str) -> Stage1CandidateSummary:
    """Stage1 통합 후보의 변수·결측·중복·출처 그룹을 집계한다 (학습 아님)."""
    target_col = STAGE1_BET_TARGET if which == "bet" else STAGE1_PORE_TARGET
    notes: list[str] = []

    # eligible_for_training 은 읽기만 하고 바꾸지 않는다.
    elig = _value_counts(df, "eligible_for_training")
    if elig and not any(k in ("True", "1", "true") for k in elig):
        notes.append("eligible_for_training 이 모두 False 입니다. 최종 학습 승인 데이터가 아닙니다.")

    # 중복: 식별·메타 열을 제외한 핵심 입력+타깃 기준
    core_cols = [c for c in (STAGE1_MANUFACTURING_INPUTS + [target_col]) if c in df.columns]
    dup = int(df.duplicated(subset=core_cols).sum()) if core_cols else 0

    src_counts = {c: _group_counts(df, c) for c in SOURCE_GROUP_COLS}

    return Stage1CandidateSummary(
        which=which,
        n_rows=len(df),
        target_col=target_col,
        target_available=int(df[target_col].notna().sum()) if target_col in df.columns else 0,
        eligible_for_training_counts=elig,
        stage1_candidate_counts=_value_counts(df, "stage1_candidate"),
        qa_status_counts=_value_counts(df, "qa_status"),
        manufacturing_input_availability=_column_availability(df, STAGE1_MANUFACTURING_INPUTS),
        feedstock_composition_availability=_column_availability(df, STAGE1_FEEDSTOCK_COMPOSITION),
        source_group_counts=src_counts,
        duplicate_rows=dup,
        feedstock_categories=_value_counts(df, "feedstock"),
        notes=notes,
    )


@dataclass
class Stage2Summary:
    """Stage2 qe 데이터의 집계 (원본·검증 대조, 승격하지 않음)."""

    n_rows: int
    eligible_for_training_counts: dict[str, int]
    qa_status_counts: dict[str, int]
    equilibrium_status_counts: dict[str, int]
    measured_qe_available: int
    unverified_capacity_available: int
    both_properties_available: int
    adsorption_input_availability: dict[str, dict[str, int]]
    property_link_availability: dict[str, dict[str, int]]
    pollutant_counts: dict[str, int]
    source_group_counts: dict[str, dict[str, int]]
    duplicate_rows: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "eligible_for_training_counts": self.eligible_for_training_counts,
            "qa_status_counts": self.qa_status_counts,
            "equilibrium_status_counts": self.equilibrium_status_counts,
            "measured_qe_available": self.measured_qe_available,
            "unverified_capacity_available": self.unverified_capacity_available,
            "both_properties_available": self.both_properties_available,
            "adsorption_input_availability": self.adsorption_input_availability,
            "property_link_availability": self.property_link_availability,
            "pollutant_counts": self.pollutant_counts,
            "source_group_counts": self.source_group_counts,
            "duplicate_rows": self.duplicate_rows,
            "notes": self.notes,
        }


def summarize_stage2(df: pd.DataFrame) -> Stage2Summary:
    """Stage2 qe 를 집계한다. 미검증 흡착량을 실측 qe 로 올리지 않는다."""
    notes: list[str] = []

    measured_available = int(df[STAGE2_QE_TARGET].notna().sum()) if STAGE2_QE_TARGET in df.columns else 0
    if measured_available == 0:
        notes.append(
            "measured_qe_mg_g 가 0 건입니다. 평형이 입증된 실측 qe 가 아직 없습니다. "
            "reported_capacity_mg_g_unverified 는 미검증 후보이며 타깃으로 승격하지 않습니다."
        )

    unverified = (
        int(df["reported_capacity_mg_g_unverified"].notna().sum())
        if "reported_capacity_mg_g_unverified" in df.columns else 0
    )
    both_props = 0
    if "both_properties_available" in df.columns:
        both_props = int(df["both_properties_available"].astype(str).isin(["True", "1", "true"]).sum())

    return Stage2Summary(
        n_rows=len(df),
        eligible_for_training_counts=_value_counts(df, "eligible_for_training"),
        qa_status_counts=_value_counts(df, "qa_status"),
        equilibrium_status_counts=_value_counts(df, "equilibrium_status"),
        measured_qe_available=measured_available,
        unverified_capacity_available=unverified,
        both_properties_available=both_props,
        adsorption_input_availability=_column_availability(df, STAGE2_ADSORPTION_INPUTS),
        property_link_availability=_column_availability(df, STAGE2_PROPERTY_LINK),
        pollutant_counts=_value_counts(df, "pollutant", limit=100),
        source_group_counts={c: _group_counts(df, c) for c in SOURCE_GROUP_COLS},
        duplicate_rows=int(df.duplicated().sum()),
        notes=notes,
    )


def build_biochar_report(base: Path | None = None) -> dict[str, Any]:
    """전체 버전 구분 + Stage1 후보 + Stage2 집계를 하나의 딕셔너리로 만든다.

    모델 학습을 실행하지 않고 eligible_for_training 을 바꾸지 않는다.
    """
    paths = resolve_biochar_root(base)
    report: dict[str, Any] = {
        "import_root": str(paths.root),
        "paths_exist": paths.exists_report(),
        "versions": {
            "original_v2": "biochar/data (빌드 산출물: Material_Master, Stage1_BET, Stage1_PoreVolume, Stage2_qe)",
            "verification_history": ["biochar/verified_v3", "biochar/verified_v4"],
            "latest_stage1_candidates_v5": "biochar/verified_v5 (BET 47 / Pore 42)",
        },
        "do_not_merge": "같은 자료의 원본·검증본·통합본이 중복이므로 폴더 간 CSV 를 합치지 않는다.",
    }

    # Stage1 최신 통합 후보
    bet = load_stage1_candidates(paths, "bet")
    pore = load_stage1_candidates(paths, "pore")
    report["stage1_bet_candidates"] = summarize_stage1_candidates(bet, "bet").to_dict()
    report["stage1_pore_candidates"] = summarize_stage1_candidates(pore, "pore").to_dict()

    # Stage2 qe (원본·검증 대조)
    if paths.stage2_qe.is_file():
        s2 = pd.read_csv(paths.stage2_qe)
        report["stage2_qe"] = summarize_stage2(s2).to_dict()
    else:
        report["stage2_qe"] = {"error": "Stage2_qe.csv 없음", "path": str(paths.stage2_qe)}

    # 예측 구조 (지시: 제조조건 → BET·Pore 각각 예측 → 흡착조건 결합 → qe)
    report["prediction_structure"] = {
        "stage1_bet": "제조조건 → BET_m2_g (회귀)",
        "stage1_pore": "제조조건 → pore_volume_cm3_g (회귀)",
        "stage2_qe": "흡착조건 + (예측 BET·Pore) → measured_qe_mg_g (회귀, 평형 검증 필요)",
        "note": "Stage1 예측 물성을 Stage2 입력으로 결합하는 연계 구조. 작업 33 조건부 설계와 대응.",
    }
    return report


# ── Stage1 학습 데이터 선정 준비 ────────────────────────────────────────────
#
# 결측이 있다는 이유로 행을 먼저 삭제하지 않는다. 누락(미보고)과 실제 '무개질'을 구분하고,
# 확인되지 않은 값을 생성하지 않는다. 학습 플래그를 바꾸지 않고 모델 학습도 하지 않는다.

# 두 입력변수 구성 (지시 작업 2)
MIN_CONFIG_INPUTS = ["feedstock", "pyrolysis_temp_C", "modification_method"]
EXTENDED_CONFIG_INPUTS = MIN_CONFIG_INPUTS + [
    "holding_time_min", "heating_rate_C_min", "atmosphere",
]

# modification_method 중 '무개질'을 명시한 원문 표현. 이 목록에 없는 값은 실제 개질로 본다.
# 결측(NaN)은 '미보고'로 남기고 '무개질'로 바꾸지 않는다 (값 생성 금지).
_UNMODIFIED_PHRASES = {
    "unmodified",
    "no added activation or chemical modification reported",
    "none",
    "no modification",
    "raw",
}


def normalize_modification(value: Any) -> str:
    """modification_method 값을 '무개질/개질/미보고'로 분류한다 (값 생성 아님, 분류만).

    Returns: "none"(무개질 명시), "modified"(실제 개질), "unreported"(미보고/결측).
    """
    if value is None:
        return "unreported"
    try:
        if pd.isna(value):
            return "unreported"
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s == "":
        return "unreported"
    if s.lower() in _UNMODIFIED_PHRASES:
        return "none"
    return "modified"


def _numeric_range(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"present_column": 0}
    num = pd.to_numeric(df[col], errors="coerce")
    present = int(df[col].notna().sum())
    numeric = int(num.notna().sum())
    info: dict[str, Any] = {
        "present_column": 1,
        "n_present": present,
        "n_numeric": numeric,
        "n_nonnumeric_nonnull": present - numeric,
        "n_missing": int(df[col].isna().sum()),
    }
    if numeric:
        info["min"] = float(num.min())
        info["max"] = float(num.max())
        info["distinct_numeric"] = sorted(float(v) for v in num.dropna().unique())
    return info


def detailed_manufacturing_check(df: pd.DataFrame, which: str) -> dict[str, Any]:
    """제조조건의 결측·범주·범위·중복 시료를 상세 점검한다 (작업 1).

    - 수치형(온도·유지시간·승온속도·유량): 범위·결측·비수치 집계.
    - 범주형(feedstock·atmosphere·modification): 값 분포·결측.
    - modification 은 무개질/개질/미보고로 분류한다.
    - 중복: canonical_material_id 와 제조 fingerprint 기준.
    """
    out: dict[str, Any] = {"which": which, "n_rows": len(df)}

    # 수치형 범위
    out["numeric"] = {
        c: _numeric_range(df, c)
        for c in ["pyrolysis_temp_C", "holding_time_min", "heating_rate_C_min", "gas_flow_mL_min"]
    }

    # 범주형 분포·결측
    cat = {}
    for c in ["feedstock", "atmosphere"]:
        if c in df.columns:
            cat[c] = {
                "n_missing": int(df[c].isna().sum()),
                "n_unique": int(df[c].dropna().nunique()),
                "values": _value_counts(df, c),
            }
    out["categorical"] = cat

    # modification: 무개질/개질/미보고 분류 (삭제하지 않음)
    if "modification_method" in df.columns:
        klass = df["modification_method"].map(normalize_modification)
        out["modification"] = {
            "class_counts": {k: int(v) for k, v in klass.value_counts().items()},
            "raw_values": _value_counts(df, "modification_method"),
            "note": "미보고(unreported)는 결측이며 '무개질'로 대체하지 않는다.",
        }

    # 중복 시료
    dup_info: dict[str, Any] = {}
    if "canonical_material_id" in df.columns:
        vc = df["canonical_material_id"].value_counts()
        dup_info["canonical_material_id_duplicates"] = {str(k): int(v) for k, v in vc.items() if v > 1}
        dup_info["n_unique_material_id"] = int(df["canonical_material_id"].nunique())
    fp_cols = [c for c in STAGE1_MANUFACTURING_INPUTS if c in df.columns]
    dup_info["manufacturing_fingerprint_duplicate_rows"] = int(
        df.duplicated(subset=fp_cols, keep=False).sum()
    )
    out["duplicates"] = dup_info
    return out


def _config_usable_mask(df: pd.DataFrame, config: list[str]) -> pd.Series:
    """구성의 필수 입력이 '사용 가능'한 행 마스크.

    - 일반 열: 값이 있어야 사용 가능 (결측이면 제외, 단 '삭제'가 아니라 집계상 제외).
    - modification_method: '무개질(none)' 또는 '개질(modified)'이면 사용 가능.
      '미보고(unreported)'는 사용 불가로 집계하되 값을 생성하지 않는다.
    """
    mask = pd.Series(True, index=df.index)
    for c in config:
        if c not in df.columns:
            mask &= False
            continue
        if c == "modification_method":
            klass = df[c].map(normalize_modification)
            mask &= klass.isin(["none", "modified"])
        else:
            mask &= df[c].notna()
    return mask


def usable_samples_by_config(df: pd.DataFrame, which: str) -> dict[str, Any]:
    """최소/확장 구성별 사용 가능 시료 수와 출처 그룹 수를 집계한다 (작업 2).

    결측 행을 먼저 삭제하지 않는다. 각 구성에서 필요한 입력이 갖춰진 행만 집계한다.
    """
    result: dict[str, Any] = {"which": which, "n_rows": len(df)}
    for name, config in (("minimal", MIN_CONFIG_INPUTS), ("extended", EXTENDED_CONFIG_INPUTS)):
        mask = _config_usable_mask(df, config)
        sub = df[mask]
        entry: dict[str, Any] = {
            "inputs": config,
            "n_usable_samples": int(mask.sum()),
            "n_excluded_missing": int((~mask).sum()),
        }
        for gcol in ("leakage_group", "original_experiment_doi"):
            if gcol in df.columns:
                entry[f"n_groups_{gcol}"] = int(sub[gcol].dropna().nunique())
        # 사용 가능 행의 원료 범주 수
        if "feedstock" in df.columns:
            entry["n_feedstock_categories"] = int(sub["feedstock"].dropna().nunique())
        result[name] = entry
    return result


def category_paper_dependency(df: pd.DataFrame, which: str) -> dict[str, Any]:
    """원료·개질 범주가 특정 논문(leakage_group)에만 존재하는지 확인한다 (작업 4).

    - 각 범주값이 몇 개 출처 그룹에 등장하는지 집계.
    - 단일 그룹에만 있는 범주(single-group categories)를 표시.
    - 그룹을 하나 제외(LOGO)했을 때, 그 그룹에만 있던 범주는 학습에서 사라지고
      테스트(제외된 그룹)에만 등장한다. 그런 '테스트 전용 범주'를 그룹별로 집계.
    """
    gcol = "leakage_group"
    out: dict[str, Any] = {"which": which, "group_col": gcol}
    if gcol not in df.columns:
        out["error"] = "leakage_group 열이 없습니다."
        return out

    for cat_col in ("feedstock", "modification_method"):
        if cat_col not in df.columns:
            continue
        # modification 은 분류값으로 치환해 집계 (미보고는 제외)
        if cat_col == "modification_method":
            series = df[cat_col].map(normalize_modification)
            work = pd.DataFrame({"cat": series, "grp": df[gcol]})
            work = work[work["cat"] != "unreported"]
        else:
            work = pd.DataFrame({"cat": df[cat_col], "grp": df[gcol]}).dropna(subset=["cat"])

        cat_to_groups = work.groupby("cat")["grp"].nunique()
        single_group_cats = sorted(cat_to_groups[cat_to_groups == 1].index.astype(str))

        # 그룹별: 그 그룹을 테스트로 뺐을 때 테스트에만 등장하는 범주
        test_only_by_group: dict[str, list[str]] = {}
        for g in work["grp"].dropna().unique():
            in_g = set(work[work["grp"] == g]["cat"].astype(str))
            in_other = set(work[work["grp"] != g]["cat"].astype(str))
            test_only = sorted(in_g - in_other)
            if test_only:
                test_only_by_group[str(g)] = test_only

        out[cat_col] = {
            "n_categories": int(work["cat"].nunique()),
            "category_to_n_groups": {str(k): int(v) for k, v in cat_to_groups.items()},
            "single_group_categories": single_group_cats,
            "test_only_categories_if_group_held_out": test_only_by_group,
        }
    return out


def logo_feasibility(df: pd.DataFrame, which: str) -> dict[str, Any]:
    """leakage_group 기준 Leave-One-Group-Out 평가 가능성을 확인한다 (작업 5).

    - 그룹 수, 그룹별 행 수.
    - 각 그룹을 제외했을 때 남는 학습 그룹 수와 행 수.
    - LOGO 는 그룹이 2개 이상이어야 의미가 있다.
    """
    gcol = "leakage_group"
    out: dict[str, Any] = {"which": which, "group_col": gcol}
    if gcol not in df.columns:
        out["error"] = "leakage_group 열이 없습니다."
        return out
    counts = df[gcol].value_counts(dropna=False)
    group_sizes = {str(k): int(v) for k, v in counts.items()}
    n_groups = int(df[gcol].dropna().nunique())
    out["n_groups"] = n_groups
    out["group_sizes"] = group_sizes
    out["logo_feasible"] = n_groups >= 2
    out["folds"] = n_groups
    # 각 fold(그룹 제외) 시 학습에 남는 행 수
    out["per_fold_train_rows"] = {
        str(g): int((df[gcol] != g).sum()) for g in df[gcol].dropna().unique()
    }
    out["note"] = (
        "전처리·결측 대체는 각 학습 fold 안에서만 학습한다 (평가 그룹 정보 미사용). "
        "그룹이 작아 일부 fold 는 특정 원료·개질 범주가 테스트에만 등장할 수 있다."
    )
    return out


def build_stage1_selection_report(base: Path | None = None) -> dict[str, Any]:
    """Stage1 학습 데이터 선정 준비 전체 집계 (작업 1·2·4·5). 학습·플래그 변경 없음."""
    paths = resolve_biochar_root(base)
    report: dict[str, Any] = {"import_root": str(paths.root)}
    for which in ("bet", "pore"):
        df = load_stage1_candidates(paths, which)
        report[which] = {
            "detailed_check": detailed_manufacturing_check(df, which),
            "usable_by_config": usable_samples_by_config(df, which),
            "category_paper_dependency": category_paper_dependency(df, which),
            "logo_feasibility": logo_feasibility(df, which),
        }
    report["preprocessing_rule"] = (
        "결측 대체·인코딩·스케일링은 Pipeline 안에서 각 학습 fold 로만 fit 한다. "
        "평가 그룹의 통계를 쓰지 않는다 (누출 방지)."
    )
    report["not_done"] = [
        "모델 학습·재학습 안 함",
        "eligible_for_training 변경 안 함",
        "결측 이유로 행 선삭제 안 함 (구성별 집계상 제외만)",
        "미보고를 무개질로 대체하지 않음",
    ]
    return report


# ── Stage2 버전별 현황 (원본 vs 검증 이력) ──────────────────────────────────


def summarize_stage2_versions(base: Path | None = None) -> dict[str, Any]:
    """Stage2 를 v2 원본(미검증)과 v3·v4 부분검증 결과로 '별도' 집계한다 (작업 6).

    검증본을 원본에 단순 추가하거나 전부 학습 승인하지 않는다. 각 버전을 구분해서 센다.
    """
    paths = resolve_biochar_root(base)
    out: dict[str, Any] = {"policy": "검증본을 원본에 합치지 않고 버전별로 별도 집계한다."}

    # v2 원본 (빌드 산출물): 평형 미검증 상태
    if paths.stage2_qe.is_file():
        s2 = pd.read_csv(paths.stage2_qe)
        out["v2_original_unverified"] = {
            "file": "biochar/data/Stage2_qe.csv",
            "n_rows": len(s2),
            "measured_qe_mg_g_available": int(s2["measured_qe_mg_g"].notna().sum())
            if "measured_qe_mg_g" in s2.columns else 0,
            "equilibrium_status": _value_counts(s2, "equilibrium_status"),
            "qa_status": _value_counts(s2, "qa_status"),
            "eligible_for_training": _value_counts(s2, "eligible_for_training"),
        }

    # v3·v4 부분검증 결과 파일 목록과 행 수 (내용 병합하지 않고 파일 단위로 센다)
    def _list_version(vdir: Path) -> dict[str, Any]:
        if not vdir.is_dir():
            return {"error": "폴더 없음", "path": str(vdir)}
        files = {}
        for p in sorted(vdir.glob("*.csv")):
            try:
                n = sum(1 for _ in p.open(encoding="utf-8", errors="ignore")) - 1
            except Exception:
                n = -1
            files[p.name] = {"rows_approx": n}
        return {"path": str(vdir.name), "n_csv": len(files), "files": files}

    out["v3_partial_verification"] = _list_version(paths.verified_v3)
    out["v4_partial_verification"] = _list_version(paths.verified_v4)
    out["note"] = (
        "v2 는 평형 미검증(measured_qe 0건). v3·v4 는 부분검증 중간 산출물이며 "
        "학습 승인 데이터가 아니다. 어느 것도 eligible_for_training 을 True 로 올리지 않았다."
    )
    return out
