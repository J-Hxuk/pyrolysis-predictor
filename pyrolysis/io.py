"""
io.py — 학습 자료 읽기, 대상별 자료 점검, 표준 양식 생성.

경로는 pathlib.Path 로 처리한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pyrolysis.contracts import CandidateTable, Issue

__all__ = [
    "TrainingData",
    "TargetAuditResult",
    "DatasetAuditReport",
    "read_training_file",
    "audit_dataset",
    "DatasetReadError",
    "TemplateError",
    "write_csv_template",
    "write_excel_template",
    "write_input_templates",
    "PredictionFileError",
    "parse_prediction_file",
]


class DatasetReadError(Exception):
    """학습 파일 읽기·파싱 실패."""


@dataclass
class TrainingData:
    """파싱된 학습 자료. 입력·목표·식별자를 분리해 보관하며 값을 수정하지 않는다."""

    inputs: pd.DataFrame
    targets: pd.DataFrame
    identifiers: pd.DataFrame
    extra_columns: list
    source_path: Path
    input_schema_id: str
    notes: list


@dataclass
class TargetAuditResult:
    """예측 대상 하나의 자료 점검 결과."""

    target_id: str
    task: str
    n_total: int
    n_available: int
    n_missing: int
    n_zero: int
    class_counts: object
    n_replicated_available: int


@dataclass
class DatasetAuditReport:
    """전체 자료 점검 결과."""

    source_path: str
    n_total_rows: int
    n_replicated_rows: int
    replicated_input_groups: int
    input_schema_id: str
    target_results: list
    excluded_columns: list
    notes: list


def read_training_file(
    path: Path,
    spec: Any,
    *,
    sheet_name=0,
    id_columns=None,
    column_map=None,
) -> TrainingData:
    """학습 파일(.csv 또는 .xlsx)을 읽어 TrainingData 로 반환한다.

    열을 입력·목표·식별자·불명으로 분류한다. 값을 수정하거나 결측을 채우지 않는다.
    알 수 없는 열은 제외하고 notes 에 기록한다. 목표 열이 없는 대상은 NaN 열로 추가하고
    notes 에 기록한다. 목표 ID 가 입력 ID 와 겹치면 데이터 누출 위험을 기록한다.
    """
    id_columns = list(id_columns or [])
    column_map = dict(column_map or {})
    notes = []

    if not path.exists():
        raise DatasetReadError("파일이 없습니다: " + str(path))

    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(path, sheet_name=sheet_name, header=0)
        elif suffix == ".csv":
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                raw = pd.read_csv(path, encoding="utf-8")
        else:
            raise DatasetReadError(
                "지원하지 않는 파일 형식: " + repr(suffix) + ". .xlsx 또는 .csv 만 허용합니다."
            )
    except DatasetReadError:
        raise
    except Exception as exc:
        raise DatasetReadError("파일 읽기 실패 (" + path.name + "): " + str(exc)) from exc

    if raw.empty:
        raise DatasetReadError("파일이 비어 있거나 헤더만 있습니다: " + path.name)

    if column_map:
        raw = raw.rename(columns=column_map)

    input_ids = set(s.id for s in spec.inputs)
    target_ids = set(t.id for t in spec.targets)
    id_col_set = set(id_columns)
    file_cols = set(raw.columns)

    leakage = target_ids & input_ids
    if leakage:
        notes.append("경고: 목표 ID가 입력 ID와 겹칩니다 (데이터 누출 위험): " + str(sorted(leakage)))

    conflict = id_col_set & (input_ids | target_ids)
    if conflict:
        notes.append("경고: 식별자 열이 규격 ID와 겹칩니다 (" + str(sorted(conflict)) + "). 식별자로 처리합니다.")
        input_ids -= conflict
        target_ids -= conflict

    extra_cols = file_cols - input_ids - target_ids - id_col_set
    if extra_cols:
        notes.append("알 수 없는 열 " + str(len(extra_cols)) + "개를 제외했습니다: " + str(sorted(extra_cols)))

    missing_input_cols = input_ids - file_cols
    if missing_input_cols:
        notes.append("규격에 있는 입력 열이 파일에 없습니다: " + str(sorted(missing_input_cols)))
    missing_target_cols = target_ids - file_cols
    if missing_target_cols:
        notes.append(
            "규격에 있는 목표 열이 파일에 없습니다: " + str(sorted(missing_target_cols))
            + " — 해당 대상의 가용 행 = 0"
        )

    avail_input_ids = sorted(input_ids & file_cols)
    avail_target_ids = sorted(target_ids & file_cols)
    avail_id_cols = [c for c in id_columns if c in file_cols]

    inputs_df = raw[avail_input_ids].copy() if avail_input_ids else pd.DataFrame(index=raw.index)
    targets_df = raw[avail_target_ids].copy() if avail_target_ids else pd.DataFrame(index=raw.index)
    id_df = raw[avail_id_cols].copy() if avail_id_cols else pd.DataFrame(index=raw.index)

    for col in sorted(missing_input_cols):
        inputs_df[col] = pd.NA
    for col in sorted(missing_target_cols):
        targets_df[col] = pd.NA

    try:
        from pyrolysis.config import build_input_schema
        schema_id = build_input_schema(spec).schema_id
    except Exception:
        schema_id = "unknown"

    return TrainingData(
        inputs=inputs_df,
        targets=targets_df,
        identifiers=id_df,
        extra_columns=sorted(extra_cols),
        source_path=path,
        input_schema_id=schema_id,
        notes=notes,
    )


def audit_dataset(data: TrainingData, spec: Any) -> DatasetAuditReport:
    """TrainingData 를 점검하고 대상별 가용 행·결측·반복 측정을 보고한다.

    결측 목표를 대체하거나 결측을 미검출로 바꾸지 않는다. 입력이 중복된 행을
    반복 측정 후보로 표시하되 제거하지 않는다. 결측(NaN)과 수치 0을 구별한다.
    """
    notes = list(data.notes)
    n_total = len(data.inputs) if not data.inputs.empty else len(data.targets)

    if not data.inputs.empty and len(data.inputs.columns) > 0:
        dup_mask = data.inputs.duplicated(keep=False)
        n_replicated = int(dup_mask.sum())
        if n_replicated > 0:
            n_dup_groups = int(data.inputs[dup_mask].drop_duplicates().shape[0])
        else:
            n_dup_groups = 0
    else:
        dup_mask = pd.Series([False] * n_total, dtype=bool)
        n_replicated = 0
        n_dup_groups = 0

    if n_replicated > 0:
        notes.append(
            "입력이 중복된 행 " + str(n_replicated) + "개 (" + str(n_dup_groups) + "개 그룹): "
            "반복 측정 또는 복제 기록 후보입니다. 원본 측정과 반복 행을 구별해야 합니다."
        )

    target_results = []

    for tgt in spec.targets:
        if tgt.id not in data.targets.columns:
            target_results.append(TargetAuditResult(
                target_id=tgt.id, task=tgt.task, n_total=n_total,
                n_available=0, n_missing=n_total, n_zero=0,
                class_counts=None, n_replicated_available=0,
            ))
            continue

        col = data.targets[tgt.id]
        null_mask = col.isna()
        avail_mask = ~null_mask
        n_available = int(avail_mask.sum())
        n_missing = int(null_mask.sum())

        if tgt.task == "regression":
            n_zero = int((col[avail_mask] == 0).sum())
            class_counts = None
        else:
            n_zero = 0
            avail_vals = col[avail_mask]
            class_counts = {str(k): int(v) for k, v in avail_vals.value_counts(dropna=True).items()}

        if len(dup_mask) == len(avail_mask):
            n_rep_avail = int((avail_mask & dup_mask).sum())
        else:
            n_rep_avail = 0

        target_results.append(TargetAuditResult(
            target_id=tgt.id, task=tgt.task, n_total=n_total,
            n_available=n_available, n_missing=n_missing, n_zero=n_zero,
            class_counts=class_counts, n_replicated_available=n_rep_avail,
        ))

    return DatasetAuditReport(
        source_path=str(data.source_path),
        n_total_rows=n_total,
        n_replicated_rows=n_replicated,
        replicated_input_groups=n_dup_groups,
        input_schema_id=data.input_schema_id,
        target_results=target_results,
        excluded_columns=data.extra_columns,
        notes=notes,
    )


# ── 표준 입력 양식 생성 (작업 8) ──────────────────────────────────────────────


class TemplateError(Exception):
    """양식 생성 실패."""


def _template_frames(schema, include_candidate_id: bool):
    """양식용 입력 열 목록과 안내표(DataFrame)를 구성한다.

    목표 결과 열은 포함하지 않는다. 식별자·규격 정보는 예측 입력 열에 넣지 않는다.
    """
    specs = sorted(schema.input_specs, key=lambda s: (s.display_order, s.id))
    input_cols = [s.id for s in specs]

    guide_rows = []
    for s in specs:
        if s.dtype == "category":
            cats = ", ".join(c.value for c in s.allowed_categories)
        else:
            cats = ""
        guide_rows.append({
            "input_id": s.id,
            "label_ko": s.label_ko,
            "dtype": s.dtype,
            "unit": s.unit if s.unit is not None else "",
            "required": "예" if s.required else "아니오",
            "allowed_categories": cats,
        })
    return input_cols, guide_rows


def write_csv_template(
    schema,
    output_path: Path,
    *,
    include_candidate_id: bool = True,
) -> Path:
    """활성 입력 규격으로 빈 CSV 양식을 UTF-8 BOM 으로 저장한다.

    - 첫 행은 열 이름. `_input_schema_id` 열에 규격 식별자를 기록한다.
    - `candidate_id` 는 선택 식별자. 목표 결과 열은 넣지 않는다.
    - 데이터 행은 비운다 (헤더만 생성).
    """
    import csv

    input_cols, _ = _template_frames(schema, include_candidate_id)

    header = []
    if include_candidate_id:
        header.append("candidate_id")
    header.extend(input_cols)
    header.append("_input_schema_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 로 저장하여 Excel 에서 한글이 깨지지 않도록 한다.
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # 규격 식별자를 한 행에 안내로 기록한다 (입력 값은 비움).
        guide_row = [""] * len(header)
        guide_row[-1] = schema.schema_id
        writer.writerow(guide_row)
    return output_path


def write_excel_template(
    schema,
    output_path: Path,
    *,
    include_candidate_id: bool = True,
) -> Path:
    """활성 입력 규격으로 빈 Excel 양식을 저장한다.

    - `inputs` 시트: 첫 행에 입력 식별자. 이후 한 행에 후보 하나를 기록한다.
    - `_schema` 시트: `input_schema_id`, 변수·단위·범주 안내.
    - 목표 결과 열은 넣지 않는다.
    """
    input_cols, guide_rows = _template_frames(schema, include_candidate_id)

    inputs_header = []
    if include_candidate_id:
        inputs_header.append("candidate_id")
    inputs_header.extend(input_cols)

    inputs_df = pd.DataFrame(columns=inputs_header)

    schema_df = pd.DataFrame(guide_rows)
    meta_df = pd.DataFrame([{"key": "input_schema_id", "value": schema.schema_id}])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            inputs_df.to_excel(writer, sheet_name="inputs", index=False)
            meta_df.to_excel(writer, sheet_name="_schema", index=False, startrow=0)
            schema_df.to_excel(writer, sheet_name="_schema", index=False, startrow=3)
    except Exception as exc:
        raise TemplateError("Excel 양식 저장 실패: " + str(exc)) from exc
    return output_path


def write_input_templates(
    spec,
    output_dir: Path,
    *,
    active_target_ids=None,
    basename: str = "input_template",
    include_candidate_id: bool = True,
):
    """활성 대상의 입력 규격으로 Excel·CSV 양식을 모두 생성한다.

    기존 build_input_schema 를 재사용하여 활성 대상 입력의 합집합으로 규격을 만든다.
    비활성 대상의 입력은 포함하지 않는다.

    Returns:
        (csv_path, xlsx_path, schema_id) 튜플.
    """
    from pyrolysis.config import build_input_schema

    schema = build_input_schema(spec, active_target_ids=active_target_ids)

    csv_path = output_dir / (basename + ".csv")
    xlsx_path = output_dir / (basename + ".xlsx")

    write_csv_template(schema, csv_path, include_candidate_id=include_candidate_id)
    write_excel_template(schema, xlsx_path, include_candidate_id=include_candidate_id)

    return csv_path, xlsx_path, schema.schema_id


# ── 예측 파일 파싱과 전체 파일 오류 검사 (작업 9) ─────────────────────────────


class PredictionFileError(Exception):
    """예측 파일 전체를 처리할 수 없는 수준의 오류 (형식·구조·규격 ID)."""


# 예측 파일에서 식별자·규격 열로 취급하여 입력에서 제외하는 열 이름
_RESERVED_COLS = {"candidate_id", "_input_schema_id"}


def _read_original_header(path: Path, suffix: str, sheet_name):
    """pandas 가 중복 열 이름을 바꾸기 전의 원래 헤더를 반환한다."""
    if suffix == ".csv":
        import csv as _csv
        for enc in ("utf-8-sig", "utf-8"):
            try:
                with path.open(encoding=enc, newline="") as f:
                    reader = _csv.reader(f)
                    for row in reader:
                        return list(row)
                    return []
            except UnicodeDecodeError:
                continue
        return []
    else:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=False)
        ws = wb[sheet_name] if isinstance(sheet_name, str) else wb.worksheets[sheet_name]
        header = []
        for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
            header = [("" if c is None else str(c)) for c in row]
            break
        wb.close()
        return header


def _check_excel_formulas_and_merges(path: Path, sheet_name) -> list:
    """Excel 데이터 영역의 수식·병합 셀을 검사한다. 수식을 실행하지 않는다."""
    issues = []
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=False, data_only=False)
    ws = wb[sheet_name] if isinstance(sheet_name, str) else wb.worksheets[sheet_name]

    if ws.merged_cells.ranges:
        issues.append(Issue(
            code="MERGED_CELL",
            message_ko="병합된 셀이 있습니다. 표준 양식은 병합 셀을 허용하지 않습니다.",
            severity="error",
        ))

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            v = cell.value
            if isinstance(v, str) and v.startswith("="):
                issues.append(Issue(
                    code="FORMULA_CELL",
                    message_ko="수식 셀이 있습니다. 계산된 값으로 저장하세요.",
                    severity="error",
                    row=cell.row - 1, col=str(cell.column_letter),
                ))
    wb.close()
    return issues


def parse_prediction_file(
    path: Path,
    input_schema,
    *,
    sheet_name=0,
    max_rows=None,
    max_cols=None,
    max_error_display=None,
):
    """예측 입력 파일을 읽고 전체 파일을 검사한다.

    설계 5.2 의 순서를 따른다. 모든 행의 오류를 수집하고, 오류가 하나라도 있으면
    CandidateTable 을 반환하지 않는다. 비어 있지 않은 잘못된 값을 결측으로 바꾸지 않는다.

    Args:
        path: .csv 또는 .xlsx 경로.
        input_schema: InputSchema (활성 입력 규격).
        sheet_name: Excel 시트.
        max_rows, max_cols: 처리 한도. None 이면 해당 한도 검사를 건너뛴다
            (무제한으로 간주하지 않으며, 한도 미설정 사실을 호출자가 안내한다).
        max_error_display: 화면 표시용 오류 수 제한. None 이면 전체를 반환.

    Returns:
        dict: {
            "candidates": CandidateTable | None,  # 오류 없을 때만
            "issues": list[Issue],                # 표시용 (max_error_display 적용)
            "total_error_count": int,             # 전체 오류 수
            "input_schema_id": str,
            "notes": list[str],
        }

    Raises:
        PredictionFileError: 형식·구조·규격 ID 등 파일 전체 수준 오류.
    """
    from pyrolysis.validation import validate_candidate_values

    notes: list = []
    issues: list = []

    if not path.exists():
        raise PredictionFileError("파일이 없습니다: " + str(path))

    suffix = path.suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise PredictionFileError(
            "지원하지 않는 파일 형식: " + repr(suffix) + ". .xlsx 또는 .csv 로 변환하세요."
        )

    # 1) 원래 헤더에서 중복 열 검사 (pandas 리네임 전)
    original_header = _read_original_header(path, ".csv" if suffix == ".csv" else ".xlsx", sheet_name)
    seen = set()
    dups = []
    for h in original_header:
        if h in seen and h not in dups:
            dups.append(h)
        seen.add(h)
    if dups:
        raise PredictionFileError("중복된 열 이름이 있습니다: " + str(dups))

    # 2) Excel 수식·병합 셀 검사
    if suffix in (".xlsx", ".xls"):
        issues.extend(_check_excel_formulas_and_merges(path, sheet_name))

    # 3) 본문 읽기
    try:
        if suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(path, sheet_name=sheet_name, header=0, dtype=object)
        else:
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig", dtype=object)
            except UnicodeDecodeError:
                raw = pd.read_csv(path, encoding="utf-8", dtype=object)
    except Exception as exc:
        raise PredictionFileError("파일 읽기 실패 (" + path.name + "): " + str(exc)) from exc

    # 완전히 빈 끝부분 행 제거 (일부만 채운 행은 유지)
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise PredictionFileError("데이터 행이 없습니다: " + path.name)

    # 한도 검사 (설정된 경우에만)
    if max_rows is not None and len(raw) > max_rows:
        raise PredictionFileError(
            "행 수가 한도를 초과했습니다: " + str(len(raw)) + " > " + str(max_rows)
        )
    if max_cols is not None and len(raw.columns) > max_cols:
        raise PredictionFileError(
            "열 수가 한도를 초과했습니다: " + str(len(raw.columns)) + " > " + str(max_cols)
        )
    if max_rows is None or max_cols is None:
        notes.append("행·열 처리 한도가 설정되지 않았습니다 (작업 29에서 확정).")

    file_cols = list(raw.columns)
    required_input_ids = list(input_schema.input_ids)

    # 4) 규격 ID 검사 (혼합·누락)
    schema_id_in_file = None
    if "_input_schema_id" in file_cols:
        vals = set(str(v) for v in raw["_input_schema_id"].dropna().tolist() if str(v).strip() != "")
        if len(vals) > 1:
            raise PredictionFileError("한 파일에 서로 다른 규격 식별자가 섞여 있습니다: " + str(sorted(vals)))
        if vals:
            schema_id_in_file = next(iter(vals))
    if schema_id_in_file is not None and schema_id_in_file != input_schema.schema_id:
        raise PredictionFileError(
            "양식의 규격 식별자가 현재 입력 계약과 다릅니다. 파일: " + schema_id_in_file
            + " / 필요: " + input_schema.schema_id + ". 운영자가 양식을 다시 생성해야 합니다."
        )

    # 5) 등록된 결과 열(목표)은 사용하지 않고 버림을 알림 — 호출자가 target_ids 전달 시 처리
    #    여기서는 입력 규격에 없는 열을 분류한다.
    known_non_input = _RESERVED_COLS
    unknown_cols = [c for c in file_cols if c not in required_input_ids and c not in known_non_input]
    # candidate_id, _input_schema_id 외의 미지 열은 양식 오류
    if unknown_cols:
        issues.append(Issue(
            code="UNKNOWN_COLUMN",
            message_ko="정의되지 않은 열이 있습니다: " + str(sorted(unknown_cols)),
            severity="error",
        ))

    # 필수 입력 열 누락
    missing_cols = [c for c in required_input_ids if c not in file_cols]
    if missing_cols:
        issues.append(Issue(
            code="MISSING_COLUMN",
            message_ko="필수 입력 열이 없습니다: " + str(sorted(missing_cols)),
            severity="error",
        ))

    # 6) 값 검증 (정의된 입력 열만, 열 순서 무관)
    present_input_ids = [c for c in required_input_ids if c in file_cols]
    value_df = raw[present_input_ids] if present_input_ids else raw.iloc[:, 0:0]
    issues.extend(validate_candidate_values(value_df, input_schema.input_specs))

    total_error_count = len(issues)

    # candidate_id 구성
    if "candidate_id" in file_cols:
        candidate_ids = [str(v) if str(v).strip() != "" else ("row_" + str(i + 1))
                         for i, v in enumerate(raw["candidate_id"].tolist())]
    else:
        candidate_ids = ["row_" + str(i + 1) for i in range(len(raw))]
    source_rows = list(range(1, len(raw) + 1))

    candidates = None
    if total_error_count == 0:
        candidates = CandidateTable(
            input_schema_id=input_schema.schema_id,
            source_rows=source_rows,
            candidate_ids=candidate_ids,
            data=raw[present_input_ids].reset_index(drop=True),
        )

    display_issues = issues
    if max_error_display is not None and len(issues) > max_error_display:
        display_issues = issues[:max_error_display]
        notes.append(
            "오류 " + str(total_error_count) + "개 중 " + str(max_error_display) + "개만 표시합니다."
        )

    return {
        "candidates": candidates,
        "issues": display_issues,
        "total_error_count": total_error_count,
        "input_schema_id": input_schema.schema_id,
        "notes": notes,
    }
