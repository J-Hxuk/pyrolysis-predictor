"""
cli.py - 공통 서비스의 명령행 진입점.

하위 명령 (노트북과 같은 공통 함수를 호출하며 알고리즘을 복제하지 않는다):
  audit  --dataset <file>     학습 파일 자료 점검 (작업 7)
  train  --demo               합성 설정으로 demo 학습 실행 (작업 3)
  report --demo               후보 비교·예측실측·변수 기여도 보고서 (작업 15)
  export --demo               배포 묶음 저장·재읽기 점검 (작업 4/16)
  make-template               활성 입력 규격으로 Excel·CSV 양식 생성 (작업 8)
  check-bundle --bundle <d>   묶음·환경 무결성 검사 (작업 17)
  prepare-runtime             runtime.yaml 에서 Streamlit 설정 생성 (작업 17)
  activate --bundle <d>       검사 통과 묶음을 active 포인터로 교체 (작업 17)

표준 실행 순서(합성 설정 점검):
  audit → train → report → export → make-template → check-bundle
  → prepare-runtime → activate
실제 설정 누락 시 각 명령은 필요한 항목을 안내한다.

Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    if args.demo:
        from pyrolysis.config import load_problem
        from pyrolysis.training import run_demo_training

        demo_path = Path(args.problem) if args.problem else Path("configs/demo.yaml")
        output_dir = Path(args.output) if args.output else Path("runs")

        print("[train --demo] 설정 파일: " + str(demo_path))
        demo_spec = load_problem(demo_path)
        result = run_demo_training(demo_spec, output_dir)

        print("[train --demo] run_id   : " + result["run_id"])
        print("[train --demo] 학습 행  : " + str(result["n_train"]))
        print("[train --demo] 평가 행  : " + str(result["n_eval"]))
        print("[train --demo] MAE      : " + str(round(result["mae"], 4)))
        print("[train --demo] RMSE     : " + str(round(result["rmse"], 4)))
        print("[train --demo] 기록 파일: " + result["record_path"])
        print("[train --demo] *** " + result["disclaimer"] + " ***")
        return 0
    else:
        print("train: --demo 없이 실제 학습은 작업 13에서 구현됩니다.")
        return 1


def _cmd_audit(args: argparse.Namespace) -> int:
    from pyrolysis.config import load_problem
    from pyrolysis.io import DatasetReadError, audit_dataset, read_training_file

    problem_path = Path(args.problem) if args.problem else Path("configs/demo.yaml")
    dataset_path = Path(args.dataset) if args.dataset else None

    if dataset_path is None:
        print("audit: --dataset 인자가 필요합니다.")
        return 1

    id_columns = []
    if args.id_columns:
        id_columns = [c.strip() for c in args.id_columns.split(",") if c.strip()]

    spec = load_problem(problem_path)

    try:
        data = read_training_file(
            dataset_path, spec,
            sheet_name=args.sheet if args.sheet is not None else 0,
            id_columns=id_columns,
        )
    except DatasetReadError as exc:
        print("[audit] 파일 읽기 실패: " + str(exc))
        return 1

    report = audit_dataset(data, spec)

    print("")
    print("[audit] 파일: " + report.source_path)
    print("  전체 행      : " + str(report.n_total_rows))
    print("  반복 측정 후보: " + str(report.n_replicated_rows) + "행 (" + str(report.replicated_input_groups) + "그룹)")
    if report.excluded_columns:
        print("  제외 열      : " + str(report.excluded_columns))

    if report.notes:
        print("")
        print("  참고:")
        for note in report.notes:
            print("    - " + note)

    print("")
    print("  대상별 가용 행:")
    for tr in report.target_results:
        parts = [str(tr.n_available) + "/" + str(tr.n_total) + "행 사용 가능"]
        if tr.n_missing > 0:
            parts.append("결측 " + str(tr.n_missing) + "행")
        if tr.task == "regression" and tr.n_zero > 0:
            parts.append("수치 0: " + str(tr.n_zero) + "행")
        if tr.task == "binary_classification" and tr.class_counts:
            parts.append("클래스 분포: " + str(tr.class_counts))
        if tr.n_replicated_available > 0:
            parts.append("반복 측정 후보 " + str(tr.n_replicated_available) + "행")
        print("    " + tr.target_id + " [" + tr.task + "]: " + ", ".join(parts))

    return 0


def _cmd_make_template(args: argparse.Namespace) -> int:
    from pyrolysis.config import ConfigError, load_problem
    from pyrolysis.io import TemplateError, write_input_templates

    problem_path = Path(args.problem) if args.problem else Path("configs/demo.yaml")
    output_dir = Path(args.output) if args.output else Path("templates/local")

    active_target_ids = None
    if args.targets:
        active_target_ids = [t.strip() for t in args.targets.split(",") if t.strip()]

    spec = load_problem(problem_path)

    try:
        csv_path, xlsx_path, schema_id = write_input_templates(
            spec, output_dir,
            active_target_ids=active_target_ids,
            basename=args.basename,
        )
    except ConfigError as exc:
        print("[make-template] 규격 구성 실패: " + str(exc))
        return 1
    except TemplateError as exc:
        print("[make-template] 양식 생성 실패: " + str(exc))
        return 1

    print("[make-template] 규격 식별자: " + schema_id)
    print("[make-template] CSV 양식  : " + str(csv_path))
    print("[make-template] Excel 양식: " + str(xlsx_path))
    if active_target_ids:
        print("[make-template] 활성 대상 : " + ", ".join(active_target_ids))
    else:
        print("[make-template] 활성 대상 : (전체)")
    return 0

def _cmd_report(args: argparse.Namespace) -> int:
    """로컬 연구 보고서 생성 (작업 15 공통 함수 호출).

    실제 연구 설정이 없을 때는 --demo 로 합성 설정을 사용한다. 노트북
    03_review_export.ipynb 과 같은 공통 함수(run_demo_training, analysis.save_*)를
    호출하여 같은 산출물을 만든다. 알고리즘을 복제하지 않는다.
    """
    if not args.demo:
        print("report: 실제 설정 보고서는 작업 28에서 확정합니다. "
              "지금은 --demo 로 합성 설정의 보고서 경로를 점검하세요.")
        return 1

    from pyrolysis.analysis import (
        compute_variable_importance,
        save_candidate_comparison,
        save_pred_vs_actual,
        save_variable_importance,
    )
    from pyrolysis.config import load_problem
    from pyrolysis.training import run_demo_training

    problem_path = Path(args.problem) if args.problem else Path("configs/demo.yaml")
    output_root = Path(args.output) if args.output else Path("runs")

    spec = load_problem(problem_path)
    result = run_demo_training(spec, output_root)
    run_dir = Path(result["run_dir"])
    report_dir = run_dir / "report"

    # 예측·실측 비교 (회귀 대상)
    from pyrolysis.training import make_demo_data
    from sklearn.model_selection import train_test_split

    X, y = make_demo_data(spec.inputs, spec.targets, n_samples=100, random_seed=42)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    pipeline = result["pipeline"]
    y_pred = pipeline.predict(X_eval)
    target_id = result["target_id"]

    pva_path = save_pred_vs_actual(y_eval, y_pred, report_dir, target_id,
                                   unit=result.get("unit"))

    # 후보 비교표 (demo 는 LinearRegression 단일 후보)
    cmp_rows = [{"candidate": result["model"], "score": result["mae"],
                 "ran": True, "reason": ""}]
    cmp_path = save_candidate_comparison(cmp_rows, report_dir, target_id,
                                         metric_name="mae", metric_unit=result.get("unit"))

    # 변수 기여도 (명시적 permutation 방법)
    imp = compute_variable_importance(pipeline, X_eval, y_eval,
                                      method="permutation", n_repeats=5, random_seed=42)
    jpath, ppath = save_variable_importance(imp, report_dir, target_id)

    print("[report --demo] run_id      : " + result["run_id"])
    print("[report --demo] 보고서 위치 : " + str(report_dir))
    print("[report --demo] 예측·실측   : " + str(pva_path))
    print("[report --demo] 후보 비교표 : " + str(cmp_path))
    print("[report --demo] 변수 기여도 : " + str(jpath) + ", " + str(ppath))
    print("[report --demo] *** " + result["disclaimer"] + " ***")
    print("[report --demo] " + imp["disclaimer"])
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """배포 묶음 내보내기.

    --demo 는 작업 4 의 demo 묶음 저장(save_demo_bundle)을 사용한다. 실제 연구용
    완전 묶음(manifest·SHA-256·환경 검사)은 작업 16 export_bundle 로 생성하며
    평가 완료 대상이 필요하므로 작업 28/29 에서 실제 설정으로 수행한다.
    """
    if not args.demo:
        print("export: 실제 연구용 완전 묶음은 작업 28/29 에서 실제 설정으로 생성합니다. "
              "지금은 --demo 로 합성 묶음 저장·재읽기 경로를 점검하세요.")
        return 1

    from pyrolysis.artifacts import DEMO_BUNDLE_MARKER, load_bundle, save_demo_bundle
    from pyrolysis.config import load_problem
    from pyrolysis.training import run_demo_training

    problem_path = Path(args.problem) if args.problem else Path("configs/demo.yaml")
    runs_root = Path(args.output) if args.output else Path("runs")
    bundle_root = Path(args.bundle_dir) if args.bundle_dir else Path("deployment/bundles")
    bundle_id = args.bundle_id or "demo_cli_export"

    spec = load_problem(problem_path)
    result = run_demo_training(spec, runs_root)
    pipeline = result["pipeline"]
    target_spec = spec.targets[0]

    disclaimer = result["disclaimer"]
    bundle_dir = save_demo_bundle(
        pipeline=pipeline,
        input_specs=[s for s in spec.inputs if s.id in target_spec.input_ids],
        target_spec=target_spec,
        output_dir=bundle_root,
        bundle_id=bundle_id,
        disclaimer=disclaimer,
    )

    # 저장 후 다시 읽어 확인
    reloaded = load_bundle(bundle_dir)
    btype = reloaded["contract"]["bundle_type"]

    print("[export --demo] 묶음 위치 : " + str(bundle_dir))
    print("[export --demo] 묶음 유형 : " + btype)
    print("[export --demo] 다시 읽기 : 성공 (대상 " + reloaded["contract"]["target_id"] + ")")
    if btype == DEMO_BUNDLE_MARKER:
        print("[export --demo] *** demo 묶음입니다. 연구용 활성화 경로는 이를 거부합니다. ***")
    return 0


def _cmd_check_bundle(args: argparse.Namespace) -> int:
    from pyrolysis.artifacts import check_bundle

    if not args.bundle:
        print("check-bundle: --bundle 인자가 필요합니다.")
        return 1
    result = check_bundle(Path(args.bundle))
    print("[check-bundle] 무결성: " + ("OK" if result["integrity_ok"] else "실패"))
    print("[check-bundle] 환경  : " + ("OK" if result["env_ok"] else "불일치"))
    for e in result["errors"]:
        print("  - " + e)
    if result["ok"]:
        print("[check-bundle] 통과")
        return 0
    print("[check-bundle] 활성화할 수 없습니다. 위 사유를 확인하세요.")
    return 1


def _cmd_prepare_runtime(args: argparse.Namespace) -> int:
    from pyrolysis.config import RuntimeConfigError, prepare_runtime_config

    runtime_path = Path(args.runtime) if args.runtime else Path("configs/runtime.yaml")
    config_path = Path(args.config) if args.config else Path(".streamlit/config.toml")

    try:
        result = prepare_runtime_config(
            runtime_path, config_path,
            require_all_limits=not args.allow_missing_limits,
        )
    except RuntimeConfigError as exc:
        print("[prepare-runtime] 실패: " + str(exc))
        return 1

    print("[prepare-runtime] config.toml 생성: " + str(config_path))
    if result["max_upload_size_mib"] is not None:
        print("[prepare-runtime] maxUploadSize = " + str(result["max_upload_size_mib"]) + " MiB")
    else:
        print("[prepare-runtime] maxUploadSize 미설정 (무제한 아님, 작업 29에서 확정)")
    if result["missing"]:
        print("[prepare-runtime] 미설정 한도: " + str(result["missing"]))
    return 0


def _cmd_activate(args: argparse.Namespace) -> int:
    from pyrolysis.artifacts import check_bundle, write_active_pointer

    if not args.bundle:
        print("activate: --bundle 인자가 필요합니다.")
        return 1
    bundle_dir = Path(args.bundle)
    active_path = Path(args.active) if args.active else Path("deployment/active.json")

    # 활성화 전 종합 검사. 실패 시 기존 포인터를 건드리지 않는다.
    result = check_bundle(bundle_dir)
    if not result["ok"]:
        print("[activate] 묶음 검사 실패로 활성화를 중단합니다. 기존 포인터는 유지됩니다.")
        for e in result["errors"]:
            print("  - " + e)
        return 1

    pointer = write_active_pointer(active_path, bundle_dir)
    print("[activate] 활성 묶음: " + pointer["bundle_id"])
    print("[activate] manifest 해시: " + pointer["manifest_hash"])
    print("[activate] 자동 push·배포는 수행하지 않습니다.")
    return 0

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyrolysis.cli",
        description="pyrolysis 공통 CLI",
    )
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="학습 실행")
    p_train.add_argument("--demo", action="store_true")
    p_train.add_argument("--problem", default=None)
    p_train.add_argument("--output", default="runs")

    p_audit = sub.add_parser("audit", help="학습 파일 자료 점검")
    p_audit.add_argument("--problem", default=None)
    p_audit.add_argument("--dataset", default=None)
    p_audit.add_argument("--id-columns", dest="id_columns", default=None)
    p_audit.add_argument("--sheet", default=None)

    p_tmpl = sub.add_parser("make-template", help="활성 입력 규격으로 Excel·CSV 양식 생성")
    p_tmpl.add_argument("--problem", default=None)
    p_tmpl.add_argument("--output", default="templates/local")
    p_tmpl.add_argument("--targets", default=None, help="활성 대상 ID (쉼표 구분). 생략 시 전체")
    p_tmpl.add_argument("--basename", default="input_template")

    p_check = sub.add_parser("check-bundle", help="묶음·환경 무결성 검사")
    p_check.add_argument("--bundle", default=None, help="묶음 디렉터리 경로")

    p_prep = sub.add_parser("prepare-runtime", help="runtime.yaml 에서 Streamlit 설정 생성")
    p_prep.add_argument("--runtime", default=None)
    p_prep.add_argument("--config", default=None)
    p_prep.add_argument("--allow-missing-limits", dest="allow_missing_limits",
                        action="store_true", help="한도 누락을 허용 (연구 모드 비권장)")

    p_act = sub.add_parser("activate", help="검사 통과 묶음을 active 포인터로 교체")
    p_act.add_argument("--bundle", default=None, help="묶음 디렉터리 경로")
    p_act.add_argument("--active", default=None, help="active.json 경로")

    p_report = sub.add_parser("report", help="로컬 연구 보고서 생성 (후보 비교·예측실측·변수 기여도)")
    p_report.add_argument("--demo", action="store_true", help="합성 설정으로 보고서 경로 점검")
    p_report.add_argument("--problem", default=None)
    p_report.add_argument("--output", default="runs")

    p_export = sub.add_parser("export", help="배포 묶음 내보내기")
    p_export.add_argument("--demo", action="store_true", help="합성 묶음 저장·재읽기 점검")
    p_export.add_argument("--problem", default=None)
    p_export.add_argument("--output", default="runs", help="학습 기록 저장 상위 디렉터리")
    p_export.add_argument("--bundle-dir", dest="bundle_dir", default=None,
                          help="묶음 저장 상위 디렉터리")
    p_export.add_argument("--bundle-id", dest="bundle_id", default=None,
                          help="묶음 ID (기본: demo_cli_export)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        sys.exit(_cmd_train(args))
    elif args.command == "audit":
        sys.exit(_cmd_audit(args))
    elif args.command == "make-template":
        sys.exit(_cmd_make_template(args))
    elif args.command == "check-bundle":
        sys.exit(_cmd_check_bundle(args))
    elif args.command == "prepare-runtime":
        sys.exit(_cmd_prepare_runtime(args))
    elif args.command == "activate":
        sys.exit(_cmd_activate(args))
    elif args.command == "report":
        sys.exit(_cmd_report(args))
    elif args.command == "export":
        sys.exit(_cmd_export(args))
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        print(args.command + ": 이후 작업에서 구현 예정입니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()