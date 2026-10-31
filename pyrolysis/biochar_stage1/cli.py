"""
cli.py — Stage1 BET 학습·저장·활성화 명령.

사용 (앱과 분리된 학습 단계):
  python -m pyrolysis.biochar_stage1.cli train       # 학습 + 저장 + 활성화
  python -m pyrolysis.biochar_stage1.cli evaluate     # eval37 GroupKFold 재현만
  python -m pyrolysis.biochar_stage1.cli audit        # 데이터 현황 점검만

앱 실행 시 자동 재학습하지 않는다. 학습은 이 CLI 로만 수행한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyrolysis.biochar_stage1 import data_validation, evaluation
from pyrolysis.biochar_stage1.data_loader import load_stage1_dataset
from pyrolysis.biochar_stage1.extract import extract_required_files
from pyrolysis.biochar_stage1.model_registry import Stage1Registry
from pyrolysis.biochar_stage1.training import save_bet_model, train_bet_model


def cmd_train(args) -> int:
    extract_required_files()
    ds = load_stage1_dataset()
    print(f"데이터 로드: base={len(ds.base)}, extra={len(ds.extra)}, "
          f"version={ds.dataset_version_hash}")
    model = train_bet_model(ds, run_cv=not args.no_cv)
    reg = Stage1Registry.default()
    version_dir = save_bet_model(model, reg.root)
    reg.set_active(model.version)
    print(f"학습·저장 완료: {model.version}")
    print(f"저장 위치: {version_dir}")
    if model.metadata.get("cross_validation"):
        cv = model.metadata["cross_validation"]["result"]
        print(f"탐색적 GroupKFold: R2={cv['mean_r2']:.4f}, "
              f"MAE={cv['mean_mae']:.3f}, RMSE={cv['mean_rmse']:.3f}")
    return 0


def cmd_evaluate(args) -> int:
    extract_required_files()
    ds = load_stage1_dataset()
    res = evaluation.evaluate_groupkfold(ds.base, ds.extra, model_name="RF")
    cmp = evaluation.compare_with_reference(res)
    print(json.dumps({"result": res.to_dict(), "comparison": cmp},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    extract_required_files()
    ds = load_stage1_dataset()
    audit = data_validation.audit_dataset(ds.base, ds.extra)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage1 BET 학습·평가·점검 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="학습 + 저장 + 활성화")
    p_train.add_argument("--no-cv", action="store_true", help="GroupKFold 평가 생략")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="eval37 GroupKFold 재현")
    p_eval.set_defaults(func=cmd_evaluate)

    p_audit = sub.add_parser("audit", help="데이터 현황 점검")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
