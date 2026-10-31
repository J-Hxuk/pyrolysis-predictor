"""
v15_cli.py — Stage2 학습·평가·추천 검증 명령 (앱과 분리된 학습 단계).

  python -m pyrolysis.biochar_stage2.v15_cli audit        # 데이터 검증
  python -m pyrolysis.biochar_stage2.v15_cli compare      # A/B/C 전수 비교
  python -m pyrolysis.biochar_stage2.v15_cli chen         # Chen16 추천 검증
  python -m pyrolysis.biochar_stage2.v15_cli train        # A·B·C 최종 모델 저장·활성화

앱은 저장된 모델만 로드하며 재학습하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys

from pyrolysis.biochar_stage2 import v15_loader as L
from pyrolysis.biochar_stage2 import v15_models as M
from pyrolysis.biochar_stage2 import v15_recommend as R
from pyrolysis.biochar_stage2.v15_extract import extract_v15_files
from pyrolysis.biochar_stage2.v15_train import Stage2Registry, save_stage2_model, train_stage2_model


def cmd_audit(args) -> int:
    extract_v15_files()
    ds = L.load_core_qe()
    print(json.dumps(L.validate_core(ds), ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args) -> int:
    extract_v15_files()
    ds = L.load_core_qe()
    print(json.dumps(M.compare_models(ds.df), ensure_ascii=False, indent=2))
    return 0


def cmd_chen(args) -> int:
    extract_v15_files()
    chen = L.load_chen16()
    print(json.dumps(R.recommend_benchmark(chen), ensure_ascii=False, indent=2))
    return 0


def cmd_train(args) -> int:
    extract_v15_files()
    ds = L.load_core_qe()
    reg = Stage2Registry.default()
    saved = {}
    # A·B·C 를 공통 알고리즘(기본 XGBoost)으로 저장. B 를 사용자 핵심(활성)으로.
    algo = args.algo
    for model_name in ("A", "B", "C"):
        trained = train_stage2_model(model_name, algo, ds=ds, run_cv=True)
        vdir = save_stage2_model(trained, reg.root)
        saved[model_name] = vdir.name
        print(f"saved {model_name}: {vdir.name}")
    # 활성 모델: Model B (제조조건 직접 예측, 사용자 배포 핵심)
    reg.set_active(saved["B"])
    print("active:", saved["B"])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage2 v15 CLI")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("compare").set_defaults(func=cmd_compare)
    sub.add_parser("chen").set_defaults(func=cmd_chen)
    pt = sub.add_parser("train")
    pt.add_argument("--algo", default="XGBoost", help="알고리즘 (기본 XGBoost)")
    pt.set_defaults(func=cmd_train)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
