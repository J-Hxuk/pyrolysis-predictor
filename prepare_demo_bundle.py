r"""기존 Task 3·4 함수로 로컬 화면 점검용 demo 묶음을 생성한다.

이 파일을 streamlit_app.py와 같은 폴더에 놓고 다음과 같이 실행한다.
    .\.venv\Scripts\python.exe .\prepare_demo_bundle.py --demo

실제 연구 자료나 실제 연구 설정을 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 UI 점검용 demo 묶음 생성")
    parser.add_argument("--demo", action="store_true", help="합성 자료 사용을 명시")
    args = parser.parse_args()
    if not args.demo:
        parser.error("합성 자료로 점검하려면 --demo를 지정하세요.")

    root = Path(__file__).resolve().parent
    required = ("streamlit_app.py", "configs/demo.yaml", "pyrolysis/artifacts.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(
            "이 파일을 프로젝트의 streamlit_app.py와 같은 폴더에 놓으세요.\n"
            "찾지 못한 파일: " + ", ".join(missing)
        )

    from pyrolysis.artifacts import DEMO_BUNDLE_MARKER, load_bundle, save_demo_bundle
    from pyrolysis.config import load_problem
    from pyrolysis.inference import make_single_candidate_table, predict_all
    from pyrolysis.training import make_demo_data, run_demo_training

    bundle_id = "demo_local_ui"
    output_dir = root / "deployment" / "bundles"
    destination = output_dir / bundle_id
    if destination.exists():
        raise SystemExit(
            f"같은 이름의 묶음이 이미 있습니다: {destination}\n"
            "기존 파일은 덮어쓰지 않았습니다. 이전 생성이 성공했다면 앱 실행으로 진행하세요."
        )

    spec = load_problem(root / "configs" / "demo.yaml")
    print("[1/3] 합성 자료로 demo 학습 실행")
    trained = run_demo_training(spec, root / "runs")

    print("[2/3] 앱이 읽는 위치에 모델 묶음 저장")
    bundle_dir = save_demo_bundle(
        pipeline=trained["pipeline"],
        input_specs=spec.inputs,
        target_spec=spec.targets[0],
        output_dir=output_dir,
        bundle_id=bundle_id,
        disclaimer=trained["disclaimer"],
    )

    print("[3/3] 저장한 묶음을 다시 읽고 단일 예측 확인")
    bundle = load_bundle(bundle_dir)
    if bundle["contract"]["bundle_type"] != DEMO_BUNDLE_MARKER:
        raise RuntimeError("저장한 묶음의 demo 표지가 일치하지 않습니다.")

    X, _ = make_demo_data(spec.inputs, spec.targets)
    candidates = make_single_candidate_table(X.iloc[0].to_dict(), "demo_schema")
    batch = predict_all(bundle, candidates)
    if not batch.completed or batch.error:
        raise RuntimeError(f"저장 후 예측 실패: {batch.error}")

    before = float(trained["pipeline"].predict(X.iloc[:1])[0])
    after = float(batch.predictions[spec.targets[0].id].values[0])
    # 제공된 test_demo_roundtrip.py와 같은 합성 시험 허용 오차를 사용한다.
    if not math.isclose(before, after, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("저장 전후 예측 결과가 시험 허용 오차를 벗어납니다.")

    print(f"[완료] demo 묶음: {bundle_dir}")
    print("생성 파일: model.joblib, contract.json, environment.json")
    print("저장 후 다시 읽기와 단일 예측 확인 완료.")
    print("합성 자료를 이용한 기능 점검입니다. 실제 연구 성능 검증이 아닙니다.")
    print("Streamlit 앱을 실행해 입력창과 결과 표시를 확인하세요.")


if __name__ == "__main__":
    main()
