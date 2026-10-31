"""Rebuild ONLY v46 research model .joblib files using the local Python environment.

Does not modify data, UI, or active/deployed models.
Avoids cross-version pickle failures involving NumPy BitGenerator modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from scripts.build_v46_phase8 import (
    DEPLOY,
    S1_CAT,
    S1_NUM,
    S2_A,
    S2_B,
    S2_CATS,
    features,
    load_inputs,
    reconcile,
    s1_model,
    s2_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(root: Path) -> dict:
    root = root.resolve()
    files = [
        root / "scripts" / "build_v46_phase8.py",
        root / "data" / "imports" / "v46_phase8" / "v45_Stage1_404_BASELINE.csv",
        root / "data" / "imports" / "v46_phase8" / "v45_Stage2_138_BASELINE.csv",
        root / "data" / "imports" / "v46_phase8" / "Phase8_source_verified_equilibrium_45.csv",
        root / "data" / "imports" / "v46_phase8" / "Stage1_new_13_source_staged.csv",
    ]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("v46 재학습에 필요한 파일 없음:\n" + "\n".join(missing))

    d1, d2, verified, s1 = load_inputs(root)
    merged1, merged2, extra2, extra1 = reconcile(d1, d2, verified, s1)
    if (len(merged1), len(merged2), len(extra1), len(extra2)) != (417, 140, 13, 2):
        raise ValueError("v46 원본 데이터의 행 개수가 달라 모델 저장을 중단했습니다.")

    specs = [
        ("Stage1", "BET_m2_g_training_candidate", "Poisson", merged1),
        ("Stage1", "TPV_cm3_g_training_candidate", "Poisson", merged1),
        ("Stage2", "target_value", "A", merged2),
        ("Stage2", "target_value", "B", merged2),
    ]
    model_dir = root / DEPLOY
    model_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    try:
        for stage, target, variant, frame in specs:
            if stage == "Stage1":
                model, columns, nums, cats = s1_model()
            else:
                model, columns, nums, cats = s2_model(variant)
            train = frame.loc[frame[target].notna()].copy()
            if train.empty:
                raise ValueError(f"{stage} {target}: 학습할 데이터가 없습니다.")

            train_x = features(train, columns, nums, cats)
            train_y = train[target].to_numpy(dtype=float)
            if not np.isfinite(train_y).all():
                raise ValueError(f"{stage} {target}: 비정상 타깃이 있습니다.")
            model.fit(train_x, train_y)
            check = train_x.head(min(3, len(train_x)))
            predictions = np.asarray(model.predict(check), dtype=float)
            if not np.isfinite(predictions).all():
                raise ValueError(f"{stage} {target}: 비유한 예측이 발견되었습니다.")
            if stage == "Stage1" and (predictions < 0).any():
                raise ValueError(f"{stage} {target}: 음수 예측이 발견되었습니다.")

            fname = f"V46_{stage}_{target}_{variant}_RESEARCH_NOT_ACTIVE.joblib"
            destination = model_dir / fname
            temporary = model_dir / (fname + ".local-rebuild-tmp")
            staged.append((temporary, destination, stage, target, variant, len(train)))
            joblib.dump(model, temporary, compress=3)

            # The crucial compatibility test: load with the *same* environment
            # used by the user's app and run predictions again.
            reloaded = joblib.load(temporary)
            result = np.asarray(reloaded.predict(check), dtype=float)
            if not np.isfinite(result).all() or not np.allclose(result, predictions, rtol=1e-6, atol=1e-8):
                raise RuntimeError(f"{fname}: 로컬 저장/복원 예측 일치 확인 실패")
            print(f"OK {fname}: {len(train)} rows, local deserialize and predictions verified")

        # Only change research model files after all four local rebuilds succeed.
        output_models = []
        for tmp, dst, stage, target, variant, count in staged:
            os.replace(tmp, dst)
            output_models.append({
                "file": dst.name, "stage": stage, "target": target,
                "variant": variant, "rows": count, "sha256": _sha256(dst)
            })

        metadata = {
            "purpose": "Rebuilt locally to fix NumPy BitGenerator pickle compatibility",
            "active_model_changed": False,
            "data_changed": False,
            "v46_stage1_rows": 417,
            "v46_stage2_rows": 140,
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "models": output_models,
            "input_source_sha256": {
                str(p.relative_to(root)).replace("\\", "/"): _sha256(p) for p in files[1:]
            },
        }
        manifest = model_dir / "V46_LOCAL_REBUILD_MANIFEST.json"
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Local environment manifest: {manifest}")
        return metadata
    finally:
        for tmp, *_ in staged:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    rebuild(args.root)
