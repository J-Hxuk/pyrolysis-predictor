"""Read-only Stage2 environment diagnostic. Usage: python scripts/check_stage2_environment.py"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run():
    import importlib.metadata as md
    packages = ("scikit-learn", "xgboost", "numpy", "pandas", "joblib", "streamlit")
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"
    report = {"python": sys.version.split()[0], "package_versions": versions, "stage2_models": {},
              "note": "Read-only check; files are not changed or retrained."}
    from pyrolysis.biochar_stage2.v15_train import Stage2Registry
    from pyrolysis.biochar_stage2.v15_loader import load_core_qe
    from pyrolysis.biochar_stage2.input_support import predict_from_contract
    reg = Stage2Registry.default(REPO)
    row = load_core_qe(REPO).df.iloc[0].to_dict()
    for name in reg.list_versions():
        try:
            model = reg.load(name)
            pred = predict_from_contract(model, row)
            report["stage2_models"][name] = {"status": "PASS", "raw_qe": pred[0],
                                               "capped_qe": pred[1]}
        except Exception as exc:
            report["stage2_models"][name] = {"status": "FAIL", "exception": type(exc).__name__,
                                               "details": str(exc)[:600]}
    status = "PASS" if report["stage2_models"] and all(x["status"] == "PASS"
               for x in report["stage2_models"].values()) else "FAIL"
    report["overall_status"] = status
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run())
