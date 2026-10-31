"""
test_project_structure.py — 작업 1 완료 기준 검증.

확인 항목:
1. pyrolysis 패키지와 모든 모듈을 import할 수 있다.
2. 계산 모듈이 Streamlit을 import하지 않는다.
3. pathlib.Path 가 사용 가능하다.
4. 핵심 의존성 패키지를 import할 수 있다.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


# ── 1. pyrolysis 패키지 import ────────────────────────────


def test_pyrolysis_package_importable() -> None:
    """pyrolysis 패키지를 import할 수 있어야 한다."""
    import pyrolysis

    assert pyrolysis.__version__ == "0.1.0"


def test_pyrolysis_modules_importable() -> None:
    """계산 모듈 전체를 import할 수 있어야 한다."""
    modules = [
        "pyrolysis.contracts",
        "pyrolysis.config",
        "pyrolysis.io",
        "pyrolysis.validation",
        "pyrolysis.preprocessing",
        "pyrolysis.models",
        "pyrolysis.splitting",
        "pyrolysis.training",
        "pyrolysis.evaluation",
        "pyrolysis.analysis",
        "pyrolysis.artifacts",
        "pyrolysis.inference",
        "pyrolysis.ranking",
        "pyrolysis.cli",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"{mod_name} import 실패"


# ── 2. 계산 계층의 Streamlit 비의존성 확인 ──────────────


def test_calculation_modules_do_not_import_streamlit() -> None:
    """
    계산 모듈(pyrolysis.* — ui 제외)은 Streamlit을 import하면 안 된다.

    깨끗한 하위 인터프리터에서 각 모듈을 import한 뒤 'streamlit' 이 sys.modules 에
    없는지 확인한다. 하위 프로세스를 쓰는 이유: 같은 세션의 다른 테스트(예: Streamlit
    AppTest)가 전역 sys.modules 에 streamlit 을 남길 수 있어, 테스트 순서에 영향을
    받지 않도록 격리한다. pyrolysis.ui / pyrolysis.biochar_ui 계층은 제외한다.
    """
    import subprocess

    calc_modules = [
        "pyrolysis.contracts",
        "pyrolysis.config",
        "pyrolysis.io",
        "pyrolysis.validation",
        "pyrolysis.preprocessing",
        "pyrolysis.models",
        "pyrolysis.splitting",
        "pyrolysis.training",
        "pyrolysis.evaluation",
        "pyrolysis.analysis",
        "pyrolysis.artifacts",
        "pyrolysis.inference",
        "pyrolysis.ranking",
        "pyrolysis.cli",
        # 새 Stage1/Stage2 계산 모듈 (UI 제외)
        "pyrolysis.biochar_stage1.features",
        "pyrolysis.biochar_stage1.extract",
        "pyrolysis.biochar_stage1.data_loader",
        "pyrolysis.biochar_stage1.data_validation",
        "pyrolysis.biochar_stage1.evaluation",
        "pyrolysis.biochar_stage1.training",
        "pyrolysis.biochar_stage1.inference",
        "pyrolysis.biochar_stage1.model_registry",
        "pyrolysis.biochar_stage2.schema",
        "pyrolysis.biochar_stage2.adapter",
        "pyrolysis.biochar_stage2.langmuir",
    ]
    code = (
        "import importlib, sys\n"
        f"mods = {calc_modules!r}\n"
        "for m in mods:\n"
        "    importlib.import_module(m)\n"
        "sys.exit(1 if 'streamlit' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, (
        "계산 모듈 import 과정에서 Streamlit이 로드됨. "
        "pyrolysis.ui 바깥의 모듈에서 streamlit을 import하지 않아야 한다.\n"
        + result.stdout + result.stderr
    )


# ── 3. pathlib.Path 사용 가능 ─────────────────────────────


def test_pathlib_available() -> None:
    """pathlib.Path 를 사용할 수 있어야 한다."""
    p = Path(".")
    assert p.exists()


# ── 4. 핵심 의존성 import ─────────────────────────────────


def test_core_dependencies_importable() -> None:
    """requirements.txt 의 핵심 패키지를 import할 수 있어야 한다."""
    packages = [
        "pandas",
        "numpy",
        "openpyxl",
        "pydantic",
        "yaml",       # pyyaml
        "sklearn",    # scikit-learn
        "xgboost",
        "joblib",
        "matplotlib",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"{pkg} import 실패"


def test_pandas_version() -> None:
    """pandas 2.2.x 가 설치되어 있어야 한다."""
    import pandas as pd

    major, minor, *_ = pd.__version__.split(".")
    assert int(major) == 2 and int(minor) >= 2, (
        f"pandas 2.2 이상 필요, 현재 {pd.__version__}"
    )


def test_numpy_version() -> None:
    """numpy 1.26.x 가 설치되어 있어야 한다."""
    import numpy as np

    major, minor, *_ = np.__version__.split(".")
    assert int(major) == 1 and int(minor) >= 26, (
        f"numpy 1.26 이상 필요, 현재 {np.__version__}"
    )


def test_sklearn_version() -> None:
    """scikit-learn 1.5.x 가 설치되어 있어야 한다."""
    import sklearn

    major, minor, *_ = sklearn.__version__.split(".")
    assert int(major) == 1 and int(minor) >= 5, (
        f"scikit-learn 1.5 이상 필요, 현재 {sklearn.__version__}"
    )


def test_pydantic_version() -> None:
    """pydantic 2.x 가 설치되어 있어야 한다."""
    import pydantic

    major, *_ = pydantic.__version__.split(".")
    assert int(major) == 2, (
        f"pydantic 2.x 필요, 현재 {pydantic.__version__}"
    )
