"""Check the v48 integrated project; never trains or overwrites models.

python -m scripts.verify_v48 [--full-checksums]
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "biochar_stage2_app.py",
    "pyrolysis/biochar_ui/stage2_pages.py",
    "pyrolysis/biochar_ui/stage1_bet_v20_page.py",
    "pyrolysis/biochar_ui/v48_material_properties_page.py",
    "pyrolysis/biochar_stage1/v48_research_bridge.py",
    "data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv",
    "data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv",
    "deployment/stage2_models/active.json",
    "deployment/stage1_models/active.json",
    "deployment/research_v46_NOT_ACTIVE/V46_Stage1_BET_m2_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
    "deployment/research_v46_NOT_ACTIVE/V46_Stage1_TPV_cm3_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _locally_rebuilt_v46_hashes(root: Path) -> dict[str, str]:
    """Accept only explicitly documented, sha-verified local model rebuilds."""
    directory = root / "deployment" / "research_v46_NOT_ACTIVE"
    manifest = directory / "V46_LOCAL_REBUILD_MANIFEST.json"
    if not manifest.is_file():
        return {}
    try:
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if report.get("active_model_changed") is not False or report.get("data_changed") is not False:
            return {}
        records = report.get("models", [])
        if len(records) != 4:
            return {}
        found = {}
        for entry in records:
            name = entry["file"]
            if not name.startswith("V46_") or not name.endswith("_RESEARCH_NOT_ACTIVE.joblib"):
                return {}
            rel = "deployment/research_v46_NOT_ACTIVE/" + name
            if rel in found:
                return {}
            found[rel] = entry["sha256"]
        return found
    except (KeyError, OSError, ValueError, TypeError):
        return {}


def verify(root: Path = ROOT, full_checksums: bool = False) -> list[str]:
    root = Path(root)
    problems = []
    for rel in REQUIRED:
        if not (root / rel).is_file():
            problems.append("MISSING: " + rel)
    try:
        from pyrolysis.biochar_stage1.extract import REQUIRED_FILES, IMPORT_SUBDIR
        from pyrolysis.biochar_stage2.v15_extract import _extracted_state, IMPORT_DIR, PACKAGE_ROOT
        for rel in REQUIRED_FILES:
            if not (root / IMPORT_SUBDIR / rel).is_file():
                problems.append("MISSING Stage 1 legacy input: " + rel)
        invalid, is_valid = _extracted_state(root / IMPORT_DIR / PACKAGE_ROOT)
        if invalid or not is_valid:
            problems.append("Stage 2 v15 source files missing or SHA mismatch: " + str(invalid))
    except Exception as e:
        problems.append("Legacy data check error: " + repr(e))
    manifest = root / "V48_MANIFEST.json"
    if not manifest.exists():
        problems.append("MISSING: V48_MANIFEST.json")
    elif full_checksums:
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))["file_hashes"]
            rebuilt = _locally_rebuilt_v46_hashes(root)
            for rel, expected in entries.items():
                f = root / rel
                if not f.is_file():
                    problems.append("MISSING SHA256 file: " + rel)
                    continue
                actual = file_hash(f)
                if actual != expected and (rel not in rebuilt or actual != rebuilt[rel]):
                    problems.append("SHA256 mismatch: " + rel)
            # Local manifest may not authorize changed active or raw files.
            if set(rebuilt).difference({
                    "deployment/research_v46_NOT_ACTIVE/" + n for n in (
                        "V46_Stage1_BET_m2_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
                        "V46_Stage1_TPV_cm3_g_training_candidate_Poisson_RESEARCH_NOT_ACTIVE.joblib",
                        "V46_Stage2_target_value_A_RESEARCH_NOT_ACTIVE.joblib",
                        "V46_Stage2_target_value_B_RESEARCH_NOT_ACTIVE.joblib",
                    )}):
                problems.append("Unexpected model in local rebuild manifest")
        except Exception as e:
            problems.append("Manifest read error: " + repr(e))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-checksums", action="store_true")
    args = parser.parse_args()
    errors = verify(full_checksums=args.full_checksums)
    print("Biochar v48: required files / extracted data / project hash check")
    if errors:
        for error in errors:
            print("ERROR:", error)
        return 1
    print("PASS: integrity only (not a guarantee of model accuracy or browser usability)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
