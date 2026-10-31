param(
    [string]$Python = "",
    [switch]$AllTests
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python 가상환경 없음: $Python" }
& $Python -m scripts.verify_v47 --full-checksums
if ($LASTEXITCODE -ne 0) { throw '데이터 무결성 검증 실패' }
if ($AllTests) {
    & $Python -m pytest tests -q
} else {
    # Cross-environment serialized v46 research models are NOT active UI models.
    # Test other v46 data integrity by default; rebuild locally to test those model pickles.
    & $Python -m pytest tests\test_v47_integrated.py tests\test_v46_phase8.py tests\test_v44_workflow.py tests\test_v43_scenarios.py -k "not test_saved_research_models_predict_finite" -q
}
exit $LASTEXITCODE
