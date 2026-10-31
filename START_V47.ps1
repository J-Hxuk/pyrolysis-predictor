param(
    [string]$Python = "",
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "가상환경 Python을 찾을 수 없습니다: $Python" -ForegroundColor Yellow
    Write-Host '처음 실행 시: py -3.12 -m venv .venv'
    Write-Host '그다음: .\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest'
    Write-Host '이미 사용 중인 환경이 있으면 -Python "C:\...\.venv\Scripts\python.exe" 옵션을 지정하세요.'
    exit 1
}
& $Python -m scripts.verify_v47
if ($LASTEXITCODE -ne 0) { throw '필수 데이터·모델 파일 확인 실패. 원문 자료를 임의로 생성하지 않습니다.' }
if ($CheckOnly) { exit 0 }
& $Python -m streamlit run biochar_stage2_app.py
