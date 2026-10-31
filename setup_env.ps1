# setup_env.ps1 — 개발 환경 설치 스크립트
# 실행: powershell -ExecutionPolicy Bypass -File setup_env.ps1

Set-Location -Path $PSScriptRoot

# Python 경로 탐색
$pythonPaths = @(
    "C:\Python312\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WindowsApps\python3.12.exe"
)

$pythonExe = $null
foreach ($p in $pythonPaths) {
    if (Test-Path $p) {
        $pythonExe = $p
        break
    }
}

if (-not $pythonExe) {
    # PATH 에서 찾기
    $found = Get-Command python3.12 -ErrorAction SilentlyContinue
    if ($found) { $pythonExe = $found.Source }
}

if (-not $pythonExe) {
    Write-Error "Python 3.12 을 찾을 수 없습니다. 설치 후 다시 실행하세요."
    exit 1
}

Write-Host "Python: $pythonExe"
& $pythonExe --version

# 가상 환경 생성
if (-not (Test-Path ".venv")) {
    Write-Host "가상 환경 생성 중..."
    & $pythonExe -m venv .venv
}

# 의존성 설치
$pip = ".\.venv\Scripts\python.exe"
Write-Host "의존성 설치 중..."
& $pip -m pip install --upgrade pip
& $pip -m pip install -r requirements.txt
& $pip -m pip install -r requirements-dev.txt
& $pip -m pip install -e . --no-deps

Write-Host "테스트 실행 중..."
& $pip -m pytest tests/ -v

Write-Host "완료."
