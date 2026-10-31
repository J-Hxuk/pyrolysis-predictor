# v58 패치 설치 및 실행 (v47 통합 프로젝트용)

**성격:** v57 데이터를 사용해 원인 분석, cap 특성 실험, strict-top-tie 추천 감사, 연구용 모델 4개를 생성하는 증분 패치. 배포 활성 모델은 변경하지 않음.

## PowerShell (Biochar_v47_Integrated_Project 최상위 폴더에서 실행)

```powershell
$ErrorActionPreference = "Stop"
$zip = ".\Biochar_v58_Failure_Analysis_Cap_Feature_v47_Patch.zip"
$python = Join-Path ([Environment]::GetFolderPath("Desktop")) "대학\학술대회\프로그램\.venv\Scripts\python.exe"
if (!(Test-Path $zip)) { throw "v58 ZIP이 현재 프로젝트 폴더에 없습니다." }
if (!(Test-Path $python)) { throw "기존 .venv Python을 찾을 수 없습니다." }
if (!(Test-Path ".\data\imports\v57_handoff_sources\Stage2_New36_Harmonized_RESEARCH.csv")) { throw "v57 데이터가 없습니다. v57 먼저 설치하세요." }
if (Test-Path ".\scripts\run_v58_failure_diagnostics.py") { throw "v58 코드가 이미 있습니다. 덮어쓰기 전에 확인하세요." }
Expand-Archive -LiteralPath $zip -DestinationPath "."
& $python -m scripts.run_v58_failure_diagnostics --root .
if ($LASTEXITCODE -ne 0) { throw "v58 분석·재학습 실패" }
& $python -m pytest tests/test_v58_failure_diagnostics.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw "v58 테스트 실패" }
```

`reports/v58_failure_diagnostics/RESULTS_KO.md`에서 결과를 확인. 이 연구 결과는 동일 기존 DOI에서 탐색적 반복 평가를 수행한 것이므로 확인적 성능 검증이 아님.

### 기존 모델 유지
`deployment/research_v58_NOT_ACTIVE`의 모델은 4개 연구용이며 Streamlit 활성 배포에 자동 연결되지 않는다. 특히 Chen 16조건에서는 12개 공동 1위로 단일 최적 조건 추천 불가.
