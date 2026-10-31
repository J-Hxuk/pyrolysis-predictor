# v55 원문 전체 누적 + 재학습 패치 (v47 프로젝트용)

- 다운로드 ZIP을 `Biochar_v47_Integrated_Project` 최상위 폴더에서 압축 해제.
- **기존 v47 원본 자료/활성 모델/외부 경로 수정 없음.** `data/imports/v55_*`, `reports/v55_*`, `deployment/research_v55_NOT_ACTIVE`, scripts와 tests만 추가.
- 신규 논문 원문 5편 + 보충 PDF 2개를 `data/imports/v55_original_sources/pdfs`에 보존.
- 원문 수치가 기존 학습 feature를 만족하지 않아도 출처/단위/결측/충돌을 기록하며, 가능한 타깃끼리 별도 재학습. qmax와 qe 혼합 금지.
- `reports/v55_cumulative_source_refit/RESULTS_KO.md`가 핵심 성능 분석 및 한계.
- 모델 12개 Stage2 (A/B×6 증량 시나리오), Stage1 3개(BET 전체 29, BET 선택 14, TPV 전체 10), qmax 2개 (별도) 저장.
- 기존 140행·32 DOI 기반 고정 평가와 Chen 16조건 추천 비교. 모두 연구용, 배포 승격 안 함.

## 기존 사용자 PC 가상환경 재사용

v47 폴더에서 PowerShell 실행:

```powershell
$ErrorActionPreference = 'Stop'
$desktop = [Environment]::GetFolderPath('Desktop')
$python = Join-Path $desktop '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw '기존 .venv Python 경로를 확인하세요.' }
& $python -m scripts.run_v55_source_cumulative_refit --root .
if ($LASTEXITCODE -ne 0) { throw 'v55 학습 실패' }
& $python -m scripts.run_v55_robustness_audits --root .
if ($LASTEXITCODE -ne 0) { throw 'v55 후속 검증 실패' }
& $python -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py tests/test_v54_bounded_rank.py tests/test_v55_source_cumulative_refit.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw '검증 실패' }
```

`.venv`는 사용자가 이전에 설치한 경로를 그대로 사용한다. 버전 차이로 Joblib 로드 경고가 발생하면 기존 환경에서 위 스크립트를 재실행해 연구용 모델 파일을 로컬 라이브러리로 재생성한다. **학습 후 결과 숫자는 개발 환경과 다를 수 있으므로 로컬 OOF 기준으로 해석.**

주의: 이번 패치는 기존 모델과 분리된 연구용 파일만 포함한다. Streamlit 활성 예측 UI에 자동 연결하지 않는다. `C0` 불확실성이 있는 Cocoa 및 평형 미확정 16행 포함 시나리오는 민감도 분석이다.
