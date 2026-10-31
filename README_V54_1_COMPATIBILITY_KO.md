# Stage2 v54.1 — 로컬 버전 재학습 결과 검증 패치

목적: Windows `.venv`에서 sklearn 1.5.2 등을 사용하여 재학습한 경우, 이전 Linux sklearn 1.8.0에서 나온 **고정 MAE 값을 그대로 assertion한 테스트 2건**을 수정합니다. 과거 수치를 임의로 덮거나 성능 하락을 숨기지 않습니다.

## 파일
- `tests/test_v53_evidence_tier_refit.py` — v53 OOF/MAE/DOI 분할 검증, Joblib 로딩, 비활성 상태 테스트
- `tests/test_v54_bounded_rank.py` — 동일 OOF/보고서·모델 SHA256 일치 및 모델 재로드 테스트
- `scripts/audit_v54_local_consistency.py` — v53↔v54 OOF 일치, 라이브러리 버전 확인, 데이터 SHA, 물질수지, 현행 MAE와 과거 MAE 차이를 별도 JSON으로 기록

## PowerShell: v47 최상위 폴더에서 실행

```powershell
$ErrorActionPreference = 'Stop'
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw '기존 Python 가상환경을 확인하세요.' }
$zip = '.\Biochar_v54_1_Local_Repro_Fix_20261031.zip'
if (!(Test-Path $zip)) { throw 'v54.1 패치 ZIP을 v47 폴더에 옮기세요.' }
# 테스트 원본 백업
$backup = '.\v54_1_tests_backup'
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item .\tests\test_v53_evidence_tier_refit.py $backup
Copy-Item .\tests\test_v54_bounded_rank.py $backup
# 테스트 파일 2개 및 감사 코드만 교체
Expand-Archive -LiteralPath $zip -DestinationPath '.' -Force
# 두 버전 보고서·Joblib이 동일 현재 환경으로 생성되어야 함 (순서 필수)
& $python -m scripts.run_v53_research_refit --root .
if ($LASTEXITCODE -ne 0) { throw 'v53 재학습 실패, 이후 단계 중단' }
& $python -m scripts.run_v54_bounded_rank --root .
if ($LASTEXITCODE -ne 0) { throw 'v54 재학습 실패, 이후 단계 중단' }
& $python -m scripts.audit_v54_local_consistency --root .
if ($LASTEXITCODE -ne 0) { throw 'OOF/보고서/모델 버전 일관성 감사 실패' }
& $python -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py tests/test_v54_bounded_rank.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'pytest 실패' }
```

## 해석상 주의
- 기존 Linux MAE 17.2468와 현재 Windows에서 보고한 17.9125의 차이는 0.6657 mg/g이다. 이는 실측값 변화의 증거가 아니다. 원인 후보는 sklearn/xgboost/Python 조합, 수치정밀도/스레드/라이브러리 동작 등이며, **정확한 원인 분리는 별도의 동일 버전 통제 실험 필요**.
- 기존 v53/v54 성능 수치는 과거 실행의 참고값으로 남기고 로컬 결과로 소급 덮어쓰지 않는다.
- 코드 검증은 보고서·OOF·model checksum의 *내부 일관성*만 검증한다. 예측 성능 우월성/독립 외부 검증을 입증하지 않는다.
- 승인 데이터, 활성 모델, Stage1, v52 데이터는 수정하지 않는다.
- 로컬 테스트 통과 후에도 scikit-learn/XGBoost 버전 변경 등으로 발생한 성능 변동 자체는 학술 발표에서 숨기지 않는다.
