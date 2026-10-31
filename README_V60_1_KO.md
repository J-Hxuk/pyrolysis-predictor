# v60.1 — Spearman 검증 가능 패널 수/원인 감사 수정

작성: 2026-10-31. 기존 v60 연구 모델, 학습 데이터, Streamlit/활성 모델을 변경하지 않습니다.

## 무엇이 문제였나

v60에는 제조조건 비교계열이 총 15개(9 DOI) 있지만, `paired_bootstrap()`은 두 비교 모델의 Spearman ρ 중 하나라도 `NaN`이면 해당 패널을 `dropna()`로 제외합니다. 기존 테스트는 검증 가능 패널도 15개라고 가정해, 로컬 재학습에서 한 패널이 상수 예측 등의 이유로 ρ가 정의되지 않으면 `assert (14 == 15)`로 실패했습니다. `NaN`은 상관계수 0이 아니며, 15→14 숫자만 교체하거나 누락을 묵인하면 안 됩니다.

## 바뀐 내용

- `scripts/audit_v60_effects.py`: 전체 계열 수(`n_panels_inventory`), 실제 양쪽 Spearman 유효 패널 수(`n_panels`), 제외 패널/DOI·모델·원인을 별도 보고합니다. 유효 패널로만 DOI bootstrap을 실시하고, 유효 DOI가 3개 미만이면 CI를 산출하지 않습니다.
- 새 `reports/v60_separate_effects/panel_rank_coverage.csv`: 152/188 학습 시나리오별 각 패널의 두 rho, 유효 여부, 상수 목표/상수 예측 분류.
- 새 `reports/v60_separate_effects/V60_1_LOCAL_RANK_AUDIT_KO.md`: 현재 PC 모델 기준 유효 패널 수 및 95% 구간. 기존 보고서의 값이 달라도 오류 없이 기록합니다.
- `tests/test_v60_separate_effects.py`: 전체 15개/9 DOI라는 데이터 불변조건은 유지하되 bootstrap 수는 OOF 결과에서 동적으로 계산합니다. 하나의 rho를 강제로 결측시키는 회귀 테스트를 추가했습니다.

## 설치·실행 순서

기존 `Biochar_v47_Integrated_Project` 루트에서 PowerShell 실행:

```powershell
$ErrorActionPreference = 'Stop'
$zip = '.\Biochar_v60_1_Rank_Coverage_Fix.zip'
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $zip)) { throw 'v60.1 패치 ZIP을 현재 폴더에 넣으세요.' }
if (!(Test-Path $python)) { throw '기존 .venv 경로를 확인하세요.' }
if (!(Test-Path '.\reports\v60_separate_effects\per_panel_rank_results.csv')) { throw 'v60 결과를 먼저 생성하세요.' }
New-Item -ItemType Directory -Force '.\backup_before_v60_1' | Out-Null
Copy-Item '.\scripts\audit_v60_effects.py' '.\backup_before_v60_1\audit_v60_effects.py' -Force
Copy-Item '.\tests\test_v60_separate_effects.py' '.\backup_before_v60_1\test_v60_separate_effects.py' -Force
Expand-Archive -LiteralPath $zip -DestinationPath '.' -Force
& $python -m scripts.audit_v60_effects --root .
if ($LASTEXITCODE -ne 0) { throw 'v60.1 패널 감사 실패' }
& $python -m pytest tests/test_v60_separate_effects.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'v60.1 테스트 실패' }
Get-Content '.\reports\v60_separate_effects\V60_1_LOCAL_RANK_AUDIT_KO.md' -Encoding UTF8
```

**재학습은 필요하지 않습니다.** 현재 로컬의 v60 OOF 결과를 다시 검증합니다. 보고서의 결론은 재학습된 사용자 PC 모델의 `V60_1_LOCAL_RANK_AUDIT_KO.md`를 우선해 해석하세요.

## 검증 완료

- 제작 환경 원본 v60 15/15 유효 패널: 수정 테스트 7개 통과.
- 모의 결측으로 14/15 유효 패널, 유효 DOI 8개: 누락 계열·제외 이유가 보고되는 것 확인.
- v52~v60 통합 60개 테스트 통과.

※ 위 모의 사례는 실데이터 성능 결과가 아니며, 사용자 로컬에서 누락된 정확한 DOI/패널은 로컬 감사 결과에서 확인해야 합니다.
