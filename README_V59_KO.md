# v59 연구용 DOI·제조시료 가중치 패치 — v47 통합 프로젝트

## 성격

기존 v57 188행·v55 152행 데이터에서 DOI 및 동일 제조시료의 반복 관측에 적용되는 학습 가중치 효과를 **실제 재학습**하고, 공통 기존140 DOI GroupKFold·새 DOI source-out·Chen 조건 순위에 대해 비교한다.

- v58의 `C0/dose` 특성 사용 실험과 원래 특성 사용 실험을 모두 수행한다.
- 기존 `data/`, 활성 `deployment/` 모델, 앱 설정을 수정하지 않는다. 연구 모델은 `deployment/research_v59_NOT_ACTIVE`에 저장된다.
- 패치에 포함된 Joblib 6개는 Python 3.13.5, scikit-learn 1.8.0, XGBoost 3.1.3에서 저장되었다. 사용자 PC의 이전 scikit-learn 1.5.2 환경에서는 **동봉된 스크립트를 이용해 재학습**할 것.
- `reports/v59_doi_material_weighting/RESULTS_KO.md`는 이 환경에서 얻은 결과. 로컬 환경과 수치가 다를 수 있다. 검증 기준은 로컬 재학습 후 같은 평가행 OOF를 재계산한 결과다.

## PowerShell — `Biochar_v47_Integrated_Project` 최상위에서 실행

```powershell
$ErrorActionPreference = "Stop"
$zip = ".\Biochar_v59_DOI_Material_Weighting_Actual_Refit_v47_Patch.zip"
$python = Join-Path ([Environment]::GetFolderPath("Desktop")) "대학\학술대회\프로그램\.venv\Scripts\python.exe"

if (!(Test-Path $zip)) { throw "v59 패치 ZIP을 프로젝트 폴더에 넣으세요." }
if (!(Test-Path $python)) { throw "기존 프로그램 .venv를 찾을 수 없습니다." }
if (!(Test-Path ".\scripts\run_v58_failure_diagnostics.py")) { throw "v58을 먼저 설치해야 합니다." }
if (!(Test-Path ".\data\imports\v57_handoff_sources\Stage2_New36_Harmonized_RESEARCH.csv")) { throw "v57 학습 데이터가 없습니다." }
if (Test-Path ".\reports\v59_doi_material_weighting\run_manifest.json") { throw "이미 v59 결과가 존재합니다. 백업 후 진행하세요." }

Expand-Archive -LiteralPath $zip -DestinationPath "."

& $python -m scripts.run_v59_doi_material_weighting --root .
if ($LASTEXITCODE -ne 0) { throw "v59 가중치 재학습 실패" }

& $python -m pytest tests/test_v59_doi_material_weighting.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw "v59 검증 실패" }
```

## 주요 결과

- DOI 전체를 묶는 5-fold 분할에서 140행을 공통 평가. 각 fold의 train에만 나머지 신규 관측을 추가.
- Model B + cap에서 152행 MAE 16.547 → 15.892 (DOI sqrt + 시료 균등) mg/g.
- Model B + cap에서 188행 MAE 17.124 → 16.175 (DOI sqrt + 행 균등) mg/g.
- 신규 DOI Shen에서는 가중치 적용 후 예측 악화. 제조조건 추천에서는 공동 1위 다수 남음.
- 이는 반복 연구에 따른 **탐색적 결과**이지 확증적 독립 검증이 아니므로 활성 모델 교체 불가.

## 산출물

- `scripts/run_v59_doi_material_weighting.py`: 데이터 검증·가중치·학습·평가·모델 저장
- `tests/test_v59_doi_material_weighting.py`: 가중치 불변식·원본 SHA·OOF 재계산·모델 로드 테스트 6개
- `reports/v59_doi_material_weighting/`: 재현 가능한 지표 및 문서
- `deployment/research_v59_NOT_ACTIVE/`: 연구용 모델 6개

이 패치는 **기존 v47 프로젝트에 풀기 위한 증분 패치**이며 처음부터 실행되는 독립 프로젝트 ZIP이 아니다.
