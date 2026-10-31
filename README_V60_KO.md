# v60 — 흡착조건 응답과 제조조건 순위 분리 (연구 전용)

**설치 전제:** 이전 v52–v59 패치가 들어 있는 `Biochar_v47_Integrated_Project`의 최상위 폴더에서 압축을 푸세요. 기존 승인 원본, 활성 모델, Streamlit 연결 및 앞 버전 산출물은 변경하지 않습니다.

## 무엇이 달라졌나

- **흡착조건 응답 (Stage2 A)**: 오염물질, C0, pH, dose, 온도, 시간 등 흡착조건만으로 qe 예측.
- **제조조건 순위 (신규 ranker)**: 동일 오염물질·C0·dose·pH·온도·시간 및 동일 원료·route·modification으로 묶을 수 있는 제조시료 간 `100 qe/(C0/dose)` 중심화 차이를 학습. Ridge/RandomForest의 출력은 **상대적 제거율 점수(%p)**이지 실측 qe가 아닙니다.
- **기존 B 비교군**: 흡착조건+제조조건을 동시에 입력하는 XGBoost, cap 입력 포함; 행 동일/DOI·제조시료 가중치 각각 비교.
- **농도 변화 따로 평가**: 동일한 제조시료에서 다른 C0 조건으로 측정한 관측만 묶어서 monotonicity·순위·MAE 별도로 보고.
- **검증**: 모두 DOI GroupKFold로 논문 전체를 한꺼번에 제외, 양쪽 학습 시나리오 152/188행 모델이 같은 188행 평가 표본 사용. 직접 모델 모델선택에 사용한 역사적 DOI들이므로 사후평가에 해당.

## 가장 중요한 한계

엄격한 제조조건 비교계열 15개 중 **흡착조건 전부가 원문에서 보고된 계열은 단 4개(3편 DOI)**입니다. 다른 11개 계열은 일부 흡착변수가 모두 NR인 관측이 묶여 있으며 **동일 조건이 확증된 것이 아닙니다.** 확인된 `strict qe`는 사과나무 2계열(1 DOI)뿐입니다. 따라서 유의한 제조 순위모델 개선·새로운 논문에 대한 실용적 추천을 주장하면 안 됩니다. C0 변화계열도 5개(2 DOI)에 불과합니다.

## Windows PowerShell

v47 폴더 안에 `Biochar_v60_Separated_Adsorption_Manufacturing_v47_Patch.zip`을 놓은 다음:

```powershell
$ErrorActionPreference = 'Stop'
$zip = '.\Biochar_v60_Separated_Adsorption_Manufacturing_v47_Patch.zip'
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $zip)) { throw 'v60 ZIP이 현재 폴더에 없습니다.' }
if (!(Test-Path $python)) { throw '기존 프로그램 .venv를 찾을 수 없습니다.' }
if (!(Test-Path '.\scripts\run_v59_doi_material_weighting.py')) { throw 'v59 선행 패치가 없습니다.' }
Expand-Archive -LiteralPath $zip -DestinationPath '.' -Force
& $python -m scripts.run_v60_separate_effects --root .
if ($LASTEXITCODE -ne 0) { throw 'v60 학습 또는 그룹분할 평가 실패' }
& $python -m scripts.audit_v60_effects --root .
if ($LASTEXITCODE -ne 0) { throw 'v60 원문 근거 감사 실패' }
& $python -m pytest tests/test_v60_separate_effects.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'v60 자체 테스트 실패' }
```

Windows의 scikit-learn 환경(이전 PC는 1.5.2)과 만든 환경(1.8.0)이 다르므로 **로컬 모델을 재학습한 후** 저장된 Joblib을 사용하세요. 기존 활성모델 파일은 건드리지 않습니다. `reports/v60_separate_effects` 아래에 로컬 결과가 출력됩니다.

## 디렉터리

- `scripts/run_v60_separate_effects.py`: fold-safe 실학습 + 모델 보존
- `scripts/audit_v60_effects.py`: 완전 조건·엄격 qe 판정, DOI bootstrap
- `tests/test_v60_separate_effects.py`: 평가 분할, 동일 조건, 동률, 물질수지·OOF 검증
- `reports/v60_separate_effects/RESULTS_KO.md`: 방법·수치·한계
- `reports/v60_separate_effects/*.csv`: 188 OOF 전체, 제조 비교·농도 시리즈, 원문 근거 등
- `deployment/research_v60_NOT_ACTIVE/*.joblib`: A/B 직접 예측용 2개, 제조 순위점수용 2개

## 금지·유의사항

**신규 ranker 출력 값을 qe (mg/g), Langmuir qmax, 혹은 인과적 제조온도 효과라고 표기하지 마세요.** 후보군의 동등 조건이 확인되지 않으면 추천 버튼 활성화도 보류해야 합니다. 제조/흡착 입력을 구분한 것은 성능 메커니즘을 시험하기 위한 연구설계이지, 측정 없이 원료의 제조조건만으로 성능 개선을 보증하는 모델이 아닙니다.
