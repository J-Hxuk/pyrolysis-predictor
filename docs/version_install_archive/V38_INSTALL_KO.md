# v38 연구용 설치 안내 — v37 적용 프로젝트 기준

1. 기존 `프로그램` 전체 폴더 및 `deployment/stage2_models/active.json`을 백업하세요.
2. v38 ZIP을 `C:\Users\jh030\Desktop\대학\학술대회\프로그램\`에 저장합니다.
3. PowerShell에서 아래를 실행합니다.

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\프로그램'
if (!(Test-Path '.\Biochar_v38_NewDOIs_FullRefit_Research_Patch.zip')) { throw 'v38 ZIP이 프로그램 폴더에 없습니다.' }
Expand-Archive -LiteralPath '.\Biochar_v38_NewDOIs_FullRefit_Research_Patch.zip' -DestinationPath '.' -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v38_new_dois.py tests\test_v37_full_refit.py tests\test_v37_sensitivity.py -q
```

패키지에는 이미 같은 버전 환경에서 재학습한 **연구용 모델과 결과가 포함**됩니다. 재생성은 선택 사항이며 활성 모델을 변경하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.build_v38_new_dois --root .
.\.venv\Scripts\python.exe -m scripts.plot_v38
```

v38의 `deployment/research_v38_NOT_ACTIVE/` 폴더는 기존 배포모델과 별개입니다. 모델 활성화나 `active.json` 수정을 하지 마세요. 기존 Streamlit 앱은 v37 UI 그대로 실행됩니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

보고서: `reports\v38_new_dois\RESULTS_KO.md`

주의: v38의 Stage 2 커피 12행은 보고된 uptake의 **조건부 연구용** 수치이고 pH 및 시료별 접촉시간이 미보고입니다. 엄격 평형 qe가 독립적으로 인증된 데이터가 아닙니다. Python/sklearn/XGBoost 버전이 다르면 저장된 joblib 모델은 로드하지 말고 로컬 재생성 코드로 학습하세요.
