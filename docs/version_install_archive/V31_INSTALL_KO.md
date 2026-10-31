# Biochar v31 설치 (v30 이후)

본 패치는 **분석·검증 전용**이며 UI·저장 모델·학습 데이터·`.venv`를 덮어쓰지 않습니다.

1. ZIP을 `C:\Users\jh030\Desktop\대학\학술대회\프로그램` 폴더에 저장합니다.
2. PowerShell에서 다음을 실행합니다.

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\프로그램'
Test-Path '.\Biochar_v31_Rice_Temperature_External_Check_Patch.zip'
Expand-Archive -LiteralPath '.\Biochar_v31_Rice_Temperature_External_Check_Patch.zip' -DestinationPath '.' -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v31_rice_external_temperature.py -q
.\.venv\Scripts\python.exe -m scripts.evaluate_v31_rice_temperature_external
```

3. 결과: `reports\v31_rice_temperature_external\RESULTS_KO.md`, `summary.json`, 원시 예측 CSV.
4. 이 패치는 기존 **v30 보고서 CSV**에 의존합니다. v30을 먼저 설치해야 합니다.

주의: 분석용 종속 라이브러리는 기존 프로젝트의 sklearn/xgboost 버전을 따릅니다. 코드가 사용하는 임의 분할은 없고 98행의 기존 DOI 집합에 외부 논문 DOI가 존재하지 않는지 검증합니다. 실제 Windows 추론 결과는 로컬에서 확인해야 합니다.
