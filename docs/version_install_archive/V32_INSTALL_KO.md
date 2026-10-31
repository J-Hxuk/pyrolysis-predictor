# Biochar v32 연구용 패치 설치 안내

## 전제
- 기존 프로젝트에 v31까지 설치되어 있어야 함.
- 파일을 원래의 `프로그램` 프로젝트 루트에 병합하는 방식.
- 원본 `data/`, `deployment/`, `.venv/`, 앱과 UI는 패치에 포함되지 않으며 수정하지 않음.

## 적용
1. [v32 패치 ZIP]을 `프로그램` 폴더로 이동합니다.
2. PowerShell에서 아래 명령을 실행합니다. ZIP 파일명은 실제 내려받은 파일명과 같아야 합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Test-Path ".\Biochar_v32_Temperature_Identifiability_Research_Patch.zip"
Expand-Archive -LiteralPath ".\Biochar_v32_Temperature_Identifiability_Research_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v32_temperature_identifiability.py -q
```

3. 보고서는 `reports\v32_temperature_identifiability\RESULTS_KO.md`에서 확인합니다.
4. 연구 결과를 다시 생성하려면 아래를 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_v32_temperature_identifiability
.\.venv\Scripts\python.exe -m scripts.plot_v32_temperature
```

**주의:** 기존 모델을 교체하지 않으며 Streamlit 화면도 바뀌지 않습니다. 보고서의 300/400/500°C 그래프는 한 논문의 근삿값을 사용한 외부 진단입니다. `qe` 엄격 평형 승인이나 전역 최적조건 추천으로 사용해서는 안 됩니다.
