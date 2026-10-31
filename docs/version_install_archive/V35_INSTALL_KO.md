# v35 설치/실행

전제: v34까지 설치된 기존 프로젝트입니다. v35는 **모델을 교체하지 않는 연구용 분석 패치**입니다.

1. 원본 `프로그램` 폴더를 백업합니다.
2. `Biochar_v35_Temperature_Panel_and_Ranking_Audit_Patch.zip`을 프로젝트 폴더 내부에 저장합니다.
3. PowerShell:

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Test-Path ".\Biochar_v35_Temperature_Panel_and_Ranking_Audit_Patch.zip"
Expand-Archive -LiteralPath ".\Biochar_v35_Temperature_Panel_and_Ranking_Audit_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m scripts.build_v35_temperature_panels
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_v35_temperature_panels.py" -v
```

`Test-Path`가 `False`면 ZIP 파일을 그 경로에 옮겼는지 먼저 확인하세요. .venv와 기존 deployment 폴더는 삭제하지 않습니다.

분석 결과: `reports\v35_temperature\RESULTS_KO.md` 및 CSV/JSON 파일. 기존 Stage 2 UI는 변경되지 않습니다.
