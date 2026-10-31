# v34 연구용 추가학습 패치 설치

v33까지 반영된 프로젝트를 기준으로 합니다. v34는 v33 또는 기존 v24/Stage2 검증 모듈을 대체하지 않습니다.

1. `Biochar_v34_Temperature_SourceAudit_and_ResearchTraining_Patch.zip`을 `프로그램` 폴더에 저장.
2. 앱 실행 중이면 Ctrl+C 종료, 프로젝트 폴더 백업 권장.
3. PowerShell 명령어:

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Test-Path ".\Biochar_v34_Temperature_SourceAudit_and_ResearchTraining_Patch.zip"
Expand-Archive -LiteralPath ".\Biochar_v34_Temperature_SourceAudit_and_ResearchTraining_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v34_temperature.py -q
```

결과는 `reports\v34_temperature\RESULTS_KO.md`, 연구용 모델은 `deployment\research_v34\`입니다. 테스트 모델은 현재 Python 환경 차이 때문에 Windows에서 다시 학습하는 것이 안전합니다. 재학습은 아래처럼 수행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.build_v34_temperature
```

참고: 이 명령어는 `deployment/stage2_models/active.json`을 변경하지 않습니다. v33 자료가 설치되어 있지 않으면 실행을 중단하고 기존 데이터를 먼저 복원하세요. `research_v34` 파일은 정식 배포모델이 아닙니다. GUI에 반영되지 않습니다.
