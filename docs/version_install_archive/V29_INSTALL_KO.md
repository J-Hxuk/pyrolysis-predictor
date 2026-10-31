# v29 연구용 패치 설치

**전제:** v28까지 적용된 기존 프로젝트. 기존 `.venv`, `data/`, `deployment/`, `active.json`, UI는 변경하지 않는다.

1. `Biochar_v29_FeatureRecovery_TemperatureReadiness_Patch.zip`를 `프로그램` 폴더에 내려받는다.
2. 프로젝트를 백업한 후 PowerShell을 실행한다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Expand-Archive -LiteralPath ".\Biochar_v29_FeatureRecovery_TemperatureReadiness_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v29_process_feature_recovery.py -q
```

연구결과는 `reports\v29_process_feature_recovery\RESULTS_KO.md`에서 확인한다. 앱 화면/기존 예측은 바뀌지 않는다. **기존 모델 자동 재학습·활성화 금지.**

재현 실행(선택):
```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_v29_process_feature_recovery
.\.venv\Scripts\python.exe -m scripts.audit_v29_temperature_readiness
```
