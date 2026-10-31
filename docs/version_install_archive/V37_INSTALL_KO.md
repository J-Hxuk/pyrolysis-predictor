# v37 설치 안내 — v36 적용 프로젝트 기준

## 준비
- 기존 `프로그램` 폴더와 `.venv`는 절대 삭제하지 마세요.
- `Biochar_v37_LiveUI_FullDataset_Research_Patch.zip`을 `프로그램` 폴더 안에 다운로드하세요.
- 기존 Streamlit 앱이 켜져 있다면 터미널에서 Ctrl+C로 종료하세요.

## PowerShell

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

# 기존 UI만 사본 백업
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" "..\stage2_pages_before_v37.py" -Force

$zip = ".\Biochar_v37_LiveUI_FullDataset_Research_Patch.zip"
if (-not (Test-Path $zip)) { throw "v37 ZIP을 프로그램 폴더에 저장했는지 확인하세요." }
Expand-Archive -LiteralPath $zip -DestinationPath "." -Force

# 새 기능 단위 테스트
.\.venv\Scripts\python.exe -m pytest tests\test_v37_sensitivity.py tests\test_v37_full_refit.py -q

# 연구용 전체 데이터 재학습 (선택: 별도 연구 모델 갱신 시)
.\.venv\Scripts\python.exe -m scripts.build_v37_full_refit --root .

# UI 실행
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

## 사용법

'흡착성능 예측' 화면의 pH·C₀·투입량·시간·흡착온도 슬라이더를 움직이면 즉시 **원시 qe 및 제거율** 예측을 새로 계산합니다. 하단에서 민감도 그래프 변수 1개와 구간을 선택할 수 있습니다. '현재 조건을 비교 기준으로 저장'을 누르면 이후 슬라이더 이동에 따른 qe 증감도 나타납니다.

- 기존 활성 모델은 그대로 유지됩니다. `deployment/research_v37_NOT_ACTIVE` 아래 연구 모델은 자동 활성화되지 않습니다.
- 신규 추가 데이터의 출처 및 학습 제외 사유는 `reports/v37_full_refit/source_status_manifest.csv`에서 확인합니다.
- `reports/v37_full_refit/RESULTS_KO.md`에서 평가 방식을 확인합니다.
- 실행 중 `ModuleNotFoundError`가 발생하면 `.venv`가 프로젝트의 지정 환경인지 확인하세요. **`.venv`를 삭제하지 마세요.**
- UI 렌더링은 Windows 환경에서 별도 확인해야 합니다.
