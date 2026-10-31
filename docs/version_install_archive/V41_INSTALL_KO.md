# Biochar v41 설치 (기존 v40 기반, 증분 패치)

**작업 전 Streamlit 서버를 종료**하세요. 기존 v40 패치를 설치한 `프로그램` 루트 기준입니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

# 현재 UI 백업 — 덮어쓰기 전 필수
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" `
    "..\stage2_pages_before_v41.py" -Force

# V41 패치 ZIP을 이 폴더에 저장했다는 전제
Expand-Archive -LiteralPath ".\Biochar_v41_Applicability_RankingGuard_Patch.zip" `
    -DestinationPath "." -Force

# 새 기능 테스트 + 직전 버전 회귀 테스트
.\.venv\Scripts\python.exe -m pytest `
    tests\test_v41_scope.py tests\test_v41_paired_oof.py `
    tests\test_v37_sensitivity.py tests\test_v40_new9.py -q

# 실제 활성 앱에서 v41 적용범위 확인
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

앱의 `흡착성능 예측`에서 슬라이더를 움직일 때 **입력 적용범위** 경고가 나타나는지, `제조조건 탐색`에서 동일 조건 비교와 미검증 점수 조회가 가능한지 확인하세요. A/B 예측 결과는 기존 활성 모델에서 계산되며 v40 연구모델로 자동 변경되지 않습니다.

Stage 1 연구용 OOF의 DOI-짝 비교만 재계산(모델 재학습 아님):

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_v41_paired_stage1 --root .
```

전체 회귀 테스트는 환경에 따라 오래 걸릴 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

v15/v41/v60 원본 ZIP이 누락됐다는 오류가 있으면 학습데이터를 새 버전 자료로 덮어쓰지 말고 이전과 동일하게 백업/추출본 경로를 점검하세요.

**주의**: 이 패치는 기존 `active.json`, 저장 모델, 훈련 데이터 및 논문 PDF를 수정하지 않습니다. 원료별·오염물질별 독립 DOI가 부족한 기존 Core 97에서는 검증된 최적 제조조건 순위가 차단되고, 연구용 원시 점수만 사용자 요청 시 표시되는 것이 정상입니다.
