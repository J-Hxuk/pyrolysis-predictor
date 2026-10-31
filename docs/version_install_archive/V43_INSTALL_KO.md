# Biochar v43 — 학술대회 시연 중심 기능 패치

## 목적
v43은 머신러닝 성능을 새로 주장하는 버전이 아니라, **시연 가능한 사용자 흐름을 완성**하는 UI 중심 업데이트입니다. v37의 실시간 슬라이더/그래프 및 v41의 입력 적용범위 경고를 유지합니다.

## 기능
1. `흡착성능 예측`에서 원료 → 흡착조건 슬라이더 조절 → qe·제거율·변수별 그래프 실시간 확인.
2. 조성 입력은 기본적으로 접혀 있어 필요한 경우만 펼침. 현재 Stage 2 A/B에 원료 조성값이 실제로 사용되지 않는다는 안내 유지.
3. `현재 조건 저장`으로 예측 결과/실험조건 저장(세션 내 최대 12개).
4. 저장된 조건의 A/B 제약 후 qe 막대그래프, 원시값/제약 후 값/제거율/적용범위/제약 경고 표 확인.
5. UTF-8 BOM CSV 내보내기와 **오프라인 HTML 보고서** 다운로드. 보고서에 조건 비교 막대그래프·현재 변수 스윕 그래프·수치표·해석 제한 포함. HTML 파일은 브라우저에서 인쇄→PDF 저장 가능.
6. 같은 이름 반복 저장 시 번호 자동 부여, 마지막 조건 삭제 / 전체 초기화.

## 필수 선행 버전
- **v41 UI와 v37 sensitivity 기능이 이미 적용된 프로그램** (v42 연구용 패치는 새 모델 저장만 하므로 UI 충돌 없음).
- 프로그램의 기존 `.venv`와 Streamlit, Altair, Pandas 및 활성 Stage 2 모델.

## Windows PowerShell 설치
먼저 **프로그램 종료**(Streamlit 터미널에서 Ctrl+C).

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

# 파일 백업 (기존 사용자 수정분을 덮어쓰기 전에 보존)
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" `
  "..\stage2_pages_before_v43.py" -Force

# 다운로드한 ZIP을 이 폴더에 둔 뒤 적용
Expand-Archive -LiteralPath ".\Biochar_v43_Demo_Workflow_Patch.zip" `
  -DestinationPath "." -Force

# 새 기능과 핵심 기존 기능만 간단 검증
.\.venv\Scripts\python.exe -m pytest `
  tests\test_v43_scenarios.py tests\test_v37_sensitivity.py tests\test_v41_scope.py -q

# 실행
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

## 1분 시연 순서
1. `흡착성능 예측` → 등록 원료 선택. 조성은 입력하지 않아도 됨.
2. 오염물질·C0·pH·투입량·시간 선택 → 수치와 그래프가 즉시 변하는지 확인.
3. `시나리오 이름`에 `기준 조건` → `현재 조건 저장`.
4. pH 또는 투입량 슬라이더 이동 → `비교 조건`으로 추가 저장.
5. 비교표와 막대그래프 확인 → `발표용 결과 보고서 (HTML)` 다운로드 → 브라우저에서 열어 인쇄/PDF 저장.

## 제한 및 주의
- 시나리오는 **Streamlit 세션 상태**에만 남습니다. 새로고침이나 서버 재시작 시 없어질 수 있으니 발표 전 반드시 CSV/HTML 다운로드.
- 막대그래프는 물질수지 제약 후 qe를 사용하지만, 표와 보고서에는 원시 qe와 제약 경고도 보존합니다. 제약 후 값이 '더 정확한 학습 모델'이라는 뜻은 아닙니다.
- Model A는 제조조건/원료 조성 효과를 학습하지 않았고, Model B는 미학습 원료에서 제외. 서로 다른 오염물질의 막대 크기는 순위로 해석하지 않습니다.
- 모델 파일, 활성 설정, 학습 CSV, 연구용 모델, 기존 제조조건 비교 화면은 변경하지 않습니다.
- 개발 환경(Linux)에는 Streamlit/Altair가 없어 **실제 브라우저 렌더링은 Windows 현장에서 확인 필요**. 순수 저장·보고서 기능과 관계 모듈 테스트 32개 통과.

## 파일 정리
이 패치는 UI 코드 1개, 작은 새 모듈 1개, 테스트 1개, 안내 문서로 구성됩니다. 실행에 필요한 v15/v41/v60 ZIP 또는 추출 데이터는 삭제/이동하지 마세요. 이미 설치된 과거 패치 ZIP만 `학술대회_Archive_20261030`으로 **수동 백업 후 이동**해도 됩니다. 다른 프로젝트 코드나 `deployment/active.json`은 건드리지 마세요.
