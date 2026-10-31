# Biochar v44 — BET·흡착 화면 연결 및 발표용 워크플로

## 핵심 변경

- `BET 물성 예측`(v20 원료 조성 기반 실험모델): 유효한 입력이면 BET 결과를 즉시 계산하며 온도별 예상 BET 그래프를 표시합니다. 버튼 수동 실행도 유지합니다.
- 그래프는 **공업·원소조성, 유지시간, 승온속도 고정 / 열분해온도만** 모델의 **학습 온도범위 내에서** 변경한 모델 응답입니다. 실험값이나 검증된 최적온도를 뜻하지 않습니다.
- BET 결과와 그래프를 UTF-8 BOM CSV로 다운로드할 수 있습니다.
- 버튼을 누르면 BET 화면의 **원료명·열분해온도·유지시간·승온속도**를 `흡착성능 예측`의 제조조건(Model B)에 전달합니다. 왼쪽 메뉴에서 해당 화면을 선택하면 전달된 값이 표시됩니다.
- **BET 예측값은 Stage 2 A/B의 입력에 전달하지 않습니다.** Stage 2 학습 모델이 이를 사용한다고 표시하지 않으며, 미학습 원료의 B 예측 차단도 유지합니다.
- 팀원의 Sakhiya·kelp 원문 재확인 내용을 별도 노트로 기록합니다(학습 행 수 변동 없음).
- 활성 모델·데이터·Stage 2 계산·기존 v43 조건 비교·CSV/HTML 기능은 그대로 둡니다.

## 설치

v43까지 적용된 프로그램 폴더에 이 패치 ZIP을 저장하고 Streamlit을 **Ctrl+C로 종료한 뒤** 아래 PowerShell 명령을 실행합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

# 먼저 두 화면 파일을 외부 보관 폴더에 백업
$backup = Join-Path (Split-Path (Get-Location) -Parent) "학술대회_Archive_20261030\v44_ui_backup"
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" (Join-Path $backup "stage2_pages_pre_v44.py") -Force
Copy-Item ".\pyrolysis\biochar_ui\stage1_bet_v20_page.py" (Join-Path $backup "stage1_bet_v20_page_pre_v44.py") -Force

Expand-Archive -LiteralPath ".\Biochar_v44_BET_Adsorption_Workflow_Patch.zip" -DestinationPath "." -Force

# 새 기능 + 기존 저장 기능 확인
.\.venv\Scripts\python.exe -m pytest tests\test_v44_workflow.py tests\test_v43_scenarios.py tests\test_v37_sensitivity.py tests\test_v41_scope.py -q

.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

## 1분 시연

1. 사이드바 `BET 물성 예측`: 원료명, 비활성화 단일 열분해, 온도·시간·승온속도 입력.
2. 공업분석 **dry basis** 실측 VM/Ash/FC 3항목 입력(합계 약 100%). 원소분석 C/H/N/O는 모두 있으면 **daf basis**로 추가, 없으면 전부 공란.
3. `입력 변경 시 BET·그래프 자동 갱신` 사용. 현재 예상 BET와 온도별 선그래프, CSV 다운로드 확인.
4. `제조조건을 흡착성능 화면으로 전달` 클릭 → 사이드바 `흡착성능 예측`으로 이동.
5. 전달된 제조조건에서 Model B를 선택해 pH·C0·Dose 등 흡착조건을 조절하고 v43 조건 저장/그래프/HTML 보고서 출력.

## 연구상 제한

- BET 모델은 기존 **Stage 1 실험용 v20**이며 182행·24 DOI의 탐색적 학습 모델입니다. 연구용 Stage 1 v40 전체 재학습 모델이 이 화면에 자동 배포된 것은 아닙니다.
- Stage 1 v20은 BET만 예측합니다. TPV는 v20의 타깃이 아니며, 별도 v60의 물성 예측기는 환경 호환성 제한이 있으므로 이 패치에서 자동 활성화하지 않습니다.
- 신규 원료 이름은 기록용입니다. Stage 1 v20의 RF에는 원료명 특성이 없고, Stage 2 Model B는 미학습 원료를 차단합니다.
- `can_explore_composition_ood` 조성 외삽은 기존 명시적 동의 조건이 유지됩니다. 공정조건 외삽은 차단합니다.
- Sakhiya는 물질수지 충돌로 학습 보류, Kelp는 pH 결측으로 연구용 조건부 3행 유지. 세부 사항은 `notes/V44_TEAM_SOURCE_FEEDBACK_KO.md` 참고.
- 개발환경에 Streamlit/Altair가 설치되어 있지 않아 실제 **Windows Streamlit 브라우저 렌더링은 사용자 PC에서 확인**해야 합니다.

## 파일 정리

v44 ZIP은 코드 변경 2개, 신규 헬퍼 1개, 테스트·노트·설치문서만 포함합니다. 대용량 논문 원본, 과거 patch ZIP, 모델 파일을 중복하지 않습니다. 과거 patch ZIP은 설치 확인 후 아카이브에 백업해도 되지만 `data/imports/`의 추출본, `deployment/`의 활성모델, 모델 실행에 필요한 파일을 삭제하지 마세요.
