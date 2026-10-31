# Biochar v48 — 물성·흡착 예측 사용 흐름 정리

## v47 → v48 변경

1. 왼쪽 **BET 물성 예측**에서 탭을 이용하여 v20 원료 조성 모델(BET)과 v46 확장 연구모델(BET+TPV)을 구분해 사용할 수 있습니다.
2. v46 BET/TPV 탭은 **417개 제조 시료의 원문 입력값** 중 온도가 보고된 시료만 선택합니다. 이후 같은 원료·제조경로의 실제 보고 온도 범위 내에서 온도별 예측곡선을 표시합니다. 이 범위는 경험적 분포 제한일 뿐 정량적 검증은 아닙니다.
3. 원료·온도·제조조건이 서로 일치하지 않을 때 추론 결과를 마치 새로운 측정값처럼 표시하지 않습니다. 해당 시료의 **원문 실측값**과 **현재 선택한 온도의 연구용 예측값**을 분리해 표시합니다.
4. v20 BET 화면에서 `제조조건 전달 → 흡착성능 화면 열기`를 선택하면 원료명·온도·유지시간·승온속도만 Stage 2에 전달됩니다. **BET/TPV 모델 출력은 Stage 2에 투입하지 않습니다.**
5. 기존 Stage 2 A/B(Core 97 기반)와 확장 Stage 1 연구모델을 구분하는 안내를 추가했습니다. 활성 모델 교체, 신규 학습, 데이터 수정은 없습니다.

## 설치 A — 기존 v47에 패치만 적용 (권장)

기존 v47 프로젝트 폴더에 `Biochar_v48_UI_Workflow_Patch.zip`을 복사하여 풀면 됩니다. v47을 먼저 설치해야 합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\Biochar_v47_Integrated_Project"
Expand-Archive -LiteralPath ".\Biochar_v48_UI_Workflow_Patch.zip" -DestinationPath "." -Force
$env:BIOCHAR_PYTHON = "C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe"
.\CHECK_V48.cmd
.\START_V48.cmd
```

PowerShell 스크립트 실행 정책과 관계없이 `.cmd`를 호출할 수 있습니다. 단, 조직 보안 정책은 우회하지 않으며 Python 실행 권한과 의존성 설치는 필요합니다.

## 설치 B — 팀원 공유용 통합본

`Biochar_v48_Integrated_Project_FULL.zip` 내부의 `Biochar_v48_Integrated_Project` 폴더를 통째로 압축 해제합니다. **새 폴더**에서 아래 명령을 실행합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\Biochar_v48_Integrated_Project"
# 새 PC인 경우에는 Python 3.12와 의존성을 새로 설치합니다.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.\CHECK_V48.cmd
.\START_V48.cmd
```

`http://localhost:8501`에서 화면을 확인합니다. 기존 `.venv`를 재사용하는 경우 `BIOCHAR_PYTHON` 환경변수로 경로를 지정합니다.

### Windows에서 v46 BET/TPV 연구용 모델이 열리지 않을 때

v46에서 겪었던 `numpy.random._pcg64.PCG64` 또는 scikit-learn 모델 호환성 문제가 재현될 수 있습니다. **연구용 모델만 현재 PC에서 재학습·재저장**합니다.

```powershell
# v48 프로젝트 폴더에서
$python = "C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe"
& $python -m scripts.rebuild_v46_research_models_local --root .
.\CHECK_V48.cmd
.\START_V48.cmd
```

다시 저장된 연구모델 4개는 `V46_LOCAL_REBUILD_MANIFEST.json`에 별도 해시로 기록되며 v48 검사기는 해당 로컬 재학습을 **연구용 모델에 한해서만 허용**합니다. 학습 데이터를 변경하거나 활성 모델의 해시가 바뀌면 검사는 실패합니다. 이 로컬 재학습만으로 과거 교차검증 오차가 갱신되는 것은 아닙니다.

## 모델·데이터 범위

- v20 BET: 원료 **공업분석(dry basis)** 및 필요시 **원소분석(daf)** 입력, 탐색적 RandomForest. 활성 v60 모델로 변환하지 않습니다.
- v46 BET/TPV: 417개 통합 제조시료에서 각각 BET 361행·TPV 231행을 학습한 별도 Poisson HistGradientBoosting 연구모델. 결측 제조 입력은 모델 내부 학습 대치기를 사용하며 결측 목록을 표시합니다. **저장된 모델의 정확한 패키지 버전과 다르면 Windows에서 joblib 로딩 실패 가능** — 이전 v46 로컬 호환성 복구 스크립트를 사용할 수 있습니다.
- Stage 2 A/B: **활성 Core 97 모델** 그대로 유지. v46 Stage 2 140행 연구용 데이터와 혼동하지 마세요.
- 신규 투입비의 분모가 `char`인 K₂CO₃/char 및 FeCl₃/char 는 v46 모델에서 학습하지 않았습니다. 활성화제 투입비별 예측·최적화 불가.
- Stage 1 연구모델의 동일 논문 시료 실측과 예측을 비교해도 **외부 검증이 아닙니다**. 관측 온도 범위 내부만 표시해도 미관측 조합의 정확도를 보장하지 않습니다.
- qₑ, qₜ, qmax는 동일한 target이 아닙니다. Stage 2의 현재 target은 `reported uptake`이며 엄밀한 평형 보증은 없습니다.

## 새 구조

- `pyrolysis/biochar_ui/v48_material_properties_page.py`: 두 물성 화면 탭
- `pyrolysis/biochar_stage1/v48_research_bridge.py`: 읽기 전용 특성 추출과 예측, 온도 곡선
- `tests/test_v48_material_workflow.py`: 모델 및 원본 구분 검증
- `scripts/verify_v48.py` + `V48_MANIFEST.json`: 프로젝트 상태 검증
- `CHECK_V48.cmd`, `START_V48.cmd`: `.ps1` 보안 정책이 막힌 Windows 환경에서도 실행 가능한 래퍼

## 검증 상태

코드 컴파일, 주요 회귀 단위 테스트, 실제 저장된 v46 모델 2개에서 유한한 결과 산출, 통합 ZIP 해시 확인을 수행했습니다. **Windows의 실제 Streamlit 브라우저 렌더링은 이 환경에서 확인하지 못했습니다.** v47 기존 ZIP과 원본 데이터를 삭제하지 마세요.

## 다음 단계

v49에서는 학생 발표용 예시 시나리오와 결과 설명을 실제 Windows 시연을 기준으로 조정하는 것을 권장합니다. 새로운 학습이나 대규모 DOI 검증보다 시연 오류와 이해하기 어려운 안내를 우선 처리합니다.
