# Biochar Prediction Platform — v47 통합 기준본

## 목적

기존 v17 전체 프로젝트를 바탕으로 이후 v18~v46 개선 사항을 **단일 폴더에 통합**한 연구·발표 시연용 기준본입니다.
논문 투고용 검증 완료 모델 또는 신뢰성이 보증된 제조조건 최적화 프로그램으로 해석해서는 안 됩니다.

### 실행 화면

- `biochar_stage2_app.py`: **주로 사용하는 프로그램**. 흡착성능 예측, 제조조건 후보 비교, BET 물성 예측, 데이터·모델 검증.
- `biochar_app.py`: 과거 Stage 1 단독 기능/시험용 앱. 동시에 실행할 필요는 없습니다.

### 포함된 기능

1. 실시간 흡착조건 슬라이더 조절 및 변수별 반응곡선(v37)
2. 원료·오염물질·학습영역 검사, 제한된 조건에서 연구용 추천 점수만 제공(v41)
3. 조건 저장·비교·CSV / HTML 보고서 내보내기(v43)
4. BET 온도별 예측 곡선, BET→흡착 화면 제조조건 전달(v44)
5. 확장된 원문 검증 연구 데이터와 모델(v45·v46); **활성 모델과 분리**

### 중요: 활성 모델과 연구용 모델 구분

- **실제 UI**: 기존 Stage2 Core 97 기반 활성 모델, 기존 Stage1 모델 및 v20 조성 BET 화면.
- **연구용 확대 데이터**: Stage1 제조시료 417행(탐색/미승인 포함), Stage2 reported uptake 140행(조건부 포함).
- v46의 연구용 모델 4개는 `deployment/research_v46_NOT_ACTIVE/`에 있으며 자동 활성화되지 않습니다.
- BET 예측값이 흡착 qe 예측의 입력으로 자동 연결되는 기능은 **없습니다**. 제조조건만 전달됩니다.
- 원료 조성·활성화제 투입비·pH가 미보고인 자료는 임의로 채우지 않습니다.

## Windows 설치(새 폴더 권장)

1. ZIP을 해제한 `Biochar_v47_Integrated_Project` 폴더를 준비합니다. 기존 `프로그램` 폴더에 덮어쓰지 마세요.
2. PowerShell에서 해당 폴더로 이동합니다.
3. 기존 환경을 재사용할 수도 있습니다. 우선 Python 3.12 기반 새 환경을 권장합니다.

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\Biochar_v47_Integrated_Project'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.\CHECK_V47.ps1
.\START_V47.ps1
```

개인 PC의 실제 해제 경로가 다르면 `Set-Location` 경로만 수정하세요.
PowerShell 실행 정책이 차단하면 `powershell -ExecutionPolicy Bypass -File .\START_V47.ps1`처럼 해당 프로세스에서만 일시 실행하세요.

연구용 Jupyter/개발 도구까지 필요한 경우에만 `-r requirements-dev.txt`도 추가 설치하세요. CatBoost 관련 전체 회귀 테스트를 실행하려면 `catboost`가 추가로 필요할 수 있습니다.

**기존 가상환경을 쓸 때** (예: 새 패키지 설치가 부담되는 경우):

```powershell
.\START_V47.ps1 -Python 'C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe'
```

기존 환경 재사용은 환경 의존성 차이를 확인한 후에만 권장합니다. 환경 문제가 생기면 새 venv를 설치하세요.

### 로컬 무결성 검사 및 테스트

```powershell
.\CHECK_V47.ps1     # 핵심 스모크 테스트
.\CHECK_V47.ps1 -AllTests  # 전체 구형 회귀 테스트 포함
```

`START_V47.ps1`은 먼저 필수 데이터 유무를 검사한 후 Streamlit 앱을 시작합니다. 기본 `CHECK_V47.ps1`은 새 Windows 환경에서 문제가 될 수 있는 **비활성 v46 연구용 joblib 로딩 테스트 1개만 제외**하고 데이터·UI 지원 기능을 검사합니다. 연구용 모델까지 확인하려면 아래 로컬 재학습을 실행한 뒤 `CHECK_V47.ps1 -AllTests`를 사용하세요.

### v46 연구용 모델의 Windows 호환성

과거에 `numpy.random._pcg64.PCG64` 직렬화 문제가 발생했었습니다. 연구용 모델을 Windows에서 불러올 때 동일한 문제가 생긴 경우에만 다음 명령으로 연구용 모델 4개를 재학습·저장합니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.rebuild_v46_research_models_local --root .
```

**활성 모델은 변경하지 않습니다.** 과거 통합 ZIP에 포함된 연구용 joblib이 다른 NumPy 버전에서 저장된 상태일 수 있습니다.

### 용량·파일 정리

- v15 58MB 원본 ZIP, v41 대용량 백업 ZIP 없이도 **검증된 추출본을 통해 프로그램이 동작**합니다. 원본 ZIP은 기존 보관 폴더에 유지하세요. 절대 삭제하라는 뜻은 아닙니다.
- 기존 원본 프로젝트의 예전 패치 ZIP을 옮기려면 다음을 먼저 *미리보기*로 실행합니다.

```powershell
.\ARCHIVE_OLD_PATCHES_V47.ps1 -SourceProjectRoot 'C:\Users\jh030\Desktop\대학\학술대회\프로그램' -ArchiveRoot 'C:\Users\jh030\Desktop\대학\학술대회\학술대회_Archive_20261030'
```

v47에서 앱과 전체 테스트가 동작하는지 확인된 경우에만 `-Apply` 옵션으로 이동합니다. 본 스크립트는 최상위 과거 `Biochar_vXX_*Patch.zip`만 이동하며 CSV·모델·PDF는 이동하거나 삭제하지 않습니다.

### v47 관련 테스트 결과

통합 이후 Linux 분석 환경의 관련 데이터·로직 테스트는 **61개 통과**했습니다. (v46 연구용 모델 로딩 검사는 Linux에서 성공하였으나, 저장 모델의 Windows 호환성을 보장하지 않습니다.) Windows Streamlit 화면 렌더링과 전체 테스트 통과는 사용자 PC에서 확인이 필요합니다.
