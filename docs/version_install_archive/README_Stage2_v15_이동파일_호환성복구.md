# Stage 2 v15 원본 ZIP 이동 후 회귀테스트 호환성 복구

## 진단
- 오류 원인: `pyrolysis/biochar_stage2/v15_extract.py`가 기존 추출본 존재 여부를 확인하기 전에 `Stage1_Stage2_Model_Ready_v15_20261029.zip`을 요구함.
- 원본 ZIP을 별도로 보관/이동했다면, 기존 추출본의 데이터가 온전하더라도 관련 테스트 15개가 수집/실행 단계에서 중단됨.
- 기존 프로젝트 복사본에서 `data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/`의 필수 CSV·SQLite 15종과 SHA256 매니페스트를 확인했으며 원문 SHA-256과 일치함.

## 이번 패치
- 변경 파일: `pyrolysis/biochar_stage2/v15_extract.py` 한 개.
- 추가 파일: `tests/test_v39_v15_extracted_fallback.py`.
- 추출된 **16개 필수 파일**(매니페스트 포함)의 유무와 매니페스트의 크기·SHA-256을 확인한 후, 정상이면 ZIP 없이 사용.
- 추출본이 누락/변경됐을 때만 원본 ZIP으로 안전하게 복구. 원본 ZIP이 없으면 누락 파일명을 포함하여 실패함.
- 이동된 ZIP을 사용하려면 PowerShell에서 `$env:BIOCHAR_V15_ZIP_PATH = 'D:\\backups\\Stage1_Stage2_Model_Ready_v15_20261029.zip'`처럼 경로를 지정. (예시 경로를 실제 위치로 변경.)
- 원본 `qe_reported_mg_g` 97행·25 DOI, 다른 qe/qt/qmax 데이터, 모델, UI, 재학습 코드는 변경하지 않음.
- 중첩 백업 `BASELINE_v14_DO_NOT_EDIT.zip`은 추출하지 않음.

## Windows PowerShell 설치
프로젝트 위치: `C:\Users\jh030\Desktop\대학\학술대회\프로그램`

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\프로그램'
Copy-Item '.\pyrolysis\biochar_stage2\v15_extract.py' '..\v15_extract_before_v39_2.py' -Force
Expand-Archive -LiteralPath '.\Biochar_v39_Stage2_v15_MovedArchive_Compatibility_Fix.zip' -DestinationPath '.' -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v39_v15_extracted_fallback.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_biochar_stage2_v15.py tests\test_biochar_stage2_v16.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 데이터가 정말 누락된 경우

```powershell
Get-ChildItem '.\data\imports\Stage1_Stage2_Model_Ready_v15\Stage1_Stage2_Model_Ready_v15_20261029' | Select-Object Name,Length
```

필수 추출본을 다른 폴더로 옮겼다면 원래 `data/imports/...` 경로로 복구하거나 원본 ZIP의 실제 경로를 `BIOCHAR_V15_ZIP_PATH`로 지정해라. **다른 버전 데이터로 대체하면 v15 검증 자체가 무효가 됨**.

## 환경별 테스트 범위
- 작업환경: ZIP을 임시 이동한 상태에서 v15/v16 기능·추출본 fallback 및 신규 테스트 28개 통과(2개 의도적 미실행).
- 미실행 2개는 `test_stage2_app_loads`(작업환경 Streamlit 미설치)와 `test_chen16_reproducible`(작업환경 sklearn/CatBoost 인터페이스 불일치로 별도 오류)임.
- **Windows 가상환경에서 전체 회귀검사는 사용자가 수행해야 함.**
