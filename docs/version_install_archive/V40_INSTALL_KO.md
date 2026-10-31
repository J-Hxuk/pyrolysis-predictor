# v40 연구 패치 설치 및 실행

전제: v39 및 `scripts/build_v37_full_refit.py`, `pyrolysis/biochar_stage2/v15_features.py`가 설치된 기존 프로젝트. `Python 3.11+`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `xgboost`, `pytest`를 사용한다. 모델 파일은 작성 환경의 라이브러리 버전과 다를 수 있으니 Windows 로컬에서 재학습 권장.

1. 기존 프로그램 폴더의 git commit 또는 별도 ZIP 백업을 만든다(임의의 파일 삭제 금지).
2. `Biochar_v40_NineDOIs_FullRefit_and_Hygiene_Patch.zip`을 프로젝트 폴더에 다운로드하고 압축 해제한다. v39 기준 CSV 2개도 함께 제공하므로 과거 ZIP 원본 경로 의존성이 없다.
3. PowerShell에서 다음 명령 실행:

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\프로그램'
Expand-Archive -LiteralPath '.\Biochar_v40_NineDOIs_FullRefit_and_Hygiene_Patch.zip' -DestinationPath '.' -Force
.\.venv\Scripts\python.exe -m scripts.build_v40_new9 --root .
.\.venv\Scripts\python.exe -m scripts.evaluate_v40_research --root .
.\.venv\Scripts\python.exe -m scripts.build_v40_envelope --root .
.\.venv\Scripts\python.exe -m pytest tests\test_v40_new9.py -q
```

4. 프로젝트 전체 테스트:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

5. **읽기 전용 폴더 정리 진단**:

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_project_hygiene_v40 --root . --out reports\v40_hygiene\my_pc_audit.json
```

이 명령은 **어떤 파일도 삭제하지 않는다.** 과거에 적용한 ZIP 패치들을 보관 폴더로 이동할 때는 추출된 활성 데이터, 모델 파일, DOI 원문·증빙 파일을 함께 옮기지 않는다. ZIP 이동 후 프로그램+전체 테스트 통과가 최종 확인되기 전에는 원본 백업을 삭제하지 않는다.

데이터 구조:
- `data/research_v40_inputs/`: v39를 고정한 읽기 전용 연구용 입력 스냅샷
- `reports/v40_new9/`: 9편 DOI 원문 감사, 신규행, 누적 학습 후보, 비교검증
- `deployment/research_v40_NOT_ACTIVE/`: Windows 재학습 후 생성될 6개 연구 모델, 배포 금지
- `scripts/`: 데이터 생성·검증·진단 코드
- `tests/`: v40 재발 방지 테스트

**주의:** v40은 **증분 패치**이며 Streamlit 활성 모델을 교체하지 않는다. 최신 ZIP만 무작정 단독 실행 파일로 취급하거나, v15/v41 과거 추출본을 삭제하면 구형 테스트의 파일 의존성 문제가 재발할 수 있다.
