# v39 전체 회귀 테스트 복구 패치

## 문제 확인
- `test_stage2_app_loads`: v28에서 메뉴 4개로 개편했으나 v15 테스트가 구형 라벨을 비교.
- `test_biochar_app.py`, `test_biochar_stage1.py`: 원본 v41 백업 ZIP을 전달용에서 제외했으나, 코드가 기존 추출본이 있어도 무조건 ZIP을 요구함.

## 수정
- `pyrolysis/biochar_stage1/extract.py`: 모든 필수 자료가 `data/imports/Biochar_Complete_Project_v41`에 존재하면 ZIP 없이 로드. 일부 누락되면 ZIP이 있을 때 누락분만 추출, 없으면 명시적으로 실패. 원자료 합성/임의 복구 없음.
- `tests/test_biochar_stage2_v15.py`: 현재 사이드바 4개 라벨로 업데이트. 모델·평가 규칙 테스트는 그대로 유지.
- `tests/test_v39_legacy_archive_fallback.py`: ZIP 없는 완전 추출본, ZIP 없는 불완전 추출본, ZIP이 있는 부분 복구 테스트.

기존 v41 원본 ZIP을 삭제할 것을 권장하지 않음. v41의 추출 파일 18개가 실제로 존재하는지 먼저 확인 필요. 현재 패치는 학습 데이터나 모델을 수정하지 않음.

## 확인
```
.\.venv\Scripts\python.exe -m pytest tests\test_v39_legacy_archive_fallback.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_biochar_stage2_v15.py tests\test_biochar_app.py tests\test_biochar_stage1.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```
