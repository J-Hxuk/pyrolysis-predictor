# Stage 2 UI v17 확인된 범위

- 기준: 사용자가 전달한 `Biochar_Review_Package.zip`의 복사본. 원본 ZIP과 기존 배포 모델, 데이터셋, Stage1 코드, v15/v16 학습·검증 코드 변경 없음.
- 환경: Python 3.13, scikit-learn 1.8.0, XGBoost 3.1.3, NumPy 2.3.5, Streamlit 미설치.
- `python -m pytest -q tests/test_biochar_stage2_input_support.py`: **10 passed**.
- `python -m pytest -q tests/test_biochar_stage2_v16.py`: **8 passed**.
- `python -m pytest -q tests/test_biochar_stage2_v15.py -k 'not stage2_app_loads'`: **14 passed, 1 failed, 1 deselected**. 실패는 UI가 아니라 설치된 `CatBoostRegressor`가 scikit-learn 1.8의 `__sklearn_tags__`와 호환되지 않아 발생. 원본 학습 코드는 수정하지 않음.
- 저장된 Stage2 A/B/C joblib 모델을 현재 환경에서 `joblib.load`하면 `AttributeError: Can't get attribute '_RemainderColsList'` 발생. 저장 당시 scikit-learn 1.5.2와 버전 불일치. **이 환경에서 모델 수치 추론 성공을 주장하지 않음.**
- `streamlit` 패키지가 현재 환경에 없으며 패키지 설치를 시도했지만 외부 연결 실패. 따라서 Streamlit 화면 자체의 렌더링 / `AppTest`는 **미실행**. 코드 문법 검사는 성공.
- Windows 사용자의 기존 `.venv`에서도 `scripts/check_stage2_environment.py`를 실행해 저장 모델 로드와 샘플 예측을 점검해야 함. 기존 `.venv` 삭제/재설치 금지.

## 추후 모델 관련 과제 (본 UI 변경과 별개)

1. Stage1 v60 실제 추론은 기존 파일 명세상 조성변수를 전처리에서 제외함. 조성 입력을 실제 오차 보정에 사용하려면 별도 모델 학습과 DOI 독립 검증이 필요.
2. Stage2 Core 97에는 raw 조성이 대체로 비어 있어 구성성분 입력으로 바로 개선된 계수를 적용할 근거가 없음.
3. Stage1 v60 원본 패키지 ZIP은 전달용 ZIP에서 제외됐고 현재 작업 복사본에도 확인되지 않음. **v60 예측기를 직접 재현 테스트하지 못함**. 현재 v60의 정적 입력 명세와 보고된 검증 결과만 확인함.
4. Style CSS와 새로운 입력 위젯은 Streamlit 1.40(프로젝트의 `requirements.txt`)에서 사용자 컴퓨터의 실제 브라우저 검증이 필요.
