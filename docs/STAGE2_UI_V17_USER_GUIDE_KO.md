# Stage 2 v17 — 실행 및 사용 가이드

## 실행

PowerShell에서 기존 `프로그램` 프로젝트 폴더로 이동한 후:

```powershell
.\.venv\Scripts\python.exe scripts\check_stage2_environment.py
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

- 모델 로드 검사 결과가 `FAIL`이면 앱을 업데이트하거나 모델을 덮어쓰지 말고 Python / scikit-learn / xgboost 버전 및 오류 메시지를 확인하세요.
- 저장된 Stage 2 모델은 scikit-learn 1.5.2, XGBoost 2.1.3에서 생성되었습니다. 다른 버전에서 joblib 로드가 실패할 수 있습니다.
- 현재 작업환경(비 Windows)에서 `scikit-learn 1.8.0`으로 직접 검증했을 때 `_RemainderColsList` 역직렬화 오류가 발생했습니다. 기존 Windows `.venv`에서 호환성을 별도 확인해야 합니다.
- `.venv`를 삭제하거나 덮어쓰지 마세요. 필요한 경우 **별도 가상환경**을 만들어 구버전 모델 호환성을 검증하세요.

## 화면 흐름

1. **B. Adsorption & Manufacture**: 원료 이름을 한국어/영어로 자유 입력. `왕겨`, `rice hull` 같은 동의어는 같은 *등록 원료명*에 연결되지만, 다른 원료를 임의 대체하지 않습니다.
2. 등록 원료와 일치하면 접을 수 있는 **문헌 관측치** 표가 나타납니다. 이 표는 실측 문헌 참고자료이지 예측 모델의 출력이 아닙니다.
3. 오염물질, C0, pH, dose, 흡착온도, 접촉시간을 지정합니다. 제조조건은 Model B에만 사용합니다.
4. 원료 원소분석/공업분석/구성성분은 아는 부분만 입력할 수 있습니다. **현재 배포된 Model A/B는 조성을 사용하지 않으며, 조성에 의한 오차보정도 수행하지 않습니다.**
5. `머신러닝 예측`에서 **원시 qe**, **제약 보정 후 qe**, **C0/dose 상한**, **DOI Nested CV** 지표를 확인하세요. 숫자는 학습 모델의 추정치이며 엄격 평형 qe로 인증된 값이 아닙니다.
6. **C. Condition Comparison**은 Model B의 온도별 추정 응답을 비교합니다. 최적 제조조건을 검증하거나 실제 실험하는 기능은 아닙니다.
7. **D. Validation**은 v16 그룹 Nested CV 및 Chen16 한정 추천 평가를 보여줍니다.

## 모델 구분

- Stage 1 구형 (`biochar_app.py`): C/H/N/O (daf), VM/Ash/FC (dry) + 열분해조건 → BET. 별도의 제한된 문헌 데이터셋 기반.
- Stage 1 v60: 원료 이름·제조/활성화조건 기반 BET/TPV. 현재 메타데이터상 원료 조성변수는 ColumnTransformer에서 제외됨. 잠금 외부검증은 음의 R².
- Stage 2 Model A: 흡착조건만 → 탐색적 reported qe.
- Stage 2 Model B: A + 원료명·제조공정 → 탐색적 reported qe. 외부 DOI 검증에서 A보다 열세.
- Stage 2 Model C: B + **실측** BET/TPV, 물성 추가의 진단 모델.
- Model D: Stage 1 예측 BET/TPV를 Stage 2로 전달하는 경로. 현재 비활성.

**원료 이름을 입력한다고 원소조성이 생성되지 않습니다.** 신규 원료에 Model A가 예측한 값은 원료 고유 특성을 반영한 추정값이 아닙니다. Model B는 미학습 이름을 받아들여 실행 가능해도 원료 외삽 검증이 충분하지 않습니다.

## 코드 변경 범위

- `pyrolysis/biochar_stage2/input_support.py` (신규 순수 함수)
- `pyrolysis/biochar_ui/stage2_pages.py` (Stage2 UI)
- `scripts/check_stage2_environment.py` (신규 환경 호환성 진단)
- `tests/test_biochar_stage2_input_support.py` (신규 단위 테스트)

**기존 데이터·저장 모델·Stage1 UI·v15/v16 분석 코드는 수정하지 않았습니다.**
