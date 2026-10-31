# v46 Windows NumPy/joblib 모델 호환성 복구

## 원인
v46 연구용 `.joblib` 모델은 다른 Python/NumPy/scikit-learn 환경에서 저장되었습니다.
`numpy.random._pcg64.PCG64 is not a known BitGenerator module` 오류는
`joblib.load()`가 모델 내부 난수 상태를 복원하지 못할 때 발생할 수 있습니다.
학습 데이터 오류를 의미하지 않습니다.

## 적용
아래 ZIP을 `프로그램` 폴더에 놓고 압축 해제한 뒤 Windows PowerShell에서 실행합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Expand-Archive -LiteralPath ".\Biochar_v46_Local_Model_Compatibility_Fix.zip" -DestinationPath "." -Force

# v46 원본 데이터를 이용하여 연구용 4모델만 Windows 환경에서 로컬 재학습/저장
.\.venv\Scripts\python.exe -m scripts.rebuild_v46_research_models_local --root .

# 기존 v46 테스트
.\.venv\Scripts\python.exe -m pytest tests\test_v46_phase8.py tests\test_v46_local_model_compat.py -q
```

## 안전성
- v46의 원문·통합 CSV를 변경하지 않습니다.
- `deployment/research_v46_NOT_ACTIVE` 연구용 모델 4개만 새로 저장합니다.
- 기존 활성 모델과 Streamlit UI는 변경하지 않습니다.
- 기존 연구용 `.joblib`의 원형은 이미 설치 ZIP에서 복구 가능합니다.
- 4개 모델을 모두 학습/임시 로딩 검사한 뒤에만 최종 파일을 교체합니다.
- 재학습은 원 데이터로 하는 **로컬 전체 재학습**이며, 이전 배포 모델 이어학습이 아닙니다.
- 모델 직렬화 호환성 검증 결과를 `V46_LOCAL_REBUILD_MANIFEST.json`에 남깁니다.

## 참고
이전 환경과 다른 라이브러리 버전이라 예측값의 마지막 자리 또는 모델 특성이
달라질 수 있습니다. 이전 평가 보고서의 교차검증 점수를 이 로컬 재저장만으로
갱신한 것으로 해석하면 안 됩니다.
