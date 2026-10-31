# Stage 1 v18 조성변수 실험 — 설치 및 실행

이 패치는 v17 프로젝트에 **읽기 전용 성능 검증 코드와 결과 보고서**를 추가한다. 기존 v60 모델/Stage 2 모델, UI, 데이터는 변경하지 않는다.

## 설치
1. 기존 `프로그램` 폴더 전체를 백업한다.
2. ZIP을 `프로그램` 폴더에 압축 해제해 `pyrolysis/`, `scripts/`, `tests/`, `reports/`, `docs/` 위치에 추가/병합한다. (`.venv`는 절대 삭제하지 않는다.)
3. 프로젝트 폴더에서 아래를 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stage1_composition_v18.py -q
.\.venv\Scripts\python.exe -m scripts.run_stage1_composition_v18
.\.venv\Scripts\python.exe -m scripts.plot_stage1_composition_v18
```

- 결과 확인: `reports/stage1_composition_v18/RESULTS_KO.md`와 `model_comparison.csv`, `fig_*.png`.
- 이 실험은 v60 배포 모델을 재학습하거나 기존 앱의 예측 수치를 바꾸지 않는다.
- **실험 모델 배포 전** 별도 잠금 DOI 외부 검증과 조성 basis 검증이 필요하다.
- 실행 환경 라이브러리 버전에 따라 수치가 조금 달라질 수 있다. 기본 원본 데이터는 `data/imports/Biochar_Complete_Project_v41/eval37/base.csv` 와 v60 master에서 읽는다.
