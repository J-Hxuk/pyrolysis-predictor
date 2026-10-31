# v53 Stage2 실제 재학습 패치 (v47 프로젝트 루트용)

2026-10-31 작업. 기존 `data/`, `deployment/stage2_models/active.json`, 기존 v52 모델 등은 수정하지 않는다.

## 이 ZIP의 성격
- 이미 **실제 학습 완료**된 연구용 A/B XGBoost Joblib 4개(148 및 164 시나리오), 원문 확인 자료, 재현 스크립트, 예측/지표 CSV, 검증 보고서 포함.
- `vNext10` 사과나무 8행은 v52 24행 내부와 **전부 중복**. 신규 행으로 재추가하지 않는다.
- `vNext11` 신규 승인 0행, 과거 tea branch 5건 strict-qe 근거 보강만 존재. 중복 추가하지 않는다.
- 148행 연구 시나리오는 target qe 근거 8행을 쓰지만 **TPV cross-source 동일 배치 확인이 부족해 학습용 승인 완료 상태가 아님.** 164행은 불확실 qt 16행이 추가되어 특히 민감도 분석 전용.
- 활성 배포모델 교체 **금지**. 평가 보고서에 상한 clipping 전 위반율 반드시 함께 표시.

## 설치 — 기존 v47 폴더의 PowerShell에서 실행
ZIP 파일을 v47 폴더에 직접 옮기고 **v47 프로젝트 폴더에서** 터미널 열기.

```powershell
$ErrorActionPreference = "Stop"
$zip = ".\Biochar_v53_Actual_Refit_v47_Patch.zip"
if (!(Test-Path $zip)) { throw "이 폴더에 v53 ZIP이 없습니다." }
if (!(Test-Path ".\scripts\build_v46_phase8.py")) { throw "현재 폴더가 v47 프로젝트 루트가 아닙니다." }
if (Test-Path ".\deployment\research_v53_NOT_ACTIVE\manifest_v53.json") {
    throw "이미 v53 연구 모델이 존재합니다. 덮어쓰기 전 백업·확인이 필요합니다."
}
Expand-Archive -LiteralPath $zip -DestinationPath "."
Get-Content ".\reports\v53_evidence_tier_refit\RESULTS_KO.md" -Encoding UTF8
```

학습된 연구 모델 4개는 `deployment/research_v53_NOT_ACTIVE/`에 있다. **ZIP을 압축 해제하면 이미 학습된 결과가 들어오므로 재학습을 해야 하는 것은 아니다.**

## 검증 및 선택적 재현

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py -q --tb=short
```

환경 버전 차이로 Joblib 실행이 실패하면, 활성 모델은 그대로 두고 **연구용 모델만** 현지 환경에서 다시 생성:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_v53_research_refit --root .
```

추가 설치 라이브러리: pandas, numpy, scikit-learn, xgboost, joblib, pytest. 개발 실행환경: Python 3.13.5 / scikit-learn 1.8.0 / xgboost 3.1.3.

## 사용해야 할 보고서
- `reports/v53_evidence_tier_refit/RESULTS_KO.md` 논문 제출용 주의사항 포함.
- `v53_fixed140_metrics.csv`: 공통 140행 5fold MAE/RMSE/R²/물리성.
- `v53_fixed140_predictions.csv`: 개별 OOF 예측·원시 모델 출력·DOI.
- `doi_cluster_paired_bootstrap.csv`: DOI 단위 paired bootstrap CI.
- `apple_preinclusion_single_doi_metrics.csv`: 원래 140행만으로 새 사과나무 DOI 시험(독립 external 아님).
- `deployment/research_v53_NOT_ACTIVE/manifest_v53.json`: 학습 모델별 SHA256.

## 후속 우선순위
1. clipping에 의존하는 기존 회귀의 약점 해결: 물리적 bounded fraction 모형과 비교.
2. strict qe 엄격집합을 단계적으로 정의해 exploratory 데이터 성능과 분리.
3. 제조온도 추천의 순위 지표를 같은 DOI·흡착조건 내부에서 평가.
