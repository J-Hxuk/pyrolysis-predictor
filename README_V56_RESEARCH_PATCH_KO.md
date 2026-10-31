# Biochar Stage1/Stage2 v56 — 신규 DOI Source-out 검증 및 미래 검증 봉인

버전: 2026-10-31 / 연구용 / 기존 v47·v55 통합 프로젝트 위에 설치.

## 왜 이 패치를 만들었나?

v55에서 고정 140행에 신규 DOI를 학습 자료로 추가해 MAE를 비교했지만, 그런 평가는 **추가한 논문의 신규 시료에 대한 성능**을 측정하지 않는다. v56은 신규 DOI 관측을 모두 시험 집단으로 분리하여 해당 DOI를 전혀 학습시키지 않고 예측하는 사후 source-out 시험을 제공한다. 이미 논문 결과를 본 뒤 분석했으므로 v56 결과를 “독립 사전 봉인 blind test”라 부르지 않는다.

## 포함된 분석

1. Stage2 Model A/B: 기존 140행 기반 vs 다른 신규 DOI 8/4행 추가 학습 vs 조건 불확실 Cocoa 포함 민감도. 평가 대상 사과나무 Cu/Zn 8행, Sugarcane Pb 4행, Cocoa Cd 4행.
2. Stage2 제조온도 순위: 같은 논문·같은 금속·같은 흡착조건의 조건계열별 Spearman ρ, top-1 regret 및 동일값(tie) 불능 판정. Model A가 제조조건을 받지 않아 동일 예측을 내는 경우 순위 성능으로 간주하지 않는다.
3. Stage1: 신규 DOI 5편에서 원문 BET 29행, TPV 10행에 대해 기존 Stage1만 학습한 경우와 다른 신규 DOI를 추가한 경우 비교.
4. 별도 Langmuir `qmax`: 기존 32행 모델 vs 다른 신규 DOI 추가, Ragweed/Horseweed 및 Cocoa에서 source-out. `qmax`는 `qe`와 절대 병합하지 않음.
5. **미래 신규 DOI 검증 봉인**: 모델 파일 SHA256, 원본 입력 SHA256, 과거 DOI 및 추가 인용 소재 출처를 봉인한다. 봉인한 **이후** 처음 들어온 실험 평형 qe 논문만 대상으로 비갱신 평가.

## 설치

ZIP을 `Biochar_v47_Integrated_Project` 루트에서 압축 해제. `scripts/`, `tests/`, `reports/v56_source_out_prospective_seal/`, `data/imports/v56_template/`만 추가된다. 기존 DB/활성 배포/모델은 수정하지 않는다. v55 설치 및 해당 학습 자료 파일들이 필요하다.

기존 `.venv` 예:

```powershell
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
& $python -m scripts.run_v56_source_holdout --root . --mode retrospective
& $python -m scripts.run_v56_source_holdout --root . --mode seal
& $python -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py tests/test_v54_bounded_rank.py tests/test_v55_source_cumulative_refit.py tests/test_v56_source_holdout.py -q --tb=short
```

**중요:** 데이터 또는 v55 연구모델의 로컬 재학습/변경을 마친 **후에** `--mode seal`을 실행한다. 그 후 봉인한 모델은 신규 DOI 평가 전에 재학습하거나 재선택하지 않는다. 기존 봉인 파일과 입력 해시가 바뀌면 덮어쓰지 않고 오류를 발생시킨다.

미래에 정말 새로운 논문을 원문 확인한 경우(사전 봉인 후, 기존 DOI·동일 물리 시료 계보에 노출되지 않은 경우)만 다음처럼 실행한다:

```powershell
& $python -m scripts.run_v56_source_holdout --root . --mode prospective --candidate '.\data\imports\v56_template\new_verified_qe.csv'
```

템플릿은 `data/imports/v56_template/FUTURE_QE_SOURCE_TEMPLATE.csv` (헤더만 제공, 가상 측정값 없음). `qe_source_evidence_status=SOURCE_VERIFIED_EXPERIMENTAL_EQUILIBRIUM_QE`, `target_kind=EXPERIMENTAL_EQUILIBRIUM_QE_MG_G`, `sample_batch_independent=true`, `source_lineage_doi=NONE_VERIFIED` 또는 `;` 구분 DOI 목록 등의 원문 검증을 요구한다. 자동 검사가 외부 출처의 실제 독립성까지 증명하는 것은 아니므로 논문별 수동 검증도 필요하다.

## 재현성과 한계

- 제작 환경: Python 3.13.5, scikit-learn 1.8.0, XGBoost 3.1.3. 사용자 PC 3.12/1.5.2에서는 재학습된 모델의 예측·MAE가 달라질 수 있으며 v56은 값을 고정해 통과시키지 않는다.
- 제작 환경 통합 테스트 **33 passed**. `reports/v56_source_out_prospective_seal/` 하위 CSV는 제작 환경에서 계산한 참고 결과이며, 로컬 실행 시 갱신된다.
- 사후 source-out만 수행했고 새로운 실측 실험은 없었다. 이것은 공통 기존 DOI 내 삽입 평가와 다르지만 선택 편향을 제거한 완전한 prospective 검증은 아니다.
- 결과는 논문별 표본 4~12개로 작고, 특히 Cocoa 초기농도 충돌과 사과나무 2017 소재 계보 노출 이슈가 있다.
