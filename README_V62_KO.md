# v62 — 제조온도 순위 예측의 한계 분석 및 Pairwise 모델 독립 재학습

## 목적

v61에서 확인한 물질수지 상한 clipping/공동 1위 문제의 원인을 진단하고, 서로 다른 제조시료 두 개의 관측 우열을 직접 학습하는 **Pairwise Logistic** 모델을 별도 연구용으로 재학습한다. v59 DOI/제조시료 가중 Model B, v60 제조조건 RandomForest, v60 clipping 전후 Model B와 **같은 v60 188행 DOI GroupKFold**의 비교계열 15개에서 비교한다.

### 모델 및 지표 정의

- Pairwise 학습 타깃: 동일한 *학습 DOI* 제조비교계열에서 관측 제거율 순서. 동일 panel 평균을 사용하는 것은 **학습 fold에 한정**한다.
- Pairwise 입력: 원료·열분해/개질 조건의 차이. 전처리는 **각 train fold에서만** fit. 짝 (i,j)와 (j,i)를 동시에 넣고 `fit_intercept=False`로 역방향 확률의 대칭성을 확보한다.
- DOI/시료 종속성을 고려해 학습 쌍의 가중치를 정함. 후보쌍을 독립 논문 수로 취급하지 않는다.
- 출력: 후보 panel 내부 **상대 우열 점수(평균 win probability)**; `qe`/`qmax`/제거율 절대값이 아니다.
- 평가: Spearman ρ (정의되지 않는 경우 NA), pairwise 일치율 (예측 동률 0.5점), 공동 1위 수, regret, DOI bootstrap CI. 비교 시 모든 모델이 실제로 유효한 panel의 교집합만 사용한다.
- 기존 Model B `raw` 점수는 물질수지 위반 가능성이 있어 **순위 실패 메커니즘 진단**만 하며 흡착량 예측/실용 추천에 활용하지 않는다.

## 필요한 기존 프로젝트 파일

- `data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv`
- `data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv`
- `reports/v60_separate_effects/all188_doi_oof_predictions.csv`
- `reports/v60_separate_effects/panel_evidence_quality.csv`
- 기존 `scripts/run_v59_doi_material_weighting.py`, `scripts/run_v60_separate_effects.py` 및 하위 코드.

## PowerShell (v47 프로젝트 최상위 폴더에서)

```powershell
$ErrorActionPreference = 'Stop'
$zip = '.\Biochar_v62_Pairwise_Temperature_Ranking_v47_Patch.zip'
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $zip)) { throw 'v62 ZIP을 v47 폴더로 이동하세요.' }
if (!(Test-Path $python)) { throw '기존 프로그램\.venv Python 환경을 찾을 수 없습니다.' }
if (!(Test-Path '.\reports\v60_separate_effects\all188_doi_oof_predictions.csv')) { throw 'v60 OOF가 없습니다.' }
if (!(Test-Path '.\scripts\run_v60_separate_effects.py')) { throw 'v60 코드가 없습니다.' }
Expand-Archive -LiteralPath $zip -DestinationPath '.' -Force
& $python -m scripts.run_v62_pairwise_diagnostics --root .
if ($LASTEXITCODE -ne 0) { throw 'v62 Pairwise 재학습·분석 실패' }
& $python -m pytest tests/test_v62_pairwise_diagnostics.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'v62 테스트 실패' }
Get-Content '.\reports\v62_pairwise_manufacturing\LOCAL_RESULTS_KO.md' -Encoding UTF8
```

**Windows 로컬에선 v60.1의 14/15 Spearman 유효 계열 결과가 재현될 수 있다.** 기준 숫자를 고정하지 않고 그 컴퓨터의 OOF를 우선 평가하므로, 기존 제작환경의 15/15 값과 혼용하지 않는다.

## 신규 산출물

- `reports/v62_pairwise_manufacturing/LOCAL_RESULTS_KO.md`: 설치 PC 재실행 시 생성되는 수치 요약
- `panel_oof_predictions.csv`: 제조 panel 59행의 pairwise OOF 상대점수와 기존 v60 OOF
- `panel_rank_metrics.csv`: 동일 panel별 4개 방식 비교
- `pairwise_feature_ambiguity.csv`: 입력 특징상 구별 불가능한 관측쌍 여부
- `manufacturing_feature_spreads.csv`: 제조 조건별 계열 내부 변동/결측
- `model_comparison.csv`, `paired_panel_rhos.csv`, `manifest_v62.json`, `fold_training_audit.csv`
- `deployment/research_v62_NOT_ACTIVE/pairwise_logit_188_RELATIVE_SCORE_NOT_QE.joblib`: 전체 학습 DOI 모델. OOF 성능 계산에 이 artifact를 재사용하지 않음.

## 주요 한계와 판단

- 기존 15개 제조조건 panel은 **열분해온도만 panel 내부에서 변화**한다. 유지시간·승온·분위기 등 다른 제조변수 효과는 직접 검증하지 못했다.
- 15개 panel 중 흡착조건 완전 보고 4개, strict qe 2개는 같은 1개 DOI. DOI 독립 표본 부족.
- 보고서 비교는 과거 DOI가 모델 개선/선택에 사용된 **사후 평가**. Pairwise 접근이 RF보다 정확하다는 증거는 없음. 추후 신규 DOI에 대한 사전 봉인평가 전까지 최적 제조조건 추천 사용 금지.
- `data/` 원본, 기존 Joblib, 활성 Streamlit 코드/설정 **수정하지 않음**. v62는 연구용 새 모델과 보고서만 추가.
