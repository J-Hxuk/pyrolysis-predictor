# v55 신규 5편 누적 원문 데이터 + Stage1/Stage2/qmax 재학습 결과

2026-10-31 · **연구용, 배포/활성 모델 변경 없음**

## 데이터 처리 원칙 및 실제 누적

사용자가 전달한 `biochar_new5(1).zip`의 논문 원문 PDF 5편, supplementary PDF 2편을 `data/imports/v55_original_sources/pdfs/`에 원본 그대로 보존했다. 모든 출처와 해시는 `Source_Registry_5.csv`에서 조회 가능하다. 기존 v46 Stage2 140행, v52 사과나무 24행, 기존 Stage1 417행을 변경하지 않았다.

1. **Stage2 qe/24h 실험 흡착량 8행**: Sugarcane bagasse/Pb 4행(2014 Ding et al., p4, C0 80 mg/L, dose 2 g/L, 24h, 250/400/500/600°C, 15.9/10.7/11.4/8.3 mg/g); Cocoa pod hull/Cd 4행(2026, Table3 `qeexp` 5.93/2.47/2.80/3.27 mg/g). 후자는 50 mg/50mL 및 실험 제거율을 사용하면 C0=10 mg/L가 역산되지만, Methods kinetics 문단은 C0=5 mg/L라고 서술한다. **C0는 `INFERRED`, 원문 수치 충돌 보존, 민감도 분석에만 사용**. Cocoa commercial activated carbon CSAC는 원료·제조 조건이 다른 상용재라 4행에 포함하지 않음.
2. **Stage1 특성 29개 BET/10개 TPV**: Cocoa 4 BET/TPV, Ragweed/Horseweed 6 BET/TPV, Sugarcane 4 BET, Rice Straw/anaerobically digested Rice Straw 8 BET, Municipal sludge 7 BET. 원료 조성 입력에는 biochar 제조 후 원소분석을 혼입하지 않음. 공정조건 미보고는 `NaN` 유지.
3. **분리 타깃 `Langmuir qmax` 16행**: Ragweed/Horseweed Cd/Pb 12행(2020 Table3), Cocoa 4행(2026 Langmuir fit). 기존 별도 qmax 32행과만 결합(최대 48행). qe와 혼합 금지.
4. **Sludge Qmax 14행**: 2013 Table2의 표기 단위 `mg kg−1`이 다른 연구의 `mg/g`과 일치하지 않아, 원문값과 인쇄 단위를 `Sludge_14_QMAX_UNRESOLVED_UNIT_SOURCE_LEDGER.csv`에 그대로 기록. 억지 1,000배 변환이나 다른 타깃에 자동 혼입하지 않음. 원문은 버리지 않음.
5. Rice 및 Sludge 문헌의 그림/kinetic fitted qe는 **측정 qe와 구별**한다. 도표에서 좌표 근삿값을 자동 생성해 학습값으로 간주하지 않았다.

## Stage2 A/B: 고정 기존 140행·32 DOI, 동일 5-fold GroupKFold

| 학습 시나리오 | A MAE (mg/g) | B MAE (mg/g) | 역할 |
|---|---:|---:|---|
| 기존140 | 17.247 | 17.690 | 기존 기준선 |
| 기존+사과나무 8 = 148 | 17.190 | 17.643 | 기존 v53 연구 시나리오 |
| 기존+사과나무 24 = 164 | 16.760 | 17.761 | 미확정 평형 16행 포함 민감도 |
| 148+Sugarcane 4 = **152** | **16.588** | 17.530 | 새 원문 직접 관측 주 분석 |
| 152+Cocoa 4 = **156** | 16.666 | **17.386** | Cocoa C0 추정 민감도 |
| 164+Sugarcane 4 = **168** | **16.155** | 17.487 | 미확정 평형 포함 민감도 |
| 168+Cocoa 4 = **172** | 16.325 | 17.431 | 두 불확실성 포함 민감도 |

*평가 대상은 항상 기존 140행으로 고정했고, 신규 DOI 시료는 각 fold의 학습 쪽으로만 추가했으므로, **이 평가는 새로운 DOI에 대한 독립 외부검증이 아니라 학습 증량 민감도 분석**이다.* 기존 140행에도 일부 exploratory/conditional 자료가 포함되어 있으므로 strictly verified qe-only 성능이 아니다.

Stage2 A 152행의 기존 대비 MAE 감소는 0.659 mg/g(약 3.82%). DOI 단위 bootstrap 95% 구간은 **[−0.020, 1.426] mg/g**으로 0을 포함. 168행은 기존보다 1.092 mg/g 낮지만 95% 구간 **[−0.218, 2.935] mg/g**으로 0을 포함. 그러므로 일반화 향상의 통계적 확증은 없음. 전체 수치는 `stage2_same140_group_oof_metrics.csv`, OOF 원자료는 `stage2_same140_group_oof_predictions.csv` 및 `doi_paired_bootstrap_v55.csv` 참고.

초기 140행으로만 학습해 새 논문을 보기 전 평가했을 때 Sugarcane 4행에 대한 A MAE 14.48, B MAE 16.95 mg/g, Cocoa 4행 A/B MAE 6.38 mg/g이었다. 이는 DOI별 표본 4개씩의 탐색적 external-source challenge일 뿐, 사용자 정의 엄격한 진짜 외부 검증 세트가 아니다.

## Stage1: 고정 기존 관측 GroupKFold

| BET 증량 | 평가 표본 | MAE (m²/g) | 비고 |
|---|---:|---:|---|
| 기존 | 361 / 49 DOI | 247.971 | 기존 기준선 |
| Cocoa+Ragweed/Horseweed+Sugarcane 14 | 동일 | **240.276** | 가장 낮은 탐색적 MAE |
| 위 14 + Rice 8 | 동일 | 244.296 | 효과 약화 |
| 위 14 + Sludge 7 | 동일 | 243.199 | 효과 약화 |
| **모든 신규 BET 29** | 동일 | 248.692 | 기존보다 악화 |

TPV는 기존 231행 /35 DOI MAE **0.17019 cm³/g**, 신규 TPV 10행을 더하면 **0.17310 cm³/g**으로 악화. **모든 원문값을 보존**하지만 실제 모델 승격은 별도로 판단해야 한다. 신규 Stage1 완전 누적 연구 모델은 446행(원료조성 다수 NR), Stage1 베이스 417 변경 없음. 이 신규 데이터 조합은 고정된 같은 평가 집단에서 사후 선택했으므로 실제 외부 개선의 독립 증거는 아니다.

## 분리 Qmax 회귀 (절대 qe와 혼합 금지)

기존 32행/3 DOI 같은 평가 집단에서 Qmax MAE는 **30.19 mg/g**. Ragweed/Horseweed 12행 추가 후 **67.56 mg/g**, Cocoa까지 16행 추가 후 **67.37 mg/g**. 학습용 Joblib을 저장했지만, 새 출처 간 스케일·금속·원료 차이가 커서 **기존 qmax 모델 대비 자동 승격 금지**. 실제 Qmax는 Langmuir 피팅의 파라미터이며 물질수지의 단일 C0/dose 상한과 동일한 제약을 적용할 수 없다.

## 제조조건 추천: Chen 16조건

| Stage2 B 학습 | Spearman ρ | Top-1 regret (%p) |
|---|---:|---:|
| 기존140 | 0.842 | 0.00 |
| 148 | 0.834 | 0.00 |
| **152** | **0.443** | **65.55** |
| 156 (Cocoa C0 추정) | 0.674 | 0.00 |
| 168 (미확정 평형 포함) | 0.535 | 0.00 |
| 172 (두 불확실성 포함) | 0.515 | 65.55 |

실험 pH 미보고로 결측치 대체. 16개 제조조건으로만 구성된 논문 내부 benchmark이며, 사전에 별도 holdout으로 봉인된 데이터는 아님. **Stage2 MAE 감소가 제조조건 추천 성능을 보증하지 않는다는 강한 반례**. A는 이 16개 조건을 구분할 제조조건 입력이 없어서 tie로 순위 평가 불가.

## 물리성 및 적용 판단

모든 Stage2 A/B 결과에 대해 추론 후 `0 ≤ qe ≤ C0/dose` clipping을 유지했지만, raw prediction에는 37–50개의 상한 위반이 관찰된다(모델/시나리오에 따라 다름). 물질수지 제약이 모형 구조적으로 학습되지는 않았고, v54 bounded regression도 기존보다 MAE가 컸다. 예측 상한 위반율과 clipping 후 MAE를 모두 표시한다.

**권고:** 152행 Model A는 연구용 성능 개선 후보, BET 14행 증량 모델은 탐색적 후보. Model B의 추천 용도는 기존 140/148 학습본을 우선 보존. 신뢰할만한 qmax/TPV 개선은 이번에 확인되지 않음. 가장 낮은 MAE 모델을 곧바로 활성화하거나 독립 검증 성능이라고 발표하지 않는다. 모델 선택 후의 독립 DOI 평가, source-boundary 실험 및 소수 DOI에서의 견고성이 추가 필요.

## 로컬 실행과 파일

- `python -m scripts.run_v55_source_cumulative_refit --root .`
- `python -m scripts.run_v55_robustness_audits --root .`
- `python -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py tests/test_v54_bounded_rank.py tests/test_v55_source_cumulative_refit.py -q --tb=short`
- Python 3.13.5/scikit-learn 1.8.0/XGBoost 3.1.3에서 **23 passed**.
- v55 `joblib`은 다른 scikit-learn 버전(사용자 PC 1.5.2)에서 로드될 때 경고/예측 불일치 가능. 사용자 PC `.venv`로 스크립트 재학습해서 동일 환경의 모델을 만드는 것이 적절함. 숫자 자체를 상수로 강제하는 테스트는 사용하지 않음.
- 모든 신규 모델은 `deployment/research_v55_NOT_ACTIVE/`; 활성 `deployment/active.json`은 수정하지 않음.
