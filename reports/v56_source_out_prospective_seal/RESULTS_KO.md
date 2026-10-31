# v56 신규 DOI Source-out 재평가 및 미래 외부 검증 봉인

**상태:** 2026-10-31 완료 · 연구용, 배포 불승격 · v55 원본/승인 DB/활성모델 무변경.

## 1. 이번 버전의 검증 범위

기존 v55의 Stage2 `same140` 평가는 *기존 DOI를 테스트*하고 새 논문을 학습에 추가하는 설계였다. 이번에는 **새 논문 전체를 시험집단으로 빼고** 나머지 DOI만 학습했다. 각 수치는 `scripts/run_v56_source_holdout.py`가 실제로 fit/predict한 결과이며 학습에 시험 DOI가 없음을 검사한다. 이미 모든 논문을 v55에서 확인하고 연구 모델을 선택한 뒤에 수행하므로 **독립적인 사전등록 blind test는 아니다**.

기존 140행/32 DOI는 일부 source-conditional/exploratory 자료가 포함된 연구 학습본이다. 새로운 Stage2 평가집단은 사과나무 Cu/Zn 8행, Sugarcane Pb 4행, Cocoa Cd 4행이다. Cocoa는 Table3 실험값과 kinetics 문단의 초기농도 불일치가 있으므로 조건추정 민감도에만 속한다.

## 2. Stage2 DOI 전체 제외 예측 (MAE, mg/g)

| 시험 DOI 계열 | 기존 140만 학습 A | 다른 source-verified DOI 추가 A | 기존 140만 학습 B | 다른 source-verified DOI 추가 B | 0 예측 기준선 |
|---|---:|---:|---:|---:|---:|
| 사과나무 Cu/Zn 8행 | 5.628 | 5.210 | **1.451** | 1.740 | 7.273 |
| Sugarcane Pb 4행 | 14.482 | 13.435 | 16.952 | 17.309 | **11.575** |
| Cocoa Cd 4행 (조건 충돌) | 6.383 | 6.383 | 6.383 | 6.383 | **3.618** |

- Sugarcane 시험에서 `other_verified_sources`는 사과나무 8행을, 사과나무 시험에서는 Sugarcane 4행을 훈련자료에 추가한 것. 시험 DOI는 제외했다.
- Sugarcane은 0 예측 기준선보다 두 모델 모두 더 큰 오차를 기록했다. 논문 간 전이 성능이 충분하지 않다.
- Cocoa는 네 예측값이 모델 raw 상한 위반 후 물질수지로 clipping되어 모두 상한에 도달했다. 두 모델의 동일 MAE는 좋은 일반화가 아니라 **경계값 예측 실패**를 반영한다.
- 사과나무 Model B는 새 DOI 시험에서 유리했지만, 이 논문은 기존 Stage1에서 관련 2017 DOI가 노출된 소재 계보가 있으므로 전체 파이프라인의 독립 holdout으로 사용할 수 없다.
- Cocoa를 **다른 DOI의 학습자료에 추가**하면 Sugarcane MAE가 A 3.230, B 7.145로 개선되지만 Cocoa 초기농도 충돌을 이용한 증량 민감도 결과이므로 주 결론 또는 독립 개선으로 주장하지 않는다.

## 3. 동일 DOI 내 제조조건 순위, Model B

| 제조계열 (각 4조건) | 기존 140 모델 Spearman ρ | top-1 regret (mg/g) |
|---|---:|---:|
| 사과나무 Cu | 0.800 | 0.83 |
| 사과나무 Zn | 0.800 | 1.12 |
| Sugarcane Pb | **−0.800** | **7.60** |
| Cocoa Cd | 계산 불가 | 계산 불가 |

Cocoa는 모든 예측이 동률이다. 동률을 임의 첫 행으로 깨서 regret=0이라고 보고하지 않았다. 제조조건이 입력에 없는 Model A는 해당 조건들의 순위를 식별하는 모델이 아니므로 추천 성능 우열 비교에서 제외해야 한다. 데이터가 증가해도 신규 DOI에서 추천 정확도가 유지된다는 근거는 없다.

## 4. Stage1 신원료 Source-out

| 평가 DOI | BET 기존 361행 MAE (m²/g) | BET 다른 DOI 증량 후 | 수치 확인 BET n |
|---|---:|---:|---:|
| Cocoa | 435.638 | 418.719 | 4 |
| Ragweed/Horseweed | 43.152 | 18.959 | 6 |
| Rice Straw | 111.721 | **220.730 (악화)** | 8 |
| Sludge | 71.547 | 51.422 | 7 |
| Sugarcane | 102.049 | 79.380 | 4 |

TPV는 Cocoa 4행에서 0.18365 → 0.17947 cm³/g, Ragweed 6행에서 0.05657 → 0.04086 cm³/g. 다만 표본이 작고 원료별 물성 분포가 달라 일반적 개선으로 판정하기 어렵다. BET Cocoa는 0 예측의 MAE(47.25 m²/g)보다도 매우 큰 오차를 기록하여 신원료 물성 외삽에 심각한 문제가 있다.

## 5. 별도 Langmuir qmax DOI 전체 제외

| 평가 DOI | 기존 Qmax 32행 MAE | 다른 신규 DOI를 훈련에 추가한 MAE | 시험 n |
|---|---:|---:|---:|
| Cocoa | 50.347 | 27.822 mg/g | 4 |
| Ragweed/Horseweed | 97.503 | 93.425 mg/g | 12 |

Cocoa의 `qmax` 평균은 약 13.46 mg/g이므로 0 예측보다도 오차가 크다. 합성/측정 조건의 차이를 해결하지 못한 상태에서 학습량 증가만으로 `qmax` 일반화 성능을 확보했다고 볼 수 없다. `qmax`는 평형등온식의 피팅 파라미터로 `qe`와 병합하지 않았다.

## 6. 앞으로의 진정한 외부평가: v56 모델 봉인

출처 레지스트리와 학습본 등에서 발견한 **103 DOI**를 역사적 노출 이력으로 등록하고, 관련 2017 사과나무 DOI도 포함했다. 모델은 *기존 v55에서 이미 선택한* 연구용 후보 `Stage2 A 152`, `Stage2 B 148`, `Stage1 BET 선택14`, `Stage1 TPV 전체10`이며 v56에서 새로 피팅한 source-out fold 모델을 배포하거나 활성화하지 않았다.

사용자의 로컬 scikit-learn 버전에서 v55 최종 모델을 맞춘 **후에** `--mode seal`을 실행해야 한다. SHA256로 모델과 학습 자료를 봉인하고, 진짜 새로운 DOI가 들어오면 해당 모델을 재학습하거나 선택에 사용하지 않은 채 `--mode prospective`를 사용한다. 같은 물리적 제조배치, 과거 DOI 관련 소재, source-unverified `qt` 또는 fitted `qmax`는 외부검증 대상으로 인정하지 않는다.

## 7. 결론 및 후속 우선순위

1. **신규 DOI 일반화는 부족하다.** 기존 Stage2 고정 DOI 내부 MAE 16~17 mg/g를 근거로 신원료 흡착량 예측이 검증됐다고 말하면 안 된다.
2. **투입 데이터 질과 추천 목적을 분리해야 한다.** Cocoa의 C0 조건을 정밀 검증하지 않은 상태에서는 개선된 숫자에 의미를 부여하기 어렵다.
3. **연구 발표의 정직한 기여점은 한계 정량화:** 전례없는 원료/논문에서 MAE·원료별 순위·물질수지 위반을 함께 공개하는 것.
4. 차기 실제 원문 실험 계열은 `FUTURE_QE_SOURCE_TEMPLATE.csv`에 엄격한 qe, C0, dose, 원료·제조배치 계보, 논문 DOI를 기록해 **모델 변경 없이 검증**한다.

### 실행 파일

`reports/v56_source_out_prospective_seal/` 아래 `stage2_retrospective_source_out_metrics.csv`, `stage2_retrospective_source_out_predictions.csv`, `stage2_retrospective_source_out_rank.csv`, `stage1_retrospective_source_out_metrics.csv`, `stage1_retrospective_source_out_predictions.csv`, `qmax_retrospective_source_out_metrics.csv`, `qmax_retrospective_source_out_predictions.csv`, `V56_RESEARCH_SUMMARY.json`.

**제작 검증:** Python 3.13.5 / scikit-learn 1.8.0 / XGBoost 3.1.3; v52~v56 통합 pytest 33개 통과. 사용자 PC 3.12 / scikit-learn 1.5.2는 재실행 결과가 달라질 수 있으며 사전에 고정한 점수와 강제 일치시키지 않는다.
