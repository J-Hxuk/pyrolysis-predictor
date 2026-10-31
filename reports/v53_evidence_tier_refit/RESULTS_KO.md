# Stage2 v53 증거 등급별 실재학습 및 원문 대조 결과

기준: 2026-10-31. **실제 XGBoost 학습과 DOI GroupKFold 평가를 실행했으며 모델 4개를 직렬화했다.** 모델은 전부 `deployment/research_v53_NOT_ACTIVE/`에 별도 저장했고 활성 모델 변경 없음.

## 데이터 및 승인 지위

- 기존 기준 140행/32 DOI. 기존 140행 자체에 `CORE_EXPLORATORY` 97건, 그림 근삿값/조건부 승인 자료 등이 섞여 있다. 이 140행을 전부 엄격한 qe 승인 데이터라고 부르지 않는다.
- Zhao et al. (2020) DOI `10.3390/en13133498`의 v52 데이터 24건(33번째 DOI)은 **Table 3 실험 qe 8건**과 **Section 3.1 흡착 24h 관측량 16건**으로 분리.
- vNext10의 8건은 **v52 Table 3의 8건과 8/8 동일 관측값**. 신규 데이터 8건으로 가산 금지. vNext10의 `TRAINING_ELIGIBLE`은 `CANDIDATE_PENDING_CROSS_SOURCE_MATERIAL_PROPERTY_REVIEW`; Stage2 6-input 학습 승인은 여전히 보류. A/B에서 일부 값 결측 대체를 허용하는 것은 연구 실험이지 학습 승인 판정이 아님.
- vNext11 신규 검토 6 DOI의 신규 training-ready 데이터 0건. 차나무 가지 5건은 과거 Stage1/Stage2 review에 노출된 데이터의 strict qe 근거 보강이며 신규 추가·독립 holdout 모두 불가. TPV 정의 보류. 본 패치에 넣지 않았음.
- 평가 외부 독립 holdout **0건**. 실제 새 실험 검증을 수행했다는 뜻이 아님.

## 고정 평가 결과: 원래 140개 관측/32 DOI에 동일한 GroupKFold(5)

| 시나리오 | Model A MAE (mg/g) | Model B MAE (mg/g) | 비고 |
|---|---:|---:|---|
| 140 기준 | 17.246811 | 17.690416 | 비교 기준 |
| 기존 140+Table3 8 | 17.189800 | 17.642990 | **주 연구 시나리오**; 학습 적격성 최종승인 아님 |
| 기존 140+전체 24 | 16.760127 | 17.760963 | 평형 근거 추가 확인 필요한 16건 포함, 민감도 분석 전용 |

MAE 향상(기준 minus 추가모델) 및 **DOI 그룹 bootstrap 95% percentile interval**:

| 모델/추가 시나리오 | 향상 mg/g | 95% 구간 mg/g |
|---|---:|---|
| A +8 | +0.0570 | [-0.289, +0.375] |
| A +24 | +0.4867 | [-0.794, +2.138] |
| B +8 | +0.0474 | [-0.626, +0.875] |
| B +24 | -0.0705 | [-0.776, +0.898] |

모든 bootstrap 구간에서 0이 포함된다. 성능 향상이 확증되지는 않았다. DOI 수가 32이고 일부 시료 중복·출처 의존성이 존재할 가능성을 모두 제거한 독립 검증이 아니다.

## 중요: 물리상한 clipping에 가려진 예측 문제

모든 평가 예측은 기존 v52 규칙대로 `0 ≤ qe_pred ≤ C0/dose` clipping. 평가 140행 중 clipping 전 **상한 위반 수**는 A 기준 43행, A+8 41행, A+24 42행; B 기준 50행, B+8 46행, B+24 46행이다. 사후 위반 0건은 모델이 원래 물리적인 출력을 생성한다는 의미가 아니다. `raw_upper_violations` 및 `cap_hits_after_clip` 기록을 반드시 함께 제시해야 한다.

## 신규 DOI 선제 시험 (학습 이전 140행 모델 → 사과나무 DOI)

- 5 g/L 실험 qe 8건에서 A MAE=5.6275, B MAE=1.45085 mg/g. **단일 DOI의 역사적 Stage1 관련 논문 노출이 존재하므로** 독립 신규 홀드아웃 일반화 성능이라 부르지 않는다.
- 기존 Stage2 A는 8개 값 모두 물질수지 상한 clipping되어 제조온도 순위를 구별하지 못한다. B는 이 단일 DOI의 절대오차가 더 작아도 내부 제조온도 최적 온도를 안정적으로 맞춘다는 근거는 부족하다.

## 모델 및 재현 방법

- 새 저장모델: A/B × (148행 연구 주 시나리오 / 164행 민감도 시나리오) = 4개 Joblib.
- Stage1 BET/TPV는 신규 승인 레코드 0건이므로 이번 패치에서 **재학습하지 않았다**. Model C/D 및 UI도 교체하지 않았다.
- 재현: v47 프로젝트 루트에서 `python -m scripts.run_v53_research_refit --root .`.
- 라이브러리/학습 입력 해시 및 저장 모델의 sha256: `v53_summary.json`, `deployment/research_v53_NOT_ACTIVE/manifest_v53.json`.
- Python 3.13.5 / scikit-learn 1.8.0 / xgboost 3.1.3 환경에서 수행. Windows 로컬 환경에서 Joblib 바이너리 사용 전 의존성 호환성과 inference 테스트 필수. 기존 자동 배포·승격 **금지**.

## 다음 필요

1. Stage2 모델의 높은 clipping 의존도를 개선하는 bounded fraction 접근 / 비교 기준 검증.
2. 논문 내 제조온도 비교에서 `top1 regret`, `rank correlation`을 유효 비교계열에 한해 검증.
3. 데이터셋의 기존 탐색적 140행 중 strict qe 확인되지 않은 출처를 분리한 별도 민감도 평가.
4. 새 DOI 유래 **실제 독립 실험 관측** 확보 후 외부 검증.
