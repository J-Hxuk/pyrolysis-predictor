# v62 로컬 DOI별 제조조건 모델 검증 결과

이 보고서는 설치된 컴퓨터의 v60 OOF와 누적 학습 데이터를 이용해 새 Pairwise Logit 모델을 DOI GroupKFold로 학습·평가한 **사후 연구용 결과**다.
절대 qe 예측과 상대 우열 점수를 혼합하지 않는다. 신규 모델은 학술·실용 배포 검증을 통과하지 않았다.

- 전체 제조조건 비교계열: 15개 / 9 DOI
- 계열 관측: 59행; 후보 쌍 95개 (서로 독립인 관측 수가 아님)
- 모델 입력으로 구별할 수 없는 후보 쌍: 0개
- 전체 188행에서 clipping 전 물질수지 상한 초과: 61행

## 모델별 제조조건 순위 검증

|모델|Spearman 유효 계열|평균 Spearman ρ (유효 계열)|Pairwise 일치율|고유 1위 계열|평균 최악 Regret (mg/g)|
|---|---:|---:|---:|---:|---:|
|기존 Model B clipped qe|14 / 15|0.569|0.715|9|5.630|
|기존 Model B raw (물리 위반 가능, 진단만)|15 / 15|0.479|0.710|13|5.528|
|기존 분리 RF (상대 점수)|15 / 15|0.557|0.741|14|3.645|
|신규 Pairwise Logistic (상대 점수)|15 / 15|0.478|0.708|15|5.528|

## DOI 재표집 비교 (순위 Pairwise − 비교모델)

- 기존 분리 RF (상대 점수): Δρ=-0.079; 95% DOI bootstrap 구간 [-0.335, 0.183]; 유효 15계열/9DOI
- 기존 Model B clipped qe: Δρ=-0.093; 95% DOI bootstrap 구간 [-0.259, 0.001]; 유효 14계열/8DOI
- 기존 Model B raw (물리 위반 가능, 진단만): Δρ=-0.001; 95% DOI bootstrap 구간 [-0.006, 0.002]; 유효 15계열/9DOI

## 데이터 완전성 및 예측 불가능성

- 흡착조건 완전 보고 계열: 4개
- 엄격 평형 qe 계열: 2개 (별도 DOI 수가 충분하지 않음)
- 비교계열 내부의 제조조건 수치 변동 수 (계열 15개 기준): process_step_count=0, pyrolysis_temperature_C=15, pyrolysis_hold_min=0, heating_rate_C_min=0, secondary_temperature_C=0, secondary_hold_min=0
- 원료·개질·분위기 등의 범주변수 계열 내부 변동 수: pollutant=0, feedstock_label_normalized=0, manufacturing_route=0, pyrolysis_atmosphere=0, activation_or_modification=0
- 제조조건이 관측상 구별되는 것과 새로운 DOI에서 정확히 우열을 예측하는 것은 별개다.
- 연구용 원시 qe는 물질수지를 초과할 수 있으므로 물리적으로 가능한 흡착량으로 사용하지 않는다.
- 단독 1위 수는 **정답률이 아니며**, 순위 예측의 정확성은 Spearman, pairwise, regret으로 별도 확인한다.

## 해석 제한

- 전체 15개 중 11개는 흡착조건 일부 미보고. 완전 보고 4개 중 엄격 평형 2개는 같은 DOI에서 나왔다.
- Pairwise 쌍은 동일 제조 비교계열 안에서 상호의존적이므로 95쌍을 독립 표본 수로 보고하지 않는다.
- 본 모델 구조 및 분석은 이전 DOI와 결과를 본 뒤 제안된 사후 실험이므로, 전향·완전 독립 검증이 아니다.
- 다른 로컬 환경에서 재학습된 v60 RF/B OOF를 넣으면 해당 PC에서 생성한 v62 보고서를 기준으로 판단해야 한다.
- 배포 모델을 수정하지 않고 `deployment/research_v62_NOT_ACTIVE/`에 별도 저장했다.
