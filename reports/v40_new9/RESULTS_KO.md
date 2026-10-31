# v40 결과 보고 — 신규 9편 통합 + 전체 연구용 재학습

## 데이터 수

- ZIP 본문 PDF 9편: 기존 DOI 2편 중복 제외, 신규 7 DOI.
- Stage 1 연구용 시료: **351 → 381** (+30). BET 관측 **313 → 336** (+23, 새 4 DOI). TPV **205 → 213** (+8, 새 3 DOI). BET 측정이 없는 제조 시료 7건은 메타데이터로만 둔다.
- Stage 2 reported uptake: **125 → 128** (+3 kelp experimental qe, 새 1 DOI), 29 → **30 DOI**. 초기 pH 미보고 및 qmax와의 보고값 불일치를 조건부로 관리한다.
- fitted qmax 7건, 초기 부피 오류 가능성이 있는 Pb/Zn 흡착량 2건은 **별도 타깃/보류**. 콩짚·왕겨·커피 원료 조성 참고파일은 교차 DOI 결합하지 않는다.

## DOI 그룹 5-fold 교차검증 (전체 후보)

| 모델 | n | DOI | MAE | R² | 음수 예측 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| BET Ridge | 336 | 45 | 249.47 m²/g | 0.519 | 66 | 음수 물리 위반 |
| BET Poisson(공정) | 336 | 45 | **239.44 m²/g** | 0.490 | 0 | 연구 후보 |
| BET Poisson(+원료조성) | 336 | 45 | 241.02 m²/g | 0.481 | 0 | 결측/분석 basis 문제, 후보 |
| TPV Ridge | 213 | 33 | 0.18732 cm³/g | 0.345 | 6 | 음수 물리 위반 |
| TPV Poisson(공정) | 213 | 33 | 0.18073 cm³/g | 0.363 | 0 | 연구 후보 |
| TPV Poisson(+원료조성) | 213 | 33 | **0.17911 cm³/g** | 0.382 | 0 | 제한적 탐색 결과 |
| Stage 2 Model A XGBoost | 128 | 30 | **18.66 mg/g** | 0.660 | 1 | mass balance 위반 총 29/128 |
| Stage 2 Model B XGBoost | 128 | 30 | 21.26 mg/g | 0.583 | 0 | mass balance 위반 총 44/128 |

모든 전처리는 각 학습 fold 안에서 수행. Stage 2 클리핑 전 원시 예측을 평가. **정량값 개선은 해당 DOI 집합의 탐색적 평가로서, 반복적 모델 개발에 이미 사용된 source들이므로 확정적인 외부 검증이 아님.**

## 고정된 기존 검증 DOI에 추가 논문만 학습시킨 비교 (공정성 주의)

| 모델 | 추가 전 MAE | 신규 DOI를 학습에 추가 후 MAE | Δ (후−전) |
|---|---:|---:|---:|
| BET Ridge | 258.91 | 247.48 | **−11.43** m²/g |
| BET Poisson(공정) | 248.63 | 257.33 | +8.70 m²/g |
| BET Poisson(+조성) | 250.81 | 265.24 | +14.43 m²/g |
| TPV Poisson(공정) | 0.18361 | 0.18999 | +0.00639 cm³/g |
| TPV Poisson(+조성) | 0.19005 | 0.18281 | −0.00725 cm³/g |
| Stage 2 A | 20.64 | 20.28 | −0.36 mg/g |
| Stage 2 B | 22.22 | 23.56 | +1.33 mg/g |

**BET Ridge의 개선은 DOI-cluster bootstrap 95% CI −21.65~-3.12 m²/g이나, 새 논문 중 일부를 직접 확인하며 방법론을 반복 조정한 뒤의 탐색적 결과이므로 독립 블라인드 개선이라고 주장하지 않음.** 다른 비교 CI는 대부분 0 포함. 이전 v39 결과와 fold 구성·집합이 달라 직접 비교해서 정확도 상승으로 해석하면 안 된다.

## 신규 DOI를 예측 대상으로 둔 추가 진단

기존 v39 자료만 학습한 후 신규 DOE를 별도 예측: BET 23행/4 DOI에서 Ridge MAE **78.94**, Poisson 공정 **56.29** m²/g. TPV 8행/3 DOI에서 Ridge **0.173**, Poisson 공정 **0.129** cm³/g. Stage 2 kelp 3행/1 DOI는 Model A **25.73**, Model B **50.77** mg/g. 이 역시 **검증 설계 과정에서 원문 결과를 이미 확인했으므로 진정한 blind test가 아님**. 3행/1 DOI로 일반화 주장 불가.

## 운영 결정

- 기존 배포 모델/Streamlit UI/active.json은 건드리지 않는다.
- 6개 연구용 모델을 고정된 설정으로 **전체 데이터에서 처음부터 학습**하여 `deployment/research_v40_NOT_ACTIVE/`에 저장한다.
- `research_training_envelopes.json`과 범위 경고 모듈은 범위·미보고·미학습 범주를 표시하지만, **입력값이 개별 범위 안에 있다는 사실이 물리적 유효성과 모델 신뢰성을 증명하는 것은 아님**. 현재 UI에 자동 연결하지 않는다.
- 다음 데이터 수집은 초기 pH, 접촉시간, C0·V·m, 제조 전 원료 조성 basis, 같은 biochar ID의 BET/TPV+실측 qe가 완비된 **신규 3개 이상의 독립 DOI**를 최우선으로 한다. 검증용 DOI는 모델 선정에 쓰지 않고 사전에 잠가 둔다.

정량 재현 파일: `model_checks/all_DOI_OOF_metrics.csv`, `model_checks/paired_old_DOI_with_new_training.csv`, `model_checks/new_DOI_prospective_challenge.csv`.
