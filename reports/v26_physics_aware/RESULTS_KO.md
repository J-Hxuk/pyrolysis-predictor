# Biochar v26 — 물리정보 반영 및 저BET 예측 개선 실험

작성일 2026-10-30 · **연구용 실험** · **기존 배포 Stage 1 v60 / Stage 2 A·B·C 수정 없음**

## 1. 연구 질문과 데이터 계약

- Stage 1: Phase 8의 기존 승인 자료에서 BET 262행(32 DOI), TPV 199행(29 DOI)으로 고정 RandomForest(기존 v23 연구용 기준)와 로그 타깃 RandomForest를 같은 DOI 단위 LODO 분할로 비교. 원료 조성은 분석 basis 불완전으로 추가하지 않음.
- Stage 1 신규 33행/6 DOI는 **학습에서 완전히 제외**하고 외부 도전셋으로 평가. 이 코호트는 v25에서 이미 사용됐으므로 여기서 신규 후보를 선택했다는 사후탐색 효과를 피할 수 없음. 일부 제조 경로(HTC, 공열분해, 개질)는 기존 feature 계약 바깥.
- Stage 2: Phase 8의 기존 98행/26 DOI로 Model A XGBoost 원시 예측, 동일 예측을 물질수지로 clip한 값, f=qe/(C0/dose)의 Logit-Ridge, fold별 평균 f 물리 baseline, 100% 제거 상한 baseline 비교. 전처리와 타깃 변환은 각 DOI fold 학습 부분만 이용하여 학습.
- Stage 2 외부: 기존 v25의 신규 가금분 51행/1 DOI, 100 g/L 투입량. **학습에서 제외**. 강한 OOD 스트레스 테스트이지 검증된 신규 DOI 일반화가 아님.
- 타깃은 엄격 평형 인증 qe가 아닌 `reported uptake`이며 침전이 기여할 수 있는 사례 포함.

## 2. Stage 1 결과

| 지표 | 기존 RF | 로그 타깃 RF | 변화 |
|---|---:|---:|---:|
| BET LODO MAE, 262행 | 315.39 m²/g | 277.08 m²/g | −12.1% |
| BET LODO R² | 0.256 | 0.303 | +0.047 |
| TPV LODO MAE, 199행 | 0.20264 cm³/g | 0.18694 cm³/g | −7.7% |
| TPV LODO R² | 0.163 | 0.171 | +0.008 |
| BET 외부 33행 MAE | 152.61 m²/g | 107.93 m²/g | −29.3% |
| TPV 외부 33행 MAE | 0.06874 cm³/g | 0.06439 cm³/g | −6.3% |
| BET 외부 **단일 건식 열분해 17행** MAE | 114.41 | 42.90 | 개선 경향 |
| TPV 외부 **단일 건식 열분해 17행** MAE | 0.06490 | 0.05025 | 개선 경향 |

- DOI-cluster bootstrap (로그 − 기존) MAE 차이: **BET −38.30 m²/g, 95% CI [−84.65, +2.64]**; **TPV −0.01570 cm³/g, 95% CI [−0.04205, +0.00965]**. 두 CI 모두 0 포함. 통계적으로 확정된 개선이라고 주장 불가.
- 낮은 BET 영역(실측 ≤30 m²/g) LODO 46행에서 RF 106.15 → 로그 RF **84.32 m²/g**. 여전히 절대 오차가 실제 크기보다 큰 경우가 있으므로, 예측이 '실용적으로 정확하다'고 볼 수 없음.
- 외부 33행 BET R²는 로그 모델에서도 **−6.57**로 크게 음수이며, TPV 외부 R²는 기존 −0.14 → 로그 −0.24로 오히려 악화. MAE 개선 하나만으로 모델 우수성을 확정하면 안 됨.
- 로그 변환은 타깃 스케일을 안정화하는 실험일 뿐 원료 조성/활성화제/혼합비 정보를 회복시켜 주지 않음. **v60 정식 모델을 교체하지 않음**.

## 3. Stage 2 결과

| 방법 | 98행 LODO MAE (mg/g) | 원시 물질수지 위반율 | 51행 가금분 OOD MAE (mg/g) |
|---|---:|---:|---:|
| Model A XGBoost 원시 예측 | 19.71 | 32.65% | 67.85 |
| 같은 XGBoost를 [0, C0/dose]로 절단 | 16.83 | 0% (사후 보정) | 0.01514 |
| Logit-Ridge 흡착률 f, q=cap×sigmoid(z) | **30.27** | 0% (구조적 보장) | 0.01514 |
| fold별 학습 평균 흡착률 × cap | 26.81 | 0% | 1.16548 |
| 100% 제거 가정 q=C0/dose | 40.80 | 0% | **0.01514** |

- Logit-Ridge는 물리 상한을 구조적으로 보장하지만 **일반 DOI에서 XGBoost보다 정확도가 떨어짐**. 도입 금지.
- 가금분 51행의 Logit-Ridge 예측 f는 평균 약 100%로 포화하고, 이것이 OOD에서 100% 제거 baseline과 거의 같은 MAE를 만드는 이유다. **흡착 메커니즘의 예측 성공이라고 주장 불가**.
- 신규 가금분은 학습 최대 dose 10 g/L와 달리 100 g/L, Zn(II)는 미학습 오염물질, pH 변화·침전 위험 존재. 시험 결과를 일반화 성능으로 보고하지 않는다.
- XGBoost clipping의 낮은 오차도 원시 모델의 물리 위반을 숨겨서는 안 된다. 항상 raw/corrected/baseline 병기.

## 4. 제조조건 추천 게이트(코드만 추가, UI 미연결)

신규 모듈 `pyrolysis/biochar_stage2/v26_screening_policy.py`는 미학습 원료·오염물질, 학습 min/max 밖 C0/dose/pH/흡착온도/접촉시간, 제조 후보 2개 미만인 경우 **탐색적 순위 시도조차 불허**한다. `후보 비교`가 허용되어도 최적조건 확정, 신규 원료 일반화, 신뢰구간 보장은 아님. v19 후보 점수화 코드를 수정하지 않고 호출 전 안전성 검사로만 준비했다.

## 5. 다음 단계 결정

1. **배포 모델 유지**: Stage 1 로그 RF는 연구용 추가 검증 대상, Stage 2 Logit-Ridge는 기각.
2. 검증된 새 원료의 원료조성(VM/Ash/FC, CHON 분석 basis)과 활성화·혼합비 데이터를 먼저 확보. 조성 없는 Stage 1 로그 개선만으로 원료 일반화 문제 해결 불가능.
3. Stage 2는 데이터 계약과 수치 범위의 적용범위 표시를 우선 강화. 고투입량/미학습 오염물질에 대한 정량 최적화 금지.
4. 조건별 추천의 불확실성은 여러 독립 DOI의 조건 랭킹 홀드아웃에서 평가해야 하며 현재 **정량 신뢰구간 없음**.

## 6. 재현성/검증

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_v26_physics_aware
.\.venv\Scripts\python.exe -m pytest tests\test_v26_physics_aware.py -q
```

- 신규 테스트: 9/9 passed.
- 기존 관련 테스트 24개 포함: **33/33 passed** (v25, v19 후보, v18 조성, v23 관련).
- 기존 전체 회귀 스위트와 Windows UI 렌더링: **미실행**.
- 데이터·모델 파일 및 기존 UI **변경하지 않음**.

## 7. 파일

- `stage1_lodo_log_vs_raw.csv`: 원료 시료 단위 LODO 두 모델 OOF
- `stage1_external33_log_vs_raw.csv`: 원문 검증 신규 6 DOI OOD (raw RF v25 benchmark + log RF)
- `stage2_lodo_bounded_fraction.csv`: 98행 DOI별 원시/clip/fraction/baseline
- `stage2_external51_bounded_fraction.csv`: 가금분 51행 100 g/L OOD
- `summary.json`: 세부 지표 및 DOI bootstrap
