# Biochar v34 — 신규 논문 4 DOI 감사 및 연구용 재학습 결과

**결론: 새 논문 패키지에서 수치 데이터를 추출·분리했고 연구용 모델 학습은 수행했지만, 기존 Stage 2 성능이 개선되지 않아 배포 모델은 교체하지 않았습니다.**

## 논문 출처 및 중복
- ZIP은 본문 5편, Supplement 2개 구성. Jia 2018 (10.3390/app8071019)은 v33과 중복이므로 제외. 신규 DOI 4편.
- Zhou 2022 (10.1038/s41598-022-17901-w): Box–Behnken Design 원시 17회 × Cu/Zn = **34개 240-min qt(흡착량)**. 본문 PDF p4 Table 1, p5 Table 2 및 pp2–3 Methods. 같은 중심조건 5반복을 합쳐 원료·오염물질별 유일 제조조건 **13점 × 2 = 26점**.
- Li 2022 (10.3390/plants11233234): 원료 rice husk R300/R500/R700, BET=64.78/257.5/330.0 m²/g, p5 Table 1. **원료 조성 아님, 열분해 후 바이오차 BET**. Cd/Se 실제 qe는 그래프 추가 판독 전까지 학습 보류.
- Shen 2019 (10.1016/j.scitotenv.2018.11.282): 원료 rice straw RSB300/500/700, BET=6.766/22.383/115.465 m²/g, p3 Table 1. 표 2는 kinetic fitted qe, 원시 측정 qe와 혼합 금지.
- Rodríguez-Vila 2018 (10.1007/s11356-017-1047-2): cedar wood 250–700°C의 Cu/Zn Langmuir fitted Csm **20행** 별도 저장, p5 Table 2. 일부 R²가 0.2~0.77로 낮아 `LOW_FIT_R2` 구분. Supplementary isotherm 그래프는 PDF 아니라 DOCX, qmax 표 값의 검증 근거로만 보존.

## v34 실제 연구용 학습
- **Stage 1:** v33 327행에 신규 6개 BET만 등록한 **333행 통합 후보표** 생성. TPV NR, 조성 basis NR. Stage1 배포모델 재학습은 하지 않았습니다.
- **Stage 2 A/B:** Phase8 **Core98 (26 DOI) + v33 Wang/Hu 실측·제거율 유래 15행 (2 DOI)** → **113행/28 DOI**로 연구용 재학습. *이번 ZIP의 Zhou qt는 이 qe 학습표에 0행만 반영*.
- **Zhou BBD 모델:** 34개 qt 원시기록을 26개 고유 제조조합으로 집계해 Cu/Zn 별도 회귀 surrogate 연구용 학습. 하나의 DOI 내 leave-one-unique-design-out CV일 뿐 DOI 외삽 검증 아님.
- 모델은 `deployment/research_v34/`에 별도 저장하였고, `deployment/stage2_models/active.json` 및 v60 모델은 그대로 유지.

## 98개 공통 홀드아웃의 DOI LODO — 새 15행 학습 추가 효과
두 학습 설정의 **동일한 원래 Core98 검증행 및 DOI 홀드아웃**에 대한 짝지은 비교. 점추정치가 좋아지지 않았습니다. Nested CV나 v16 배포 성능이 아닙니다.

| 방법 | 기존 98 MAE | 113행 학습 MAE | 변화 (113−98) | 113행 원시 물리 위반 |
|---|---:|---:|---:|---:|
| A XGBoost | 19.71 | 20.36 | +0.65 mg/g | 33/98 |
| B Ridge | 29.05 | 32.15 | +3.11 mg/g | 50/98 |
| B XGBoost | 21.35 | 21.67 | +0.32 mg/g | 37/98 |

DOI-cluster bootstrap (expanded−old MAE) 95% CI:
- A_XGBoost: +0.65 mg/g, CI [-2.35, +3.69]
- B_Ridge: +3.11 mg/g, CI [+0.23, +6.41]
- B_XGBoost: +0.32 mg/g, CI [-2.31, +2.86]

**해석:** B Ridge의 오차 증가 신뢰구간은 0을 넘습니다. 기존 훈련 DOI와 다른 두 논문을 추가하는 것만으로 모델 일반화가 향상되지 않았습니다.

## 새 DOI Wang/Hu를 홀드아웃한 온도 추천 검증
공통 baseline Core98만 학습 / 다른 신규 DOI의 데이터 1편만 추가하여 학습한 경우를 비교. 학습에 홀드아웃 DOI 사용 금지.
| 모델 | 학습 설정 | 신규 DOI 15행 MAE | 물리 위반 | Top-1 온도 적중 |
|---|---|---:|---:|---|
| A XGBoost | 기존 98 | 15.48 | 8/15 | 온도 입력 없음 |
| B Ridge | 기존 98 | 69.45 | 15/15 | 0/4 |
| B XGBoost | 기존 98 | 15.23 | 13/15 | 0/4 |
| A XGBoost | 다른 신규 DOI 포함 | 12.31 | 4/15 | 온도 입력 없음 |
| B Ridge | 다른 신규 DOI 포함 | 66.48 | 15/15 | 0/4 |
| B XGBoost | 다른 신규 DOI 포함 | 12.51 | 6/15 | 0/4 |

**기존 Model B는 온도 상위 1개 선택 0/4를 기록**했습니다. 4개 계열이지만 독립 DOI는 **2편**이므로 일반화 적중률로 해석하지 않습니다.

## Zhou BBD 240분 qt 연구 (qe와 분리)
| 대상·모델 | 고유 제조조건 LOO MAE (mg/g) | R² |
|---|---:|---:|
| Cu(II)_Mean | 0.90 | -0.174 |
| Cu(II)_LinearRidge | 0.77 | 0.056 |
| Cu(II)_QuadraticRidge | 1.00 | -0.283 |
| Zn(II)_Mean | 1.50 | -0.174 |
| Zn(II)_LinearRidge | 1.53 | -0.037 |
| Zn(II)_QuadraticRidge | 1.11 | 0.294 |

**해석:** Cu에서 선형 Ridge는 평균 예측보다 MAE가 조금 낮지만, 13개 고유 조건에 대한 DOI 내부 검증이고 R²가 낮습니다. Zn의 2차 Ridge R²도 약 0.29에 그칩니다. 논문 내부 조건 보간 이상으로 일반화 주장 불가. 원시 34행을 반복 중심점 그대로 무작위 분할하지 않았습니다.

## 연구적 제약 및 후속 작업
1. `qe`와 일정 시간의 `qt` 및 Langmuir `qmax/Csm`는 서로 다른 타깃. 단순 합산/학습 금지.
2. v33 Wang/Hu 15행은 연구용 조건부 흡착량으로, 2인 원문 독립 검증·흡착 평형 승인 후 정식 학습에 편입 필요. Wang의 pH는 원문 4.5–5.0 **범위**로, numeric 입력에 임의 평균을 주지 않고 결측으로 처리하며 원문 범위를 별도 보존.
3. Zhou의 Cu·Zn 반응에서 침전 가능성: 실제 초기/최종 pH 및 금속 수지 검증 없이는 순수 흡착기작 모델이라고 단정 불가.
4. Stage1 v34 새 6행 BET는 오직 원료 이름 + 공정 + 제조 후 BET이며 열분해 전 원료 C/H/O/Ash 조성이 없는 자료.
5. 다른 DOI에서 온도 비교군을 지속 확보하고, 논문 단위 홀드아웃에서 Top-1/Top-2/regret 평가.
6. 이번 실험에서 배포 승격 기준 미충족. 기존 UI·배포 모델 변경 없음.

## 재현
```powershell
.\.venv\Scripts\python.exe -m scripts.build_v34_temperature
.\.venv\Scripts\python.exe -m pytest tests\test_v34_temperature.py -q
```

기술 메모: 학습코드는 v33가 이미 설치된 프로젝트와 Phase8 Core98를 사용합니다. 원시 PDF는 이 패치에 중복 포함하지 않고 출처 SHA256으로 추적합니다. Python 3.13 / sklearn 1.8.0 / XGBoost 3.1.3 환경에서 연구용 joblib 파일을 저장했으므로, 사용자 Windows Python 3.12 환경에서 재학습하는 것을 권장합니다.
