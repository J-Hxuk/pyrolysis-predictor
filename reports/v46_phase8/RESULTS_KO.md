# Biochar v46 — Phase 8 원문검증·연결복구 DB v2 반영 및 연구용 추가 학습

- 원자료: `261030_바이오차_Phase8_원문검증·연결복구_DB_v2.zip.zip`, 내부 `Biochar_Phase8_Data_Reliability_Upgrade_v2.zip` 및 `Stage2_Training_Ready_Verified_v2.csv`.
- 기준본: Biochar v45 연구용 제조시료 404행, Stage2 reported uptake 138행.
- **활성 모델 및 Streamlit UI는 변경하지 않았음.** 연구용 재학습 모델 4개를 별도로 저장.

## 1. 실제로 늘어난 데이터

| 데이터 | v45 | v46 후보 | 비고 |
|---|---:|---:|---|
| Stage1 제조시료 | 404 | 417 | 새 13개는 같은 DOI의 대나무 원료 활성화 계열; source-approved 측정 물성이지만 기존 모델 학습 자동 승인 아님 |
| Stage1 BET | 348 | 361 | 13개 모두 원문 표 실측 |
| Stage1 TPV | 218 | 231 | 13개 모두 원문 표 실측 |
| Stage2 reported uptake | 138 | 140 | 1개는 새 Cu 실측, 1개는 이전 Ni 관측 복구 |
| Stage2 DOI | 32 | 32 | 독립 DOI 증가 **0** |
| Phase8 source-verified equilibrium 부분집합 | — | 45 | 기존 43행과 매칭됨; 신규 학습행 45개가 아님 |

추가 흡착 두 건은 `UPG2_OAK_CU_PH3` (Cu(II), 17.11 mg/g, pH 3, C0 30 mg/L, dose 1.4 g/L, time 미보고) 및 `UPG_NI_MSBC_KINETIC_18H` (Ni(II), 24.55 mg/g, C0 100 mg/L, dose 1 g/L, 18h). 원문 수치를 수정하거나 미보고 시간을 임의 채움하지 않음.

45행의 `source-verified`는 **원문 상 평형 흡착량으로 판단하는 근거**를 확보했다는 의미이며, 45행 전부 모든 실험조건이 독립적으로 완벽히 재검증됐다는 의미는 아님. 상세 QC 경고는 원본 CSV와 `phase8_source_qa_warnings`에 보존.

## 2. 모델 평가

모든 수치는 탐색적 DOI-group 5-fold 평가다. `old_fixed` 두 전략은 **완전히 동일한 v45 관측 행에서** 비교했으며, 새 행이 테스트 DOI와 같을 때에는 해당 행을 학습 fold에서 **제외**했다. 이 조건을 지키지 않고 같은 DOI의 추가 실측값을 테스트 데이터 누출시키면 결과가 왜곡된다.

| 모델 | 고정 v45 관측 MAE (기존 학습) | 고정 v45 관측 MAE (+ 신규 학습) | 해석 |
|---|---:|---:|---|
| Stage1 BET Poisson | 237.394 m²/g | 226.961 m²/g | 4.4% 낮아짐, 단 13개 고 BET 시료는 특수 다단계 공정 |
| Stage1 TPV Poisson | 0.181560 cm³/g | 0.181678 cm³/g | 사실상 동일 |
| Stage2 A XGBoost | 21.910 mg/g | 21.980 mg/g | 오차 소폭 증가 |
| Stage2 B XGBoost | 23.901 mg/g | 23.877 mg/g | 변화 미미 |

새 데이터까지 검증 집합에 포함한 전체 DOI group-5 결과는 비교 대상이 달라져 위 표의 개선율과 직접 비교하면 안 된다.

| 전체 집합(연구용) | MAE | R² | 원시 물질수지 위반 |
|---|---:|---:|---:|
| BET 361행 | 247.971 m²/g | 0.555 | 음수 예측 0 |
| TPV 231행 | 0.170191 cm³/g | 0.482 | 음수 예측 0 |
| Stage2 A 140행 | 23.123 mg/g | 0.605 | 44/140 |
| Stage2 B 140행 | 24.944 mg/g | 0.576 | 50/140 |

Stage1 신규 DOI 13개를 기존 404행만으로 학습한 모델에 외부 입력했을 때 BET MAE **535.43 m²/g**, TPV MAE **0.2483 cm³/g**. 이는 해당 특수 활성화 계열이 기존 학습 분포에서 어려운 영역임을 시사하며, 외부 실험에서 정량 보장 가능한 예측이란 의미는 아님.

## 3. 핵심 입력 표현 문제

대나무 13개 제조시료는 600°C/120분 1차 탄화 이후 K₂CO₃ 활성화 온도, `K2CO3_to_CHAR_mass_ratio`, `FeCl3_to_CHAR_mass_percent` 등이 달라지는 실험이다. 기존 Stage1 숫자 입력은 `activation_agent_FEEDSTOCK_mass_ratio`이며 **분모가 다르므로 대입하지 않았다**. 이 결과 13행은 현재 Stage1 모델 입력에서 **단 6개 서로 다른 특징 벡터**로 표현된다. 특히 900°C에서 K₂CO₃계 5행, FeCl₃ 병용계 4행은 실제 투입비가 달라도 모델이 동일 입력으로 본다.

따라서 이번 13행 반영은 **연구용 민감도 학습**이며, 제조 투입비 최적화나 신규 원료 일반화 성능을 주장하지 않는다. 사용자 입력에 동일 basis 비율을 추가하고 독립 DOI가 확보되기 전까지 정식 모델 교체는 권장하지 않는다.

## 4. 학술대회 실무 결론

1. Phase8 v2는 주로 43개 기존 행의 원문 신뢰성 향상. 새로운 45행을 통째로 더하지 않고 중복을 제거함.
2. 추가 학습한 새 모델은 연구용으로 보존. 현재 UI 활성 모델은 유지.
3. Stage1에서 직접 측정된 BET/TPV는 계속 축적하되 다단계 활성화의 약품/char 비율을 **원문 basis 그대로** 관리.
4. Stage2에서 엄격한 평형 qₑ가 확인된 45행은 `phase8_source_verified_v2` 열로 필터링 가능. 다른 조건부 reported uptake와 구분해 표시.
5. 이번 파일에는 독립적 신규 DOI가 없으므로 추천 정확도 향상이나 독립 외부 검증 확보를 주장하지 않음.

### 팀원이 추가로 확보하면 효율적인 자료

- **우선:** 서로 다른 DOI에서 원료와 제조온도 3수준 이상, K₂CO₃/char 또는 FeCl₃/char 비율을 명시하고 BET·TPV를 표로 보고한 자료. 1차 탄화 vs 2차 활성화 공정도 모두 기록.
- **Stage2:** 같은 제조시료들의 실제 qe(C₀, dose, 초기 pH, 접촉시간 포함)를 보유한 논문. 흡착량 vs qmax·qt 구분.

## 5. 재현 및 테스트

```powershell
.\.venv\Scripts\python.exe -m scripts.build_v46_phase8 --root .
.\.venv\Scripts\python.exe -m pytest tests\test_v46_phase8.py -q
```

관련 v46 신규 테스트 7개 통과, 기존 전체 회귀 테스트 및 Windows UI 렌더링은 미확인. 원본 Phase8 대용량 ZIP은 이번 패치에 중복 첨부하지 않고, 핵심 CSV 및 기존 v45 입력의 고정 복제본만 포함함.
