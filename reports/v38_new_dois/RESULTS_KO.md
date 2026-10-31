# Biochar v38 — 신규 DOI 편입, 전체 재학습 및 제조경로/원료 편중 감사

- 작성 기준: 2026-10-30
- 상태: **연구용**. 기존 배포 Stage 1 v60, Stage 2 A/B/C 및 `active.json` 변경 없음.
- 재현 코드: `python -m scripts.build_v38_new_dois --root .`
- 이 연구에서 `qe_reported_mg_g`는 **reported uptake**이며 엄격 평형 인증이 아니다.

## 1. 최초 질문의 답

새 논문이 기존 모델에 잘 맞지 않더라도 **원문 데이터는 폐기하지 않는다**. 기존 모델에 행을 기계적으로 더하는 학습 방식 대신, *출처 확인 → DOI/시료 중복 제거 → 타깃 정의별 분리 → 전체 호환 데이터로 from-scratch 재학습*이 타당하다. 단, 효과 검증은 기존 DOI 동일 홀드아웃에 대한 **짝지은 평가**와 별도 신규 DOI 진단으로 분리해야 한다. 다른 정의의 `qt`, 피팅 `qmax`, 다성분 동시 용액을 `qe_reported`에 강제 합치는 것은 금지한다.

## 2. 팀원 신규 ZIP 감사

| DOI | 논문/원료 | 새 Stage1 시료 | 새 BET | 새 TPV | Stage2 취급 |
|---|---|---:|---:|---:|---|
| `10.1016/j.chemosphere.2022.134051` | Ke et al. 2022, spent coffee grounds, 300/500/700°C × fresh/H2O2-aged | 6 | 6 | 6 | Main text에서 보고된 단일금속 Cd/Zn 평형 uptake 12행. **미검증 조건부 연구용**; 초기 pH, 시료별 접촉시간 NR |
| `10.1016/j.colsurfa.2021.128100` | Yang et al. 2022, wheat straw, N2/air limitation × 300/450/600/750°C | 8 | 8 | 0 | 보충자료 Qm은 Langmuir 피팅값, 현재 reported qe에 미편입 |
| `10.21577/0103-5053.20200038` | de Andrade et al. 2020, sugarcane bagasse, 300/400/500/600°C | 4 | 0 | 0 | 6개 금속 동시 용액에서 BC500의 제거율만 별도 보관; **단일금속 데이터 아님** |
| `10.1016/j.arabjc.2024.105946` | Lv et al. 2024, soybean straw | 0 | 0 | 0 | **v33 기존 3개 시료와 중복**. 추가 행 생성 금지 |

### 값 출처 및 과학적 구분

- Coffee BET와 기공부피는 **Supplement Table S7**에서 표의 실수치를 추출했다. 예: CGB300 BET=1.127 m²/g, TPV=0.0003822 cm³/g. **본 논문의 pore volume 수치를 실제로 기재한 그대로** 사용하며 기존 TPV 측정법 간 이질성은 남는다.
- Coffee uptake는 **본문 p4의 평형 시점 수치**이며 보충자료 Tables S1–S2의 pseudo-second-order/pseudo-first-order fitted `qe`를 채택하지 않았다. 다만 초기 pH 및 시료별 실제 평형시간이 불분명하므로 배포학습 승인값이 아니라 조건부 연구용이다.
- Wheat BET 값은 **Main Table 1**의 N2 1.77/8.47/68.09/22.83, oxygen-limited 2.91/27.98/135.22/25.35 m²/g이다. 보충자료의 `Qm`은 qe와 혼합하지 않는다.
- Sugarcane에서는 500 µg/L = 0.5 mg/L(각 금속), 0.2g/100mL = 2g/L; 금속당 물질수지 최대 0.25 mg/g이다. BC500의 제거율 6개를 환산한 값을 **별도 다성분 데이터셋**에 보관한다. C0를 500 mg/L로 오독하면 1000배 오류가 된다.
- 논문의 개질 **후** C/H/O, EDS 등의 분석은 **열분해 전 원료 조성**으로 사용하지 않았다. DOI/시료명과 타깃 정의를 유지한다.

## 3. 데이터셋 규모

| 데이터 | v37 | v38 |
|---|---:|---:|
| Stage1 제조시료 master(관측이 아닌 메타데이터 포함) | 333 | **351** |
| Stage1 BET 수치 후보 | 299 | **313** |
| Stage1 TPV 수치 후보 | 199 | **205** |
| Stage2 reported uptake **조건부 연구용** | 113 | **125** |
| Stage2 DOI | 28 | **29** |

Stage1 351행 전부가 BET/TPV 학습 대상인 것은 아니며, 원본 QA 미해결 자료도 포함되어 있다. Stage2 125행 중 새 커피 12행은 초기 pH가 **모두 NR**로, fold 내부 결측 대체를 거친 *탐색적 연구* 자료다. 신뢰할 수 있는 125개 엄격 평형 qe가 확보됐다는 의미가 아니다.

## 4. **정확히 동일한 이전 DOI 검증행**에서 데이터 추가 전후 효과

그룹 단위 5분할은 **기존 DOI에서 고정**했다. 같은 DOI의 관측행은 검증과 학습에 동시에 들어가지 않는다. 신규 DOI 자료는 보강 모델의 학습 fold에만 추가한다. 하이퍼파라미터는 v37 고정값으로 선택하며 새 DOI에서 선택하지 않았다. 데이터 추가 효과만 짝지어 본 실험이다.

| 대상/알고리즘 | 기존 모델 MAE | 신규 DOI 추가 후 from-scratch MAE | 증가분(신규−기존) | DOI Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Stage1 BET / Ridge | 255.82 | **251.83** m²/g | −3.99 | [−12.79, +3.28] |
| Stage1 TPV / RF(log) | 0.17082 | **0.16835** cm³/g | −0.00247 | [−0.01009, +0.00549] |
| Stage2 A / XGBoost | 19.80 | **18.19** mg/g | −1.61 | [−3.34, +0.08] |
| Stage2 B / XGBoost | **19.74** | 19.82 mg/g | +0.08 | [−1.26, +1.37] |

**모든 95% CI가 0을 포함**하므로 정확도 개선을 확정할 수 없다. Stage2 A의 물질수지 원시 예측 위반은 34/113 → 21/113, B는 43/113 → 35/113이다. 물리 제약을 모델에 반영한 것이 아니라 단지 재학습으로 바뀐 수치이며 개선의 보증이 아니다. clipping을 모델 정확도 개선으로 취급하지 않았다.

## 5. v38 편중 진단: *신규 DOI*에 대한 원 모델의 예측

신규 DOI를 모두 제외한 v37 학습자료로 모델을 적합하고, 신규 자료를 **진단용 평가**에 사용했다. 이번 작업에서 보고서/변수 분석에 이미 노출된 DOI이므로 완전히 잠근 미래 외부 검증이라고 주장하지 않는다.

| 신규 DOI 별 평가 | 실측 행 | 기존 연구용 모델 예측 MAE | 문제 |
|---|---:|---:|---|
| Coffee + Wheat BET | 14 | **201.09 m²/g** | Ridge 원시 BET **14/14 음수** |
| Coffee TPV | 6 | 0.228 cm³/g | 측정 TPV 약 10⁻⁴ cm³/g대에 비해 과대예측 |
| Coffee Cd/Zn Stage2 A | 12 | 27.23 mg/g | 미학습 Zn, 초기 pH NR, 산화 노화공정 문제 |
| Coffee Cd/Zn Stage2 B | 12 | 20.55 mg/g | 동일 한계, 개질/노화 특성 표현 부족 |

**결론:** v37의 BET 모델이 일반적인 평균 MAE에서는 우세해도, 낮은 BET·미학습 원료/분위기에 대해 물리적으로 불가능한 음수값을 낸다. 신규 데이터가 이를 드러낸 것이므로 버릴 이유가 아니라 다음 모델 설계의 근거다. 무조건 음수 → 0 clipping을 성능개선으로 주장하지 않는다.

### 기존 DOI에서의 편중 (v37 기준)

- Stage1 BET: 측정 BET 0–25 m²/g 50행 MAE 117.69; 25–200 m²/g 80행 MAE 186.52; 200 m²/g 초과 169행 MAE 329.49. **절대 MAE 규모만으로 상대 성능을 단정하면 안 됨**.
- Stage2 Model B: `SINGLE_STAGE_PYROLYSIS` 40행/12 DOI MAE 12.91 mg/g; `CHEMICAL_POST_MODIFICATION` 15행/4 DOI 37.59 mg/g. DOI/오염물질/농도 분포가 교란되므로 개질 그 자체의 인과적 열세라고 해석 불가.
- Stage2 Model B 오염물질별 기존 OOF: Cd(II) MAE 13.13 (56행/10 DOI), Pb(II) 42.08 (14행/7 DOI), Ni(II) 40.11 (11행/4 DOI). 서로 다른 농도/흡착량 스케일에 의존한다.
- 소수 DOI 집단(또는 개별 DOI만 포함한 원료·경로)은 서술적 진단에만 사용하며 그룹별 일반화 성능으로 보고하지 않는다.

## 6. 학습·배포 정책

- `deployment/research_v38_NOT_ACTIVE/`에 **v38 Stage1 BET/TPV·Stage2 A/B 총 4개 연구용 새 모델** 저장. 이는 실제 전체 호환 후보 데이터로 처음부터 재학습한 것이다.
- **모델 선택/배포 자동 전환 금지**. 기존 활성 모델과 UI는 그대로 둔다.
- JavaScript 등 프론트엔드 수정 없음. 기존 v37 실시간 슬라이더 UI 유지.
- 새 ZIP의 논문 원문 PDF는 중복으로 포함하지 않는다. SHA-256은 `uploaded_paper_sha256.csv`, source→target 표는 `source_audit_4_DOIs.csv`에 보존.
- 최신 모델은 Python 3.13 / scikit-learn 1.8.0에서 생성됐다. Windows의 의존성 버전이 다르면 joblib 직접 로드 금지. 해당 컴퓨터에서 스크립트로 재학습 권장.

## 7. 다음 연구 결론 (v39 추천)

1. **BET 모델의 음수 출력을 학습단에서 방지할 후보**(log1p target, gamma/tweedie positivity-aware 모형 등)를 포함해 신규 DOI 진단과 기존 DOI를 구분해 비교. v38 신규 DOI를 근거로 초매개변수를 선택하면 그 DOI는 더는 독립 테스트가 아니다.
2. Stage2 새 커피 12행의 **원문 초기 pH·시료별 접촉시간**을 확보하면 연구용 조건부 → 품질 승인 승격을 재검토. 미보고를 임의 숫자로 채우지 말 것.
3. Zn(II), 산화노화 H2O2 공정, 다성분 경쟁흡착은 새 입력 계약과 **독립적인 DOI**를 더 확보하기 전 별도 영역으로 유지.
4. 최고 제조조건 추천은 원료·흡착조건이 고정된 온도 비교군과 DOI 외부 검증에서만 평가. v38의 포괄 재학습 결과만으로 신뢰성 있다고 보고할 수 없음.

## 8. 검증 가능 파일

- `Stage1_Master_351_RESEARCH_STAGING.csv`, `Stage2_reported_uptake_125_RESEARCH_CONDITIONAL.csv`
- `paired_old_DOI_evaluation.csv`, `paired_predictions_and_new_DOI.csv`, `error_slices.csv`, `summary.json`
- `fig_stage2_paired_MAE.png`, `fig_new_DOI_BET_failure.png`
- 모델 `deployment/research_v38_NOT_ACTIVE/`, 원문 전사 파일 `data/imports/v38_new_dois/`

전체 스위트는 이번 실험에서 실행하지 않았다. 관련 v38 및 v37 테스트를 실행하고 그 결과를 설치 가이드에 기재한다.
