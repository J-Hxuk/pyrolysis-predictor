# Stage 2 v16 독립 방법론 감사 및 Kiro 인계

작성일: 2026-10-29

## 검토 범위
- 근거: 사용자가 제공한 Stage 2 v15·v16 작업 보고의 요약 수치.
- **실제 v16 코드, `reports/stage2_v16/` CSV/PNG와 `summary.json` 원본은 확보되지 않았으므로 재실행·재계산·버그 확정은 불가능.**
- 수학적/실험설계상 명확한 문제와 보고서에서 확인할 항목을 분리했다.

## 1. 최우선: Chen16 LOO 추천 평가 해석 오류 가능성

16개의 실측값을 `y[0], ..., y[15]`라고 할 때, 평균 예측 DummyRegressor를 Leave-One-Out으로 적용하면

`yhat_i = (sum_j y_j - y_i) / 15`.

따라서 `y_i`와 `yhat_i`의 순위는 정확히 반대가 된다(비상수 `y`의 Spearman=-1). 보고된 Dummy Spearman=-1은 **LOO 평균 예측 방식이 자동으로 만드는 구조적 결과**일 수 있다. 이를 학습 모델 대비 실제 순위 성능의 타당한 비교로 해석하면 안 된다.

예시: 실제 `[10,20,30,40]`일 때 3개씩 평균한 LOO 예측은 `[30,26.667,23.333,20]`이고 Spearman=-1이다.

또한 16개를 LOO로 얻은 예측값은 서로 다른 16개 fitted model의 출력이다. 이를 합쳐 상위 3개를 뽑는 행위는 **한 번 학습한 하나의 모델로 후보 16개를 동시 추천하는 상황의 평가가 아니다**. Ridge Spearman=0.903과 Top-3=3/3, regret=4.4%p도 단일 모델 추천 성공의 증거로 단정하지 말 것.

**수정 권고**: 원래 LOO 수치는 삭제하지 말고 `cross-fold pooled diagnostic (not single-fit ranking)`으로 재라벨. 추가로 16개에서 반복 고정시드 분할(예: train 12/test 4, 조건별 test 출현 수 기록)을 만들고 각 split에서 **하나의 모델**로 test 4개를 동시에 예측하여 pairwise ranking accuracy, held-out top-1 hit, regret 및 제거율 MAE를 계산한다. Dummy 등 상수 예측의 동률은 Spearman 0 또는 -1로 강제하지 말고 `undefined / tie`로 처리하며, top-1에서는 랜덤 tie-break 기대값 또는 고정 시드 다회 시뮬레이션으로 비교한다. n=16의 제한과 반복 split 간 의존성 명시. Top-3 (test 4개 중 3개)는 우연 적중률이 높으므로 주지표로 사용하지 않는다.

**주의**: 12/4 홀드아웃은 새로운 원료 일반화 검증이 아니라 동일 논문·동일 설계 내 미관측 조건에 대한 제한적인 순위 예측 검증이다. 16개 직교배열에서 관측하지 않은 64개 전체 조합을 검증한 것처럼 서술 금지.

## 2. Ablation: '범주형은 도움이 된다' 서술 근거 부족

보고된 A 기준 LODO XGB R²=0.689 (nested A R²=0.675), ablation 추가 결과 `manufacturing_route=0.667`, `feedstock=0.608`, `step_count=0.597`로 **같은 평가 기준일 경우 A보다 모두 낮다**. `고결측 수치형이 나쁘고 결측 없는 범주형이 도움`이라는 결론은 현재 보고된 숫자와 모순된다.

확인: ablation의 A-only baseline, 동일 Fold/행, 알고리즘/하이퍼파라미터, 동일 점수 계산 방법과 예측 타깃을 출력하여 `delta = R²(A+feature) - R²(A in same ablation)` 재계산. 서로 다른 평가 절차의 점수를 섞지 않는다. 결측률과 기여도 사이의 연관성만으로 **고결측이 성능 저하의 주원인**이라고 인과적으로 단정하지 않는다. 특히 대조 비교가 feature 외 변화(알고리즘 변경 등)를 함께 포함하는지 확인한다.

## 3. LOFO ≠ LODO: feedstock→DOI 내포가 동등성을 의미하지 않음

`30 feedstock each occurs in only 1 DOI`라 해도 **하나의 DOI에 여러 feedstock이 등장할 수 있다**. LOFO에서 특정 feedstock을 holdout하는 동안 같은 DOI의 다른 feedstock은 train에 남을 수 있으므로 DOI 독립성이 보장되지 않는다. 총 feedstock 30종/DOI 25개라면(양쪽 전부 실제 등장했다는 조건 하에) 하나 이상의 DOI는 여러 feedstock을 포함한다.

확인: LOFO 각 Fold에서 `train_doi ∩ test_doi`, `train_feedstock ∩ test_feedstock`을 계산·보고. DOI 공유가 있으면 LOFO를 `new feedstock within partly seen DOI`로 분류, LODO와 동등하거나 독립적인 외삽 검증이라고 주장하지 않는다. DOI와 원료가 내포돼 있으면 새로운 DOI 효과와 새로운 원료 효과를 통계적으로 분리하기 어렵다.

## 4. 흡착률 f 경로의 R² 재해석

`f = qe × dose / C0` (dose가 g/L, C0가 mg/L, qe가 mg/g이며 둘 다 양수일 때). `f`를 학습한 모델의 예측을 다시 `qe`로 변환하여 계산한 **R²(qe)=0.19**만으로 **R²(f)=0.19**라고 말할 수 없다. 예측 타깃과 평가 공간을 구분해야 한다.

동일 held-out sample에 대해 각각 `(R²(qe), MAE(qe), violation_rate)` 및 `(R²(f), MAE(f))`를 따로 산출. raw qe, clipped qe, bounded f→qe를 모두 기록. 높은 R²(qe)가 C0/dose의 분산에 의존한다는 설명은 합리적 가설이나, `C0/dose only`, `f constant`와 `f model` 등 대조 결과를 함께 제시해 기여를 정량화해야 한다.

## 5. 정당하게 유지할 수 있는 v16 결과 (원본 확인 전 조건부)

- Nested DOI CV에서 A R²=0.675, MAE=18.64 mg/g: 보고된 절차대로라면 모델 선정 편향을 억제한 성능 추정치이며 의미 있음.
- A vs 물리 Baseline: R² 0.689 vs 0.19, MAE 18.56 vs 26.56 mg/g: 같은 test fold를 쓴 경우 예측 성능 개선 근거.
- B는 A보다 낮음: paired DOI bootstrap MAE_B−MAE_A=+4.99 mg/g, 95% CI [0.58,10.64]. 표본과 bootstrap 방법이 적절했다면 A의 상대적 우위 근거. 그러나 **제조조건이 본질적으로 무용하다** 또는 **결측이 유일한 원인이다**의 증거는 아님.
- qe는 전부 `EQUILIBRIUM_NOT_CERTIFIED`: **equilibrium capacity 예측으로 표기 불가**. `reported uptake` 사용.
- 원시 qe 제약 위반 29.9%: 물리적 유효성의 핵심 제한, clipping과 raw 성능을 분리 보고하는 것은 적절.

## 최소 수정 구현 순서 (기존 v15/v16 결과 보존)

1. Chen16의 LOO Dummy 역상관 수학적 원인 및 split별 predictor 다름을 문서화. 단일-fit 반복 12/4 holdout 평가 추가 (동률 처리 명시). 실제 특성별 데이터 사용/시험조건 누출 감사.
2. Ablation A 기준 점수를 **같은 Fold·알고리즘**으로 재출력하고 각 변수 ΔR²/ΔMAE 재해석.
3. LOFO 각 split DOI 중복 집계, DOI 독립 여부에 따라 설명 수정.
4. f와 qe의 타깃별 메트릭 분리, 잘못된 'R²(f)=0.19' 단정 제거.
5. 전체 회귀 테스트 `python -m pytest tests -q --tb=short` 재실행. 환경 오류면 stderr/exit code를 보존해 '환경 문제'를 추측이 아닌 근거로 확인.

**미실행:** 원본 v16 산출물·코드를 직접 확인하지 못했으므로 이 문서의 테스트는 작성 시점에 수행되지 않았다.

---

# Kiro 수정 결과 보고 (2026-10-29)

원본 v16 코드·`reports/stage2_v16/` 산출물을 직접 확보해 4개 문제를 **최소 수정**으로 처리했다.
기존 v15/v16 모델·데이터·UI·배포 모델은 변경하지 않았고 신규 기능은 추가하지 않았다.
수정은 평가·진단·보고 경로(검증 모듈·리포트 생성기·추천 검증)에 한정했다.

## 수정한 파일
- `pyrolysis/biochar_stage2/v15_recommend.py` (Chen16)
- `pyrolysis/biochar_stage2/v16_validate.py` (Ablation delta, LOFO 감사)
- `pyrolysis/biochar_stage2/v16_report.py` (summary.json·CSV 보고 분리/추가)

산출물은 `reports/stage2_v16/` 를 재생성해 반영했다(기존 수치 보존, 신규 필드/파일 추가).

## 1. Chen16 LOO 순위평가 오류 — 확인 및 수정 완료
- **수학적 원인 재확인(실데이터)**: Chen16 n=16, 타깃(제거율) 분산 589.89(비상수). 평균예측 Dummy 를
  LOO 로 적용하면 `yhat_i=(Σy−y_i)/15` 이고 해석적 `Spearman(y,yhat)=−1.0` 이 코드 출력 `−1.0` 과 일치.
  즉 Dummy 의 `Spearman=−1·top3=0` 은 **LOO 평균예측이 만드는 구조적 산물**임을 확정.
- **수정**:
  - `recommend_benchmark` 출력에 `diagnostic_type="CROSS_FOLD_POOLED_DIAGNOSTIC (not single-fit ranking)"`,
    `pooled_loo_caveat`, `dummy_anti_correlation_note` 추가. 기존 LOO 수치는 **삭제하지 않고** 재라벨만 했다.
  - 단일 모델 추천 평가로 `recommend_single_fit_holdout()` 신설: 고정시드 반복 **train 12/test 4**
    ShuffleSplit(기본 50회), 각 split 에서 **하나의 모델**로 test 4개를 동시 예측·순위화. 조건별 test
    출현 횟수 기록, 반복 split 의존성·일반화 한계 명시. 주지표는 **pairwise ranking accuracy**(실제 동률
    쌍 제외, 예측 동률 쌍은 불일치로 보수적 처리), top-1 은 예측 동률 시 균등 tie-break 기대적중확률(1/동률수).
    Top-3(test 4개 중 3개)는 우연적중이 높아 보조지표로만 둠.
- **재검증 수치(실데이터, 50회)**:
  - pooled LOO: Dummy Spearman=−1.0(구조적), Ridge 0.903 — 기존 값 그대로 보존.
  - single-fit 12/4 holdout: **Ridge** pairwise=0.90, top1_hit=0.92, regret=0.51%p, MAE=13.9%p /
    **Dummy** pairwise=0.0, top1_hit=0.25(tie-break), regret=30.97%p. → 구조적 −1 이 사라지고 Dummy 의
    순위정보 없음이 정직하게 드러남. ExtraTrees pairwise=0.82, RandomForest 0.80, XGBoost 0.68.
- **신규 산출물**: `reports/stage2_v16/chen16_single_fit_holdout.csv`,
  `summary.json` 의 `chen16_pooled_loo` / `chen16_single_fit_holdout`(기존 `chen16` 키는 pooled LOO 로 보존).

## 2. Ablation 기준 점수 재검증 — 확인 완료, delta 재계산 추가
- **확인**: `ablation_manufacturing` 은 `A_only`·`B_full`·`add_one_to_A`·`drop_one_from_B` 를 모두
  **동일 `eval_cols`**(동일 algo=Ridge, 동일 scheme=LODO, 동일 행/fold)로 산출한다 → 평가 기준 일치.
  `baseline_consistency` 필드로 이를 명시.
- **수정**: 같은 run 의 `A_only` 기준 `delta_r2_vs_A`/`delta_mae_vs_A` 를 변수별로 산출·보고(`delta_vs_A_only`).
- **재검증 수치(동일 기준 A_only R²=0.570, MAE=26.10)**: `manufacturing_route` ΔR²=+0.097·ΔMAE=−3.64,
  `feedstock` ΔR²=+0.038·ΔMAE=−2.02 (같은 기준에서 **범주형은 A 대비 도움**). 반면 고결측 수치형
  `pyrolysis_hold_min` ΔR²=−0.367, `heating_rate` ΔR²=−0.145 (악화).
  - **원인 규명**: 인계 문서가 지적한 모순(“범주형이 A보다 낮다”)은 **서로 다른 평가절차의 점수를 섞은 것**
    (ablation 의 `A_only`=0.570 vs nested-A=0.675)에서 비롯됐다. 같은 ablation 기준으로 통일해 비교하면
    범주형 추가가 A 를 넘어서는 것이 맞다. 다만 결측률–기여도 연관만으로 “고결측이 성능저하의 주원인”
    이라고 **인과 단정하지 않는다**는 주석을 보고에 포함.
- **신규 산출물**: `ablation.csv` 에 `delta_mae_vs_A`/`delta_r2_vs_A` 열, `summary.json` 의
  `ablation.baseline_consistency`·`ablation.delta_vs_A_only`·`interpretation_note`.

## 3. LOFO의 DOI 중복 확인 — 확인 완료
- **확인(실데이터)**: Core 97행 / DOI 25 / feedstock 30. 각 feedstock 은 1개 DOI 에만 등장하나,
  **2개 DOI 는 2개 이상 feedstock 을 포함**. LOFO 30개 fold 중 **7개 fold** 에서
  `train_doi ∩ test_doi ≠ ∅`.
- **수정**: `lofo_doi_overlap_audit()` 신설 — fold 별 `train/test` 의 DOI·feedstock 교집합을 집계하고
  `doi_independent` 판정. 결과 `doi_independent=False`, 분류 =
  `"new feedstock within partly seen DOI (DOI 독립 아님; LODO 와 동등 주장 불가)"`.
- **결론**: LOFO 를 LODO 와 동등하거나 독립적 외삽 검증으로 서술하면 안 된다(DOI 공유로 새 DOI 효과와
  새 원료 효과 분리 곤란). 보고에 명시.
- **신규 산출물**: `reports/stage2_v16/lofo_doi_overlap.csv`, `summary.json` 의 `lofo_doi_overlap`.

## 4. f와 qe 평가 지표 분리 — 수정 완료
- **확인**: `fraction_target_experiment` 는 이미 `qe_direct`/`f_rate_then_qe`(qe 공간)와 `f_target_metrics`
  (무차원 f 공간)를 분리 계산하나, 보고(CSV)에는 qe 공간 수치만 기록돼 `R²(f)=0.19` 혼동 소지가 있었다.
- **수정**: 보고에 `target_space`(qe/f) 열을 추가해 평가공간을 명시하고 f 공간 지표를 CSV·summary 에 포함.
  `fraction_experiment_note` 로 “f 를 학습한 예측을 qe 로 환산한 R²(qe) 를 R²(f) 라고 말하지 않는다” 명기.
- **재검증 수치(GroupKFold_DOI, Ridge)**: **R²(qe)=0.443**(qe 공간) vs **R²(f)=−0.451**(f 공간),
  MAE(f)=0.307. LODO 는 R²(qe)≈0.194, R²(f)≈−0.340. → 두 값은 명백히 다른 지표이며 혼용 불가.
- **신규 산출물**: `fraction_experiment.csv` 에 `target_space` 열 + f 공간 행, `summary.json` 의
  `fraction_experiment_note`.

## 테스트 통과 여부
- **v16 재검증 테스트 통과**: `python -m pytest tests/test_biochar_stage2_v16.py` → **8 passed in 2.72s (exit 0)**.
  내 수정이 v16 평가 로직·누출방지·구조 테스트를 깨지 않음을 확인.
- **편집 파일 정적 진단**: `v15_recommend.py`·`v16_validate.py`·`v16_report.py` 모두 오류/경고 없음.
- **전체 리포트 재생성 성공**: `v16_report.run_v16_report` 가 신규 필드 포함해 `reports/stage2_v16/`
  전체를 오류 없이 재생성(summary.json + 신규 CSV). 위 재검증 수치는 이 재생성 산출이다.
- **전체 `pytest tests` 미완료(환경 성능 제약)**: 전체 스위트는 다수 파일에서 모델(RF/XGBoost 등)을 학습해
  이 환경에서 1회 실행이 매우 길어(참고: v16 리포트 1회 ≈ 5분) 제한 시간 내 완료되지 않아 **전체 통과 여부는
  미확인**이다. 이는 코드 실패가 아니라 실행시간 문제이며, 변경 범위를 직접 검증하는 v16 테스트는 통과했다.
  재현 명령: `python -m pytest tests -q`(충분한 시간 확보 권장). 가능하면 `pytest-timeout` 설치 후
  테스트별 상한을 두고 느린 테스트를 식별할 것.

## 보존 확인
- v15 배포 모델 설정(`v15_models.available_algorithms` 등), 데이터 로더, 입력계약, Streamlit UI 는
  수정하지 않았다. 신규 평가 함수만 추가했고 기존 공개 API/출력 키는 보존(하위호환)했다.
