# Stage 1 v18 — 원료 조성의 추가 예측 기여도 (탐색적 독립 DOI 평가)

## 한 줄 결론
비활성화 바이오매스 BET 평가 데이터 182행·24 DOI에서 **동일한 LODO 검증**으로 RandomForest에 원료 공업·원소분석을 추가한 경우, 제조조건만 사용한 비교모델보다 MAE가 감소했다. 그러나 **이 결과는 현재 v60 배포 모델의 개선을 입증하거나 TPV/Stage 2 개선을 입증하지 않는다.**

## 데이터·정의
- 출처: `data/imports/Biochar_Complete_Project_v41/eval37/base.csv`.
- 182개 관측, 24개 `provenance_group`, 원료 조성 패턴 44개, 공정조건 패턴 91개. 필수 10개 변수/타깃 결측 0.
- 타깃: **BET 비표면적(m²/g)**. 입력: HTT(°C), 유지시간(min), 승온속도(°C/min); 조성은 원료 `C/H/N/O` (dry ash-free basis)와 `VM/Ash/FC` (dry basis).
- 이 자료의 적용영역은 비활성화 바이오매스 열분해. v60의 활성화 소재·제조경로 전체에 적용할 수 없다.
- 원료명, 원료 활성화조건, 최종 바이오차 조성은 해당 실험의 입력에 없다. 이를 반영한 것으로 표시하지 않는다.

## 방법
- LODO(`LeaveOneGroupOut`, DOI provenance 단위), 훈련/시험 DOI는 엄격히 분리. 모든 구성 동일 182개 시험 행·동일 홀드아웃.
- **고정 모델**: RandomForest(120 trees, max_depth=7, min_samples_leaf=3, seed=20261030), Ridge(alpha=10). 테스트 점수로 하이퍼파라미터를 변경하지 않음.
- 전처리는 각 학습 Fold 내부에서만 fit. OOF 예측을 합쳐 pooled R², MAE, RMSE 및 DOI-macro MAE 평가.
- DOI 단위 재표본 Cluster Bootstrap 2,500회(seed=20261030)로 `MAE(추가 변수) − MAE(제조조건만)`의 95% Percentile CI를 계산. 이 CI는 동일 자료에서 여러 변수세트를 살펴본 이후의 탐색적 CI이므로 확증적 유의성으로 주장하지 않는다.

## 결과 — RandomForest

| 입력 | OOF R² | MAE(m²/g) | DOI-macro MAE(m²/g) |
|---|---:|---:|---:|
| 제조조건만 | −0.460 | 97.75 | 106.05 |
| 제조조건 + 원소분석 | −0.099 | 85.67 | 88.37 |
| 제조조건 + 공업분석 | 0.253 | 69.52 | 76.17 |
| 제조조건 + 전체 조성 | **0.257** | **66.42** | **80.48** |

- **전체 조성 vs 제조조건만**: 행 단위 MAE 차이 −31.32 m²/g, DOI-cluster Bootstrap 95% CI **[−62.72, −1.34]**; 상대 감소 약 **32.0%**.
- DOI별 평균 MAE가 감소한 그룹은 **17/24 DOI**. 다만 DOI 동등가중(macro) MAE 차이 −25.58 m²/g의 Bootstrap CI **[−57.50, +5.26]**은 0을 포함하므로 논문별 개선이 일관적으로 확인됐다고 단정할 수 없다.
- 공업분석만 추가해도 R² 0.253으로 전체 조성과 유사했다. 추가적인 원소분석 정보의 유익성이 확정되지 않았다.
- Ridge는 전 구성에서 R²가 크게 음수이며, 조성 추가의 일관된 성능 개선을 보이지 않았다. 즉 **효과는 모델 알고리즘에 의존**한다.
- 본 결과는 *고정 후보 모델 간 탐색적 성능 비교*이지, 사후 모델 선택을 포함한 Nested CV나 잠금 외부 Test의 점수가 아니다.

## Stage 1 v60과의 관계 — 지금 바로 연결 불가
- v60 master: 256행/30 DOI. v60 BET 모델은 주로 활성화·제조공정 정보를 쓰는 **다른 데이터·아키텍처**다. 조성값은 현재 실제 v60 추론 Feature에서 drop된다.
- 원료 C/H/O/N 네 값 완전: 36행. Ash/VM/FC 완전: 17행. **둘 다 완전한 자료는 17행, 단 1 DOI**. 따라서 이 17행만으로 v60에서 조성을 더할 때 신규 DOI 일반화가 개선된다는 주장을 검증할 수 없다.
- 조성 basis의 혼합·미확인 값이 있어 임의로 전체 표본을 중앙값 보충하거나 다른 원료의 평균값을 주입해 학습하지 않는다.
- 기존 v60 연구 잠금 외부 Challenge의 BET R² −0.891, TPV R² −3.331 문제는 이 실험으로 해결되지 않는다.

## TPV 별도 점검
- `TPV_default_train_pool.csv` 748행 중 ACCEPTED 545행. 원소분석·공업분석·온도·유지시간이 모두 있는 483행은 **전부 원실험 원문 검증이 완료되지 않은 2차 편찬 자료**다.
- 그 483행에는 승온속도와 명시된 원료명이 모두 결측이고, 표시된 공정·조성 입력이 일치하는 중복 조합이 264행 있다. 활성화 조건이 누락되어 서로 다른 TPV를 같은 입력에 매칭했을 가능성을 배제할 수 없다.
- 따라서 TPV 조성 보조모델은 이번에 학습하거나 배포하지 않았다. TPV 결과 개선 여부는 **미확인**이다.

## UI·모델 운영에 대한 결정
1. 자연어 원료명 입력과 선택적 조성 입력 UI는 유지한다.
2. 해당 조성값이 현재 v60 추론에 반영되지 않는다는 문구를 유지한다.
3. v18은 기존 Stage 1 v60, Stage 2 v16/17 모델을 **변경하지 않고**, 기존 코드에 독립 검증 모듈만 추가했다.
4. 추후 비활성화 단일 열분해 BET의 **독립 실험용 Composition-assisted 모델**을 만들 수 있으나, 배포 전 별도 잠금 DOI Test와 입출력 basis 검증이 필요하다.
5. 2차 데이터 TPV, 활성화된 제품, 신규 원료에 이 결과를 일반화하지 않는다.

## 재현
```powershell
.\.venv\Scripts\python.exe -m scripts.run_stage1_composition_v18
.\.venv\Scripts\python.exe -m scripts.plot_stage1_composition_v18
.\.venv\Scripts\python.exe -m pytest tests/test_stage1_composition_v18.py -q
```

### 파일
- `model_comparison.csv`: 동일 OOF 시험 행 성능
- `oof_predictions.csv`: DOI별 실측/모델 예측
- `paired_doi_bootstrap.csv`: 짝지은 DOI 재표본의 MAE 차이
- `v60_composition_coverage.csv`: v60 입력 가용성 감사
- `summary.json`: 데이터·방법·수치·TPV 실행 보류 사유
- `fig_*.png`: 성능 및 논문별 이질성 시각화

**검증 상태:** 신규 v18 단위 테스트 5/5 통과; 관련 Stage 2 입력·v16 회귀 테스트 포함 23/23 통과. 전체 프로젝트 테스트 및 Windows 실제 UI 실행은 이 결과에서 검증하지 않았다.
