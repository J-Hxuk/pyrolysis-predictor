# Stage 2 (qe) 독립 검증(봉인) 프로토콜

**버전:** v1 (2026-10-31 확정)
**역할:** 평가 설계 문서. 활성 배포 모델·학습 데이터를 변경하지 않음.
**목적:** Stage 2 흡착량(qe) 모델의 **진짜 일반화 성능**을 측정하기 위한, 사후 편향이 없는
봉인(sealed) 검증 체계를 사전에 확정한다. 데이터가 준비되면 이 규칙대로 **단 한 번** 평가한다.

---

## 1. 배경 — 왜 봉인 검증이 필요한가

현재 보고된 모든 Stage 2 성능 수치(그룹 CV R² 0.55~0.72, 분포 분석의 DOI별 음수 R² 다발)는
**사후(post-hoc) 평가**다. 평가에 쓰인 36개 DOI는 모델 선택·튜닝 과정에서 이미 관찰된
논문들이므로, 낙관적으로 편향됐을 가능성이 있다. 기존 코드도 이를 명시한다:
`run_v59` 의 source-out은 주석에 *"still retrospective, previously examined DOIs"*,
`run_v57` 의 프로토콜 문자열은 `PRE_INCLUSION_SOURCE_OUT_POSTHOC_NOT_FUTURE_BLIND`.

실제로 Stage 1 에서는 봉인 챌린지(3 DOI, 23행)가 BET R²=**−0.891** 로 나와,
**사후 CV(R²≈0.34)와 봉인 성능 사이에 큰 격차**가 있음이 확인됐다. Stage 2 도
같은 종류의 봉인 검증이 없으면 일반화 성능을 신뢰할 수 없다.

## 2. 현재 데이터로는 '진짜 봉인'이 불가능하다 (근거)

Stage 2 188행/36 DOI 중 "새 DOI"로 추가된 `new36` 은 사실상 **2개 DOI뿐**이다:
- `10.1007/s11356-017-8847-2` (Shen 2017): 29행, 단일 등온선 스윕(4 재료, C0 시리즈) — 조건 고도 상관
- `10.1016/j.scitotenv.2019.07.102` (Cao 2019): 7행, 48h qTotal(평형 미확인)

DOI 2개(그중 29/36행이 한 논문)로는 DOI-그룹 봉인 평가의 **유효 표본이 ~2 그룹**에 불과해
통계적으로 의미 있는 봉인 검증이 성립하지 않는다. 또한 Stage 2 CSV에는 **publication year
컬럼이 없어** "최신 DOI 봉인"도 불가능하다 (lineage 신호는 `model_use_tier` 의 `V##` 배치 태그뿐).

**결론: 지금 있는 자료로 Stage 2 봉인 세트를 억지로 만들면 '가짜 봉인'이 되어 오히려 잘못된
안심을 준다.** 따라서 본 프로토콜은 **규칙을 사전 확정**하고, 적격 데이터가 확보되는 즉시
그대로 실행하는 방식으로 설계한다.

## 3. 봉인 대상 DOI (사전 지정)

`data/imports/v57_handoff_sources/V57_SOURCE_REGISTRY.csv` 에 등록돼 있으나 **아직 학습·선택에
한 번도 쓰이지 않은(STAGED)** DOI 를 봉인 후보로 사전 지정한다:

| DOI | 논문 | 상태 | 비고 |
|---|---|---|---|
| 10.1021/acs.est.7b06261 | Alam 2018 | SOURCE_STAGED | pH 의존 그래프 추출 + Zn 용출 QC 대기 |
| 10.1016/j.ecoenv.2014.04.042 | Sun 2014 | SOURCE_STAGED | 등온선 Figure1 C0 재구성 QC 대기 |
| 10.1016/j.scitotenv.2020.136894 | Zhao 2020 | SOURCE_STAGED | Figure3 실험 추출 QC 대기 |

(참고: `10.1021/jf9044217` Uchimiya 2010 은 SOURCE_HOLD 로 본문 미포함 — 봉인 대상 제외.)

**적격 기준(봉인 세트에 포함되려면 모두 충족):**
1. DOI 가 현재 활성 학습 데이터(188행)와 **완전히 disjoint**.
2. 물리 제약 통과: `0 ≤ qe ≤ C0/dose` (`physical_cap_pass`).
3. 평형 qe 로 보고되거나, 최소한 `target_origin`/`target_qualification` 에 측정 근거 기록.
4. 필수 입력(16개 Model B 피처)이 결측 없이 또는 명시적 결측으로 존재.

## 4. 봉인 규칙 (잠금 절차)

Stage 1 v60 챌린지 프로토콜(`05_V60_Challenge_Protocol.json`)의 잠금 언어를 Stage 2 에 그대로 적용한다:

1. **모델 동결**: 평가에 사용할 활성 Stage 2 모델 아티팩트(현재
   `deployment/stage2_models/stage2_B_XGBoost_ebb4783561d745f8`)의 **SHA-256 을 기록**하고,
   그 해시의 모델로만 예측한다. 봉인 평가를 위해 **재학습·재선택하지 않는다.**
2. **누수 차단**: 봉인 DOI 의 어떤 행도 모델 학습·하이퍼파라미터 선택·피처 선택·임계값
   조정에 사용되지 않았음을 `assert held_doi ∉ train.doi` 로 코드에서 검증한다.
3. **단 한 번 평가**: 봉인 세트는 **딱 한 번** 점수화한다. 점수를 본 뒤 모델을 고치면 그
   세트는 오염되며, 새 봉인 세트가 필요하다.
4. **메트릭 동결**: 평가 직후 예측·지표·SHA-256·환경 버전을 `run_manifest.json` 에 기록하고
   변경하지 않는다.
5. **역할 태그**: 봉인 행에 `evaluation_role = STAGE2_SEALED_CHALLENGE_V1` 을 부여해 학습
   데이터와 식별 분리한다.

## 5. 평가 지표 (사전 확정)

봉인 세트가 작으므로 **집계 + DOI 분포 + 물리 타당성 + 베이스라인 대비**를 함께 본다:

- **집계**: MAE, RMSE, R² (capped 예측 기준).
- **DOI별**: DOI 단위 MAE·R² 와 그 분포(중앙값/최악). DOI-그룹 부트스트랩 95% 구간.
- **물리 타당성**: clipping 전 상한 위반율.
- **베이스라인 대비**: `predict=0` MAE, `predict=global_median` R² 와 비교해 모델이 자명한
  기준보다 나은지 확인 (Stage 1 챌린지가 이 비교를 포함했음 — 음수 R² 해석의 기준).
- **route별**: `v59_route_family`(none/physical/chemical)별 성능 — Stage 1 에서 활성화 공정
  외삽이 실패 원인이었으므로 Stage 2 도 route별로 확인.

## 6. 성공·실패 해석 기준 (사전 확정)

- 봉인 R² > 0 이고 DOI 중앙값 MAE 가 사후 CV 수준과 비슷하면 → 일반화 신뢰도 상향.
- 봉인 R² < 0 (평균값보다 못함)이면 → 사후 CV 가 낙관 편향임을 확증. 현실 성능은 봉인값.
- route별로 none 은 양호하나 activation(physical/chemical)에서 무너지면 → Stage 1 과 동일한
  **활성화 공정 외삽 한계**로 진단하고, 적용 범위를 비활성화 영역으로 제한 권고.

## 7. 실행 절차

데이터 준비 후:
1. 봉인 CSV 를 `data/imports/stage2_sealed_v1/Stage2_Sealed_Challenge_V1.csv` 로 배치
   (§3 적격 기준·§4 역할 태그 충족).
2. `python -m scripts.run_stage2_sealed_eval --root .` 실행
   (구현 틀: `scripts/run_stage2_sealed_eval.py`).
3. 산출물은 `reports/stage2_sealed_validation/` 에 기록되고, 메트릭은 동결된다.

## 8. 현재 상태

- 봉인 규칙·대상·지표: **확정됨(본 문서).**
- 봉인 데이터: **미확보** — STAGED DOI 3개의 qe 추출(디지털화)과 QC 가 선행돼야 함.
- 실행 코드: `scripts/run_stage2_sealed_eval.py` 에 틀 작성(데이터 없으면 명확히 안내 후 중단).
