# Biochar Stage1 Expansion v5 — 데이터 반영 현황

작성 기준: 첨부 ZIP `Biochar_Stage1_Expansion_v5.zip` 를 별도 데이터 버전으로 추가하고
변수·결측·중복·출처 그룹을 집계한 결과. **모델 학습·재학습은 수행하지 않았고,
`eligible_for_training` 플래그를 변경하지 않았다.**

## 1. 위치 정리 (기존 보존, 별도 버전 추가)

- 원본 ZIP: `data/Biochar_Stage1_Expansion_v5.zip` (기존 `data/261008_final_test data.xlsx` 보존)
- 해제 위치: `data/imports/Biochar_Stage1_Expansion_v5/` (67개 파일)
- 집계 산출물: `runs/biochar_import_v5/aggregate_report.json` (로컬 연구 기록, Git 제외)
- 로더 모듈: `pyrolysis/biochar_data.py` (읽기 전용 집계. 기존 코드 미수정)

`data/`·`runs/` 는 `.gitignore` 로 Git 에서 제외된다. 실제 자료는 저장소에 올라가지 않는다.

## 2. 데이터 버전 구분 (합치지 않음)

같은 자료의 원본·검증본·통합본이 중복으로 들어 있어 **폴더 간 CSV 를 합치지 않는다.**

| 버전 | 위치 | 성격 |
|---|---|---|
| 원본 v2 | `biochar/data/` | 빌드 산출물: Material_Master(454), Stage1_BET(259), Stage1_PoreVolume(77), Stage2_qe(7177) |
| 검증 이력 v3 | `biochar/verified_v3/` | 원문 대조 중간 결과 (최종 학습 데이터 아님) |
| 검증 이력 v4 | `biochar/verified_v4/` | 원문 대조 추가 결과 (Jaffari·Zhu·Shen·PB 등) |
| 최신 Stage1 통합 후보 v5 | `biochar/verified_v5/` | **BET 47 / 총 공극부피 42** source-checked 후보 |

로더는 최신 통합 후보(v5)와 Stage2(원본 대조)를 **구분해서** 읽는다. 원본·검증본을 섞지 않는다.

## 3. Stage1 최신 통합 후보 집계 (v5)

제조조건 → 물성(BET, 총 공극부피) 예측용 후보. **모두 `eligible_for_training=False`,
`qa_status=SOURCE_CHECKED`** (최종 학습 승인 아님). 이번 신규 전사 `stage1_candidate=True` 29건.

| 항목 | BET 후보 | 총 공극부피 후보 |
|---|---:|---:|
| 행 수 | 47 | 42 |
| 타깃 값 보유 | 47 | 42 |
| stage1_candidate=True (신규) | 29 | 29 |
| 완전 중복 행 | 0 | 0 |
| 출처 그룹(leakage_group) | 8 | 7 |
| 원실험 DOI 그룹 | 8 | 7 |

제조조건 입력 확보율 (BET 47 기준):

| 입력 | 확보 | 결측 |
|---|---:|---:|
| feedstock(원료 종류) | 47 | 0 |
| pyrolysis_temp_C | 47 | 0 |
| holding_time_min | 42 | 5 |
| heating_rate_C_min | 36 | 11 |
| atmosphere | 42 | 5 |
| gas_flow_mL_min | 19 | 28 |
| modification_method | 42 | 5 |

원료 조성 확보율(BET 47): C 25 / H 13 / O 5 / N 25 / Ash·VM·FC 각 8.
→ 원료 전체 조성을 필수로 요구하면 표본이 급감한다. 기본 입력은
**원료 종류·온도·유지시간·승온속도·분위기·개질 여부**를 우선 검토하고, 조성은 확장 모델에서
별도 평가하는 것이 타당하다 (README 권고와 일치).

원료 범주(BET 47): rubber wood sawdust 8, pruned tea branch 5, mango peel 5, corn straw 4,
spent Pleurotus/shiitake substrate 각 3, poplar leaf/twig/bark 각 3, rice straw 3 등 15종.
원료·논문 수가 적어 새로운 원료에 대한 예측 신뢰성을 주장할 수 없다.

## 4. Stage2 qe 집계 (원본·검증 대조)

흡착조건 + (예측 물성) → 평형 흡착량 qe 예측. **원본과 검증 결과를 대조한다.**

| 항목 | 값 |
|---|---:|
| 행 수 | 7,177 |
| eligible_for_training | 전부 False |
| equilibrium_status | 전부 UNVERIFIED |
| measured_qe_mg_g (검증 실측 qe) | **0** |
| reported_capacity (미검증 후보) | 5,856 |
| BET·Pore 둘 다 보유 | 5,285 |
| qa_status | REVIEW 6,184 / EXCLUDE 730 / HOLDOUT 263 |

**미검증 흡착량을 실측 평형 qe 로 취급하지 않는다.** 평형이 입증된 qe 가 0건이므로
Stage2 는 현재 학습 확정 행이 없다 (README·HANDOFF 와 일치).

흡착조건 입력 확보율: pollutant 7,161 / initial_conc_mg_L 5,330 / initial_pH 6,657 /
adsorbent_dose_g_L 1,642 / adsorption_temp_C 6,629 / contact_time_min 5,291.
오염물질 40여 종(IBU·Sr(II)·IBF·CBZ·Cd·Pb 등). 단일/경쟁 흡착, 단위 혼재가 남아 있다.

## 5. 예측 구조 (유지)

```
제조조건 ──► Stage1-BET (회귀) ─┐
제조조건 ──► Stage1-Pore(회귀) ─┤
                               ├─► 흡착조건과 결합 ──► Stage2-qe (회귀, 평형 검증 필요)
```

Stage1 에서 예측한 BET·총 공극부피를 Stage2 입력으로 결합하는 **연계 구조**다.
본 프로젝트 설계의 조건부 작업 33(작용기·기공 예측을 성능 예측에 연결)과 대응한다.
외부 평가 독립성·누수 그룹 처리가 전제이며, 아직 구현·학습하지 않는다.

## 6. 현재 프로젝트와 연결하려면 필요한 것

이 데이터를 본 프로그램(configs/local + 학습 파이프라인)에 연결하려면 아래가 필요하다.
모두 **연구 판단·검증이 선행**해야 하며 지금은 수행하지 않았다.

1. **대상 승인 기준 확정(작업 25–28)**: `eligible_for_training` 을 어떤 조건에서 True 로
   올릴지 연구 담당자가 정한다. 현재는 Stage1·Stage2 모두 False.
2. **Stage1 feature 확정**: 기본 모델 입력(원료 종류·온도·유지시간·승온속도·분위기·개질)과
   조성 확장 모델을 분리. 결측·범주 분포를 근거로 선정.
3. **출처 그룹 분할 정의**: `leakage_group`/`original_experiment_doi` 로 논문·공유 시료가
   학습·평가에 동시에 들어가지 않도록 그룹 분할. Stage2 는 leakage_group 열이 없어 생성 필요.
4. **Stage2 평형 검증**: 원문에서 평형·단위·단일/경쟁 흡착을 확인해 `measured_qe_mg_g` 로
   승격. 승격 전에는 Stage2 학습 불가.
5. **단위·범주 정규화 규칙**: atmosphere 표기, 공극부피 환산 등 (README 에 기록된 불일치 반영).
6. **본 프로그램 설정 작성**: 위가 정해지면 `configs/local/problem.yaml`·`dataset.yaml`·
   `training.yaml` 로 반영하고 `audit` → (승인 후) 학습으로 진행.

## 7. 이번 작업에서 하지 않은 것 (명시)

- 모델 학습·재학습: **하지 않음.**
- `eligible_for_training` 변경: **하지 않음** (전부 원본 그대로 False).
- 폴더 간 CSV 병합, 미검증 흡착량의 qe 승격: **하지 않음.**
- 기존 프로젝트 파일·데이터·모델 수정: **하지 않음** (읽기 전용 로더만 추가).


---

# Stage1 학습 데이터 선정 준비 (상세 점검)

근거 산출물: `runs/biochar_import_v5/stage1_selection_report.json`,
`stage2_versions.json`. 로더: `pyrolysis/biochar_data.py` (읽기 전용).
**학습 플래그 변경·모델 학습은 하지 않았다.**

## 1. 제조조건 상세 점검 (BET 47 / Pore 42)

수치형 (둘 다 비수치 혼입 0, 중복 시료 0):

| 변수 | 범위 | 결측 (BET / Pore) |
|---|---|---|
| pyrolysis_temp_C | 300–800 (7개 값) | 0 / 0 |
| holding_time_min | 60–240 (60/120/150/180/240) | 5 / 5 |
| heating_rate_C_min | 5–10 (5/10) | 11 / 11 |
| gas_flow_mL_min | 30–200 (30/60/200) | 28 / 28 |

범주형:
- `feedstock`: BET 15종 / Pore 14종, 결측 0.
- `atmosphere`: N2, `hypoxia (gas not specified)`, `oxygen-limited muffle furnace` + 결측 5.
  hypoxia 를 N2 로 간주하지 않는다.
- `modification_method` 분류 (무개질/개질/**미보고**):
  - BET: none 35, modified 7, **미보고 5**
  - Pore: none 35, modified 2, **미보고 5**
  - 원문 `unmodified`·`no added modification` = 무개질(none). **미보고(결측)는 none 으로 대체하지 않는다.**

중복: `canonical_material_id` 전부 고유(47/42), 제조 fingerprint 중복 행 0.

## 2. 입력 구성별 사용 가능 시료·출처 그룹 (결측 선삭제 없음)

필수 입력이 갖춰진 행만 집계(미보고 결측은 해당 구성에서 제외하되 삭제하지 않음):

| 구성 | BET 시료 | BET 그룹 | Pore 시료 | Pore 그룹 |
|---|---:|---:|---:|---:|
| 최소 (원료·온도·개질) | 42 | 7 | 37 | 6 |
| 확장 (+유지시간·승온속도·분위기) | 36 | 6 | 31 | 5 |

- 최소 구성에서 제외 5건은 modification 미보고 행(= `SMS_PC_SC_SHARED_2018_2019` 추정).
- 확장 구성은 승온속도·분위기 결측으로 그룹 1개가 더 빠진다.

## 3. 범주의 논문 종속성 (중요)

- **feedstock 의 모든 범주가 단일 출처 그룹에만 존재**한다 (BET 15/15, Pore 14/14 모두
  `n_groups=1`). 즉 **LOGO 로 어떤 그룹을 빼든 그 그룹의 원료 범주는 전부 테스트 전용**이 된다.
  예: `10.1111/gcbb.13080` 을 빼면 poplar leaf/twig/bark·rice straw 가 학습에 없는 원료가 된다.
  → 원료를 one-hot 범주로만 쓰면 held-out 논문의 원료를 "미학습 범주"로 예측한다.
  원료 조성·물성 기반 표현(수치 feature)으로 대체하는 구성을 함께 검토해야 한다.
- `modification_method`: BET 는 none 5그룹/modified 2그룹(단일그룹 없음). Pore 는 modified 가
  단일그룹(`10.4491/KSEE.2020.42.2.29`) → 그 그룹을 빼면 '개질' 범주가 테스트 전용.

## 4. Leave-One-Group-Out 가능성

- BET: 8그룹 → LOGO 8 fold 가능. 그룹 크기 2~12. fold별 학습 행 35~45.
- Pore: 7그룹 → LOGO 7 fold 가능. 그룹 크기 2~12. fold별 학습 행 30~40.
- **전처리·결측 대체는 각 학습 fold 안에서만 fit** 한다(평가 그룹 통계 미사용). 3번 때문에
  일부 fold 는 테스트 전용 원료 범주를 포함하므로, 인코더는 미지 범주를 처리할 수 있어야 한다
  (예: one-hot 의 handle_unknown, 또는 원료 수치 표현).

## 5. Stage2 버전별 현황 (별도 집계, 합치지 않음)

- **v2 원본(미검증)** `biochar/data/Stage2_qe.csv`: 7,177행, `measured_qe_mg_g` 0건,
  equilibrium_status 전부 UNVERIFIED, qa REVIEW 6,184/EXCLUDE 730/HOLDOUT 263.
- **v3 부분검증**: CSV 15개 (Stage2_verified_qe_partial_conditions 357, Soria 345 등).
- **v4 부분검증**: CSV 14개 (JAF23 3,757 리뷰, SHEN24/ZHU19 계열, PB2020 실측 qe 4건 등).
- v3·v4 는 **중간 검증 산출물**이며 원본에 합치지 않고 전부 학습 승인하지 않는다.
  미검증 흡착량을 실측 평형 qe 로 승격하지 않았다.

## 6. problem.yaml 초안

`configs/local/problem.yaml` (Git 제외) 에 Stage1 초안을 작성했다.
- `status: draft`, `is_training_ready()=False` (프로젝트 로더로 검증).
- 입력: 확장 구성 6개. feedstock/atmosphere/modification 은 **관측된 범주만** 기록.
- 대상: `BET_m2_g`, `pore_volume_cm3_g` (둘 다 회귀, 비음수).
- modification 은 none/modified 만 범주로 두고, 미보고는 결측으로 처리(none 대체 금지).
- 온도·유지시간·승온속도의 physical_constraints 는 '관측 범위'로 표시(실제 물리 한계는
  연구 판단으로 확정).

## 7. 이번 단계에서 하지 않은 것

- 모델 학습·재학습, `eligible_for_training` 변경: **하지 않음.**
- 결측 행 선삭제: **하지 않음** (구성별 집계상 제외만).
- 미보고를 무개질로 대체: **하지 않음.**
- v3·v4 검증본을 원본에 병합하거나 학습 승인: **하지 않음.**

## 8. 다음 결정 필요 (연구 판단)

1. 원료 범주의 논문 종속성 대응: one-hot + handle_unknown 유지 vs 원료 조성/물성 수치 표현.
2. 최소 vs 확장 구성 선택 (시료 수 ↔ 정보량 교환). 작은 그룹 수에서 LOGO 신뢰도 한계 인지.
3. modification 미보고 5건 처리 방침(제외 유지 / 원문 재확인).
4. atmosphere 의 hypoxia·muffle 범주를 별도로 둘지, 연구적으로 어떻게 다룰지.
5. 승인 기준 확정 후에만 `eligible_for_training` 과 `status: configured` 로 전환.


---

# Stage1 탐색적 평가 — 새 원료·새 논문에 대한 일반화 검증

**이 결과는 '탐색적 평가 — 새 원료·새 논문에 대한 일반화 검증'이다. Stage2 흡착 성능이나
전체 연계 성능이 아니며, 최종 배포 모델 선정이 아니다.**

- 모듈: `pyrolysis/biochar_stage1_eval.py` (읽기 전용, 승인 플래그 미변경)
- 산출물: `runs/biochar_import_v5/stage1_exploratory_eval.json`,
  `runs/biochar_import_v5/manifests/eval_manifest_{bet,pore}_{minimal,extended}.csv`
- 방법: `leakage_group` 기준 LOGO(Leave-One-Group-Out). 각 fold 에서 전처리를 학습 fold 로만
  fit(중앙값·최빈값 대치, 표준화, one-hot `handle_unknown=ignore`). 고정 보수적 설정,
  테스트 fold 로 튜닝하지 않음. R² 는 테스트 그룹 분산이 0이거나 n<2면 NA.

## 분위기·장치 정규화 (원문 표기 보존)

- `N2` → `inert_N2`
- `hypoxia (gas not specified)` → `oxygen_limited` (가스 미지정, 10.1111/gcbb.13080)
- `oxygen-limited muffle furnace` → `oxygen_limited` + 장치 `muffle_furnace` (SMS 그룹)
- 결측(10.1016/heliyon) → `unreported` (추정하지 않음)
- muffle 만으로 분위기를 추정하지 않는다. 원문 표기는 `atm_raw` 로 보존.

## 평가용 데이터 목록·제외 사유 (별도 저장, 플래그 보존)

`eval_manifest_*.csv` 에 시료별 `included`/`exclusion_reasons` 와 **원본
`eligible_for_training_original`** 을 함께 담았다. 제외는 삭제가 아니라 '해당 구성에서
사용 불가' 표시이며 원본 후보 파일은 바꾸지 않았다. 주요 제외 사유: modification 미보고,
atmosphere 미보고, 승온속도 결측 등.

## BET (타깃 BET_m2_g, 단위 m²/g)

적격 시료: 최소 42 / 확장 36. 실제 남은 그룹으로 LOGO 재생성(최소 7 fold, 확장 6 fold).

[A] 각 구성 전체 적격 데이터 (구성마다 시료·그룹이 다름):

| 구성 | 모델 | fold | MAE | RMSE | R²(OOF) |
|---|---|---:|---:|---:|---:|
| minimal | mean_baseline | 7 | 112.0 | 148.4 | -0.08 |
| minimal | random_forest | 7 | 81.4 | 127.2 | +0.21 |
| minimal | xgboost | 7 | 69.7 | 123.6 | +0.25 |
| extended | mean_baseline | 6 | 118.5 | 157.4 | -0.10 |
| extended | random_forest | 6 | 92.4 | 139.8 | +0.13 |
| extended | xgboost | 6 | 92.9 | 152.4 | -0.03 |

[B] 확장 구성 동일 시료(36, 6 fold)에서 최소 vs 확장 통제 비교:

| 구성 | 모델 | MAE | RMSE | R²(OOF) |
|---|---|---:|---:|---:|
| minimal | random_forest | 86.6 | 136.8 | +0.17 |
| minimal | xgboost | 71.4 | 132.7 | +0.22 |
| extended | random_forest | 92.4 | 139.8 | +0.13 |
| extended | xgboost | 92.9 | 152.4 | -0.03 |

→ BET 는 트리 모델이 평균 기준모델보다 전체 OOF 에서 낫다(R² 양수). **동일 시료 비교에서
최소 구성이 확장 구성보다 더 좋다** (확장이 추가 조건을 넣어도 작은 데이터에서 과적합·
분산 증가). 단, 특정 작은 그룹(예: 2개짜리 KSEE)에서는 모든 모델의 그룹별 R² 가 크게 음수다.

## Pore (타깃 pore_volume_cm3_g, 단위 cm³/g)

적격 시료: 최소 37 / 확장 31. LOGO(최소 6 fold, 확장 5 fold).

[A] 전체 적격 데이터:

| 구성 | 모델 | fold | MAE | RMSE | R²(OOF) |
|---|---|---:|---:|---:|---:|
| minimal | mean_baseline | 6 | 0.075 | 0.128 | -0.15 |
| minimal | random_forest | 6 | 0.077 | 0.128 | -0.15 |
| minimal | xgboost | 6 | 0.116 | 0.157 | -0.75 |
| extended | mean_baseline | 5 | 0.080 | 0.139 | -0.17 |
| extended | random_forest | 5 | 0.107 | 0.153 | -0.43 |
| extended | xgboost | 5 | 0.246 | 0.272 | -3.49 |

→ **Pore 는 어떤 모델도 평균 기준모델을 넘지 못한다(전체 OOF R² 전부 음수).** 즉 현재 적격
시료·입력으로는 새 논문의 총 공극부피를 일반화 예측하지 못한다. xgboost 는 특히 나쁘다.

## 해석과 한계

- feedstock 범주가 **논문 종속**이라 LOGO 에서 held-out 논문의 원료는 대부분 미학습 범주다.
  `handle_unknown=ignore` 로 미학습 원료는 사실상 '원료 정보 없음'처럼 처리되어, 모델이
  주로 온도 등 수치에 의존한다. 이것이 BET 에서 최소 구성이 더 나은 이유와 맞닿는다.
- 작은 그룹(n=2 등)의 그룹별 R² 가 극단적 음수인 것은 표본이 작아 생기는 현상이며 전체
  OOF 지표로 함께 본다.
- 이 수치를 실제 성능 보장·Stage2 성능·공개 배포 성능으로 해석하지 않는다.

## 이번 단계에서 하지 않은 것

- `eligible_for_training` 변경·원본 후보 파일 수정: **안 함.**
- 테스트 fold 로 하이퍼파라미터 선택: **안 함** (고정 보수적 설정).
- 최종 배포 모델 선정: **안 함** (결과 확인 후 별도 진행).
- Stage2·전체 연계 평가: **안 함.**

## 다음 단계 (결과 확인 후 결정)

1. BET: 최소 구성 + 트리 모델이 후보. 원료의 논문 종속성 때문에 "새 원료 일반화"는
   아직 약하므로, 원료를 조성/물성 수치로 표현하는 입력 개선을 검토.
2. Pore: 현재 일반화가 성립하지 않음 → 입력·정의(측정범위 불일치) 재검토 또는 대상 보류.
3. 위 판단 후에만 최종 배포 모델 선정과 `eligible_for_training`·`status: configured` 전환.


---

# Stage 2 흡착성능(qe) ML 개발 현황 — v15 (2026-10-29 패키지 기준)

데이터: `data/imports/Stage1_Stage2_Model_Ready_v15/`로 필요한 CSV·SQLite만 선택 추출
(백업 `BASELINE_v14_DO_NOT_EDIT.zip`는 **중복 압축 해제하지 않고 원형 보존**). 기존 Stage1
v60 모델·데이터는 변경하지 않았다. `data/`는 .gitignore 대상.

## 1. 변경·신규 모듈
- `pyrolysis/biochar_stage2/v15_extract.py` — 필요한 파일만 추출(백업 제외).
- `pyrolysis/biochar_stage2/v15_features.py` — Model A/B/C/D 입력 계약, 금지 필드.
- `pyrolysis/biochar_stage2/v15_loader.py` — Core 97 로더·검증, 타깃 혼합 차단.
- `pyrolysis/biochar_stage2/v15_models.py` — A/B/C × 알고리즘 × 그룹 CV, 물리제약.
- `pyrolysis/biochar_stage2/v15_recommend.py` — Chen16 추천 검증(조건 홀드아웃).
- `pyrolysis/biochar_stage2/v15_train.py` + `v15_cli.py` — 최종 모델 학습·저장·활성화(앱과 분리).
- `pyrolysis/biochar_ui/stage2_pages.py` + `biochar_stage2_app.py` — 공학 스타일 UI 4화면.
- `tests/test_biochar_stage2_v15.py` — 지시 §8 의 12개 항목 자동 검증.
- 저장 모델: `deployment/stage2_models/stage2_{A,B,C}_XGBoost_<hash>/` + `active.json`(=Model B).

## 2. 실제 사용 입력·결측률 (Core 97)
- 97행 / 25 DOI / 60 material / 30 원료 / 5 오염물질. observation_id 고유, material이 여러
  DOI에 걸치지 않음. 단일 타깃 `qe_reported_mg_g`(탐색적 reported uptake).
- 흡착조건 결측: adsorption_T_C·contact_time 27.8%, pollutant/C0/pH/dose 0%.
- 제조조건 결측: heating_rate 57.7%, activation 49.5%, secondary 71.1%, pyro_temp 19.6%.
- **원료 조성 확보율 극히 낮음(raw_C 2행, raw_ash·O·VM·FC 0행)** → 조성 기반 일반화 불가.
  원료명에서 조성을 생성하지 않으며 조성은 필수 입력으로 강제하지 않음.
- 물질수지: target ≤ C0/dose 위반 0건, 기록 cap과 계산 cap 97/97 일치.

## 3. Model A/B/C 성능 (그룹 OOF CV, 무작위 분할 아님)
알고리즘: Dummy/Ridge/RandomForest/ExtraTrees/XGBoost. CatBoost는 환경 미설치로 제외.
최우수는 **XGBoost·Ridge(강한 정규화)**. 트리 앙상블(RF/ET)은 97행에서 과적합·열세.

| 분할 | 모델 | 최우수 알고 | R² | MAE(mg/g) | RMSE |
|---|---|---|---|---|---|
| LODO | A | XGBoost | 0.695 | 18.74 | 33.01 |
| LODO | B | Ridge | 0.637 | 22.65 | 36.03 |
| LODO | C | Ridge | 0.525 | 25.90 | 41.19 |
| GroupKFold(DOI) | A | XGBoost | 0.621 | 19.78 | 36.78 |
| GroupKFold(DOI) | B | XGBoost | 0.549 | 23.52 | 40.14 |
| LOFO | A | XGBoost | 0.701 | 18.34 | 32.67 |
| LOFO | B | XGBoost | 0.681 | 19.09 | 33.76 |

## 4. DOI 단위 vs 바이오매스 단위 검증
- LODO·GroupKFold(DOI): 같은 논문 자료가 학습·검증에 섞이지 않음(누출 테스트 통과).
- LOFO: 정규화 원료명 단위 외삽. 미학습 원료의 조성 기반 신뢰도는 보장하지 않음.
- 세 분할 모두에서 결론 일관: 아래 §5.

## 5. 제조조건 추가의 성능 변화 (핵심, 정직한 보고)
**제조조건을 추가한 Model B 가 흡착조건만의 Model A 보다 반드시 개선되지 않는다.** 오히려
세 분할 모두에서 A 가 B 와 같거나 더 낫다(예: LODO R² A 0.695 vs B 0.637/0.646). Model C
(실측 BET/TPV 추가)도 개선 없음. 원인 추정: 97행 소표본·제조조건 고결측(heating_rate 58%,
activation 49%)·원료의 논문 종속성. 결과를 숨기지 않고 그대로 보고하며, **현재 데이터로는
'제조조건 직접 예측 모델이 흡착조건 모델보다 우수하다'고 주장할 수 없다.**

## 6. Chen16 제조조건 추천 성능 (조건 홀드아웃 LOO, 옥수숫대 Cd(II))
in-paper 조건 보간 검증(새 원료 일반화 아님). 16개 전체 학습 후 자기 순위 맞춤이 아님.

| 알고리즘 | 제거율 MAE(%p) | Spearman | Top-1 | Top-3 겹침 | regret(%p) |
|---|---|---|---|---|---|
| Ridge | 11.67 | 0.903 | 미적중 | 3/3 | 4.42 |
| ExtraTrees | 17.39 | 0.824 | 미적중 | 3/3 | 4.42 |
| XGBoost | 11.66 | 0.659 | 미적중 | 0/3 | 25.81 |
| RandomForest | 16.92 | 0.662 | 미적중 | 0/3 | 36.07 |

해석: Ridge/ExtraTrees는 상위권 조건 선별(Top-3 3/3, regret 4.4%p, Spearman 0.82~0.90)에
유효하나, **단일 최적조건(Top-1) 정확 지목은 실패**. 온도가 지배적 요인(400→700°C 제거율
26→96%)이라 순위 경향은 재현되지만 미세 최적은 불확실. 추천 기능은 '유망 조건 shortlist
보조'로만 타당하며 전역 최적 정량 추천으로 제시하지 않는다.

## 7. 물리 제약 위반 현황
0 ≤ qe ≤ C0/dose. 원시 예측의 위반율은 분할·모델에 따라 16~63%로 높다(소표본·외삽). 앱·
평가에서 **원시 예측과 제약 보정 예측을 구분**해 보고하며, clipping으로 오류를 은폐하지 않는다.
학습 데이터 자체는 위반 0건.

## 8. 미학습 바이오매스 처리
자연어 입력을 정규화해 학습 원료와 대조한다. 정확 일치→사용, 부분 일치→후보만 제시(자동
대체 안 함), 불일치→**적용범위 밖(OOD) 경고**. 이름에서 조성을 생성하지 않는다. 성능 검증
없이 임의 바이오매스의 최적조건을 정량 추천하는 기능은 넣지 않았다.

## 9. 기존 Stage 1 기능 영향
없음. Stage1 v60·v41 모듈·모델·데이터 미변경. 전체 테스트 420 통과/1 skip(환경 게이트)로
회귀 없음 확인. Stage1→Stage2(Model D) 예측 물성 전달은 **비활성** 상태.

## 10. 테스트 통과/실패/미실행
- 신규 `test_biochar_stage2_v15.py` 16개 모두 통과(지시 §8 의 12항목 + 보강).
- 전체 스위트 **420 passed, 1 skipped**(skip = v60 env-호환 재현, 현재 venv 비호환이라 설계상 skip).
- CatBoost 비교: 환경 미설치로 **미실행**(지시: 환경 허용 시에만). 설치 시 알고리즘 레지스트리에
  자동 포함되도록 구현됨.

## 11. Stage 1–2 최종 통합(End-to-End) 전 해결 과제
1. Stage1 v60 라이브러리 호환성(sklearn 1.8/xgboost 3.4 필요)과 독립검증 음수 R² 한계.
2. Model D 활성화 시 Stage1 예측 BET/TPV의 **오차 전파**와 **학습 DOI 중복 누출**(fold별 Stage1
   재학습·OOF 필요, v60 고정 artifact를 통합 holdout에 그대로 쓰면 안 됨).
3. 제조조건 추가 모델(B)이 A를 넘지 못하는 현재 한계 → 데이터 확충·제조조건 결측 보완 필요.
4. 엄격한 평형 qe 승인(현재 0행)·원료 조성 basis 감사.

**현 시점에서 Stage 1–2 End-to-End 모델은 완성으로 보고하지 않는다.** 데이터 로더와 Model
A/B/C의 검증 가능한 학습·평가 파이프라인, Chen16 추천 검증, Stage2 UI 1차를 완료했고, 그
결과(특히 B가 A를 넘지 못함, 추천은 shortlist 수준)를 근거로 추천 기능 범위를 '탐색적 보조'로
한정했다.


---

# Stage 2 v16 — 성능 독립 재검증 및 Baseline Benchmark (2026 추가)

v15 모델·데이터·UI 를 보존한 채 예측 성능을 독립 재검증했다. 새 배포 모델을 추가하지 않고
Stage1–2 통합도 진행하지 않았다. 신규 모듈: `pyrolysis/biochar_stage2/v16_validate.py`,
`v16_report.py`. 산출물: `reports/stage2_v16/`(model_comparison.csv, per_doi_eval.csv,
ablation.csv, fraction_experiment.csv, nested_cv.csv, bootstrap_ci.csv, fig_*.png, summary.json).
테스트: `tests/test_biochar_stage2_v16.py`. 기존 Stage1/v15 모델은 수정하지 않았다.

## 1. 데이터·누출 감사 결과
- DOI 25개, 행/ DOI 평균 3.88(최대 13, 1행 DOI 5개). 오염물질 Cd 41·Cu 22·Pb 13·Ni 11·Phenol 10.
- 타깃 `qe_reported_mg_g` 전부 `REPORTED_EXPERIMENTAL_UPTAKE_EQUILIBRIUM_NOT_CERTIFIED`,
  `CORE_EXPLORATORY`. 다수 행에 `UNVERIFIED_CONDITIONS`·`TEMPERATURE_NR`·`CONTACT_TIME_NR`
  경고. **엄격한 평형 승인이 아니다.**
- 전처리(결측 대체·인코딩·스케일링)는 모두 Pipeline 내부에서 fold 별 fit — 검증 데이터 누출 없음.
- A/B/C 는 동일 97행·동일 그룹 fold 사용(행 필터링 없음) — 공정 비교 확인.
- DOI/원료명/식별자는 모델 입력에서 제외(FORBIDDEN). 다만 **집단평균이 분산의 57%(DOI),
  57%(원료), 38%(route)를 설명** → 이들은 강한 신호이나 입력 금지(누출 방지). 원료는 DOI 에
  완전히 내포(원료 30종 모두 단일 DOI) → LOFO ≈ LODO.
- **핵심 구조적 사실**: `qe/(C0/dose)`(물질수지 상 제거율)의 중앙값 0.70, 범위 0.037~1.0.
  즉 qe 는 C0/dose 스케일에 강하게 지배된다. 이는 R² 해석에 결정적(아래 5).

## 2. Baseline Benchmark (동일 fold)
LODO 기준 OOF:

| 추정기 | R² | MAE(mg/g) | RMSE |
|---|---|---|---|
| Dummy(평균) | -0.08 | 42.85 | 62.0 |
| **물리 baseline** qe=(C0/dose)×평균흡착률 | **0.19** | **26.56** | 53.7 |
| A: Ridge | 0.57 | 26.10 | 39.2 |
| A: RandomForest | 0.38 | 31.45 | 47.1 |
| **A: XGBoost** | **0.689** | **18.56** | 33.3 |
| B: Ridge | 0.637 | 22.65 | 36.0 |
| B: XGBoost | 0.635 | 20.57 | 36.1 |
| B: RandomForest | 0.20 | 37.04 | 53.6 |

Chen16 조건 홀드아웃(LOO): Dummy Spearman **-1.0**/MAE 23.5%p/regret 68.9; Ridge Spearman
**0.903**/MAE 11.7%p/Top3 3/3/regret 4.42; ExtraTrees 0.824/11.7.../3-3/4.42. 모델이 단순
baseline 을 크게 능가.

## 3. 평가 신뢰성 — Nested CV + Cluster Bootstrap
- **Nested CV**(내부 DOI 그룹 CV 로 알고리즘 선택, 외부 LODO 평가):
  - **A nested R²=0.675, MAE=18.64**(선택 알고리즘: XGBoost 24/25, Ridge 1).
  - B nested R²=0.570, MAE=23.63.
  - → **A 의 R²≈0.69 는 알고리즘 cherry-pick 때문이 아니다.** 모델 선택을 포함해도 0.675 유지.
- **A vs B DOI cluster bootstrap**(MAE_B − MAE_A): 점추정 **+4.99 mg/g, 95% CI [0.58, 10.64]**
  → 0 을 포함하지 않음 → **B 가 A 보다 통계적으로 유의하게 나쁨**(개선 아님, 오히려 악화).

## 4. 제조조건 기여도 Ablation (동일 fold, Ridge)
A_only R²0.57(MAE26.1) → A 에 변수 하나 추가 시 MAE(낮을수록 좋음):

| 추가 변수 | MAE | R² | 결측% | 사용표본 |
|---|---|---|---|---|
| manufacturing_route | 22.45 | 0.667 | 0 | 97 |
| feedstock_label_normalized | 24.08 | 0.608 | 0 | 97 |
| process_step_count | 23.14 | 0.597 | 0 | 97 |
| pyrolysis_temperature_C | 25.63 | 0.604 | 19.6 | 78 |
| activation_or_modification | 25.39 | 0.583 | 49.5 | 49 |
| pyrolysis_atmosphere | 27.63 | 0.358 | 33.0 | 65 |
| heating_rate_C_min | 30.13 | 0.425 | 57.7 | 41 |
| pyrolysis_hold_min | 31.50 | 0.204 | 19.6 | 78 |
| secondary_temperature_C | 27.75 | 0.489 | 71.1 | 28 |

**결측 효과 vs 실제 기여 분리**: 결측 0% 범주형(route·feedstock·step_count)만 A 를 개선한다.
결측률 높은 수치 공정변수(hold_min 19.6%, heating_rate 57.7%, secondary 71.1%)는 추가 시
오히려 악화 → **B 가 A 를 못 넘는 주원인은 제조조건의 높은 결측·이질성**이지, 제조조건이
본질적으로 무용하다는 뜻은 아니다. (route/feedstock 은 사실상 DOI 대리변수일 가능성도 있음 —
원료가 DOI 에 내포되므로.)

## 5. 물리 제약 실험 (qe 직접 vs 무차원 흡착률 f=qe·dose/C0)
LODO, Ridge:

| 경로 | R²(qe) | MAE | 물리위반율 |
|---|---|---|---|
| qe 직접 예측 | 0.57 | 26.10 | **0.299** |
| f 예측(0≤f≤1)→qe 복원 | 0.19 | 27.44 | **0.0** |

**중대한 해석**: qe 직접 예측은 원시 예측의 **30% 가 0≤qe≤C0/dose 를 위반**(clipping 없으면
비물리적). f 경로는 위반 0 을 보장하지만 R² 가 0.57→0.19 로 급락한다. 이는 **qe 의 높은 R² 상당
부분이 C0/dose 스케일에서 오고, 흡착률 자체(f)의 예측력은 훨씬 낮음**을 보여준다. 보고 시
원시·clipping·f-경로를 구분한다.

## 6. 최종 질문 답변
1. **A 의 R²≈0.695 는 모델 선택을 고려해도 유지되는가?** → **예(≈0.675).** Nested CV 에서
   내부 선택이 XGBoost 를 24/25 fold 에서 고르며 외부 OOF R²=0.675. cherry-pick 아님.
2. **A 는 단순/물리 baseline 보다 우수한가?** → **예.** A(XGB) R²0.689/MAE18.6 vs 물리 0.19/26.6
   vs Dummy −0.08/42.9. 다만 물리 baseline 도 MAE 26.6 로 상당 — qe R² 의 큰 부분은 C0/dose
   스케일 기여(5번). 흡착률 f 로 보면 예측력은 R²0.19 수준으로 제한적.
3. **제조조건 추가(B)가 개선 못한 주원인?** → 제조조건의 **높은 결측·이질성**. 결측 0% 변수만
   도움, 고결측 수치변수는 악화. bootstrap 상 B 가 A 보다 유의하게 나쁨. 또한 원료가 DOI 에
   내포되어 LODO 에서 제조/원료 정보가 held-out DOI 로 일반화되기 어렵다.
4. **Chen16 추천은 단순 방법보다 나은가?** → **예(크게).** Ridge Spearman 0.90·regret 4.4%p vs
   Dummy Spearman −1.0·regret 68.9%p. 단, in-paper 조건 보간이며 Top-1 단일 최적 지목은 실패,
   새 원료 일반화는 미검증.
5. **신뢰 가능한 예측·추천 범위?** → (a) 흡착조건+오염물질 기반 qe 의 **스케일·대략 수준** 예측
   (R²는 스케일 기여 큼, 흡착률 자체 예측력은 제한적). (b) 학습 원료·동일 논문 조건 내에서의
   **제조조건 순위 선별(shortlist)**. 범위 밖: 미학습 원료 정량 예측, 단일 최적조건 확정,
   엄격 평형 qe 보장, 조성 기반 예측.

## 7. 발견된 데이터·정의 이슈(보정하지 않고 보고)
- qe 가 엄격 평형 미인증(reported uptake). 임의 보정하지 않음.
- qe 가 C0/dose 스케일에 지배됨 → R² 가 흡착화학 학습을 과대평가할 수 있음(f-실험으로 노출).
- 원료∈DOI 완전 내포 → LOFO 가 독립적 원료 외삽 검증이 되지 못함(DOI 검증과 사실상 동일).
- 원시 qe 예측의 30% 물리위반 → clipping 은 오류를 가릴 수 있어 위반율을 별도 보고.

## 8. 테스트
신규 `tests/test_biochar_stage2_v16.py`(물리 baseline 누출 없음, nested CV 선택 보고, bootstrap
CI 구조, ablation 결측 분리, A>baseline 등). 전체 회귀 스위트 실행 결과는 변경 이력에 기록.
