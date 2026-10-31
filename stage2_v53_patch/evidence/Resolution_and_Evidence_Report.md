# 원문 보완 감사 — 대나무 SI 및 사과나무 가지 Cu/Zn 자료

작성일: 2026-10-31
범위: 사용자가 제공한 bamboo SI PDF 및 공개 원문 DOI 10.3390/en13133498. 기존 모델·승인 DB는 수정하지 않았으며 재학습도 하지 않음.

## 결과 요약

- **사과나무 가지:** 2020년 논문 Table 3에서 `qe,exp` 8개를 확인했다(BC300/400/500/600 × Cu/Zn). 각 값은 평균±표준편차, n=3으로 보고된다. 본문은 1440분에 흡착 포화에 도달했다고 설명한다. 따라서 8개는 원문 실험 평형 qe로 승인 가능하다.
- **흡착 조건:** Table 2/Methods 2.2.2 및 Figure 5에서 Cu C0=64 mg/L, Zn C0=65 mg/L, pH 5, dose 5 g/L, 25°C, 1440분(24 h)을 대응시킬 수 있다. qexp 자체는 표의 실험값이다. Ce는 해당 관측별로 직접 표에 없으므로 NR이며 계산치로 대체하지 않았다.
- **물성 연결:** 2020년 논문 Table 1은 BET를 보고하고 TPV는 보고하지 않는다. 논문은 해당 biochar 특성값을 Reference 27로 돌리며, Ref. 27은 DOI 10.3390/en10091293 (2017)이다. 그 논문 Stage1 staging에는 TPV가 있다. 다만 BC500 BET가 2020 값 37.25와 2017 값 37.24 m²/g으로 0.01 차이가 나므로 cross-source property linkage는 명시적 인용 기반의 연결 후보로 기록하고 최종 배치 동일성은 확인 필요로 남겼다.
- **학습/검증:** 2020 DOI는 프로젝트 파일에서 검색되지 않았지만, 인용된 2017 DOI는 여러 Stage1 학습·평가 산출물에 노출되어 있다. 그러므로 이 8행 또는 두 제조조건 시리즈는 독립 외부 holdout으로 선언할 수 없다. Stage2 목표 스키마의 training candidate로는 보관하되 TPV/동일 배치 확인 전 실제 학습 적격 확정은 보류한다.
- **비교계열:** 동일 DOI 안에서 Cu 4조건, Zn 4조건의 두 내부 비교계열이 있다. 동일 C0, pH, dose, 온도, 접촉시간을 적용하고 qe,exp를 비교하므로 제조조건 순위 비교 후보로 적합하다. 다만 이는 독립 외부 검증이 아니라 개발용 내부 비교다.
- **대나무:** SI는 원료 온도계열에 대해 C0 50 mg/L, 용액 120 mL, pH 6, 25°C, 150 rpm, 24 h를 복구한다. 그러나 흡착제 질량/dose 값은 기재되지 않고, SI의 식은 타깃을 q_t로 정의한다. 24 h를 평형으로 증명하는 plateau 근거도 해당 원료 온도계열에는 없다. 따라서 이 값들은 strict-qe 또는 Stage2 학습 데이터로 승격하지 않는다. 수정 KAM 시리즈의 투입량을 원료 바이오차 시리즈에 옮기지 않았다.

## 정량 변경

| 항목 | 이번 보완 | 판정 |
|---|---:|---|
| 신규 원문에서 직접 확인한 실험 qe | 8 | 사과나무 가지 2020 Table 3, strict qe source-verified |
| 신규 완전 Stage2 학습 확정 | 0 | TPV의 cross-source 동일 배치 검토 필요 |
| 제조조건 비교 후보 | 2 series | Cu 4개 조건, Zn 4개 조건; 같은 DOI 내부 비교 |
| 독립 외부 holdout | 0 | 2017 참조 DOI의 Stage1 노출 확인 |
| 대나무 strict-qe 승인 | 0 | dose 및 평형 증거 없음; 24h 값은 q_t로만 확인 |
| 신규 Stage1 승인 소재 | 0 | 2020 Table1 조성은 열분해 후 biochar 값이며 raw feedstock 입력이 아님; recipe 항목 일부 NR |

## 추출 및 원문 대조 상세

### 사과나무 가지 논문

논문: Zhao, Ta & Wang, “Absorption of Cu(II) and Zn(II) from Aqueous Solutions onto Biochars Derived from Apple Tree Branches,” *Energies* 13 (2020), 3498. [DOI](https://doi.org/10.3390/en13133498) · [공개 원문 PDF](https://psecommunity.org/wp-content/plugins/wpor/includes/file/2303/LAPSE-2023.24104-1v1.pdf)

`qe,exp` (mg/g): Cu—BC300 4.96±0.18, BC400 7.56±0.27, BC500 9.70±0.33, BC600 8.87±0.42; Zn—BC300 4.30±0.10, BC400 5.69±0.13, BC500 9.11±0.28, BC600 7.99±0.22. Table 3의 `qe,exp` 행만 가져왔으며 PF/PS kinetic fitted qe와 Table 4 Langmuir Qmax는 추출 타깃에서 제외했다.

원문은 q_t=(C0−Ct)V/m로 정의하고, t가 평형 접촉시간과 같을 때 Ct=Ce 및 qt=qe라고 명시한다. Kinetic results는 1440분에서 saturation에 도달했다고 기술한다. 즉 표의 qe,exp는 fitted qe가 아니라 실험값으로 처리했다. 개별 Ce는 표에 없으므로 NR로 둔다.

### 대나무 논문 SI

제공 SI `toxics-2033459-supplementary.pdf`에서 온도별 원료 바이오차의 흡착 실험 항목을 확인했다. SI Eq. (4)는 q_t를 시각 t에서의 흡착량으로 정의한다. 24 h는 contact time이지, 그 자체로 equilibrium 증명은 아니다. mass가 “fixed”라고만 되어 있고 값이 없어 dose 계산이 불가능하다.

## 원본/프로젝트 상태

- 기존 Core97 및 승인 DB 원본: 변경 없음.
- v50/활성 모델: 변경·재학습 없음.
- 2020 사과나무 논문 DOI: 추출된 프로젝트 트리의 CSV/MD/JSON/XLSX에서 DOI 문자열 미발견. 따라서 현재 프로젝트 DB의 Stage2 학습 반영 기록은 찾지 못함.
- 2017 사과나무 가지 DOI `10.3390/en10091293`: `Stage1_Master_381_RESEARCH_STAGING.csv` 및 여러 Stage1 OOF/report 파일에 존재. `DOI_Exposure_and_Leakage_Audit.csv`에서 historically exposed로 분류되어 외부 holdout 금지.

## 다음 필요한 확인

1. 2017 Table 2의 TPV가 2020년 adsorption에 쓰인 BC300–600과 완전히 동일한 제조 배치인지 저자/원문으로 확인. 2020 본문이 Ref. 27을 물성 출처로 직접 인용하지만, 배치 ID는 명시하지 않는다.
2. 2020 paper의 sample preparation 세부를 Ref. 27에서 보충할 수 있는 항목(holding time, heating rate, atmosphere)은 재사용 관계 확인 후에만 연결한다. 원료 C/H/O/N/S, VM/Ash/FC는 열분해 전 시료 측정치가 직접 확보되지 않으면 NR 유지.
3. TPV와 동일 물리 시료 link가 확인되면 이 8행을 신규 6-input Stage2 training candidate로 등록할 수 있다. 독립 holdout은 계속 불가.
