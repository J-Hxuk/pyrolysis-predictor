# v60.1 로컬 제조조건 순위 검증 기록

이 기록은 설치된 PC의 `per_panel_rank_results.csv`로 재계산한 값이다. 이전 제작 환경의 15개 유효 계열 통계와 혼용하지 않는다.

- 전체 제조 비교계열: 15개 / 9 DOI
- Spearman ρ가 정의되지 않는 비교계열은 0으로 대체하지 않고 CI에서만 제외한다.
- 전체 결과: `panel_rank_coverage.csv`
- DOI bootstrap: `paired_doi_bootstrap.csv`

- 188행 모델의 양측 Spearman 유효 계열: 14 / 15
- 양측 유효 DOI: 8 / 9
- 평가 제외 계열: 10.3390/su14159022:14
- 평균 ρ 차이(RF−B): -0.008710146064228985
- DOI bootstrap 95% 구간: [-0.22403536772259125, 0.1853116730221384]

## 188행 모델의 제외 원인
- DOI `10.3390/su14159022` / panel 14: CONSTANT_BASELINE_PREDICTIONS; RF rho=0.5, B rho=nan

이 결과는 사후 DOI 분할 평가이다. 데이터 추가·모델 선택에 이미 사용된 DOI를 미래의 독립 검증으로 간주하지 않는다.
