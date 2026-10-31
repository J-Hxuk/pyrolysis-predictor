# v51 추가 문헌 검토자료 인입 및 학습 적합성 판정

## 입력 파일
- 사용자가 전달한 `Biochar_Additional_Paper_Review_vNext10.zip` (CSV 2개, XLSX 1개, Markdown 1개).
- **논문 PDF 원문은 이 ZIP에 포함되어 있지 않음**. 데이터 판단은 동봉된 추출·검증 기록을 우선한다.
- 관련 공개 원문(대나무): https://doi.org/10.3390/toxics10120787

## 판정

| DOI | 확인된 관측 | 누락/이슈 | 승인된 Stage 2 신규 qe |
|---|---|---|---:|
| 10.3390/toxics10120787 | Cu 4개 본문 수치, Cd 그림 근삿값 4개 | dose·접촉/평형 시간 불명, Cd 본문/그림 불일치, BET·TPV BC500만 | **0** |
| 10.1016/j.biortech.2013.03.186 | 제조온도 300/400/500/600°C 및 초록 최대용량 | 온도별 qe/조건 미확보, 레거시 활성 Stage 1에 이전 노출 | **0** |
| 10.2166/wst.2020.309 | 제조온도 × 유지시간 조합, 초록 최대용량 | 시료별 qe/조건 미확보, 최대용량과 실측 qe는 다름 | **0** |

- 입력물 총량: **4개 파일 / 3편 DOI / 8개 흡착량 후보** (Cu4 + Cd4).
- 모두 **REVIEW**. `strict_qe` 승인 없음. Stage1 신규 승인 행 없음.
- v46 데이터 **Stage 1 417행, Stage 2 140행을 변경하지 않음**. 새 학습·모델 저장 없음.
- 현 통합 CSV에서 DOI 3편이 없더라도, 억새 DOI는 `deployment/stage1_models/.../metadata.json` 등 이전 활성 모델 이력에 존재해 신규 외부 검증으로 간주하지 않음.
- **BC500 BET 4.2422 m²/g / BJH total pore capacity 0.000693 cm³/g**는 source-staged 정보로만 저장; BJH pore capacity를 무비판적으로 기존 TPV 정의와 동일시하지 않음.

## 다음으로 실제 필요한 최소 자료
1. **대나무:** 원문 Supplementary Materials (Methods)의 무개질 BC400/500/600/700 테스트에 해당하는 **흡착제 dose, 접촉/평형 시간, Figure 1의 시료별 Ce 또는 qe 계산 근거** 확보. Cd 본문과 Figure 1 불일치 확인.
2. **억새:** 기존 이력의 수치 및 출처와 대조·원본 회수. 신규 독립 홀드아웃에는 불가.
3. **폐찻잎:** 온도별 실측 qe·C0·dose·pH·time을 포함하는 원문/보충자료. Langmuir qmax·kinetic fitted qe와 구분.

## 프로그램 상태
`data/imports/v51_newpaper_review/`에 **원본 4개 파일을 그대로** 보관. `scripts/audit_v51_newpapers.py`가 가정값 생성 없이 읽기 전용으로 승인·누락·DOI 상태를 재확인한다.

이 자료만으로 v50 모델을 다시 학습해 MAE를 비교하는 것은 의미가 없고, 신규 학습 0행이므로 기존 성능이 그대로다.
