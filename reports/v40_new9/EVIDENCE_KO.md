# v40: 신규 9편 원문 수치·데이터 분리 감사

**분석 대상:** 사용자가 제공한 `Biochar_9_New_DOIs_Main.zip`의 PDF 9편. 원문 PDF는 이 패치 ZIP에 중복 포함하지 않으며 각 원본의 SHA-256은 `source_pdf_sha256_manifest.json` 참조. 원문 페이지는 PDF 상 표기 페이지 기준. 모든 후보는 연구용 staging이며 자동 배포 승인 상태가 아니다.

| DOI | 근거 위치 | 확정 입력·측정 / 처리 |
|---|---|---|
| `10.3390/ma17123030` Jiang (2024) | Table 1 p5, Table 2 p7, Methods p3 | coconut shell, rice husk, cow dung 각각 300–800°C 6점, 총 18시료. 원료 조성은 **동일 DOI 원료**의 실제 수치. 제조 0 min 유지(피크 도달 즉시 종료), 승온 10°C/min, N2 500 mL/min. BET 16개; 우분 300/400°C는 결측(`--`), 0이 아님. TPV 표 수치 없음. |
| `10.3390/app11199223` Araya (2021) | Methods pp3–4, Table 2 p6, Table 4 p11, Table 5 p13 | kelp biochar 300/450/600°C BET=2.0/4.0/4.0 m²/g, TPV=0.009211/0.012871/0.018898 cm³/g. 125 mg/L Cu, dose 1 g/L, 23±2°C, 24h, **실험 qe(exp)** =71.91/119.9/101.9 mg/g. 원문에 초기 pH 미보고. N2 4 L/min은 *냉각 과정*의 공급량으로 기재되어 있으므로 제조 유량 변수에는 넣지 않음. Langmuir qm=49.24/58.00/93.55 mg/g은 별도 테이블에 보존. 특히 450°C kinetic 실측 qe(119.9)가 isotherm fitted qm(58.00)보다 커서 원문의 측정조건·피팅 적합성 재검토 필요. |
| `10.1016/j.biteb.2021.100920` Sakhiya (2022) | Methods p2, Table 1 p2, Table 3 p4, Figure 1 p5 | rice straw 300/400/500°C, BET 12.78/26.52/64.56, TPV 0.0087/0.0142/0.0303. 원료 VM/FC/Ash 실측 있음. 그러나 초기온도 비교 흡착실험에 `0.1 g + 5 mL × 100 mg/L`로 적혀 있어 **상한은 5 mg/g**인데 Pb=17.93, Zn=25.73 mg/g을 보고. 다른 실험에는 50 mL 사용. 5/50 mL 원문 충돌 해결 전 qe 전부 학습 차단. 원료 O=56.39%는 C/H/N 외 차감 방식으로 ash basis 일관성 미확인, 조성모델 O 입력 차단. |
| `10.1016/j.egyr.2021.07.032` Kumabe (2021) | Methods p2, Table 1 p4, Table 6 p6 | coffee grounds char 무활성/steam 활성 두 시료. 제조 450°C 60min, 9°C/min N2 500mL/min. Steam activation은 800°C 30min으로 별도 이력. 무활성 BET=`ND`로 null, TPV=0.482; steam BET=445, TPV=0.266. **낮은 BET에서 높은 TPV가 물성 측정상 가능한지 재검토 필요**, ND를 0으로 바꾸지 않음. 원료 proximate a.b., ultimate d.a.f., O+S 합계만 보고되어 분리 O값 금지. |
| `10.1039/C9RA08199B` Yan (2019) | Methods p2, Table 1 p2, Table 2 pp4–5, abstract p1 | chicken manure 200/400/600/800°C 4시료 제조 ID·조성(바이오차 제조 **후**) 식별. BET/TPV는 본문 수치 미확인으로 null. qmax 180.21/200.80/239.59/242.57 mg/g, **Langmuir 피팅값만 별도 보관**. Abstract 마지막 시료를 CMB200으로 잘못 쓰고 있으므로 원본 Table 2 및 시료명을 확인. |
| `10.1021/acsomega.1c04717` Tahir (2021) | Table 1 p2 | soybean straw 원료 VM=73.61%, FC=14.59%, ash=5.03%, moisture=6.77%; C=53.68%, H=5.38%, O=37.17%, N=2.59%, S=1.17%. **다른 DOI 대두짚의 직접 조성 입력으로 사용하지 않음.** |
| `10.3390/en13226033` Gao (2020) | Table 1 p3 | rice husk proximate **as-received**와 ultimate **daf** 조성. 같은 원료명이라도 다른 DOI의 시료에 병합 금지. |
| `10.3390/ijerph16050845` Wang (2019) | 원문 비교 | **v33에 이미 존재**. 중복 배제. |
| `10.3390/su14159022` Hu (2022) | 원문 비교 | **v33에 이미 존재**. 중복 배제. |

### 데이터 승인 정책

1. Stage 1에 신규 30개 제조시료를 **연구용 후보**로 병합하되, 실측 BET 23개, TPV 8개만 수치형 타깃으로 둔다.
2. Stage 2 실험 qe는 kelp 3개만 추가한다. 초기 pH 미보고와 qmax와의 불일치를 `RESEARCH_CONDITIONAL_NOT_CERTIFIED`로 기록한다. 이들 값은 최종 배포·공정 추천 근거로 확정하지 않는다.
3. 닭분뇨·켈프의 qmax 7개와 왕겨가 아닌 *rice straw* 논문의 물질수지 불일치 Pb/Zn 2개는 별도 CSV `separate_target_and_blocked_9.csv`에 보존한다. 데이터는 버린 게 아니다.
4. 원료 조성 문헌 2편은 독립 참고자료이며 원료 이름만으로 서로 다른 DOI의 제조·흡착 시료에 조성값을 복사하는 행위를 금지한다.
5. 전부 연구용 스테이징. DOI 단위 분할/새 출처 도메인 검증/불확실성 표시 후 배포를 결정한다.
