# Biochar v48 릴리스 노트

- UI: v20 BET / v46 연구용 BET·TPV 병렬 탭, 동일 제조조건 그래프, 원문 실측과 ML 예측 구분, 결과 CSV.
- Navigaton: v20 제조조건을 흡착성능 화면으로 직접 이동 (Streamlit widget callback 처리).
- Safety: v46 모델을 비활성 연구용으로 명시하고 Stage2 A/B와 직접 연결하지 않음. 원문에 없는 제조온도 및 활성화제 비율을 임의 생성하지 않음.
- Installation: Windows PowerShell 실행 정책과 관계없는 CMD 실행 래퍼 추가.
- Model registry: v47 활성 모델 JSON, 저장된 모델 학습계수, 모든 원본 CSV는 변경하지 않음.
- Quality: 기존 v47 체크섬 `V47_MANIFEST.json`은 역사적 스냅샷. **v48에서는 `scripts.verify_v48` 및 `V48_MANIFEST.json`을 사용합니다.** `CHECK_V47.ps1 --full-checksums`는 코드 변경으로 인해 불일치할 수 있음.
- Caveats: v46 모델 pickle이 사용자 가상환경에서 호환되지 않으면 해당 연구용 탭만 오류. v20 화면 및 Stage2 활성 모델은 별개.
