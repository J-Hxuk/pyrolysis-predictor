# Biochar v51: 신규 논문 3편 인입 (원본 보존 / 학습 보류)

**이 패치는 기존 모델을 갱신하지 않습니다.** v50까지의 활성 모델·모델 평가 파일·기존 데이터는 수정하지 않습니다.

### 설치 (기존 v47 + v48~v50 설치 폴더)

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\Biochar_v47_Integrated_Project"
Expand-Archive -LiteralPath ".\Biochar_v51_NewPaper_Intake_SAFE_Patch.zip" -DestinationPath "." -Force
$python = "C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe"
& $python -m scripts.audit_v51_newpapers --root . --out reports/v51_newpaper_review/local_audit.json
& $python -m pytest tests/test_v51_newpapers.py -q
```

### 파일
- `data/imports/v51_newpaper_review/`: 업로드된 검토 ZIP의 4개 원본 파일 (그대로 저장)
- `reports/v51_newpaper_review/RESULTS_KO.md`: 각 논문별 학습 가능 여부 및 추가 확보 항목
- `scripts/audit_v51_newpapers.py`: 읽기 전용 중복·결측·승인상태 검사
- `tests/test_v51_newpapers.py`: 데이터 및 비승인 불변성 검사

### 향후 학습
실측 qe, 농도, 투입량, 접촉시간이 실험 조건별로 확인된 후 **새 버전에서 승인된 행만 통합해 동일 DOI 분할로 재학습**합니다. 현재 8개 REVIEW 후보는 절대 자동 병합하지 않습니다.
