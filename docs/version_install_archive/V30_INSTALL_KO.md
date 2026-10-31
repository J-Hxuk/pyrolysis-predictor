# Biochar v30 설치 및 확인

이 패치는 **모델을 학습하거나 Streamlit UI를 변경하지 않습니다.** v29가 적용된 기존 프로젝트 위에 연구용 CSV/코드/테스트를 추가합니다.

## 설치

1. `Biochar_v30_SourceGraph_ExperimentalQE_Research_Patch.zip`을 `프로그램` 폴더 안에 저장합니다.
2. Windows PowerShell에서 다음을 실행합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

$zip = ".\Biochar_v30_SourceGraph_ExperimentalQE_Research_Patch.zip"
if (-not (Test-Path $zip)) { throw "ZIP 파일을 먼저 프로그램 폴더에 저장하세요: $zip" }

Expand-Archive -LiteralPath $zip -DestinationPath "." -Force

.\.venv\Scripts\python.exe -m pytest tests\test_v30_source_graph_evidence.py -q
```

## 출처연결 데이터 재생성 (선택)

`data/imports/v24_source_audit/Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv`가 있으면 왕겨 제조시료 9개와 흡착 관측 18개를 DOI+시료명으로 정확히 연결한 CSV도 재생성합니다.

```powershell
.\.venv\Scripts\python.exe scripts\build_v30_source_graph_evidence.py
```

## 결과 확인

`reports\v30_graph_evidence\RESULTS_KO.md`

**주의:** `APPROX_NOT_TRAINED`와 `DOSAGE_CONFLICT` 파일은 기존 학습 파이프라인에 넣으면 안 됩니다. v30의 `training_approved` 열은 모두 `False`입니다. 실측 그래프점, 피팅 qmax, 계산 추정치의 역할을 혼합하지 마세요.

현재 배포 모델, 데이터, `.venv`, `active.json`, UI는 전혀 수정하지 않습니다.
