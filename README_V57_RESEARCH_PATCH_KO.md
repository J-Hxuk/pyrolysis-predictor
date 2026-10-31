# Biochar v57 | Handoff CurrentBatch — 실제 누적 재학습 패치

2026-10-31, 연구 전용, **활성 모델/기존 데이터 무변경**.

새 `Biochar_Kiro_Handoff_CurrentBatch.zip`에 있던 Shen 2017 실험 그래프 29행 + Cao 2019 실험 Table2 7행을 모두 원문/CSV 보존하고, Stage2 A/B 모델에 누적·재학습했습니다. BET는 독립 시료 4개의 Stage1 연구용 학습자료로만 추가했습니다. TPV 0건. 추가 원문 4건의 수치화되지 않은 부분은 staging PDF로 보관했습니다.

**기본 참조:** `reports/v57_handoff_cumulative_refit/RESULTS_KO.md`를 먼저 읽어주세요. 36행 전체 누적 시 오차가 악화되므로 자동 활성화하지 않습니다. 41개 테스트 통과는 코드/평가 재현성 테스트이지 독립 실제 적용성 보증은 아닙니다.

## 기존 v47 프로젝트 폴더에서 적용

1. ZIP을 `Biochar_v47_Integrated_Project` 최상위에 넣고 압축 해제합니다. 
2. Python은 기존 `바탕 화면/대학/학술대회/프로그램/.venv/Scripts/python.exe` 재사용.
3. **이미 재학습된 Joblib 17개가 포함**되어 있으므로 기본적으로 재학습을 다시 할 필요가 없습니다. Windows의 scikit-learn 버전 차이를 해소하려면 다음 선택 명령을 실행하세요.

```powershell
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
& $python -m scripts.run_v57_handoff_cumulative_refit --root .
if ($LASTEXITCODE -ne 0) { throw 'v57 DOI 누적 재학습 실패' }
& $python -m scripts.run_v57_source_weight_sensitivity --root .
if ($LASTEXITCODE -ne 0) { throw 'v57 가중치 실험 실패' }
& $python -m pytest tests/test_v57_handoff_cumulative_refit.py -q --tb=short
```

*주의*: 재학습하면 환경별로 OOF 결과와 Joblib 해시가 갱신됩니다. `reports/v57_handoff_cumulative_refit/`에 로컬 결과를 보존하고, 과거 수치와 다르면 그 차이를 보고해야 합니다. 연구 결과를 검증 없이 발표 수치로 확정하거나, 새 모델을 자동 배포하지 않습니다.

## 구성

- `data/imports/v57_handoff_sources/`: 전수 36행 원본 CSV·원본 QC xlsx·원문 PDF·6 DOI Source registry·Stage1 450/Stage2 159/181/188/192/204/208 증량 데이터셋
- `scripts/run_v57_handoff_cumulative_refit.py`: DOI GroupKFold + 36행 누적 + OOF + sourceout + BET + 연구 모델 저장
- `scripts/run_v57_source_weight_sensitivity.py`: Shen DOI 다운웨이팅 실험
- `deployment/research_v57_NOT_ACTIVE/`: Stage2 A/B 12개, Shen DOI 가중치 A/B 4개, Stage1 BET 1개 Joblib, 메타데이터
- `reports/v57_handoff_cumulative_refit/`: 모든 수치, OOF 예측, 순위, paired bootstrap, 단계별 해석
- `tests/test_v57_handoff_cumulative_refit.py`: 검증 코드 8개

DOI별 신뢰성/결측·평형 증명 상태는 출력 특성으로 임의 전환하지 않았으며, qmax를 qe로 편입하지 않았습니다.
