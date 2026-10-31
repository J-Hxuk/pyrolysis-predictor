# Stage2 v53 — v47 폴더 직접 설치 및 재학습 준비

**대상:** 이미 로컬에 압축 해제되어 있는 v47 통합 프로젝트 폴더. 

## 역할과 상태
- 사용자는 압축을 **v47 폴더 안에 푸는 작업** 및 필요시 v47 프로젝트 통합본 업로드만 수행한다.
- 데이터셋 통합, 중복/물성/평형 판정, Stage2 모델 학습과 DOI OOF 성능검증, 모델 비교, 후속 배포 패치 제작은 ChatGPT가 담당한다.
- 기존 v52 실제 학습 CSV/모델파일은 이 패치에 포함되지 않았다. 따라서 패치 설치나 내부 source audit만으로 실측 재학습 완료라고 주장하지 않는다.
- 기존 Core97/승인 DB, v50/활성 모델, v52 실험 산출물을 임의 변경하지 않는다.
- 이전 `KIRO_TASK_v53.md`의 Kiro 위임은 이 수정 패키지에서는 사용하지 않는다.

## Windows PowerShell — 이미 v47 디렉터리 내에서 실행
```powershell
$ErrorActionPreference = 'Stop'
if ((Split-Path (Get-Location).Path -Leaf) -notmatch '(?i)v47') { throw '먼저 v47 프로젝트 폴더로 이동하세요.' }
$zip = Join-Path $env:USERPROFILE 'Downloads\Stage2_v53_for_v47_20261031.zip'
if (-not (Test-Path -LiteralPath $zip)) { throw "다운로드 파일을 찾지 못함: $zip" }
if (Test-Path -LiteralPath '.\stage2_v53_patch') { throw '이미 stage2_v53_patch가 있습니다. 중복 덮어쓰기 중단.' }
Expand-Archive -LiteralPath $zip -DestinationPath '.'
Get-ChildItem '.\stage2_v53_patch'
```

## 선행 검증(선택)
```powershell
.\.venv\Scripts\python.exe -m pytest stage2_v53_patch/tests/test_v53_patch.py -q --tb=short
.\.venv\Scripts\python.exe stage2_v53_patch/scripts/v53_stage2_patch.py audit --output stage2_v53_patch/reports/source_audit_local.json
```

## 실제 연구 워크플로
1. v47 통합본·최신 v52 CSV/모델/설정/OOF를 원본 그대로 받아 실제 버전과 해시/파일계보 확인.
2. 2020 Zhao 논문의 24행 가운데 5 g/L 실험 qe 8행은 vNext10와 중복되는지 1:1 대조. 대나무 SI는 strict qe 불승인.
3. 결측 입력이 포함된 8행을 무리하게 Model B/C에 넣지 않고 검증 가능한 입력만 사용. 2017/2020 공유 시료 교차 논문 누수 제거.
4. 동일 테스트 DOI 및 같은 전처리/모델·피처로 140/140+8/140+24(민감도) 비교. ROC 등 무관한 지표 사용하지 않음.
5. MAE·RMSE·R²·DOI별 성능·물리 위반율·부트스트랩 CI 보고. 개선이 재현되면 별도 배포 후보 저장. 활성 모델 자동 승격 금지.

## 주의
본 패치의 단독 코드가 사용하는 예시 추정기(Mean/RandomForest/ExtraTrees)와 실제 v52 학습 모델이 다를 수 있다. 정확한 모델을 확인해 동일 조건으로 비교하지 않으면 과거 보고된 MAE와 직접 비교하지 않는다.
