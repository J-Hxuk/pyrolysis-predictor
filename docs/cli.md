# CLI·노트북 실행 안내

이 문서는 명령행(CLI)과 노트북이 **같은 공통 함수**를 호출하여 같은 규격·결과를
내는 실행 경로를 정리한다. 노트북에 학습·전처리·평가 알고리즘을 복제하지 않는다.
실제 자료 확보 전에는 합성 설정(`configs/demo.yaml`)으로만 기능을 점검한다.

모든 명령은 가상환경의 Python 으로 실행한다:

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli <명령> [옵션]
```

## 표준 실행 순서

합성 설정 점검 기준의 권장 순서다. 실제 설정에서는 자료·평가 계획이 확정된 뒤
같은 순서를 따른다 (작업 25–30).

```
audit → train → report → export → make-template → check-bundle → prepare-runtime → activate
```

실제 설정 값이 없으면 각 명령은 중단하지 않고 **필요한 항목을 안내**한다. 미설정
한도·규격을 무제한·기본값으로 자동 대체하지 않는다.

## 명령 목록

### `audit` — 학습 파일 자료 점검 (작업 7)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli audit `
  --problem configs/demo.yaml `
  --dataset data/local/<파일>.xlsx `
  --id-columns source_id,sample_id,measurement_id `
  --sheet 0
```

- `--dataset` 가 없으면 안내 후 종료 코드 1.
- 대상별 가용 행·결측·수치 0·클래스 분포·반복 측정 후보를 보고한다.
- 결측 목표를 대체하거나 결측을 미검출로 바꾸지 않는다.

### `train` — 학습 실행

```powershell
# 합성 설정 점검
.\.venv\Scripts\python.exe -m pyrolysis.cli train --demo --problem configs/demo.yaml --output runs
```

- `--demo` 는 합성 설정으로 학습·평가 경로를 연결하고 `runs/<run_id>/demo_run.json` 에
  기록을 남긴다. 산출물에 "시험용 합성 자료" 표지가 붙는다.
- `--demo` 없이 실행하면 실제 대상별 학습은 작업 13(설정 확정 후)에서 연결된다는 안내가 나온다.
- 동일 합성 설정·난수이면 노트북 `02_train_evaluate.ipynb` 과 MAE·RMSE 가 일치한다.

### `report` — 로컬 연구 보고서 (작업 15)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli report --demo --output runs
```

- 후보 비교표, 예측·실측 비교 그림, 변수 기여도(명시적 `permutation` 방법)를
  `runs/<run_id>/report/` 에 저장한다.
- 서로 다른 단위의 오차를 하나로 합산하지 않는다 (대상별 분리 저장).
- 변수 기여도는 모델 관점의 상대적 영향이며 인과관계·반응 메커니즘이 아니라는
  주의 문구를 산출물에 남긴다.
- 노트북 `03_review_export.ipynb` 과 같은 함수(`analysis.save_*`)를 호출한다.

### `export` — 배포 묶음 내보내기 (작업 4/16)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli export --demo `
  --bundle-dir deployment/bundles --bundle-id demo_cli_export
```

- `--demo` 는 합성 묶음을 저장하고 다시 읽어 재현을 점검한다. 묶음에
  `demo-synthetic` 표지가 붙어 **연구용 활성화 경로가 이를 거부**한다.
- 실제 연구용 완전 묶음(manifest·SHA-256·환경 검사)은 평가 완료 대상이 필요하므로
  작업 28/29 에서 실제 설정으로 생성한다. `--demo` 없이 실행하면 그 안내가 나온다.

### `make-template` — 입력 양식 생성 (작업 8)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli make-template `
  --problem configs/demo.yaml --output templates/local --targets synthetic_y1
```

- 활성 대상의 입력 규격으로 `.xlsx`(inputs·_schema 시트)와 CSV(_input_schema_id 열)를
  만든다. 비활성 대상 입력과 목표 결과 열은 넣지 않는다.
- `--targets` 를 생략하면 전체 대상의 입력 합집합을 사용한다.

### `check-bundle` — 묶음·환경 무결성 검사 (작업 17)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli check-bundle --bundle deployment/bundles/<bundle_id>
```

- 파일 해시·상대 경로·누락과 저장 당시 라이브러리 버전(major.minor)을 검사한다.
- 환경 경고를 무시하고 통과시키지 않는다. 통과하지 못하면 종료 코드 1.

### `prepare-runtime` — Streamlit 설정 생성 (작업 17)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli prepare-runtime `
  --runtime configs/runtime.yaml --config .streamlit/config.toml
```

- `limits.max_upload_size_mib` 로 `server.maxUploadSize` 를 쓰고,
  `client.disableDataExport=true`, `server.enableStaticServing=false` 를 적용한다.
- 필수 운영 한도가 비어 있으면(`null`) 실패한다. 미설정 한도를 무제한으로 두지 않는다.
  실측 전 점검만 하려면 `--allow-missing-limits` 를 쓸 수 있다(연구 모드 비권장).

### `activate` — active 포인터 교체 (작업 17)

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli activate --bundle deployment/bundles/<bundle_id>
```

- 활성화 전 `check-bundle` 과 같은 검사를 수행하고, 실패하면 기존 포인터를 건드리지 않는다.
- `deployment/active.json` 을 임시 파일로 쓴 뒤 교체하여 불완전 활성화로 손상되지 않게 한다.
- 자동 GitHub push·배포는 하지 않는다.

## 노트북

| 노트북 | 호출하는 공통 함수 | CLI 대응 |
|---|---|---|
| `01_data_audit.ipynb` | `config.load_problem`, `io.read_training_file`, `io.audit_dataset` | `audit` |
| `02_train_evaluate.ipynb` | `config.load_problem`, `training.run_demo_training` | `train --demo` |
| `03_review_export.ipynb` | `analysis.save_*`, `artifacts.save_demo_bundle`/`load_bundle` | `report --demo`, `export --demo` |

- 커널은 `pyrolysis-research` 를 선택한다 (등록: `docs/setup.md` 4절).
- **커밋 전 Kernel → Restart & Clear Output** 으로 실제 셀 출력을 지운다.
- 노트북 실행만으로 GitHub push·배포는 발생하지 않는다.

## 오류 대응

| 증상 | 원인·대응 |
|---|---|
| `audit: --dataset 인자가 필요합니다.` | 학습 파일 경로를 지정한다. |
| `report`/`export` 가 "작업 28/29 에서..." 안내 | 실제 설정이 아직 없다. `--demo` 로 경로를 점검한다. |
| `check-bundle` 환경 불일치 | 저장 당시와 다른 라이브러리 버전. 고정 의존성으로 재설치한다. |
| `prepare-runtime` 한도 누락 실패 | `configs/runtime.yaml` 의 운영 한도를 실측으로 채운다(작업 29). |
| 모델 교체 | 새 묶음을 `export`→`check-bundle`→`activate` 순으로 바꾼다. 실패 시 이전 포인터가 유지된다. |

## 실제 자료·비밀값 비배포

- 실제 학습 자료(`data/`), 상세 실행 기록(`runs/`), 로컬 설정(`configs/local/`),
  비밀값(`.streamlit/secrets.toml`)은 Git 추적 대상이 아니다 (`.gitignore` 확인).
- 노트북에는 실제 셀 출력을 저장하지 않는다. 커밋 전 `git status` 로 확인한다.
