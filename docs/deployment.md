# 첫 공개 배포 안내 (작업 31–32)

이 문서는 실제 모델이 준비된 뒤(작업 28–30 완료) 비공개 GitHub 저장소를 구성하고
Streamlit Community Cloud 로 공개 배포하는 절차와 인수 점검 항목을 정리한다.

> **선행 조건.** 아래 절차는 작업 30(실제 묶음 로컬 수용 점검·배포 준비 완료)이
> 끝난 뒤에 실행한다. 이 문서는 안내서로서 미리 작성되었고, 실제 저장소·배포
> 작업과 체크 표기는 **운영자의 계정 작업**과 **실제 활성 대상 확정(작업 28)** 이후에
> 수행한다. 안내서 작성만으로 작업 31·32 를 완료로 표시하지 않는다.

담당 구분:
- **Kiro**: 코드·명세·고정 의존성·설정 원형·안내서 준비, 추적/제외 대상 점검 명령 제시.
- **운영자**: GitHub 비공개 저장소 생성·push, Streamlit Cloud 연결·공개 전환.
- **연구 담당자**: 공개할 실제 정의·지원 범위·설명 확인(작업 30).

---

## A. 배포에 포함·제외할 것

### 포함 (Git 추적)

- 애플리케이션 코드: `streamlit_app.py`, `pyrolysis/` 패키지.
- 명세·문서: `.kiro/specs/`, `docs/`, `README.md`.
- 고정 의존성: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`.
- 설정 원형: `configs/*.template.yaml`, `configs/runtime.yaml`, `.streamlit/config.toml`.
- 배포 묶음: `deployment/bundles/<bundle_id>/`, 포인터 `deployment/active.json`.
- 입력 양식 안내 문서(실제 양식 파일 `templates/local/` 은 제외).

### 제외 (`.gitignore` 로 이미 처리됨 — 커밋 전 재확인)

- 실제 학습 자료: `data/`
- 상세 실행 기록: `runs/`
- 로컬 전용 설정(실제 경로·변수명 포함): `configs/local/`
- 실제 입력 양식: `templates/local/`
- 비밀값: `.streamlit/secrets.toml`, `.env`, `secrets.*`
- 노트북 실제 셀 출력(커밋 전 Kernel → Restart & Clear Output)

---

## B. 작업 31 — 비공개 GitHub 저장소와 배포 변경 준비

### B-1. 추적/제외 대상 점검

커밋 전 아래 명령으로 실제 자료·비밀값·노트북 출력이 스테이징에 없는지 확인한다.

```powershell
# 스테이징 예정 파일 목록
git status
git add <필요한 파일만 지정>   # git add -A 대신 명시적으로 추가
git diff --cached --name-only

# 금지 경로가 포함되지 않았는지 확인 (출력이 없어야 정상)
git diff --cached --name-only | Select-String -Pattern '^data/|^runs/|^configs/local/|^templates/local/|secrets'

# 노트북 셀 출력이 비어 있는지 확인
Select-String -Path notebooks\*.ipynb -Pattern '"output_type"' -Quiet
```

- 마지막 명령이 `True` 면 노트북에 출력이 남아 있는 것이다. Kernel → Restart &
  Clear Output 후 다시 저장한다.

### B-2. 비공개 저장소 구성 (운영자)

1. GitHub 에서 **비공개(private)** 저장소를 새로 만든다.
2. 로컬 저장소를 연결하고 배포 가능한 변경을 `main` 에 올린다.

   ```powershell
   git remote add origin <비공개-저장소-URL>
   git push -u origin main
   ```

   > Kiro 는 자동 push 를 수행하지 않는다. 이 단계는 운영자가 직접 실행한다.

### B-3. 되돌릴 수 있는 조합 식별

배포 가능한 상태를 하나의 조합으로 식별할 수 있어야 한다.

- 커밋 해시(코드·명세·설정 원형)
- `deployment/active.json` 의 `bundle_id` + `manifest_hash`
- `requirements.txt` 의 고정 버전

실패 시 **이 조합으로 복구**한다(아래 D-3).

### B-4. 작업 31 완료 기준 (운영자 확인 후 체크)

- [ ] 운영자가 확인한 비공개 저장소 `main` 에 배포 가능한 변경이 반영됨.
- [ ] 설치·검사 명령과 되돌릴 코드·환경·묶음 조합을 식별할 수 있음.
- [ ] 실제 자료·상세 기록·비밀값·로컬 경로·노트북 출력이 제외됨.

> 원격 작업이 실행되지 않았으면 안내서 작성만으로 완료 처리하지 않는다.

---

## C. 작업 32 — Streamlit 공개 배포

### C-1. Cloud 앱 생성 (운영자)

Streamlit Community Cloud 에서 새 앱을 만들고 다음을 지정한다.

| 항목 | 값 |
|---|---|
| 저장소 | 위 비공개 저장소 (비공개 유지) |
| 브랜치 | `main` |
| 메인 파일 | `streamlit_app.py` |
| Python 버전 | 3.12 |
| 의존성 | `requirements.txt` (고정 버전) |

- 저장소는 **비공개로 유지**하고, 앱 공유 설정만 **공개(public)** 로 전환한다.
- 비밀값이 필요하면 Cloud 의 Secrets 로 설정한다. 저장소에 비밀값을 넣지 않는다.

### C-2. 운영 한도 재확인

배포 전 로컬에서 운영 설정을 생성·동기화한다(작업 17·29).

```powershell
.\.venv\Scripts\python.exe -m pyrolysis.cli prepare-runtime --runtime configs/runtime.yaml --config .streamlit/config.toml
```

- 실제 처리 한도(업로드 MiB·행·열·추론 스레드·대기)는 작업 29 에서 실측으로 채운다.
- Cloud 의 로딩·처리 자원에서 다시 확인하고 필요하면 측정에 따라 조정한다.

### C-3. 공개 배포 인수 점검 (비로그인 브라우저)

합성 후보로 아래를 확인한다(실제 연구 행을 입력하지 않는다).

- [ ] 로그인 없이 접속되고 실제 활성 대상의 예측을 사용할 수 있다.
- [ ] 로컬과 **같은 묶음·결과**를 사용한다(`bundle_id`·`manifest_hash` 일치).
- [ ] 단일·일괄 예측, 수치형 순위, 전체 실패(마지막 행 오류), 범위 이탈 표시.
- [ ] 두 브라우저 세션의 입력·결과가 섞이지 않는다.
- [ ] 결과 다운로드·복사 내보내기가 차단된다(`disableDataExport`).
- [ ] 중단된 요청이 완료로 표시되지 않는다.
- [ ] 서버 재시작 후 **재학습 없이 같은 묶음**을 읽는다.

### C-4. 작업 32 완료 기준 (실제 Cloud 확인 후 체크)

- [ ] 로그인 없이 실제 활성 대상 예측 사용, 로컬과 같은 묶음·결과.
- [ ] 중단 요청이 완료로 표시되지 않음.
- [ ] 운영 안내에 모델 교체·실패 시 준비 상태·복구 절차·확인한 한도가 기록됨.
- [ ] 상시 가동·특정 처리 시간은 보장하지 않음을 명시.

---

## D. 운영 절차

### D-1. 모델 교체

새 실제 묶음으로 교체할 때:

```powershell
# 1) 새 묶음을 export (작업 29, 실제 설정)
# 2) 무결성·환경 검사
.\.venv\Scripts\python.exe -m pyrolysis.cli check-bundle --bundle deployment/bundles/<new_bundle_id>
# 3) active 포인터 교체 (검사 통과 시에만, 원자적)
.\.venv\Scripts\python.exe -m pyrolysis.cli activate --bundle deployment/bundles/<new_bundle_id>
# 4) 커밋·push 후 Cloud 재배포 (운영자)
```

- `activate` 는 검사 통과 시에만 포인터를 바꾸고, 실패하면 **기존 포인터를 유지**한다.

### D-2. 시작 실패 시

- 앱이 묶음을 읽지 못하면 자동으로 demo 나 다른 묶음으로 전환하지 않는다.
  준비되지 않은 상태와 사유를 표시한다.
- 운영자가 D-3 으로 명시적 복구를 수행한다.

### D-3. 이전 호환 조합으로 복구

1. 직전에 정상 동작한 **커밋 해시 + `active.json`(bundle_id·manifest_hash) + 고정 의존성** 조합을 확인한다(B-3).
2. 해당 커밋으로 되돌리거나 `active.json` 을 직전 묶음으로 교체한다.
3. `check-bundle` 로 재검증 후 Cloud 를 재배포한다.

### D-4. 보장하지 않는 것

- 상시 가동(uptime)·특정 처리 시간(latency)을 보장하지 않는다.
- 예측 결과의 과학적 정확성·최적 조건을 보장하지 않는다(연구 판단은 별도).

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 안내서 작성 | ✅ 완료 (이 문서) |
| `.gitignore` 제외 규칙 | ✅ 적용됨 (data/·runs/·configs/local/·templates/local/·secrets) |
| 실제 활성 대상 확정 | ⬜ 작업 28 대기 |
| 실제 배포 묶음·운영 한도 | ⬜ 작업 29 대기 |
| 비공개 저장소 push | ⬜ 운영자 작업 대기 |
| Streamlit Cloud 공개·인수 점검 | ⬜ 운영자 작업 대기 |

`deployment/active.json` 은 아직 `no_bundle` 상태다. 실제 묶음 활성화(작업 29·30) 후
`bundle_id`·`manifest_hash` 가 채워진다.
