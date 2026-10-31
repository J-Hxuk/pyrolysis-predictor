# tests/fixtures

이 폴더는 테스트 전용 가상(합성) 검증 자료를 보관한다.

## 규칙

- **실제 연구 데이터를 이 폴더에 넣지 않는다.**
- 파일명에 `synthetic_` 또는 `demo_` 접두어를 사용하여 가상 자료임을 명시한다.
- 가상 자료의 수치는 실제 촉매 성능이나 측정값을 나타내지 않는다.
- 이 폴더의 모든 파일은 Git에 포함된다 (`.gitignore` 에서 제외하지 않음).

## 현재 파일

- `test_draft_spec.yaml`, `test_minimal_configured_spec.yaml` — 설정 구조 검사용 (작업 2).
- `synthetic_spec_a.yaml` — 통합 점검용 규격 A: 회귀(wt%) + 검출 분류 (작업 24).
- `synthetic_spec_b.yaml` — 통합 점검용 규격 B: 단위가 다른 회귀 2개 + 미지원 분류 (작업 24).

규격 A·B 는 입력 ID·대상·단위가 서로 달라 "코드 수정 없이 설정으로만 바뀜"을
`tests/test_end_to_end.py` 에서 확인한다.
