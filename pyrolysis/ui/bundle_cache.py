"""
bundle_cache.py — 읽기 전용 배포 묶음의 st.cache_resource 재사용 (작업 22).

설계 10.3 [S1]:
  읽기 전용 배포 묶음만 st.cache_resource 로 재사용한다.
  캐시 키에는 묶음 ID 와 manifest 해시를 포함한다.
  사용자 입력·파일·결과·순위는 캐시에 넣지 않고 st.session_state 에만 둔다.
  사용자 자료를 st.cache_data·전역 변수에 저장하지 않는다.

캐시 대상은 모델·계약 등 묶음 내용뿐이며 요청별 데이터는 담지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["bundle_cache_key", "load_cached_bundle"]


def bundle_cache_key(bundle_id: str, manifest_hash: str) -> str:
    """묶음 캐시 키. ID 와 manifest 해시를 함께 쓴다.

    묶음이 교체되거나 manifest 가 바뀌면 키가 달라져 새로 로드된다.
    """
    return f"{bundle_id}@{manifest_hash}"


def load_cached_bundle(bundle_dir: Path, bundle_id: str, manifest_hash: str) -> dict[str, Any]:
    """읽기 전용 묶음을 st.cache_resource 로 캐시하여 반환한다.

    st.cache_resource 는 반환 객체를 공유하므로, 호출자는 이 묶음을 수정하거나
    fit 을 호출하지 않는다. 요청별 입력·결과는 세션에 두고 여기에 담지 않는다.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load(_cache_key: str, _dir: str) -> dict[str, Any]:
        # _cache_key(=ID@해시)만 캐시 식별에 쓰고, 실제 로드는 디렉터리에서 한다.
        from pyrolysis.artifacts import load_research_bundle

        return load_research_bundle(Path(_dir))

    key = bundle_cache_key(bundle_id, manifest_hash)
    return _load(key, str(bundle_dir))
