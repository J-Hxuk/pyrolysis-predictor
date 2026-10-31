from __future__ import annotations
from typing import Any
import streamlit as st

__all__ = ['render_input_widgets']

def render_input_widgets(input_specs: list[Any], key_prefix: str = 'input') -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for spec in input_specs:
        input_id = spec.id
        label = spec.label_ko
        unit_text = f' ({spec.unit})' if spec.unit else ''
        full_label = f'{label}{unit_text}'
        widget_key = f'{key_prefix}_{input_id}'
        if spec.dtype == 'int':
            min_val = int(spec.min_value) if spec.min_value is not None else 0
            max_val = int(spec.max_value) if spec.max_value is not None else 100
            default_val = int(spec.default_value) if spec.default_value is not None else min_val
            value = st.number_input(full_label, min_value=min_val, max_value=max_val, value=default_val, step=1, key=widget_key)
            inputs[input_id] = int(value)
        elif spec.dtype == 'float':
            min_val = float(spec.min_value) if spec.min_value is not None else 0.0
            max_val = float(spec.max_value) if spec.max_value is not None else 100.0
            default_val = float(spec.default_value) if spec.default_value is not None else min_val
            value = st.number_input(full_label, min_value=min_val, max_value=max_val, value=default_val, step=0.1, key=widget_key, format='%.2f')
            inputs[input_id] = float(value)
        elif spec.dtype == 'category':
            options = spec.categories if spec.categories else ['unknown']
            default_idx = 0
            if spec.default_value and spec.default_value in options:
                default_idx = options.index(spec.default_value)
            value = st.selectbox(full_label, options=options, index=default_idx, key=widget_key)
            inputs[input_id] = value
        else:
            st.warning(f'지원하지 않는 입력 형식: {spec.dtype}')
            value = st.text_input(full_label, value=str(spec.default_value) if spec.default_value else '', key=widget_key)
            inputs[input_id] = value
    return inputs
