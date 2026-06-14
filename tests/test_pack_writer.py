"""Registry guards (formerly pack_writer guards).

Pins the behavior that replaced the latent fallthrough bug: per-task content (instruction / output_schema
/ metric_config) used to live in pack_writer functions that fell through to the dimred default for an
unknown task_type. That content now lives in the task registry, one record per task, and an unknown
task_type raises instead of silently producing the wrong pack.
"""

from __future__ import annotations

import pytest

from biopulse.tasks.registry import get

OP_TASKS = ("label_projection", "spatially_variable_genes", "denoising", "dimensionality_reduction")
PROSE_FIELDS = ("instruction", "output_schema", "metric_config")


@pytest.mark.parametrize("field", PROSE_FIELDS)
def test_records_have_distinct_prose_for_op_tasks(field: str) -> None:
    values = [getattr(get(task_type), field) for task_type in OP_TASKS]
    assert all(values), f"every OP task must define {field}"
    assert len(set(values)) == len(OP_TASKS), f"{field} must be distinct per task (no fallthrough)"


def test_get_raises_on_unknown_task_type() -> None:
    with pytest.raises(ValueError):
        get("denosing_typo")  # one transposed letter must not silently resolve to a default record


def test_aliases_resolve_to_canonical_task_type() -> None:
    assert get("svg").task_type == "spatially_variable_genes"
    assert get("dimred").task_type == "dimensionality_reduction"
    assert get("lp").task_type == "label_projection"
    assert get("denoise").task_type == "denoising"
