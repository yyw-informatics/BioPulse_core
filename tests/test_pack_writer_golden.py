"""Golden master tests for the pack builder's generated text artifacts.

These tests pin the generated instructions, output schemas, metric configs, score shims, and task
templates for each task. The artifact text is intentionally stable because downstream benchmark packs
and reviewer-facing examples rely on these contracts.

Generated Markdown and environment artifact names are stored with a ``.txt`` suffix so the public
repository can keep root ``README.md`` as its only tracked Markdown file and avoid committing
environment manifests. Regenerate only for an intentional content change with:
    BIOPULSE_UPDATE_GOLDEN_PACKS=1 python -m pytest tests/test_pack_writer_golden.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from biopulse.builder import pack_writer
from biopulse.tasks.registry import get as get_task

GOLDEN_DIR = Path(__file__).parent / "golden_packs"
TASK_TYPES = ["label_projection", "spatially_variable_genes", "denoising", "dimensionality_reduction"]
FIXED_DATASET_ID = "GOLDEN_DATASET"  # task.yaml's source_dataset_id comes from discovery; pin it so the template is stable
GOLDEN_NAME_OVERRIDES = {
    "environment.yml": "environment.yml.txt",
    "instruction.md": "instruction.md.txt",
    "output_schema.md": "output_schema.md.txt",
}


def _artifacts(task_type: str) -> dict[str, str]:
    """Return the per-task text artifacts generated from the task registry."""
    record = get_task(task_type)
    return {
        "task.yaml": pack_writer._task_yaml(record, FIXED_DATASET_ID),
        "instruction.md": record.instruction,
        "output_schema.md": record.output_schema,
        "metric_config.yaml": record.metric_config,
        "score.py": pack_writer._metric_script(record),
        "evidence_spec.yaml": pack_writer._evidence_spec(task_type),
        "environment.yml": pack_writer._environment_yaml(),
    }


def _cases() -> list[tuple[str, str]]:
    return [(task_type, name) for task_type in TASK_TYPES for name in sorted(_artifacts(task_type))]


def _golden_name(artifact: str) -> str:
    return GOLDEN_NAME_OVERRIDES.get(artifact, artifact)


@pytest.mark.parametrize("task_type,artifact", _cases(), ids=[f"{t}:{a}" for t, a in _cases()])
def test_generated_pack_artifact_matches_golden(task_type: str, artifact: str) -> None:
    produced = _artifacts(task_type)[artifact]
    path = GOLDEN_DIR / task_type / _golden_name(artifact)
    if os.environ.get("BIOPULSE_UPDATE_GOLDEN_PACKS", "") not in ("", "0", "false"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(produced, encoding="utf-8")
        pytest.skip(f"golden updated: {task_type}/{artifact}")
    assert path.exists(), f"no golden for {task_type}/{artifact}; regenerate with BIOPULSE_UPDATE_GOLDEN_PACKS=1"
    assert produced == path.read_text(encoding="utf-8")
