"""build_rare_celltype_pack must consume the registry, not a parallel copy.

rare_celltype is derived (copytree of the label_projection pack) but its instruction / metric_config /
labels are owned by the task registry. A prior bug: the derive-script inherited label_projection's
metric_config (primary_metric: accuracy) while the scorer grades on macro_f1. This pins that the built
pack's registry-owned files come from the registry record, hermetically (no real pack needed).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from biopulse.tasks.registry import get

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_rare_celltype_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_rare_celltype_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_label_projection_pack(packs_dir: Path) -> None:
    """A minimal stand-in for the label_projection pack: enough for copytree, with a metric_config that
    says 'accuracy' so we can prove the derive-script overwrites it with the registry's macro_f1."""
    pack = packs_dir / "op_label_projection_mini"
    (pack / "public" / "input").mkdir(parents=True)
    (pack / "metrics").mkdir(parents=True)
    (pack / "hidden" / "ground_truth").mkdir(parents=True)
    (pack / "task.yaml").write_text("task_id: op_label_projection_mini\ntask_type: label_projection\n", encoding="utf-8")
    (pack / "public" / "instruction.md").write_text("# Label Projection Task\n", encoding="utf-8")
    (pack / "metrics" / "metric_config.yaml").write_text("primary_metric: accuracy\n", encoding="utf-8")
    (pack / "hidden" / "ground_truth" / "solution.h5ad").write_bytes(b"")
    (pack / "public" / "input" / "train.h5ad").write_bytes(b"")


def test_rare_pack_is_built_from_the_registry(tmp_path: Path) -> None:
    _fake_label_projection_pack(tmp_path)
    module = _load_script()
    assert module.main(["--packs-dir", str(tmp_path)]) == 0

    record = get("rare_celltype")
    dst = tmp_path / "op_rare_celltype_mini"

    # metric_config came from the registry (macro_f1), NOT the inherited label_projection 'accuracy'
    metric_config = (dst / "metrics" / "metric_config.yaml").read_text(encoding="utf-8")
    assert metric_config == record.metric_config
    assert "primary_metric: macro_f1" in metric_config
    assert "accuracy" not in metric_config

    # instruction came from the registry (the rare-specific, macro-F1 text)
    instruction = (dst / "public" / "instruction.md").read_text(encoding="utf-8")
    assert instruction == record.instruction
    assert "macro-F1" in instruction

    # task.yaml derives its identity fields from the record (no drift) + keeps the derived-task markers
    task_yaml = (dst / "task.yaml").read_text(encoding="utf-8")
    assert f"title: {record.title}" in task_yaml
    assert "is_openproblems: false" in task_yaml
    assert "task_type: rare_celltype" in task_yaml
