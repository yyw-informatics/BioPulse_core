#!/usr/bin/env python
"""Derive the rare cell-type benchmark pack from the label-projection pack.

Same data (inputs + hidden solution) and prediction format, but a task.yaml task_type of
``rare_celltype`` (scored on macro-F1 — rare-class sensitive) and an instruction that emphasizes the
rare populations. Run after building the benchmark packs (it copies op_label_projection_mini).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# rare_celltype is a BioPulse-derived task. It reuses the label-projection data and prediction format,
# then applies rare-cell-type framing, instructions, and macro-F1 metric configuration from the registry.


def _task_yaml(record) -> str:
    # Most task metadata comes from the registry record; data_source and source_dataset_id identify the
    # source pack used for this derived benchmark.
    return (
        f"task_id: {record.task_id}\n"
        f"task_type: {record.task_type}\n"
        f"title: {record.title}\n"
        f"source: {record.source}\n"
        f"is_openproblems: {str(record.is_openproblems).lower()}\n"
        "data_source: openproblems\n"
        "source_dataset_id: cxg_immune_cell_atlas\n"
        "public_dir: public\n"
        "hidden_dir: hidden/ground_truth\n"
        "metrics_dir: metrics\n"
    )


def main(argv: list[str] | None = None) -> int:
    from biopulse.tasks.registry import get

    parser = argparse.ArgumentParser(description="Derive the rare cell-type pack from label projection")
    parser.add_argument("--packs-dir", default="benchmark_packs", type=Path)
    args = parser.parse_args(argv)

    record = get("rare_celltype")
    src = args.packs_dir / "op_label_projection_mini"
    dst = args.packs_dir / "op_rare_celltype_mini"
    if not src.exists():
        print(f"ERROR: {src} not found — run scripts/build_benchmark_packs.py first.", file=sys.stderr)
        return 1
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "task.yaml").write_text(_task_yaml(record), encoding="utf-8")
    (dst / "public" / "instruction.md").write_text(record.instruction, encoding="utf-8")
    # Replace the copied label-projection metric config with the rare-cell-type macro-F1 config.
    (dst / "metrics" / "metric_config.yaml").write_text(record.metric_config, encoding="utf-8")
    print(f"Derived {dst} from {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
