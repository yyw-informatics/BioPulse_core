"""Build the full-dataset ``dkd`` label_projection pack from op_cache, then enrich it with the OP
leaderboard + method menu + contamination blacklist. No methods are rerun.

The op_cache layout (``op_cache/task_label_projection/dkd/log_cp10k/{train,test,solution}.h5ad``) does
not match what ``discover_label_projection`` expects, so we hand-build the discovery dict and call
``write_pack`` directly (the same pattern the OP adapter would otherwise produce). The three enrichment
files are written at the pack ROOT — never under ``public/`` — so they reach the harness, never the agent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from biopulse.builder.pack_writer import write_pack
from biopulse.runner.harness import CONTAMINATION_BLACKLIST
from biopulse.tasks.registry import get


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    base = _repo_root()
    parser = argparse.ArgumentParser(description="Build + enrich the full dkd label_projection pack")
    parser.add_argument("--data", type=Path, default=base / "op_cache/task_label_projection/dkd/log_cp10k")
    parser.add_argument("--boards", type=Path, default=base / "op_cache/task_label_projection/boards/label_projection_dkd")
    parser.add_argument("--out", type=Path, default=base / "benchmark_packs")
    args = parser.parse_args(argv)

    record = get("label_projection_dkd")
    assert record.task_id == "op_label_projection_dkd", f"registry collision: {record.task_id}"

    discovery = {
        "source": "openproblems",
        "repo_name": "task_label_projection",
        "repo_path": str(args.data),
        "resource_root": str(args.data),
        "selected": {
            "task_type": "label_projection",
            "dataset_id": "cellxgene_census/dkd",
            "root": str(args.data),
            "normalization_id": "log_cp10k",
            "files": {"train": "train.h5ad", "test": "test.h5ad", "solution": "solution.h5ad"},
        },
        "candidates": [],
        "note": "full cellxgene_census/dkd dataset (not the OP test resource); pulled from op_cache",
    }

    pack_dir = write_pack("label_projection_dkd", discovery, args.out)
    print("built pack:", pack_dir)

    # Enrich at pack ROOT (contamination-safe: copy_public_to_workspace never stages pack-root files).
    shutil.copy2(args.boards / "leaderboard.yaml", pack_dir / "leaderboard.yaml")
    shutil.copy2(args.boards / "methods.yaml", pack_dir / "methods.yaml")
    (pack_dir / "blacklist.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "biopulse.blacklist.v1",
                "hosts": list(CONTAMINATION_BLACKLIST),
                "forbidden_files": ["solution.h5ad"],
                "note": "L2+ contamination blacklist — answer-revealing hosts blocked at the network layer",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("enriched (pack root):", sorted(p.name for p in pack_dir.glob("*.yaml")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
