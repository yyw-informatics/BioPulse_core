"""Build the full openproblems_v1/immune_cells denoising pack from op_cache, then enrich it with the
OP leaderboard + method menu + contamination blacklist. No methods are rerun.

Hand-builds the discovery dict (op_cache layout != what discover_denoising expects) and calls
``write_pack`` directly. Denoising's hidden truth is the held-out molecular split, so the record's
``solution_role='test'`` maps the discovery's ``test`` role to ``hidden/ground_truth/solution.h5ad``.
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
    parser = argparse.ArgumentParser(description="Build + enrich the full immune_cells denoising pack")
    parser.add_argument("--data", type=Path, default=base / "op_cache/task_denoising/immune_cells/log_cp10k")
    parser.add_argument("--boards", type=Path, default=base / "op_cache/task_denoising/boards/denoising_immune")
    parser.add_argument("--out", type=Path, default=base / "benchmark_packs")
    args = parser.parse_args(argv)

    record = get("denoising_immune")
    assert record.task_id == "op_denoising_immune", f"registry collision: {record.task_id}"

    discovery = {
        "source": "openproblems",
        "repo_name": "task_denoising",
        "repo_path": str(args.data),
        "resource_root": str(args.data),
        "selected": {
            "task_type": "denoising",
            "dataset_id": "openproblems_v1/immune_cells",
            "root": str(args.data),
            "normalization_id": "log_cp10k",
            "files": {"train": "train.h5ad", "test": "test.h5ad"},
        },
        "candidates": [],
        "note": "full openproblems_v1/immune_cells denoising dataset; pulled from op_cache",
    }

    pack_dir = write_pack("denoising_immune", discovery, args.out)
    print("built pack:", pack_dir)

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
