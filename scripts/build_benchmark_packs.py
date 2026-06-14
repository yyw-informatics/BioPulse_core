from __future__ import annotations

import argparse
from pathlib import Path

from biopulse.builder.pack_writer import build_packs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BioPulse benchmark packs from Open Problems resources")
    parser.add_argument("--openproblems-root", default="external/openproblems", type=Path)
    parser.add_argument("--out", default="benchmark_packs", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["label_projection", "spatially_variable_genes", "denoising", "dimensionality_reduction"],
    )
    parser.add_argument("--prefer", default="resources_test")
    parser.add_argument("--dataset-id")
    args = parser.parse_args()

    packs = build_packs(args.openproblems_root, args.out, args.tasks, prefer=args.prefer, dataset_id=args.dataset_id)
    for pack in packs:
        print(f"built {pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
