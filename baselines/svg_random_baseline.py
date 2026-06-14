#!/usr/bin/env python
"""Random baseline for the spatially-variable-genes task.

Assigns each gene a uniform random spatial-variability score (seeded for reproducibility) — the lower
reference the leaderboard normalizes against. Reads ``input/dataset.h5ad`` and writes
``outputs/output.h5ad``.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


def main() -> int:
    dataset = ad.read_h5ad("input/dataset.h5ad")
    rng = np.random.RandomState(0)
    scores = rng.rand(dataset.n_vars)

    var = dataset.var.copy()
    var["pred_spatial_var_score"] = scores
    output = ad.AnnData(var=var, uns={"method_id": "svg_random"})
    output.var_names = dataset.var_names

    Path("outputs").mkdir(exist_ok=True)
    output.write_h5ad("outputs/output.h5ad")
    Path("outputs/report.md").write_text(
        "# Random baseline\n\nAssigns each gene a uniform random spatial-variability score (seeded).\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
