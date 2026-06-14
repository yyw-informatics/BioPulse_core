#!/usr/bin/env python
"""Expression-variance baseline for the spatially-variable-genes task.

Scores each gene by the variance of its expression across spots — a non-spatial proxy for spatial
structure. Reads ``input/dataset.h5ad`` and writes ``outputs/output.h5ad`` with
``var['pred_spatial_var_score']``.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def main() -> int:
    dataset = ad.read_h5ad("input/dataset.h5ad")
    matrix = _dense(dataset.layers["normalized"] if "normalized" in dataset.layers else dataset.X)
    scores = np.asarray(matrix, dtype=np.float64).var(axis=0)

    var = dataset.var.copy()
    var["pred_spatial_var_score"] = scores
    output = ad.AnnData(var=var, uns={"method_id": "svg_variance"})
    output.var_names = dataset.var_names

    Path("outputs").mkdir(exist_ok=True)
    output.write_h5ad("outputs/output.h5ad")
    Path("outputs/report.md").write_text(
        "# Expression-variance baseline\n\nScores each gene by its expression variance across spots, a "
        "non-spatial proxy for spatial variability.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
