#!/usr/bin/env python
"""Majority-class baseline for the label-projection task.

Predicts the most frequent training label for every test cell. Reads the public inputs from the run
workspace (``input/train.h5ad``, ``input/test.h5ad``) and writes ``outputs/prediction.h5ad`` plus a
report. A trivial lower reference for the scorer — a real classifier should beat it.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> int:
    train = ad.read_h5ad("input/train.h5ad")
    test = ad.read_h5ad("input/test.h5ad")
    label_col = "label" if "label" in train.obs else "cell_type"
    majority = Counter(train.obs[label_col].astype(str)).most_common(1)[0][0]

    prediction = ad.AnnData(
        obs=pd.DataFrame({"label_pred": [majority] * test.n_obs}, index=test.obs_names),
        uns={"method_id": "label_projection_majority"},
    )
    Path("outputs").mkdir(exist_ok=True)
    prediction.write_h5ad("outputs/prediction.h5ad")
    Path("outputs/report.md").write_text(
        "# Majority-class baseline\n\nPredicts the most frequent training label for every test cell, "
        "establishing the floor a real classifier must beat.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
