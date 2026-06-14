#!/usr/bin/env python
"""k-nearest-neighbours baseline for the label-projection task.

Fits a kNN classifier on the labelled training cells and predicts each test cell's label. Uses the PCA
embedding when both splits carry ``obsm['X_pca']``, otherwise the normalized layer, otherwise ``X``.
Reads ``input/train.h5ad`` / ``input/test.h5ad`` and writes ``outputs/prediction.h5ad``.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _features(train, test) -> tuple[np.ndarray, np.ndarray]:
    if "X_pca" in train.obsm and "X_pca" in test.obsm:
        return np.asarray(train.obsm["X_pca"]), np.asarray(test.obsm["X_pca"])
    if "normalized" in train.layers and "normalized" in test.layers:
        return _dense(train.layers["normalized"]), _dense(test.layers["normalized"])
    return _dense(train.X), _dense(test.X)


def main() -> int:
    train = ad.read_h5ad("input/train.h5ad")
    test = ad.read_h5ad("input/test.h5ad")
    label_col = "label" if "label" in train.obs else "cell_type"

    x_train, x_test = _features(train, test)
    k = min(15, max(1, train.n_obs))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, train.obs[label_col].astype(str).to_numpy())
    pred = model.predict(x_test)

    prediction = ad.AnnData(
        obs=pd.DataFrame({"label_pred": pred}, index=test.obs_names),
        uns={"method_id": f"label_projection_knn_k{k}"},
    )
    Path("outputs").mkdir(exist_ok=True)
    prediction.write_h5ad("outputs/prediction.h5ad")
    Path("outputs/report.md").write_text(
        f"# kNN baseline (k={k})\n\nFits a k-nearest-neighbours classifier on the training cells and "
        "predicts each test cell from its nearest neighbours in feature space.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
