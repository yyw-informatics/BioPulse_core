"""Scorer for the Open Problems dimensionality-reduction task.

The agent embeds a normalized expression matrix into 2D (``obsm['X_emb']``). We score how well the
embedding preserves the high-dimensional neighborhood structure via **trustworthiness** (sklearn):
the fraction of each point's low-dim neighbors that were also true high-dim neighbors, in [0, 1],
higher is better. As a secondary signal we report the silhouette of the embedding by the hidden
cell-type labels — does the 2D map keep biological types separable. ``final_score = trustworthiness``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .common import (
    base_result,
    finalize_result,
    outputs_for_run,
    read_task_yaml,
    report_present,
    require_anndata,
    scan_output_text_for_forbidden_refs,
    scan_workspace_safety,
    workspace_for_run,
)


def score(benchmark_dir: Path | str, run_dir: Path | str, run_id: Optional[str] = None) -> dict:
    benchmark = Path(benchmark_dir)
    run = Path(run_dir)
    task = read_task_yaml(benchmark / "task.yaml")
    task_id = str(task.get("task_id", benchmark.name))
    result = base_result(task_id=task_id, run_id=run_id or run.name)
    outputs = outputs_for_run(run)
    workspace = workspace_for_run(run)

    safety_ok, safety_violations = scan_workspace_safety(workspace)
    result["violations"].extend(safety_violations)
    result["violations"].extend(scan_output_text_for_forbidden_refs(outputs))
    result["safety_gate_passed"] = safety_ok and not any("Forbidden reference" in item for item in result["violations"])

    output_path = outputs / "embedding.h5ad"
    solution_path = benchmark / "hidden" / "ground_truth" / "solution.h5ad"
    result["metrics"]["report_present"] = report_present(outputs, result)

    if not output_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing required output: outputs/embedding.h5ad")
        return finalize_result(result)
    if not solution_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing hidden ground truth: hidden/ground_truth/solution.h5ad")
        return finalize_result(result)

    adata = require_anndata(result)
    if adata is None:
        result["metrics"]["schema_valid"] = 0.0
        return finalize_result(result)

    prediction = adata.read_h5ad(output_path)
    solution = adata.read_h5ad(solution_path)

    schema_valid = 1.0
    if "X_emb" not in prediction.obsm:
        schema_valid = 0.0
        result["violations"].append("embedding.h5ad missing obsm['X_emb']")
    if "method_id" not in prediction.uns:
        schema_valid = 0.0
        result["violations"].append("embedding.h5ad missing uns['method_id']")
    if "normalized" not in solution.layers:
        schema_valid = 0.0
        result["violations"].append("solution.h5ad missing layers['normalized']")
    if schema_valid == 1.0 and prediction.n_obs != solution.n_obs:
        schema_valid = 0.0
        result["violations"].append(
            f"embedding has {prediction.n_obs} cells but solution has {solution.n_obs} (must match, same order)"
        )
    elif schema_valid == 1.0 and not (prediction.obs_names == solution.obs_names).all():
        # OP aligns embedding<->solution by position; trustworthiness compares row i of X_emb to row i of
        # the normalized layer, so a reordered embedding must not be silently mis-scored.
        schema_valid = 0.0
        result["violations"].append(
            "embedding obs_names do not match solution obs_names (cells must be in the same order)"
        )

    result["metrics"]["schema_valid"] = schema_valid
    if schema_valid != 1.0:
        return finalize_result(result)

    emb = np.asarray(prediction.obsm["X_emb"], dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] < 2:
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append(f"obsm['X_emb'] must be 2D with >=2 columns, got shape {emb.shape}")
        return finalize_result(result)
    if not np.isfinite(emb).all():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("obsm['X_emb'] contains non-finite values")
        return finalize_result(result)

    # OP's trustworthiness metric (task_dimensionality_reduction/src/metrics/trustworthiness): sklearn
    # trustworthiness at n_neighbors=15 against the high-dim normalized layer. Synthetic mini-cases may
    # have too few cells for k=15, so k is capped to sklearn's valid range; real OP data uses k=15.
    high_dim = _dense(solution.layers["normalized"])
    if solution.n_obs < 3:
        result["warnings"].append("trustworthiness requires at least 3 cells; returning 0.0")
        result["metrics"]["trustworthiness"] = 0.0
    else:
        max_neighbors = max(1, (solution.n_obs - 1) // 2)
        n_neighbors = min(15, max_neighbors)
        result["metrics"]["trustworthiness"] = _trustworthiness(high_dim, emb, n_neighbors, result)
    result["metrics"]["silhouette_celltype_diagnostic"] = _silhouette(emb, solution, result)
    result["metrics"]["n_cells"] = float(solution.n_obs)
    result["metrics"]["emb_dims"] = float(emb.shape[1])
    result["final_score"] = result["metrics"]["trustworthiness"]
    return finalize_result(result)


def _dense(matrix) -> np.ndarray:
    array = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return array.astype(np.float64)


def _trustworthiness(high_dim: np.ndarray, emb: np.ndarray, n_neighbors: int, result: dict) -> float:
    try:
        from sklearn.manifold import trustworthiness

        return float(trustworthiness(high_dim, emb, n_neighbors=n_neighbors))
    except Exception as exc:  # pragma: no cover - sklearn missing/edge
        result["warnings"].append(f"trustworthiness unavailable: {exc}")
        return 0.0


def _silhouette(emb: np.ndarray, solution, result: dict) -> float:
    if "cell_type" not in solution.obs:
        result["warnings"].append("solution missing obs['cell_type']; silhouette skipped")
        return 0.0
    try:
        from sklearn.metrics import silhouette_score

        labels = solution.obs["cell_type"].astype(str).to_numpy()
        if len(set(labels)) < 2:
            return 0.0
        return float(silhouette_score(emb, labels))
    except Exception as exc:  # pragma: no cover
        result["warnings"].append(f"silhouette unavailable: {exc}")
        return 0.0


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score a dimensionality-reduction run")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(score(args.benchmark, args.run_dir, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
