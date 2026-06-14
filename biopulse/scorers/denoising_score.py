"""Scorer for the Open Problems denoising task (molecular cross-validation).

The agent receives a noisy ``train`` count matrix (one molecular split) and must produce a denoised
matrix of the SAME cells x genes. We score it against the held-out ``test`` split (the hidden
solution). Following the Open Problems denoising metric, both matrices are library-size normalized to
a fixed target sum of 10,000 and log1p-transformed, then compared by mean squared error. Lower MSE is
better, so ``final_score = -mse`` keeps the "higher is better" convention used across BioPulse scorers.
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

    output_path = outputs / "denoised.h5ad"
    solution_path = benchmark / "hidden" / "ground_truth" / "solution.h5ad"
    result["metrics"]["report_present"] = report_present(outputs, result)

    if not output_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing required output: outputs/denoised.h5ad")
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
    if "denoised" not in prediction.layers:
        schema_valid = 0.0
        result["violations"].append("denoised.h5ad missing layers['denoised']")
    if "method_id" not in prediction.uns:
        schema_valid = 0.0
        result["violations"].append("denoised.h5ad missing uns['method_id']")
    if "counts" not in solution.layers:
        schema_valid = 0.0
        result["violations"].append("solution.h5ad missing layers['counts']")
    if schema_valid == 1.0 and prediction.shape != solution.shape:
        schema_valid = 0.0
        result["violations"].append(
            f"denoised shape {prediction.shape} != solution shape {solution.shape} (must be same cells x genes)"
        )
    elif schema_valid == 1.0 and not (
        (prediction.obs_names == solution.obs_names).all()
        and (prediction.var_names == solution.var_names).all()
    ):
        # OP aligns denoised<->test by position and asserts identical obs/var names; a reordered matrix
        # would otherwise be silently mis-scored (task_denoising metrics operate on row/col-aligned AnnData).
        schema_valid = 0.0
        result["violations"].append(
            "denoised obs_names/var_names do not match solution (cells and genes must be in the same order)"
        )

    result["metrics"]["schema_valid"] = schema_valid
    if schema_valid != 1.0:
        return finalize_result(result)

    pred = _dense(prediction.layers["denoised"])
    true = _dense(solution.layers["counts"])

    # OP's mse metric (mirrors task_denoising/src/metrics/mse/script.py): library-size normalize
    # (target_sum=10000) + log1p on both, then mean squared error. Negatives are not clipped; they make
    # log1p non-finite, which the guard below reports as a failure rather than a spurious 0.0.
    mse = float(np.mean((_norm_total_log(pred) - _norm_total_log(true)) ** 2))
    # OP's poisson metric (task_denoising/src/metrics/poisson/script.py): rescale denoised by
    # (test_total / train_sum), then the Poisson NLL. Requires uns['train_sum'] on the solution.
    poisson = _poisson_nll(pred, true, solution.uns.get("train_sum"))

    if not np.isfinite(mse) or (poisson is not None and not np.isfinite(poisson)):
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append(
            "denoised.h5ad produced a non-finite metric (NaN/inf — negative/degenerate denoised values?)"
        )
        return finalize_result(result)

    result["metrics"]["mse"] = mse
    if poisson is not None:
        result["metrics"]["poisson"] = poisson
    result["metrics"]["n_cells"] = float(pred.shape[0])
    result["metrics"]["n_genes"] = float(pred.shape[1])
    result["final_score"] = -mse
    return finalize_result(result)


def _dense(matrix) -> np.ndarray:
    array = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return array.astype(np.float64)


def _norm_total_log(matrix: np.ndarray, target_sum: float = 10000.0) -> np.ndarray:
    """normalize_total(target_sum) + log1p — matches OP's mse metric (scanpy.pp.normalize_total): scale
    each cell to `target_sum` total counts, then log1p. (We reimplement in numpy; OP imports scanpy.)"""
    lib = matrix.sum(axis=1, keepdims=True)
    safe = np.where(lib > 0, lib, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log1p(matrix / safe * target_sum)


def _poisson_nll(denoised: np.ndarray, test: np.ndarray, train_sum) -> "float | None":
    """OP's poisson metric (task_denoising/src/metrics/poisson/script.py): rescale denoised by
    (test_total / train_sum), then (test - denoised * log(test + 1e-6)).mean(). None if train_sum absent."""
    if train_sum is None:
        return None
    try:
        rescaled = denoised * float(test.sum()) / float(train_sum)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float((test - rescaled * np.log(test + 1e-6)).mean())
    except Exception:
        return None


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score a denoising run")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(score(args.benchmark, args.run_dir, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
