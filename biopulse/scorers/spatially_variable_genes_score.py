from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import (
    base_result,
    finalize_result,
    load_control_outputs,
    normalize_to_controls,
    outputs_for_run,
    read_task_yaml,
    report_present,
    require_anndata,
    scan_output_text_for_forbidden_refs,
    scan_workspace_safety,
    spearman_fallback,
    top_k_overlap,
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

    output_path = outputs / "output.h5ad"
    solution_path = benchmark / "hidden" / "ground_truth" / "solution.h5ad"
    result["metrics"]["report_present"] = report_present(outputs, result)

    if not output_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing required output: outputs/output.h5ad")
        return finalize_result(result)
    if not solution_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing hidden ground truth: hidden/ground_truth/solution.h5ad")
        return finalize_result(result)

    ad = require_anndata(result)
    if ad is None:
        result["metrics"]["schema_valid"] = 0.0
        return finalize_result(result)

    prediction = ad.read_h5ad(output_path)
    solution = ad.read_h5ad(solution_path)

    schema_valid = 1.0
    if "pred_spatial_var_score" not in prediction.var:
        schema_valid = 0.0
        result["violations"].append("output.h5ad missing var['pred_spatial_var_score']")
    if "method_id" not in prediction.uns:
        schema_valid = 0.0
        result["violations"].append("output.h5ad missing uns['method_id']")
    if "true_spatial_var_score" not in solution.var:
        schema_valid = 0.0
        result["violations"].append("solution.h5ad missing var['true_spatial_var_score']")

    result["metrics"]["schema_valid"] = schema_valid
    if schema_valid != 1.0:
        return finalize_result(result)

    pred_scores = _var_scores(prediction, "pred_spatial_var_score")
    true_scores = _var_scores(solution, "true_spatial_var_score")
    overlap = sorted(set(pred_scores) & set(true_scores))
    result["metrics"]["gene_overlap_count"] = float(len(overlap))
    if not overlap:
        result["violations"].append("No overlapping genes between output and solution")
        return finalize_result(result)

    # Diagnostics (not the OP metric): a global rank-correlation over all features, kept for reference.
    xs = [pred_scores[key] for key in overlap]
    ys = [true_scores[key] for key in overlap]
    kendall_tau = _kendall_tau(xs, ys)
    spearman_r = _spearman_r(xs, ys)
    top_50 = top_k_overlap(pred_scores, true_scores, k=50)
    if kendall_tau is not None:
        result["metrics"]["kendall_tau"] = kendall_tau
    else:
        result["warnings"].append("Kendall tau unavailable; using Spearman/top-k fallback")
    if spearman_r is not None:
        result["metrics"]["spearman_r"] = spearman_r
    result["metrics"]["top_50_overlap"] = top_50

    # OP's `correlation` metric (the headline final_score): per-gene Kendall tau grouped by
    # var['orig_feature_name'], averaged across groups — mirrors task_spatially_variable_genes/
    # src/metrics/correlation/script.py. Falls back to the global tau only when the solution carries
    # no orig_feature_name grouping (synthetic / degenerate input).
    correlation = _op_grouped_correlation(prediction, solution)
    if correlation is not None:
        result["metrics"]["correlation"] = correlation
        result["final_score"] = correlation
    else:
        result["warnings"].append(
            "solution lacks var['orig_feature_name']; final_score is the global Kendall tau, "
            "not OP's grouped correlation"
        )
        result["final_score"] = (
            kendall_tau if kendall_tau is not None else spearman_r if spearman_r is not None else top_50
        )

    # OP normalization: rescale the headline correlation against the stored random/oracle controls
    # (random->0, oracle->1), reported alongside the raw score. No-op if the pack ships no controls.
    controls = load_control_outputs(benchmark, ad)
    if correlation is not None and controls:
        cmin = _op_grouped_correlation(controls["random"], solution)
        cmax = _op_grouped_correlation(controls["oracle"], solution)
        if cmin is not None and cmax is not None:
            scaled = normalize_to_controls(correlation, cmin, cmax)
            if scaled is not None:
                result["metrics"]["correlation_scaled"] = scaled
                result["metrics"]["control_random_score"] = cmin
                result["metrics"]["control_oracle_score"] = cmax
    return finalize_result(result)


def _op_grouped_correlation(prediction, solution):
    """OP's SVG metric: merge prediction.var <-> solution.var on `feature_id`, group by
    `orig_feature_name`, take the per-group Kendall tau of pred vs true spatial-var scores, and average
    across groups. Returns None when the solution has no `orig_feature_name` (so the caller falls back)."""
    if "orig_feature_name" not in solution.var:
        return None
    try:
        import pandas as pd

        pv = prediction.var.copy()
        sv = solution.var.copy()
        if "feature_id" not in pv:
            pv = pv.assign(feature_id=list(prediction.var_names))
        if "feature_id" not in sv:
            sv = sv.assign(feature_id=list(solution.var_names))
        df = pd.merge(
            pv[["feature_id", "pred_spatial_var_score"]],
            sv[["feature_id", "orig_feature_name", "true_spatial_var_score"]],
            how="left",
            on="feature_id",
        )
        per_group = df.groupby("orig_feature_name", observed=True)[
            ["pred_spatial_var_score", "true_spatial_var_score"]
        ].apply(lambda g: g["pred_spatial_var_score"].corr(g["true_spatial_var_score"], method="kendall"))
        value = float(per_group.mean())
        return None if value != value else value  # NaN -> fall back
    except Exception:
        return None


def _var_scores(adata, column: str) -> dict[str, float]:
    if "feature_id" in adata.var:
        keys = [str(value) for value in adata.var["feature_id"].tolist()]
    else:
        keys = [str(value) for value in adata.var_names.tolist()]
    values = [float(value) for value in adata.var[column].tolist()]
    return dict(zip(keys, values))


def _kendall_tau(xs: list[float], ys: list[float]) -> Optional[float]:
    try:
        from scipy.stats import kendalltau

        value = kendalltau(xs, ys, nan_policy="omit").statistic
        return None if value != value else float(value)
    except Exception:
        return None


def _spearman_r(xs: list[float], ys: list[float]) -> Optional[float]:
    try:
        from scipy.stats import spearmanr

        value = spearmanr(xs, ys, nan_policy="omit").statistic
        return None if value != value else float(value)
    except Exception:
        return spearman_fallback(xs, ys)


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score a spatially variable genes run")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(score(args.benchmark, args.run_dir, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
