from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import (
    base_result,
    finalize_result,
    load_control_outputs,
    macro_f1,
    normalize_to_controls,
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

    prediction_path = outputs / "prediction.h5ad"
    solution_path = benchmark / "hidden" / "ground_truth" / "solution.h5ad"
    result["metrics"]["report_present"] = report_present(outputs, result)

    if not prediction_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing required output: outputs/prediction.h5ad")
        return finalize_result(result)
    if not solution_path.exists():
        result["metrics"]["schema_valid"] = 0.0
        result["violations"].append("Missing hidden ground truth: hidden/ground_truth/solution.h5ad")
        return finalize_result(result)

    ad = require_anndata(result)
    if ad is None:
        result["metrics"]["schema_valid"] = 0.0
        return finalize_result(result)

    prediction = ad.read_h5ad(prediction_path)
    solution = ad.read_h5ad(solution_path)

    schema_valid = 1.0
    if "label_pred" not in prediction.obs:
        schema_valid = 0.0
        result["violations"].append("prediction.h5ad missing obs['label_pred']")
    if "method_id" not in prediction.uns:
        schema_valid = 0.0
        result["violations"].append("prediction.h5ad missing uns['method_id']")
    label_col = "label" if "label" in solution.obs else "cell_type" if "cell_type" in solution.obs else None
    if label_col is None:
        schema_valid = 0.0
        result["violations"].append("solution.h5ad missing obs['label'] or obs['cell_type']")
    if prediction.n_obs != solution.n_obs:
        schema_valid = 0.0
        result["violations"].append(f"Prediction cell count {prediction.n_obs} does not match solution cell count {solution.n_obs}")
    elif not (prediction.obs_names == solution.obs_names).all():
        # OP asserts identical obs_names so prediction and solution align row-for-row (we compare by
        # position); a reordered prediction would otherwise be silently mis-scored. (OP f1/accuracy scripts.)
        schema_valid = 0.0
        result["violations"].append("Prediction obs_names do not match solution obs_names (cells must be in the same order)")

    result["metrics"]["schema_valid"] = schema_valid
    if schema_valid != 1.0 or label_col is None:
        return finalize_result(result)

    y_true = [str(value) for value in solution.obs[label_col].tolist()]
    y_pred = [str(value) for value in prediction.obs["label_pred"].tolist()]
    correct = sum(1 for true_value, pred_value in zip(y_true, y_pred) if true_value == pred_value)
    accuracy = correct / len(y_true) if y_true else 0.0
    result["metrics"]["accuracy"] = accuracy
    # OP's f1 metric: sklearn F1 in three flavors, headline = weighted (mirrors src/metrics/f1/script.py).
    result["metrics"].update(_f1_scores(y_true, y_pred))
    result["final_score"] = accuracy

    # OP normalization: rescale accuracy against the stored random/oracle controls (random->0,
    # oracle->1), reported alongside the raw score. No-op if the pack ships no controls.
    controls = load_control_outputs(benchmark, ad)
    if controls:
        cmin = _control_accuracy(controls["random"], y_true)
        cmax = _control_accuracy(controls["oracle"], y_true)
        if cmin is not None and cmax is not None:
            scaled = normalize_to_controls(accuracy, cmin, cmax)
            if scaled is not None:
                result["metrics"]["accuracy_scaled"] = scaled
                result["metrics"]["control_random_score"] = cmin
                result["metrics"]["control_oracle_score"] = cmax
    return finalize_result(result)


def _control_accuracy(control, y_true: list[str]) -> "float | None":
    """Accuracy of a stored control prediction (label_pred, in solution row order) vs the true labels.
    Returns None if the control is malformed, so normalization is skipped rather than wrong."""
    if "label_pred" not in control.obs:
        return None
    y_pred = [str(value) for value in control.obs["label_pred"].tolist()]
    if not y_true or len(y_pred) != len(y_true):
        return None
    return sum(1 for true_value, pred_value in zip(y_true, y_pred) if true_value == pred_value) / len(y_true)


def _f1_scores(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """OP's f1 metric (task_label_projection/src/metrics/f1/script.py): sklearn F1 in macro / micro /
    weighted (headline = weighted). Also returns ``macro_f1`` as an alias for ``f1_macro``."""
    try:
        from sklearn.metrics import f1_score

        scores = {
            f"f1_{avg}": float(f1_score(y_true, y_pred, average=avg, zero_division=0))
            for avg in ("macro", "micro", "weighted")
        }
        scores["macro_f1"] = scores["f1_macro"]
        return scores
    except Exception:  # sklearn unavailable -> fall back to the lightweight macro-F1 helper
        return {"macro_f1": macro_f1(y_true, y_pred)}


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score a label projection run")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(score(args.benchmark, args.run_dir, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
