"""Rare cell-type annotation scorer.

Same data + prediction format as label projection (predict ``obs['label_pred']`` for every test
cell), but scored for **rare-population sensitivity**: the headline ``final_score`` is **macro-F1**
(every cell type weighted equally, so missing rare types hurts as much as missing common ones --
the opposite of accuracy, which a model can inflate by nailing only the common classes). It also
reports rare-class recall / F1 over the classes below a frequency threshold.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from .common import (
    base_result,
    finalize_result,
    macro_f1,
    outputs_for_run,
    read_task_yaml,
    report_present,
    require_anndata,
    scan_output_text_for_forbidden_refs,
    scan_workspace_safety,
    workspace_for_run,
)

RARE_THRESHOLD = 0.02  # a class is "rare" if it is < 2% of the test cells


def _rare_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    counts = Counter(y_true)
    rare = [cls for cls, k in counts.items() if n and k / n < RARE_THRESHOLD]
    if not rare:
        return {"n_rare_classes": 0.0, "rare_class_recall": 0.0, "rare_class_f1": 0.0}
    recalls, f1s = [], []
    for cls in rare:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(recall)
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return {
        "n_rare_classes": float(len(rare)),
        "rare_class_recall": sum(recalls) / len(recalls),
        "rare_class_f1": sum(f1s) / len(f1s),
    }


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
        result["violations"].append(f"Prediction cell count {prediction.n_obs} != solution {solution.n_obs}")
    elif not (prediction.obs_names == solution.obs_names).all():
        # Predictions are scored against the solution by position, so the cells must be in the same
        # order — mirrors the obs_names check in label_projection_score.
        schema_valid = 0.0
        result["violations"].append(
            "Prediction obs_names do not match solution obs_names (cells must be in the same order)"
        )

    result["metrics"]["schema_valid"] = schema_valid
    if schema_valid != 1.0 or label_col is None:
        return finalize_result(result)

    y_true = [str(value) for value in solution.obs[label_col].tolist()]
    y_pred = [str(value) for value in prediction.obs["label_pred"].tolist()]
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    result["metrics"]["accuracy"] = correct / len(y_true) if y_true else 0.0
    result["metrics"]["macro_f1"] = macro_f1(y_true, y_pred)
    result["metrics"].update(_rare_class_metrics(y_true, y_pred))
    # Headline = macro-F1: rewards getting rare populations right, not just the common ones.
    result["final_score"] = result["metrics"]["macro_f1"]
    return finalize_result(result)
