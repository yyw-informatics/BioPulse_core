from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_task_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        parsed: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip().strip('"')
        return parsed


def workspace_for_run(run_dir: Path) -> Path:
    workspace = run_dir / "workspace"
    return workspace if workspace.exists() else run_dir


def outputs_for_run(run_dir: Path) -> Path:
    return workspace_for_run(run_dir) / "outputs"


def scan_workspace_safety(workspace: Path) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if not workspace.exists():
        violations.append(f"Workspace missing: {workspace}")
        return False, violations
    for path in workspace.rglob("*"):
        lower_parts = {part.lower() for part in path.parts}
        if "hidden" in lower_parts:
            violations.append(f"Forbidden hidden path in workspace: {path.relative_to(workspace)}")
        if path.name.lower() == "solution.h5ad":
            violations.append(f"Forbidden solution file in workspace: {path.relative_to(workspace)}")
    return not violations, sorted(set(violations))


def scan_output_text_for_forbidden_refs(outputs: Path) -> list[str]:
    violations: list[str] = []
    if not outputs.exists():
        return violations
    forbidden = ["hidden/ground_truth", "hidden\\ground_truth", "solution.h5ad"]
    text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".log"}
    for path in outputs.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for needle in forbidden:
            if needle in text:
                violations.append(f"Forbidden reference {needle!r} found in output file {path.relative_to(outputs)}")
    return sorted(set(violations))


def base_result(task_id: str, run_id: str) -> dict:
    return {
        "task_id": task_id,
        "run_id": run_id,
        "passed": False,
        "final_score": 0.0,
        "safety_gate_passed": False,
        "metrics": {},
        "violations": [],
        "warnings": [],
        "scored_at_utc": now_utc(),
    }


def finalize_result(result: dict) -> dict:
    # dict.fromkeys dedupes while preserving first-seen order.
    violations = list(dict.fromkeys(result.get("violations", [])))
    warnings = list(dict.fromkeys(result.get("warnings", [])))
    result["violations"] = violations
    result["warnings"] = warnings
    # A run passes only on all three: safety gate clear, no violations, valid output schema.
    result["passed"] = bool(result.get("safety_gate_passed")) and not violations and float(result.get("metrics", {}).get("schema_valid", 0.0)) == 1.0
    result["final_score"] = _clean_number(result.get("final_score", 0.0))
    result["metrics"] = {key: _clean_number(value) for key, value in result.get("metrics", {}).items()}
    return result


def require_anndata(result: dict):
    try:
        import anndata as ad

        return ad
    except Exception as exc:
        result["violations"].append(f"anndata is required to score h5ad outputs: {exc}")
        return None


def report_present(outputs: Path, result: dict) -> float:
    report = outputs / "report.md"
    present = 1.0 if report.exists() and report.stat().st_size > 0 else 0.0
    if present == 0.0:
        result["violations"].append("Missing required report: outputs/report.md")
    return present


def macro_f1(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    """Unweighted mean of per-label F1 over the **union** of labels present in y_true ∪ y_pred.

    Every label that appears in either array contributes equally, including labels that appear only in
    the predictions. This is stricter on spurious labels than sklearn's default macro-F1 convention. The
    label-projection scorer uses sklearn when available and keeps this as a dependency-free fallback
    (see ``label_projection_score._f1_scores``).
    """
    true = [str(value) for value in y_true]
    pred = [str(value) for value in y_pred]
    labels = sorted(set(true) | set(pred))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def top_k_overlap(pred_scores: Dict[str, float], true_scores: Dict[str, float], k: int = 50) -> float:
    overlap = set(pred_scores) & set(true_scores)
    if not overlap:
        return 0.0
    k = min(k, len(overlap))
    pred_top = {key for key, _ in sorted(((key, pred_scores[key]) for key in overlap), key=lambda item: item[1], reverse=True)[:k]}
    true_top = {key for key, _ in sorted(((key, true_scores[key]) for key in overlap), key=lambda item: item[1], reverse=True)[:k]}
    return len(pred_top & true_top) / k if k else 0.0


def normalize_to_controls(
    raw: float, random_score: float, oracle_score: float, clamp: bool = False
) -> Optional[float]:
    """Open Problems-style normalization: random control -> 0 and oracle control -> 1.

    This mirrors how Open Problems reports leaderboard scores (a separate stage after metric
    computation, using the task's control_methods as anchors). Returns None when the controls are
    degenerate (oracle == random), so the caller reports raw only. OP does not clamp — a method worse
    than random scores < 0 and one beating the oracle scores > 1 — so ``clamp`` defaults to False.
    """
    denom = float(oracle_score) - float(random_score)
    if denom == 0:
        return None
    scaled = (float(raw) - float(random_score)) / denom
    if clamp:
        scaled = max(0.0, min(1.0, scaled))
    return float(scaled)


def load_control_outputs(benchmark: Path | str, anndata_module: Any) -> Dict[str, Any]:
    """Read a pack's stored control predictions (``baselines/baseline_outputs/{random,oracle}.h5ad``).

    These are OP control_methods outputs, generated deterministically at pack-build time (see
    ``biopulse.builder.controls``). Returns ``{"random": ad, "oracle": ad}`` only when BOTH are
    present (both anchors are needed to normalize); returns ``{}`` for packs that ship no controls
    (denoising / dimred), so those scorers report raw only.
    """
    base = Path(benchmark) / "baselines" / "baseline_outputs"
    found: Dict[str, Any] = {}
    for name in ("random", "oracle"):
        path = base / f"{name}.h5ad"
        if path.exists():
            found[name] = anndata_module.read_h5ad(path)
    return found if len(found) == 2 else {}


def spearman_fallback(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    try:
        import pandas as pd

        rx = pd.Series(xs).rank(method="average").to_numpy(dtype=float)
        ry = pd.Series(ys).rank(method="average").to_numpy(dtype=float)
        return pearson(rx.tolist(), ry.tolist())
    except Exception:
        return None


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _clean_number(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric
