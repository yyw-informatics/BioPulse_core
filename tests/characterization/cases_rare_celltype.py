"""Characterization cases for the rare_celltype scorer — one per return branch.

Sibling of ``cases_label_projection.py``: each build(tmp) constructs a minimal
(benchmark_dir, run_dir) that triggers exactly one return path in rare_celltype_score.score.
Fixtures are deterministic (fixed labels/names, no RNG) so the snapshots are stable.

rare_celltype is BioPulse-derived (is_openproblems:false): it ships no controls and, like its
label-projection sibling, requires prediction and solution obs_names to match. The headline
``final_score`` is macro-F1, and it also reports rare-class recall/F1 over classes below
RARE_THRESHOLD (2%). The happy path therefore splits in two:
classes with a rare population (n_rare_classes>0, rare metrics populated) vs. all-common classes
(``_rare_class_metrics`` returns the n_rare_classes=0.0 short-circuit).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from charsnap import Case

from biopulse.scorers.rare_celltype_score import score

# --- builders ---------------------------------------------------------------------------------------


def _pack(tmp: Path) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        "task_id: char_rare_celltype\ntask_type: rare_celltype\n", encoding="utf-8"
    )
    return pack


def _run(tmp: Path, *, with_report: bool = True) -> Path:
    out = tmp / "run" / "workspace" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if with_report:
        (out / "report.md").write_text("approach + limitations\n", encoding="utf-8")
    return tmp / "run"


def _write_solution(pack: Path, labels, *, col: str = "label", names=None) -> None:
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    sol = ad.AnnData(X=np.zeros((len(labels), 1), dtype=np.float32))
    sol.obs[col] = list(labels)
    if names is not None:
        sol.obs_names = list(names)
    sol.write_h5ad(truth / "solution.h5ad")


def _write_pred(run: Path, preds, *, names=None, with_method: bool = True) -> None:
    a = ad.AnnData(X=np.zeros((len(preds), 1), dtype=np.float32))
    a.obs["label_pred"] = list(preds)
    if with_method:
        a.uns["method_id"] = "char_method"
    if names is not None:
        a.obs_names = list(names)
    a.write_h5ad(run / "workspace" / "outputs" / "prediction.h5ad")


def _names(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


# --- branches ---------------------------------------------------------------------------------------


def _missing_prediction(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    return pack, _run(tmp)  # no prediction.h5ad written


def _missing_solution(tmp: Path):
    pack = _pack(tmp)
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"])
    return pack, run  # no hidden solution written


def _missing_label_pred(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    run = _run(tmp)
    a = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    a.uns["method_id"] = "char_method"  # has method_id but no obs['label_pred']
    a.obs_names = ["c0", "c1"]
    a.write_h5ad(run / "workspace" / "outputs" / "prediction.h5ad")
    return pack, run


def _missing_method_id(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"], with_method=False)
    return pack, run


def _solution_missing_label_col(tmp: Path):
    pack = _pack(tmp)
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    sol = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    sol.obs["other"] = ["x", "y"]  # neither 'label' nor 'cell_type'
    sol.obs_names = ["c0", "c1"]
    sol.write_h5ad(truth / "solution.h5ad")
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"])
    return pack, run


def _cell_count_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B", "C"], names=["c0", "c1", "c2"])
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"])  # 2 vs 3
    return pack, run


def _obs_names_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c1", "c0"])  # same count, cells reordered
    return pack, run


def _happy_with_rare(tmp: Path):
    # 51 cells: 'A' x25, 'B' x25, 'R' x1. R is 1/51 = 1.96% < 2% -> rare. n_rare_classes>0.
    names = _names(51)
    true = (["A"] * 25) + (["B"] * 25) + ["R"]
    pred = (["A"] * 25) + (["B"] * 24) + ["A"] + ["R"]  # one B wrong, R correct
    _write_solution(pack := _pack(tmp), true, names=names)
    run = _run(tmp)
    _write_pred(run, pred, names=names)
    return pack, run


def _happy_no_rare(tmp: Path):
    # 4 balanced classes, 13 cells each (52 total): every class is 13/52 = 25% >= 2%.
    # _rare_class_metrics hits the `if not rare` short-circuit -> n_rare_classes=0.0 only.
    classes = ["A", "B", "C", "D"]
    true: list[str] = []
    for cls in classes:
        true.extend([cls] * 13)
    names = _names(len(true))
    pred = list(true)
    pred[0] = "B"  # one wrong so metrics are not a trivial 1.0
    _write_solution(pack := _pack(tmp), true, names=names)
    run = _run(tmp)
    _write_pred(run, pred, names=names)
    return pack, run


def _label_col_celltype(tmp: Path):
    # alternate label column 'cell_type'; 51 cells with a rare class so it reaches the happy path.
    names = _names(51)
    true = (["A"] * 25) + (["B"] * 25) + ["R"]
    pred = list(true)
    _write_solution(pack := _pack(tmp), true, col="cell_type", names=names)
    run = _run(tmp)
    _write_pred(run, pred, names=names)
    return pack, run


def _safety_gate_triggered(tmp: Path):
    names = _names(51)
    true = (["A"] * 25) + (["B"] * 25) + ["R"]
    _write_solution(pack := _pack(tmp), true, names=names)
    run = _run(tmp)
    _write_pred(run, list(true), names=names)
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # forbidden answer artifact
    return pack, run


def _report_missing(tmp: Path):
    # valid schema + happy path, but no report.md (report_present is a metric, not a gate).
    names = _names(51)
    true = (["A"] * 25) + (["B"] * 25) + ["R"]
    _write_solution(pack := _pack(tmp), true, names=names)
    run = _run(tmp, with_report=False)
    _write_pred(run, list(true), names=names)
    return pack, run


CASES = [
    Case("rare_celltype__missing_prediction", score, _missing_prediction),
    Case("rare_celltype__missing_solution", score, _missing_solution),
    Case("rare_celltype__missing_label_pred", score, _missing_label_pred),
    Case("rare_celltype__missing_method_id", score, _missing_method_id),
    Case("rare_celltype__solution_missing_label_col", score, _solution_missing_label_col),
    Case("rare_celltype__cell_count_mismatch", score, _cell_count_mismatch),
    Case("rare_celltype__obs_names_mismatch", score, _obs_names_mismatch),
    Case("rare_celltype__happy_with_rare", score, _happy_with_rare),
    Case("rare_celltype__happy_no_rare", score, _happy_no_rare),
    Case("rare_celltype__label_col_celltype", score, _label_col_celltype),
    Case("rare_celltype__safety_gate_triggered", score, _safety_gate_triggered),
    Case("rare_celltype__report_missing", score, _report_missing),
]
