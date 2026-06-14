"""Characterization cases for the label_projection scorer — one per return branch.

Reference module for the other scorers' case files: each build(tmp) constructs a minimal
(benchmark_dir, run_dir) that triggers exactly one return path in label_projection_score.score.
Fixtures are deterministic (fixed labels/names, no RNG) so the snapshots are stable.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from charsnap import Case

from biopulse.scorers.label_projection_score import score

# --- builders ---------------------------------------------------------------------------------------


def _pack(tmp: Path) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        "task_id: char_label_projection\ntask_type: label_projection\n", encoding="utf-8"
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


def _write_controls(pack: Path, random_preds, oracle_preds, names) -> None:
    base = pack / "baselines" / "baseline_outputs"
    base.mkdir(parents=True, exist_ok=True)
    for fname, preds in (("random", random_preds), ("oracle", oracle_preds)):
        obj = ad.AnnData(
            obs=pd.DataFrame({"label_pred": list(preds)}, index=list(names)),
            uns={"method_id": f"control_{fname}"},
        )
        obj.write_h5ad(base / f"{fname}.h5ad")


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
    _write_pred(run, ["A", "B"], names=["c1", "c0"])  # same count, reversed names
    return pack, run


def _happy_no_controls(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B", "A", "C"], names=["c0", "c1", "c2", "c3"])
    run = _run(tmp)
    _write_pred(run, ["A", "B", "A", "X"], names=["c0", "c1", "c2", "c3"])  # 3/4 correct
    return pack, run


def _happy_with_controls(tmp: Path):
    pack = _pack(tmp)
    names = ["c0", "c1", "c2", "c3"]
    true = ["A", "B", "A", "C"]
    _write_solution(pack, true, names=names)
    _write_controls(pack, random_preds=["A", "A", "A", "A"], oracle_preds=true, names=names)
    run = _run(tmp)
    _write_pred(run, ["A", "B", "A", "X"], names=names)
    return pack, run


def _label_col_celltype(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], col="cell_type", names=["c0", "c1"])  # alternate label column
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"])
    return pack, run


def _safety_gate_triggered(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    run = _run(tmp)
    _write_pred(run, ["A", "B"], names=["c0", "c1"])
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # forbidden answer artifact
    return pack, run


def _report_missing(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["A", "B"], names=["c0", "c1"])
    run = _run(tmp, with_report=False)  # valid schema but no report.md
    _write_pred(run, ["A", "B"], names=["c0", "c1"])
    return pack, run


CASES = [
    Case("label_projection__missing_prediction", score, _missing_prediction),
    Case("label_projection__missing_solution", score, _missing_solution),
    Case("label_projection__missing_label_pred", score, _missing_label_pred),
    Case("label_projection__missing_method_id", score, _missing_method_id),
    Case("label_projection__solution_missing_label_col", score, _solution_missing_label_col),
    Case("label_projection__cell_count_mismatch", score, _cell_count_mismatch),
    Case("label_projection__obs_names_mismatch", score, _obs_names_mismatch),
    Case("label_projection__happy_no_controls", score, _happy_no_controls),
    Case("label_projection__happy_with_controls", score, _happy_with_controls),
    Case("label_projection__label_col_celltype", score, _label_col_celltype),
    Case("label_projection__safety_gate_triggered", score, _safety_gate_triggered),
    Case("label_projection__report_missing", score, _report_missing),
]
