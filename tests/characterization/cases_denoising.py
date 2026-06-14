"""Characterization cases for the denoising scorer — one per return branch.

Sibling of ``cases_label_projection.py``: each build(tmp) constructs a minimal
(benchmark_dir, run_dir) that triggers exactly one return path in denoising_score.score.
Fixtures are deterministic (fixed counts, fixed obs/var names, no RNG) so the snapshots are
stable.

Required output for this scorer: ``outputs/denoised.h5ad`` with layers['denoised'] + uns['method_id'].
The hidden solution is ``hidden/ground_truth/solution.h5ad`` with layers['counts']; the OP poisson
metric additionally needs solution.uns['train_sum'] (a float). Denoising ships NO controls, so there
is no scaled-metric branch.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from charsnap import Case

from biopulse.scorers.denoising_score import score

# --- builders ---------------------------------------------------------------------------------------


def _pack(tmp: Path) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        "task_id: char_denoising\ntask_type: denoising\n", encoding="utf-8"
    )
    return pack


def _run(tmp: Path, *, with_report: bool = True) -> Path:
    out = tmp / "run" / "workspace" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if with_report:
        (out / "report.md").write_text("approach + limitations\n", encoding="utf-8")
    return tmp / "run"


def _write_solution(
    pack: Path,
    counts,
    *,
    obs_names=None,
    var_names=None,
    train_sum=None,
    with_counts: bool = True,
) -> None:
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts, dtype=np.float32)
    sol = ad.AnnData(X=np.zeros_like(counts))
    if with_counts:
        sol.layers["counts"] = counts
    if obs_names is not None:
        sol.obs_names = list(obs_names)
    if var_names is not None:
        sol.var_names = list(var_names)
    if train_sum is not None:
        sol.uns["train_sum"] = float(train_sum)
    sol.write_h5ad(truth / "solution.h5ad")


def _write_pred(
    run: Path,
    denoised,
    *,
    obs_names=None,
    var_names=None,
    with_denoised: bool = True,
    with_method: bool = True,
) -> None:
    denoised = np.asarray(denoised, dtype=np.float32)
    a = ad.AnnData(X=np.zeros_like(denoised))
    if with_denoised:
        a.layers["denoised"] = denoised
    if with_method:
        a.uns["method_id"] = "char_method"
    if obs_names is not None:
        a.obs_names = list(obs_names)
    if var_names is not None:
        a.var_names = list(var_names)
    a.write_h5ad(run / "workspace" / "outputs" / "denoised.h5ad")


# A small deterministic count matrix (2 cells x 3 genes) reused across the happy/safety branches.
_COUNTS = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
_DENOISED = [[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]]
_OBS = ["c0", "c1"]
_VAR = ["g0", "g1", "g2"]


# --- branches ---------------------------------------------------------------------------------------


def _missing_output(tmp: Path):
    # return @ line 51: outputs/denoised.h5ad does not exist
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    return pack, _run(tmp)  # no denoised.h5ad written


def _missing_solution(tmp: Path):
    # return @ line 55: hidden/ground_truth/solution.h5ad does not exist
    pack = _pack(tmp)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run  # no hidden solution written


def _missing_denoised_layer(tmp: Path):
    # return @ line 93 (schema_valid=0): prediction has no layers['denoised']
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR, with_denoised=False)
    return pack, run


def _missing_method_id(tmp: Path):
    # return @ line 93 (schema_valid=0): prediction has no uns['method_id']
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR, with_method=False)
    return pack, run


def _solution_missing_counts(tmp: Path):
    # return @ line 93 (schema_valid=0): solution has no layers['counts']
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, with_counts=False)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _shape_mismatch(tmp: Path):
    # return @ line 93 (schema_valid=0): prediction shape != solution shape
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)  # 2 x 3
    run = _run(tmp)
    # 2 x 2 prediction: differs in gene count -> shape mismatch branch (evaluated first)
    _write_pred(run, [[1.0, 2.0], [3.0, 4.0]], obs_names=_OBS, var_names=["g0", "g1"])
    return pack, run


def _names_mismatch(tmp: Path):
    # return @ line 93 (schema_valid=0): same shape but reordered obs_names -> rejected
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=["c1", "c0"], var_names=_VAR)  # reversed cells
    return pack, run


def _nonfinite_metric(tmp: Path):
    # return @ line 112: denoised full of negatives -> log1p non-finite -> FAILED run.
    # normalize_total divides by the (negative) library sum then log1p(negative) -> NaN.
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(
        run,
        [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        obs_names=_OBS,
        var_names=_VAR,
    )
    return pack, run


def _happy_with_train_sum(tmp: Path):
    # return @ line 120: valid schema, finite metrics, train_sum present -> mse + poisson both reported
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, train_sum=18.0)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _happy_without_train_sum(tmp: Path):
    # return @ line 120: valid schema, finite metrics, train_sum absent -> mse only (poisson absent)
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)  # no train_sum
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _safety_gate_triggered(tmp: Path):
    # return @ line 120 path but passed=False: schema valid yet a smuggled solution.h5ad sits in the
    # workspace -> safety gate fails with a "Forbidden ... in workspace" violation.
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, train_sum=18.0)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # smuggled answer key
    return pack, run


def _report_missing(tmp: Path):
    # return @ line 120 path: valid schema + finite metrics but no report.md -> report violation
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, train_sum=18.0)
    run = _run(tmp, with_report=False)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


CASES = [
    Case("denoising__missing_output", score, _missing_output),
    Case("denoising__missing_solution", score, _missing_solution),
    Case("denoising__missing_denoised_layer", score, _missing_denoised_layer),
    Case("denoising__missing_method_id", score, _missing_method_id),
    Case("denoising__solution_missing_counts", score, _solution_missing_counts),
    Case("denoising__shape_mismatch", score, _shape_mismatch),
    Case("denoising__names_mismatch", score, _names_mismatch),
    Case("denoising__nonfinite_metric", score, _nonfinite_metric),
    Case("denoising__happy_with_train_sum", score, _happy_with_train_sum),
    Case("denoising__happy_without_train_sum", score, _happy_without_train_sum),
    Case("denoising__safety_gate_triggered", score, _safety_gate_triggered),
    Case("denoising__report_missing", score, _report_missing),
]
