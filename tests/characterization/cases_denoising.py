"""Characterization cases for the denoising scorer.

Each case constructs a minimal ``(benchmark_dir, run_dir)`` pair for one observable scorer outcome.
Fixtures are deterministic, with fixed counts and fixed observation/variable names, so snapshots remain
stable across platforms.

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


def _missing_output(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    return pack, _run(tmp)  # no denoised.h5ad written


def _missing_solution(tmp: Path):
    pack = _pack(tmp)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run  # no hidden solution written


def _missing_denoised_layer(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR, with_denoised=False)
    return pack, run


def _missing_method_id(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR, with_method=False)
    return pack, run


def _solution_missing_counts(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, with_counts=False)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _shape_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)  # 2 x 3
    run = _run(tmp)
    # 2 x 2 prediction: differs in gene count -> shape mismatch branch (evaluated first)
    _write_pred(run, [[1.0, 2.0], [3.0, 4.0]], obs_names=_OBS, var_names=["g0", "g1"])
    return pack, run


def _names_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=["c1", "c0"], var_names=_VAR)  # reversed cells
    return pack, run


def _nonfinite_metric(tmp: Path):
    # Negative values exercise non-finite metric handling after normalization and log1p.
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
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, train_sum=18.0)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _happy_without_train_sum(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR)  # no train_sum
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    return pack, run


def _safety_gate_triggered(tmp: Path):
    # A solution file in the workspace should be treated as a forbidden answer artifact.
    pack = _pack(tmp)
    _write_solution(pack, _COUNTS, obs_names=_OBS, var_names=_VAR, train_sum=18.0)
    run = _run(tmp)
    _write_pred(run, _DENOISED, obs_names=_OBS, var_names=_VAR)
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # forbidden answer artifact
    return pack, run


def _report_missing(tmp: Path):
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
