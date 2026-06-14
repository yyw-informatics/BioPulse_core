"""Characterization cases for the spatially_variable_genes scorer — one per return branch.

Sibling of cases_label_projection.py: each build(tmp) constructs a minimal
(benchmark_dir, run_dir) that triggers exactly one return path in
spatially_variable_genes_score.score. Fixtures are deterministic (fixed feature_ids /
scores, no RNG) so the snapshots are stable.

SVG aligns genes by feature_id via a merge (no obs_names / cell-order branch). The headline
final_score is OP's per-orig_feature_name grouped Kendall tau; it falls back to the global
Kendall tau (with a warning) when the solution carries no var['orig_feature_name'].
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from charsnap import Case

from biopulse.scorers.spatially_variable_genes_score import score

# --- builders ---------------------------------------------------------------------------------------


def _pack(tmp: Path) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        "task_id: char_spatially_variable_genes\ntask_type: spatially_variable_genes\n",
        encoding="utf-8",
    )
    return pack


def _run(tmp: Path, *, with_report: bool = True) -> Path:
    out = tmp / "run" / "workspace" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if with_report:
        (out / "report.md").write_text("approach + limitations\n", encoding="utf-8")
    return tmp / "run"


def _svg_anndata(feature_ids, scores, *, col, orig_feature_name=None, method_id=None):
    """One gene per var row, indexed by var_names == feature_ids, carrying a feature_id column."""
    n = len(feature_ids)
    a = ad.AnnData(X=np.zeros((1, n), dtype=np.float32))
    a.var_names = [str(f) for f in feature_ids]
    a.var["feature_id"] = [str(f) for f in feature_ids]
    if col is not None:
        a.var[col] = [float(s) for s in scores]
    if orig_feature_name is not None:
        a.var["orig_feature_name"] = list(orig_feature_name)
    if method_id is not None:
        a.uns["method_id"] = method_id
    return a


def _write_solution(
    pack: Path, feature_ids, scores, *, col="true_spatial_var_score", orig_feature_name=None
) -> None:
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    sol = _svg_anndata(feature_ids, scores, col=col, orig_feature_name=orig_feature_name)
    sol.write_h5ad(truth / "solution.h5ad")


def _write_pred(run: Path, feature_ids, scores, *, col="pred_spatial_var_score", method_id="char_method") -> None:
    pred = _svg_anndata(feature_ids, scores, col=col, method_id=method_id)
    pred.write_h5ad(run / "workspace" / "outputs" / "output.h5ad")


def _write_controls(pack: Path, feature_ids, random_scores, oracle_scores) -> None:
    base = pack / "baselines" / "baseline_outputs"
    base.mkdir(parents=True, exist_ok=True)
    for fname, scores in (("random", random_scores), ("oracle", oracle_scores)):
        obj = _svg_anndata(
            feature_ids, scores, col="pred_spatial_var_score", method_id=f"control_{fname}"
        )
        obj.write_h5ad(base / f"{fname}.h5ad")


# Shared deterministic fixture: 4 genes, 2 orig_feature_name groups of 2 genes each.
_FIDS = ["g0", "g1", "g2", "g3"]
_TRUE = [0.1, 0.9, 0.2, 0.8]
_GROUPS = ["A", "A", "B", "B"]
# Group A concordant, group B discordant w.r.t. true -> grouped corr mean = 0.0.
_PRED = [0.2, 0.95, 0.9, 0.1]


# --- branches ---------------------------------------------------------------------------------------


def _missing_output(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    return pack, _run(tmp)  # no output.h5ad written


def _missing_solution(tmp: Path):
    pack = _pack(tmp)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)
    return pack, run  # no hidden solution written


def _missing_pred_score(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp)
    # has method_id + feature_id but no var['pred_spatial_var_score']
    _write_pred(run, _FIDS, _PRED, col=None)
    return pack, run


def _missing_method_id(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED, method_id=None)
    return pack, run


def _solution_missing_true_score(tmp: Path):
    pack = _pack(tmp)
    # solution has feature_id/orig_feature_name but no true_spatial_var_score
    _write_solution(pack, _FIDS, _TRUE, col=None, orig_feature_name=_GROUPS)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)
    return pack, run


def _no_overlap(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, ["s0", "s1", "s2", "s3"], _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp)
    _write_pred(run, ["p0", "p1", "p2", "p3"], _PRED)  # disjoint feature_ids
    return pack, run


def _happy_grouped_no_controls(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)  # OP grouped correlation = headline; no controls in pack
    return pack, run


def _happy_global_fallback(tmp: Path):
    pack = _pack(tmp)
    # solution lacks orig_feature_name -> falls back to global kendall_tau + emits a warning
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=None)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)
    return pack, run


def _happy_with_controls(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    # oracle = perfectly concordant (corr 1.0); random = fully discordant (corr -1.0);
    # prediction corr 0.0 -> correlation_scaled = (0 - -1)/(1 - -1) = 0.5
    _write_controls(
        pack,
        _FIDS,
        random_scores=[0.9, 0.1, 0.8, 0.2],
        oracle_scores=[0.1, 0.9, 0.2, 0.8],
    )
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)
    return pack, run


def _safety_gate_triggered(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp)
    _write_pred(run, _FIDS, _PRED)
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # forbidden answer artifact
    return pack, run


def _report_missing(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, _FIDS, _TRUE, orig_feature_name=_GROUPS)
    run = _run(tmp, with_report=False)  # valid schema but no report.md
    _write_pred(run, _FIDS, _PRED)
    return pack, run


CASES = [
    Case("spatially_variable_genes__missing_output", score, _missing_output),
    Case("spatially_variable_genes__missing_solution", score, _missing_solution),
    Case("spatially_variable_genes__missing_pred_score", score, _missing_pred_score),
    Case("spatially_variable_genes__missing_method_id", score, _missing_method_id),
    Case("spatially_variable_genes__solution_missing_true_score", score, _solution_missing_true_score),
    Case("spatially_variable_genes__no_overlap", score, _no_overlap),
    Case("spatially_variable_genes__happy_grouped_no_controls", score, _happy_grouped_no_controls),
    Case("spatially_variable_genes__happy_global_fallback", score, _happy_global_fallback),
    Case("spatially_variable_genes__happy_with_controls", score, _happy_with_controls),
    Case("spatially_variable_genes__safety_gate_triggered", score, _safety_gate_triggered),
    Case("spatially_variable_genes__report_missing", score, _report_missing),
]
