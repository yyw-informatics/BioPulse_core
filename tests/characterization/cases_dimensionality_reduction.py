"""Characterization cases for the dimensionality_reduction scorer — one per return branch.

Sibling of cases_label_projection.py: each build(tmp) constructs a minimal
(benchmark_dir, run_dir) that triggers exactly one return path in
dimensionality_reduction_score.score. Fixtures are deterministic (fixed values, fixed
obs_names/var_names, np.random.RandomState(0) where randomness is unavoidable) so the
snapshots are stable.

Required output: outputs/embedding.h5ad with obsm['X_emb'] (2D, >=2 cols) + uns['method_id'].
Solution: hidden/ground_truth/solution.h5ad with layers['normalized'] (+ obs['cell_type'] for
the silhouette diagnostic). dimred ships NO controls. n is kept small (8-12 cells, 6 genes) so
trustworthiness's k = min(15, n//2-1) stays valid.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from charsnap import Case

from biopulse.scorers.dimensionality_reduction_score import score

# --- builders ---------------------------------------------------------------------------------------

N_CELLS = 10
N_GENES = 6
NAMES = [f"c{i}" for i in range(N_CELLS)]
GENES = [f"g{j}" for j in range(N_GENES)]


def _pack(tmp: Path) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        "task_id: char_dimensionality_reduction\ntask_type: dimensionality_reduction\n",
        encoding="utf-8",
    )
    return pack


def _run(tmp: Path, *, with_report: bool = True) -> Path:
    out = tmp / "run" / "workspace" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if with_report:
        (out / "report.md").write_text("approach + limitations\n", encoding="utf-8")
    return tmp / "run"


def _normalized(n: int = N_CELLS, g: int = N_GENES) -> np.ndarray:
    """Deterministic high-dim normalized matrix; structured (not constant) so trustworthiness
    is well-defined and finite."""
    rng = np.random.RandomState(0)
    return rng.rand(n, g).astype(np.float32)


def _emb(n: int = N_CELLS, cols: int = 2) -> np.ndarray:
    rng = np.random.RandomState(1)
    return rng.rand(n, cols).astype(np.float64)


def _write_solution(
    pack: Path,
    *,
    names=NAMES,
    with_normalized: bool = True,
    with_cell_type: bool = True,
    normalized=None,
) -> None:
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    n = len(names)
    sol = ad.AnnData(X=np.zeros((n, N_GENES), dtype=np.float32))
    sol.obs_names = list(names)
    sol.var_names = list(GENES)
    if with_normalized:
        mat = _normalized(n) if normalized is None else normalized
        sol.layers["normalized"] = mat
    if with_cell_type:
        # two types, deterministic split, both with >=2 members so silhouette is defined
        sol.obs["cell_type"] = ["typeA" if i % 2 == 0 else "typeB" for i in range(n)]
    sol.write_h5ad(truth / "solution.h5ad")


def _write_pred(
    run: Path,
    *,
    names=NAMES,
    emb=None,
    with_emb: bool = True,
    with_method: bool = True,
) -> None:
    n = len(names)
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    a.obs_names = list(names)
    if with_emb:
        a.obsm["X_emb"] = _emb(n) if emb is None else emb
    if with_method:
        a.uns["method_id"] = "char_method"
    a.write_h5ad(run / "workspace" / "outputs" / "embedding.h5ad")


# --- branches ---------------------------------------------------------------------------------------


def _missing_output(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    return pack, _run(tmp)  # no embedding.h5ad written


def _missing_solution(tmp: Path):
    pack = _pack(tmp)
    run = _run(tmp)
    _write_pred(run)
    return pack, run  # no hidden solution written


def _schema_missing_x_emb(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    _write_pred(run, with_emb=False)  # has method_id but no obsm['X_emb']
    return pack, run


def _schema_missing_method_id(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    _write_pred(run, with_method=False)  # has X_emb but no uns['method_id']
    return pack, run


def _schema_missing_normalized(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, with_normalized=False)  # solution has no layers['normalized']
    run = _run(tmp)
    _write_pred(run)
    return pack, run


def _cell_count_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)  # 10 cells
    run = _run(tmp)
    eight = NAMES[:8]
    _write_pred(run, names=eight, emb=_emb(8))  # 8 vs 10
    return pack, run


def _obs_names_mismatch(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    reordered = list(reversed(NAMES))  # same count, reordered names
    _write_pred(run, names=reordered, emb=_emb(N_CELLS))
    return pack, run


def _x_emb_wrong_shape(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    one_col = _emb(N_CELLS, cols=1)  # 2D but only 1 column -> shape[1] < 2
    _write_pred(run, emb=one_col)
    return pack, run


def _x_emb_non_finite(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    bad = _emb(N_CELLS, cols=2).copy()
    bad[0, 0] = np.nan  # non-finite value
    _write_pred(run, emb=bad)
    return pack, run


def _happy_with_cell_type(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, with_cell_type=True)  # trustworthiness + silhouette diagnostic
    run = _run(tmp)
    _write_pred(run)
    return pack, run


def _happy_without_cell_type(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack, with_cell_type=False)  # silhouette skipped -> warning + 0.0
    run = _run(tmp)
    _write_pred(run)
    return pack, run


def _safety_gate_triggered(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp)
    _write_pred(run)
    (run / "workspace" / "solution.h5ad").write_bytes(b"")  # forbidden answer artifact
    return pack, run


def _report_missing(tmp: Path):
    pack = _pack(tmp)
    _write_solution(pack)
    run = _run(tmp, with_report=False)  # valid schema but no report.md
    _write_pred(run)
    return pack, run


CASES = [
    Case("dimensionality_reduction__missing_output", score, _missing_output),
    Case("dimensionality_reduction__missing_solution", score, _missing_solution),
    Case("dimensionality_reduction__schema_missing_x_emb", score, _schema_missing_x_emb),
    Case("dimensionality_reduction__schema_missing_method_id", score, _schema_missing_method_id),
    Case("dimensionality_reduction__schema_missing_normalized", score, _schema_missing_normalized),
    Case("dimensionality_reduction__cell_count_mismatch", score, _cell_count_mismatch),
    Case("dimensionality_reduction__obs_names_mismatch", score, _obs_names_mismatch),
    Case("dimensionality_reduction__x_emb_wrong_shape", score, _x_emb_wrong_shape),
    Case("dimensionality_reduction__x_emb_non_finite", score, _x_emb_non_finite),
    Case("dimensionality_reduction__happy_with_cell_type", score, _happy_with_cell_type),
    Case("dimensionality_reduction__happy_without_cell_type", score, _happy_without_cell_type),
    Case("dimensionality_reduction__safety_gate_triggered", score, _safety_gate_triggered),
    Case("dimensionality_reduction__report_missing", score, _report_missing),
]
