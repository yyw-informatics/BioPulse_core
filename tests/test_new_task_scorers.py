"""Scorer tests for the two added Open Problems tasks: denoising and dimensionality_reduction.

The denoising tests pin a real bug that was caught during bring-up: a denoised matrix with negative
values (any PCA/linear reconstruction) made log1p produce NaN, which the number-cleaner turned into
0.0 — i.e. a *perfect* MSE that also passed. A non-finite score must never masquerade as the best
possible result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ad = pytest.importorskip("anndata")
np = pytest.importorskip("numpy")

from biopulse.scorers.denoising_score import score as score_denoising
from biopulse.scorers.dimensionality_reduction_score import score as score_dimred


def _task_yaml(pack: Path, task_type: str) -> None:
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(
        f"task_id: test_{task_type}\ntask_type: {task_type}\n", encoding="utf-8"
    )


def _run_with_outputs(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    outputs = run_dir / "workspace" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "report.md").write_text("method and limitations\n", encoding="utf-8")
    return run_dir


def _write_denoised(run_dir: Path, matrix) -> None:
    out = run_dir / "workspace" / "outputs"
    a = ad.AnnData(X=np.zeros(matrix.shape, dtype=np.float32))
    a.layers["denoised"] = np.asarray(matrix, dtype=np.float64)
    a.uns["method_id"] = "test"
    a.write_h5ad(out / "denoised.h5ad")


def _denoising_pack(tmp_path: Path, test_counts) -> Path:
    pack = tmp_path / "pack"
    _task_yaml(pack, "denoising")
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    sol = ad.AnnData(X=np.zeros(test_counts.shape, dtype=np.float32))
    sol.layers["counts"] = np.asarray(test_counts, dtype=np.float64)
    sol.write_h5ad(truth / "solution.h5ad")
    return pack


def test_denoising_negative_values_do_not_score_as_perfect(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    test_counts = rng.poisson(0.5, size=(12, 8)).astype(float)
    pack = _denoising_pack(tmp_path, test_counts)

    run_dir = _run_with_outputs(tmp_path)
    # A reconstruction full of negatives — the shape that triggered the original fake-perfect bug.
    denoised = np.full((12, 8), -1.0)
    _write_denoised(run_dir, denoised)

    result = score_denoising(pack, run_dir)
    # OP does not clip; negatives make the metric non-finite, which is a FAILED run — never a perfect 0.0.
    assert result["passed"] is False
    assert any("non-finite" in v for v in result["violations"])


def test_denoising_non_finite_fails_not_passes(tmp_path: Path) -> None:
    test_counts = np.ones((6, 4), dtype=float)
    pack = _denoising_pack(tmp_path, test_counts)
    run_dir = _run_with_outputs(tmp_path)
    bad = np.full((6, 4), np.inf)
    _write_denoised(run_dir, bad)

    result = score_denoising(pack, run_dir)
    assert result["passed"] is False
    assert any("non-finite" in v for v in result["violations"])


def test_denoising_rejects_reordered_cells(tmp_path: Path) -> None:
    """Position-based alignment is only safe if cells/genes line up. A prediction whose cells carry the
    same barcodes in a DIFFERENT order must fail schema (OP asserts obs/var-name identity)."""
    test_counts = np.arange(24, dtype=float).reshape(6, 4)
    pack = _denoising_pack(tmp_path, test_counts)
    sol_path = pack / "hidden" / "ground_truth" / "solution.h5ad"
    sol = ad.read_h5ad(sol_path)
    sol.obs_names = [f"cell{i}" for i in range(6)]
    sol.write_h5ad(sol_path)

    run_dir = _run_with_outputs(tmp_path)
    out = run_dir / "workspace" / "outputs"
    a = ad.AnnData(X=np.zeros((6, 4), dtype=np.float32))
    a.layers["denoised"] = test_counts.copy()
    a.uns["method_id"] = "test"
    a.obs_names = [f"cell{i}" for i in reversed(range(6))]  # same cells, reversed order
    a.write_h5ad(out / "denoised.h5ad")

    result = score_denoising(pack, run_dir)
    assert result["metrics"]["schema_valid"] == 0.0
    assert any("same order" in v for v in result["violations"])


def test_denoising_closer_matrix_scores_better(tmp_path: Path) -> None:
    rng = np.random.RandomState(1)
    test_counts = rng.poisson(1.0, size=(15, 10)).astype(float)
    pack = _denoising_pack(tmp_path, test_counts)

    run_close = _run_with_outputs(tmp_path / "a")
    _write_denoised(run_close, test_counts.copy())  # perfect
    run_far = _run_with_outputs(tmp_path / "b")
    _write_denoised(run_far, np.zeros_like(test_counts))  # all zeros

    close = score_denoising(pack, run_close)["metrics"]["mse"]
    far = score_denoising(pack, run_far)["metrics"]["mse"]
    assert close < far


@pytest.mark.external_data
def test_denoising_reproduces_op_golden_poisson(tmp_path: Path) -> None:
    """Our denoising scorer must reproduce OP's shipped golden poisson (0.3113 on cxg_immune_cell_atlas),
    mirroring task_denoising/src/metrics/{mse,poisson} in plain numpy."""
    import shutil

    pack = Path("benchmark_packs/op_denoising_mini")
    op_den = Path(
        "external/openproblems/task_denoising/resources_test/"
        "task_denoising/cxg_immune_cell_atlas/denoised.h5ad"
    )
    if not (pack.exists() and op_den.exists()):
        pytest.skip("op_denoising_mini pack or vendored OP denoised output not present")
    run_dir = _run_with_outputs(tmp_path)
    shutil.copy(op_den, run_dir / "workspace" / "outputs" / "denoised.h5ad")

    r = score_denoising(pack, run_dir)
    assert r["metrics"]["poisson"] == pytest.approx(0.3113, abs=1e-3)  # OP golden
    assert r["passed"] is True


@pytest.mark.external_data
def test_dimred_reproduces_op_trustworthiness(tmp_path: Path) -> None:
    """Our dimred scorer must reproduce OP's exact trustworthiness call (sklearn, n_neighbors=15,
    high-dim = solution.layers['normalized']). 0.5993 on cxg_mouse_pancreas_atlas — faithful-by-
    construction (OP's exact call; the vendored golden recorded NMI/ARI, which need leidenalg)."""
    import shutil

    pack = Path("benchmark_packs/op_dimensionality_reduction_mini")
    op_emb = Path(
        "external/openproblems/task_dimensionality_reduction/resources_test/"
        "task_dimensionality_reduction/cxg_mouse_pancreas_atlas/embedding.h5ad"
    )
    if not (pack.exists() and op_emb.exists()):
        pytest.skip("op_dimensionality_reduction_mini pack or vendored OP embedding not present")
    run_dir = _run_with_outputs(tmp_path)
    shutil.copy(op_emb, run_dir / "workspace" / "outputs" / "embedding.h5ad")

    r = score_dimred(pack, run_dir)
    assert r["metrics"]["trustworthiness"] == pytest.approx(0.5993, abs=2e-3)
    assert r["passed"] is True


def test_dimred_rejects_reordered_cells(tmp_path: Path) -> None:
    """An embedding whose cells are in a different order than the solution must fail schema — otherwise
    trustworthiness silently compares row i of X_emb to the wrong cell's high-dim neighborhood."""
    rng = np.random.RandomState(3)
    high = rng.normal(size=(10, 6))

    pack = tmp_path / "pack"
    _task_yaml(pack, "dimensionality_reduction")
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    sol = ad.AnnData(X=np.zeros((10, 6), dtype=np.float32))
    sol.layers["normalized"] = high.astype(np.float64)
    sol.obs["cell_type"] = np.repeat(["a", "b"], 5)
    sol.obs_names = [f"c{i}" for i in range(10)]
    sol.write_h5ad(truth / "solution.h5ad")

    run_dir = _run_with_outputs(tmp_path)
    a = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    a.obsm["X_emb"] = rng.normal(size=(10, 2))
    a.uns["method_id"] = "test"
    a.obs_names = [f"c{i}" for i in reversed(range(10))]  # same cells, reversed order
    a.write_h5ad(run_dir / "workspace" / "outputs" / "embedding.h5ad")

    result = score_dimred(pack, run_dir)
    assert result["metrics"]["schema_valid"] == 0.0
    assert any("same order" in v for v in result["violations"])


def test_dimred_pca_beats_random(tmp_path: Path) -> None:
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(2)
    # Three separated clusters in 12-dim normalized space.
    blocks = [rng.normal(loc, 0.2, size=(20, 12)) for loc in (0.0, 5.0, 10.0)]
    high = np.vstack(blocks)
    labels = np.repeat(["a", "b", "c"], 20)

    pack = tmp_path / "pack"
    _task_yaml(pack, "dimensionality_reduction")
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    sol = ad.AnnData(X=np.zeros(high.shape, dtype=np.float32))
    sol.layers["normalized"] = high.astype(np.float64)
    sol.obs["cell_type"] = labels
    sol.write_h5ad(truth / "solution.h5ad")

    def run_emb(name: str, emb) -> dict:
        run_dir = _run_with_outputs(tmp_path / name)
        a = ad.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32))
        a.obsm["X_emb"] = np.asarray(emb, dtype=np.float64)
        a.uns["method_id"] = "test"
        a.write_h5ad(run_dir / "workspace" / "outputs" / "embedding.h5ad")
        return score_dimred(pack, run_dir)

    pca = run_emb("pca", PCA(n_components=2, random_state=0).fit_transform(high))
    random = run_emb("rand", rng.normal(size=(60, 2)))

    assert pca["passed"] and random["passed"]
    assert 0.0 <= pca["metrics"]["trustworthiness"] <= 1.0
    assert pca["metrics"]["trustworthiness"] > random["metrics"]["trustworthiness"]
