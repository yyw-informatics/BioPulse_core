"""Control-normalization: the OP random->0 / oracle->1 rescaling and its deterministic controls.

Covers the pure normalize_to_controls math, the control loader's both-or-nothing contract, and the
deterministic OP control generator (self-contained, no vendored data needed). The end-to-end normalized
scores on real OP golden outputs are pinned in test_scorer_coverage.py (skip-guarded on the packs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biopulse.scorers.common import load_control_outputs, normalize_to_controls

ad = pytest.importorskip("anndata")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from biopulse.builder.controls import CONTROL_SEED, write_controls


def test_normalize_to_controls_basic() -> None:
    assert normalize_to_controls(0.5, 0.0, 1.0) == pytest.approx(0.5)
    # random anchor -> 0, oracle anchor -> 1
    assert normalize_to_controls(0.0491, 0.0491, 1.0) == pytest.approx(0.0)
    assert normalize_to_controls(1.0, 0.0491, 1.0) == pytest.approx(1.0)
    # a negative random anchor (SVG) shifts the scale
    assert normalize_to_controls(0.7225, -0.0059, 1.0) == pytest.approx(0.7241, abs=1e-3)


def test_normalize_to_controls_degenerate_returns_none() -> None:
    assert normalize_to_controls(0.5, 0.3, 0.3) is None  # oracle == random -> undefined


def test_normalize_to_controls_clamp() -> None:
    # OP does not clamp by default: worse-than-random < 0, better-than-oracle > 1
    assert normalize_to_controls(-0.2, 0.0, 1.0) == pytest.approx(-0.2)
    assert normalize_to_controls(1.5, 0.0, 1.0) == pytest.approx(1.5)
    # clamp=True pins to [0, 1]
    assert normalize_to_controls(-0.2, 0.0, 1.0, clamp=True) == 0.0
    assert normalize_to_controls(1.5, 0.0, 1.0, clamp=True) == 1.0


def _tiny_svg_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    (pack).mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text("task_id: t\ntask_type: spatially_variable_genes\n", encoding="utf-8")
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    n = 20
    ids = [f"g{i}" for i in range(n)]
    var = pd.DataFrame(
        {"feature_id": ids, "true_spatial_var_score": np.linspace(0.0, 1.0, n), "orig_feature_name": ids},
        index=ids,
    )
    sol = ad.AnnData(X=np.zeros((1, n), dtype=np.float32), var=var)
    sol.write_h5ad(truth / "solution.h5ad")
    return pack


def test_svg_controls_are_deterministic_and_anchored(tmp_path: Path) -> None:
    pack_a = _tiny_svg_pack(tmp_path / "a")
    pack_b = _tiny_svg_pack(tmp_path / "b")
    write_controls(pack_a)
    write_controls(pack_b)

    ra = ad.read_h5ad(pack_a / "baselines" / "baseline_outputs" / "random.h5ad")
    rb = ad.read_h5ad(pack_b / "baselines" / "baseline_outputs" / "random.h5ad")
    # Same seed -> byte-for-byte identical random control across builds.
    assert np.allclose(ra.var["pred_spatial_var_score"].to_numpy(), rb.var["pred_spatial_var_score"].to_numpy())
    # Reproduces a fresh RandomState(CONTROL_SEED).rand(n).
    assert np.allclose(ra.var["pred_spatial_var_score"].to_numpy(), np.random.RandomState(CONTROL_SEED).rand(20))

    # Oracle control == the true scores (so it scores a perfect 1.0 downstream).
    oracle = ad.read_h5ad(pack_a / "baselines" / "baseline_outputs" / "oracle.h5ad")
    assert np.allclose(oracle.var["pred_spatial_var_score"].to_numpy(), np.linspace(0.0, 1.0, 20))


def test_write_controls_noop_for_tasks_without_controls(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "task.yaml").write_text("task_id: t\ntask_type: denoising\n", encoding="utf-8")
    assert write_controls(pack) == []  # denoising/dimred ship no local controls


def test_load_control_outputs_requires_both(tmp_path: Path) -> None:
    base = tmp_path / "baselines" / "baseline_outputs"
    base.mkdir(parents=True)
    # only one anchor present -> normalization is impossible, so loader returns {}
    ad.AnnData(X=np.zeros((1, 1), dtype=np.float32)).write_h5ad(base / "random.h5ad")
    assert load_control_outputs(tmp_path, ad) == {}
    ad.AnnData(X=np.zeros((1, 1), dtype=np.float32)).write_h5ad(base / "oracle.h5ad")
    loaded = load_control_outputs(tmp_path, ad)
    assert set(loaded) == {"random", "oracle"}
