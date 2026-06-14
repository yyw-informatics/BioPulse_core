"""Coverage for the scorer happy paths + the safety gate.

The existing scorer tests mostly hit the missing-output / schema-failure branches. These exercise the
*passing* paths — a real prediction that scores an exact, known value — and the contamination safety
gate that gates every run's `passed` flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ad = pytest.importorskip("anndata")
np = pytest.importorskip("numpy")

from biopulse.scorers.common import scan_workspace_safety
from biopulse.scorers.label_projection_score import score as score_lp
from biopulse.scorers.rare_celltype_score import score as score_rare
from biopulse.scorers.spatially_variable_genes_score import score as score_svg


def _task(pack: Path, task_type: str) -> None:
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "task.yaml").write_text(f"task_id: test_{task_type}\ntask_type: {task_type}\n", encoding="utf-8")


def _run(tmp: Path) -> Path:
    out = tmp / "run" / "workspace" / "outputs"
    out.mkdir(parents=True)
    (out / "report.md").write_text("approach + limitations\n", encoding="utf-8")
    return tmp / "run"


def _labelled_solution(pack: Path, labels: list[str]) -> None:
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    sol = ad.AnnData(X=np.zeros((len(labels), 1), dtype=np.float32))
    sol.obs["label"] = labels
    sol.write_h5ad(truth / "solution.h5ad")


def _write_pred(run_dir: Path, preds: list[str]) -> None:
    a = ad.AnnData(X=np.zeros((len(preds), 1), dtype=np.float32))
    a.obs["label_pred"] = preds
    a.uns["method_id"] = "test"
    a.write_h5ad(run_dir / "workspace" / "outputs" / "prediction.h5ad")


def test_label_projection_happy_path_exact_accuracy(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _task(pack, "label_projection")
    _labelled_solution(pack, ["A", "B", "A", "C"])
    run_dir = _run(tmp_path)
    _write_pred(run_dir, ["A", "B", "A", "X"])  # 3 of 4 correct

    result = score_lp(pack, run_dir)
    assert result["final_score"] == pytest.approx(0.75)
    assert result["passed"] is True
    assert result["metrics"]["schema_valid"] == 1.0


def test_rare_celltype_real_beats_majority(tmp_path: Path) -> None:
    # Imbalanced labels: one common class + two rare ones. Macro-F1 must reward catching the rare ones.
    truth = ["common"] * 8 + ["rare1", "rare2"]
    pack = tmp_path / "pack"
    _task(pack, "rare_celltype")
    _labelled_solution(pack, truth)

    good = _run(tmp_path / "good")
    _write_pred(good, truth)  # perfect
    maj = _run(tmp_path / "maj")
    _write_pred(maj, ["common"] * 10)  # ignores rare classes

    good_f1 = score_rare(pack, good)["final_score"]
    maj_f1 = score_rare(pack, maj)["final_score"]
    assert good_f1 > maj_f1
    assert maj_f1 < 0.5  # majority-only collapses macro-F1


def test_svg_perfect_ranking_beats_random(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    n = 30
    true_scores = rng.rand(n)
    pack = tmp_path / "pack"
    _task(pack, "spatially_variable_genes")
    truth = pack / "hidden" / "ground_truth"
    truth.mkdir(parents=True)
    sol = ad.AnnData(X=np.zeros((1, n), dtype=np.float32))
    sol.var["true_spatial_var_score"] = true_scores
    sol.write_h5ad(truth / "solution.h5ad")

    def run_with(scores) -> dict:
        run_dir = _run(tmp_path / f"r{rng.randint(1_000_000)}")
        a = ad.AnnData(X=np.zeros((1, n), dtype=np.float32))
        a.var_names = sol.var_names
        a.var["pred_spatial_var_score"] = np.asarray(scores, dtype=float)
        a.uns["method_id"] = "test"
        a.write_h5ad(run_dir / "workspace" / "outputs" / "output.h5ad")
        return score_svg(pack, run_dir)

    perfect = run_with(true_scores)
    random = run_with(rng.rand(n))
    assert perfect["metrics"]["kendall_tau"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["final_score"] > random["final_score"]


@pytest.mark.external_data
def test_label_projection_reproduces_op_golden(tmp_path: Path) -> None:
    """Our LP scorer must reproduce OP's shipped accuracy (0.4969) and OP's headline weighted F1
    (0.4858) on cxg_immune_cell_atlas, computed sklearn's way like task_label_projection/src/metrics."""
    import shutil

    pack = Path("benchmark_packs/op_label_projection_mini")
    op_pred = Path(
        "external/openproblems/task_label_projection/resources_test/"
        "task_label_projection/cxg_immune_cell_atlas/prediction.h5ad"
    )
    if not (pack.exists() and op_pred.exists()):
        pytest.skip("op_label_projection_mini pack or vendored OP prediction not present")
    run_dir = _run(tmp_path)
    shutil.copy(op_pred, run_dir / "workspace" / "outputs" / "prediction.h5ad")

    r = score_lp(pack, run_dir)
    assert r["metrics"]["accuracy"] == pytest.approx(0.4969, abs=1e-3)      # OP golden
    assert r["metrics"]["f1_weighted"] == pytest.approx(0.4858, abs=1e-3)   # OP's headline F1
    assert r["metrics"]["f1_macro"] == pytest.approx(0.27, abs=1e-2)
    assert r["passed"] is True


@pytest.mark.external_data
def test_svg_reproduces_op_golden_correlation(tmp_path: Path) -> None:
    """Our SVG scorer must reproduce OP's shipped golden `correlation` (0.7225 on mouse_brain_coronal),
    by mirroring OP's grouped-by-orig_feature_name Kendall tau — not a global tau (which gives 0.58)."""
    import shutil

    pack = Path("benchmark_packs/op_svg_mini")
    op_out = Path(
        "external/openproblems/task_spatially_variable_genes/resources_test/"
        "task_spatially_variable_genes/mouse_brain_coronal/output.h5ad"
    )
    if not (pack.exists() and op_out.exists()):
        pytest.skip("op_svg_mini pack or vendored OP output not present")
    run_dir = _run(tmp_path)
    shutil.copy(op_out, run_dir / "workspace" / "outputs" / "output.h5ad")

    r = score_svg(pack, run_dir)
    assert r["metrics"]["correlation"] == pytest.approx(0.7225, abs=1e-3)  # OP golden
    assert r["final_score"] == pytest.approx(0.7225, abs=1e-3)
    assert r["passed"] is True


@pytest.mark.external_data
def test_label_projection_control_normalization(tmp_path: Path) -> None:
    """With the pack's stored controls, the LP scorer reports accuracy_scaled = (acc - random)/(oracle
    - random). true_labels is a perfect oracle (1.0); random_labels floors near the label entropy."""
    import shutil

    pack = Path("benchmark_packs/op_label_projection_mini")
    op_pred = Path(
        "external/openproblems/task_label_projection/resources_test/"
        "task_label_projection/cxg_immune_cell_atlas/prediction.h5ad"
    )
    controls = pack / "baselines" / "baseline_outputs" / "oracle.h5ad"
    if not (pack.exists() and op_pred.exists() and controls.exists()):
        pytest.skip("op_label_projection_mini pack, controls, or vendored OP prediction not present")
    run_dir = _run(tmp_path)
    shutil.copy(op_pred, run_dir / "workspace" / "outputs" / "prediction.h5ad")

    m = score_lp(pack, run_dir)["metrics"]
    assert m["control_oracle_score"] == pytest.approx(1.0)            # true_labels -> perfect
    assert m["control_random_score"] == pytest.approx(0.0491, abs=5e-3)
    assert m["accuracy_scaled"] == pytest.approx(0.471, abs=5e-3)
    assert m["accuracy"] == pytest.approx(0.4969, abs=1e-3)           # raw is unchanged


@pytest.mark.external_data
def test_svg_control_normalization(tmp_path: Path) -> None:
    """With the pack's stored controls, the SVG scorer reports correlation_scaled. true_ranking is a
    perfect oracle (1.0); random_ranking sits near 0, so scaled ~= raw correlation."""
    import shutil

    pack = Path("benchmark_packs/op_svg_mini")
    op_out = Path(
        "external/openproblems/task_spatially_variable_genes/resources_test/"
        "task_spatially_variable_genes/mouse_brain_coronal/output.h5ad"
    )
    controls = pack / "baselines" / "baseline_outputs" / "oracle.h5ad"
    if not (pack.exists() and op_out.exists() and controls.exists()):
        pytest.skip("op_svg_mini pack, controls, or vendored OP output not present")
    run_dir = _run(tmp_path)
    shutil.copy(op_out, run_dir / "workspace" / "outputs" / "output.h5ad")

    m = score_svg(pack, run_dir)["metrics"]
    assert m["control_oracle_score"] == pytest.approx(1.0)            # true_ranking -> perfect
    assert abs(m["control_random_score"]) < 0.05                      # random ranking ~ 0
    assert m["correlation_scaled"] == pytest.approx(0.7241, abs=5e-3)
    assert m["correlation"] == pytest.approx(0.7225, abs=1e-3)        # raw is unchanged


def test_safety_gate_flags_hidden_solution_and_passes_clean(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    (ws / "outputs").mkdir(parents=True)
    (ws / "outputs" / "prediction.h5ad").write_bytes(b"")
    ok, violations = scan_workspace_safety(ws)
    assert ok and not violations  # clean workspace passes

    # An agent that smuggled the answer key in is caught.
    (ws / "solution.h5ad").write_bytes(b"")
    (ws / "hidden").mkdir()
    bad_ok, bad_violations = scan_workspace_safety(ws)
    assert not bad_ok
    assert any("solution" in v for v in bad_violations)
    assert any("hidden" in v for v in bad_violations)
