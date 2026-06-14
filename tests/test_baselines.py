"""End-to-end tests for the deterministic baselines and the run_baseline harness.

Builds a self-contained synthetic label-projection pack (no Open Problems data required) and runs the
baselines through the full pipeline, asserting both the scoring outcome and the provenance artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("anndata")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from biopulse.runner.run_baseline import BASELINE_SCRIPTS, repo_root, run_baseline  # noqa: E402


def test_every_baseline_script_exists() -> None:
    """Guards against the runner referencing a baseline script that is not shipped."""
    for filename in BASELINE_SCRIPTS.values():
        assert (repo_root() / "baselines" / filename).is_file(), filename


def _synthetic_label_projection_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    (pack / "public" / "input").mkdir(parents=True)
    (pack / "hidden" / "ground_truth").mkdir(parents=True)
    (pack / "task.yaml").write_text(
        "task_id: synthetic_label_projection\ntask_type: label_projection\n", encoding="utf-8"
    )
    (pack / "public" / "instruction.md").write_text("Predict the test labels.\n", encoding="utf-8")

    rng = np.random.RandomState(0)
    genes = [f"g{i}" for i in range(8)]

    def _block(labels: list[str], names: list[str]):
        a = ad.AnnData(X=rng.rand(len(labels), len(genes)).astype(np.float32),
                       var=pd.DataFrame(index=genes))
        a.obs_names = names
        a.obs["label"] = labels
        return a

    train = _block((["A"] * 20) + (["B"] * 20), [f"tr{i}" for i in range(40)])
    test_names = [f"te{i}" for i in range(10)]
    test_labels = (["A"] * 5) + (["B"] * 5)
    test = _block(test_labels, test_names)
    del test.obs["label"]  # the test split is unlabelled
    solution = _block(test_labels, test_names)

    train.write_h5ad(pack / "public" / "input" / "train.h5ad")
    test.write_h5ad(pack / "public" / "input" / "test.h5ad")
    solution.write_h5ad(pack / "hidden" / "ground_truth" / "solution.h5ad")
    return pack


_PROVENANCE_ARTIFACTS = (
    "run_manifest.json",
    "agent_profile.json",
    "token_usage.json",
    "cost_summary.json",
    "evidence_bundle.json",
    "evaluator_results.json",
)


def test_majority_baseline_scores_and_emits_provenance(tmp_path: Path) -> None:
    pack = _synthetic_label_projection_pack(tmp_path)
    result = run_baseline(pack, "label_projection_majority", "run1", runs_dir=tmp_path / "runs")

    assert result["safety_gate_passed"] is True
    assert result["metrics"]["schema_valid"] == 1.0
    assert result["passed"] is True

    run_dir = tmp_path / "runs" / "run1"
    for artifact in _PROVENANCE_ARTIFACTS:
        assert (run_dir / artifact).exists(), artifact
    cost = json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))
    assert cost["total_cost_usd"] == 0.0  # deterministic baseline: no LLM, no cost


def test_bad_agent_stub_fails_schema_gate(tmp_path: Path) -> None:
    pack = _synthetic_label_projection_pack(tmp_path)
    result = run_baseline(pack, "bad_agent_stub", "run2", runs_dir=tmp_path / "runs")

    assert result["passed"] is False
    assert any("Missing required output" in violation for violation in result["violations"])
