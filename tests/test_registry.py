"""Task registry: the single source of truth that replaced ~15 task_type dispatch sites.

Pins the record contents (so a wrong FileSpec or a missing record is caught) and the dispatch helpers.
The characterization suite proves the scorers themselves enforce these required fields; here we pin that
the registry's FileSpecs match what each scorer expects (drift guard) and that lookup/aliases behave.
"""

from __future__ import annotations

import pytest

from biopulse.tasks.registry import REGISTRY, get, required_outputs

ALL_TASK_TYPES = {
    "label_projection",
    "spatially_variable_genes",
    "denoising",
    "dimensionality_reduction",
    "rare_celltype",
}

# Full-dataset / derived variants registered outside the canonical auto-discovered set. label_projection_dkd
# is built from a hand-built discovery dict (the full cellxgene_census/dkd dataset) by
# scripts/build_label_projection_dkd_pack.py, so it is not OP-adapter-discovered and is excluded from the
# canonical per-task invariant checks below (which parametrize over ALL_TASK_TYPES).
VARIANT_TASK_TYPES = {"label_projection_dkd", "denoising_immune"}

# The output contract each scorer enforces — the drift guard. If a scorer's required-field check and this
# table disagree, one of them changed without the other.
EXPECTED_OUTPUT_REQUIRED = {
    "label_projection": {"obs": ["label_pred"], "uns": ["method_id"]},
    "spatially_variable_genes": {"var": ["pred_spatial_var_score"], "uns": ["method_id"]},
    "denoising": {"layers": ["denoised"], "uns": ["method_id"]},
    "dimensionality_reduction": {"obsm": ["X_emb"], "uns": ["method_id"]},
    "rare_celltype": {"obs": ["label_pred"], "uns": ["method_id"]},
}

EXPECTED_REQUIRED_OUTPUTS = {
    "label_projection": ["outputs/prediction.h5ad", "outputs/report.md"],
    "spatially_variable_genes": ["outputs/output.h5ad", "outputs/report.md"],
    "denoising": ["outputs/denoised.h5ad", "outputs/report.md"],
    "dimensionality_reduction": ["outputs/embedding.h5ad", "outputs/report.md"],
    "rare_celltype": ["outputs/prediction.h5ad", "outputs/report.md"],
}


def test_registry_covers_every_task_type() -> None:
    assert set(REGISTRY) == ALL_TASK_TYPES | VARIANT_TASK_TYPES


@pytest.mark.parametrize("task_type", sorted(ALL_TASK_TYPES))
def test_record_invariants(task_type: str) -> None:
    record = get(task_type)
    assert record.task_type == task_type
    assert record.task_id and record.title
    assert callable(record.scorer)  # opaque callable
    # every output declares method_id + matches the scorer's enforced contract (drift guard)
    assert record.output.required == EXPECTED_OUTPUT_REQUIRED[task_type]
    assert "method_id" in record.output.required.get("uns", [])
    assert record.solution.filename == "solution.h5ad"


def test_is_openproblems_flag() -> None:
    assert get("rare_celltype").is_openproblems is False
    assert get("rare_celltype").source == "biopulse_derived"
    for task_type in ALL_TASK_TYPES - {"rare_celltype"}:
        assert get(task_type).is_openproblems is True
        assert get(task_type).source == "openproblems"


def test_discover_and_controls_presence() -> None:
    # OP tasks have source discovery; rare_celltype (derived) does not.
    assert get("rare_celltype").discover is None
    for task_type in ALL_TASK_TYPES - {"rare_celltype"}:
        assert callable(get(task_type).discover)
    # local controls exist only for label_projection + spatially_variable_genes
    assert callable(get("label_projection").controls)
    assert callable(get("spatially_variable_genes").controls)
    for task_type in ("denoising", "dimensionality_reduction", "rare_celltype"):
        assert get(task_type).controls is None


@pytest.mark.parametrize("task_type,expected", sorted(EXPECTED_REQUIRED_OUTPUTS.items()))
def test_required_outputs(task_type: str, expected: list[str]) -> None:
    assert required_outputs(task_type) == expected


def test_alias_resolution_and_unknown() -> None:
    assert get("svg").task_type == "spatially_variable_genes"
    assert get("dimred").task_type == "dimensionality_reduction"
    assert get("lp").task_type == "label_projection"
    assert get("denoise").task_type == "denoising"
    with pytest.raises(ValueError):
        get("not_a_task")


def test_registry_holds_scorers_as_opaque_callables() -> None:
    # Architectural invariant: the registry must not carry a metric-iteration / slot-reading surface.
    # A TaskRecord exposes a single `scorer` callable and no `metrics`/`fn` list.
    record = get("label_projection")
    assert not hasattr(record, "metrics")
    assert not hasattr(record, "metric_fns")
