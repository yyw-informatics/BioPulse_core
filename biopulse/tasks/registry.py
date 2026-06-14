"""Task registry: one ``TaskRecord`` per ``task_type``.

Each record is the single source of truth for task metadata that would otherwise be scattered across
``task_type ==`` branches: pack identity, public input files, output and solution contracts, scorer,
Open Problems source discovery, control generation, and the task-facing instruction/schema/config text.
Consumers (``run_baseline.score_run``, ``validate_pack``, ``pack_writer``,
``openproblems_adapter.discover_task``, ``controls.write_controls``) perform a registry lookup instead
of branching on task type.

Invariant: the registry stores scorer/discovery/control functions as callables and never introspects
their internals. Scoring remains task-specific rather than being routed through a generic metric driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from biopulse.builder import controls as _controls
from biopulse.builder.openproblems_adapter import (
    discover_denoising,
    discover_dimensionality_reduction,
    discover_label_projection,
    discover_spatially_variable_genes,
)
from biopulse.scorers.denoising_score import score as _score_denoising
from biopulse.scorers.dimensionality_reduction_score import score as _score_dimred
from biopulse.scorers.label_projection_score import score as _score_label_projection
from biopulse.scorers.rare_celltype_score import score as _score_rare_celltype
from biopulse.scorers.spatially_variable_genes_score import score as _score_svg


@dataclass(frozen=True)
class FileSpec:
    """File contract: file name plus the AnnData fields the scorer requires.

    ``required`` is keyed by AnnData slot (``obs``, ``var``, ``uns``, ``layers``, ``obsm``). The scorers
    enforce these contracts, and the drift-guard tests keep the registry and scorer checks aligned.
    Public input files that are read-only for the agent carry an empty requirement map.
    """

    filename: str
    required: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_type: str
    title: str
    inputs: tuple[FileSpec, ...]
    output: FileSpec
    solution: FileSpec
    scorer: Callable[..., dict]  # Task-specific scorer; callers should not introspect it.
    instruction: str
    output_schema: str
    metric_config: str
    aliases: tuple[str, ...] = ()
    discover: Optional[Callable[..., dict]] = None  # OP source discovery; None if not OP-built
    controls: Optional[Callable[[Path], list]] = None  # control generator; None if no local controls
    #: which discovered source role becomes the hidden solution.h5ad. Usually "solution"; denoising's
    #: held-out molecular split is the "test" role (the train split is the agent's input).
    solution_role: str = "solution"
    source: str = "openproblems"
    is_openproblems: bool = True


# --- per-task prose (pinned by tests/test_pack_writer_golden.py) ------------------------------------

_LABEL_PROJECTION_INSTRUCTION = """# Label Projection Task\n\nYou are given a labeled single-cell training dataset and an unlabeled test dataset. Predict labels for every test cell.\n\nWrite `outputs/prediction.h5ad` with `obs[\"label_pred\"]` for every test cell and `uns[\"method_id\"]`. Also write `outputs/report.md` explaining the approach and limitations.\n\nDo not search for hidden solutions. Do not use files outside the workspace.\n"""
_SVG_INSTRUCTION = """# Spatially Variable Genes Task\n\nYou are given a spatial transcriptomics dataset. Assign a spatial variability score to each gene.\n\nWrite `outputs/output.h5ad` with `var[\"pred_spatial_var_score\"]` for every gene and `uns[\"method_id\"]`. Also write `outputs/report.md` explaining the approach and limitations.\n\nDo not search for hidden solutions. Do not use files outside the workspace.\n"""
_DENOISING_INSTRUCTION = """# Denoising Task\n\nYou are given `public/input/train.h5ad`, a single-cell count matrix in `layers[\"counts\"]` that is a noisy molecular sub-sample. Produce a denoised matrix of the SAME cells and genes (same shape, same order).\n\nWrite `outputs/denoised.h5ad` with the denoised counts in `layers[\"denoised\"]` and `uns[\"method_id\"]`. Also write `outputs/report.md` explaining the approach and limitations.\n\nYou are scored by mean squared error against a held-out molecular split (after library-size normalization and log1p), so recovering true expression structure matters more than copying the noisy input. Do not search for hidden solutions. Do not use files outside the workspace.\n"""
_DIMRED_INSTRUCTION = """# Dimensionality Reduction Task\n\nYou are given `public/input/dataset.h5ad` with a normalized expression matrix in `layers[\"normalized\"]` (and raw `layers[\"counts\"]`, `var[\"hvg_score\"]`). Embed the cells into 2D.\n\nWrite `outputs/embedding.h5ad` with the 2D coordinates in `obsm[\"X_emb\"]` (one row per cell, same order) and `uns[\"method_id\"]`. Also write `outputs/report.md` explaining the approach and limitations.\n\nYou are scored on trustworthiness — how well the 2D embedding preserves each cell's high-dimensional neighborhood. Do not search for hidden solutions. Do not use files outside the workspace.\n"""

_LABEL_PROJECTION_SCHEMA = """# Output Schema\n\nRequired files:\n\n- `outputs/prediction.h5ad`\n- `outputs/report.md`\n\n`prediction.h5ad` must be an AnnData file containing:\n\n- `obs[\"label_pred\"]`: predicted labels for test cells\n- `uns[\"method_id\"]`: method or agent identifier\n- `uns[\"dataset_id\"]` and `uns[\"normalization_id\"]` when available\n"""

# rare_celltype is derived from label_projection (same I/O) but graded on macro-F1 to emphasize rare
# populations. build_rare_celltype_pack.py reads this record, keeping the registry as the source of truth.
_RARE_CELLTYPE_INSTRUCTION = """# Rare Cell-Type Annotation Task

You are given a labeled single-cell training dataset and an unlabeled test dataset. Predict a
cell-type label for every test cell.

Write `outputs/prediction.h5ad` with `obs["label_pred"]` for every test cell and `uns["method_id"]`.
Also write `outputs/report.md` explaining your approach and limitations.

**You are scored on macro-F1** — every cell type counts equally, so correctly identifying the RARE
populations matters as much as the common ones. A classifier that ignores rare classes (optimizing
plain accuracy) will score poorly; account for class imbalance.

Do not search for hidden solutions. Do not use files outside the workspace.
"""
_SVG_SCHEMA = """# Output Schema\n\nRequired files:\n\n- `outputs/output.h5ad`\n- `outputs/report.md`\n\n`output.h5ad` must be an AnnData file containing:\n\n- `var[\"pred_spatial_var_score\"]`: numeric spatial variability score for each gene\n- `uns[\"method_id\"]`: method or agent identifier\n- `uns[\"dataset_id\"]` when available\n"""
_DENOISING_SCHEMA = """# Output Schema\n\nRequired files:\n\n- `outputs/denoised.h5ad`\n- `outputs/report.md`\n\n`denoised.h5ad` must be an AnnData file with the SAME shape as the input, containing:\n\n- `layers[\"denoised\"]`: denoised counts for every cell and gene\n- `uns[\"method_id\"]`: method or agent identifier\n- `uns[\"dataset_id\"]` when available\n"""
_DIMRED_SCHEMA = """# Output Schema\n\nRequired files:\n\n- `outputs/embedding.h5ad`\n- `outputs/report.md`\n\n`embedding.h5ad` must be an AnnData file containing:\n\n- `obsm[\"X_emb\"]`: 2D embedding coordinates, one row per cell in the input order\n- `uns[\"method_id\"]`: method or agent identifier\n- `uns[\"dataset_id\"]` and `uns[\"normalization_id\"]` when available\n"""


def _metric_config(primary: str, output: str, fallback: Optional[str] = None) -> str:
    fb = f"fallback_metric: {fallback}\n" if fallback else ""
    return (
        f"primary_metric: {primary}\n{fb}required_output: {output}\nrequired_report: outputs/report.md\n"
        "forbidden_names:\n  - hidden\n  - solution.h5ad\n"
    )


# --- the records ------------------------------------------------------------------------------------

_RECORDS: tuple[TaskRecord, ...] = (
    TaskRecord(
        task_id="op_label_projection_mini",
        task_type="label_projection",
        title="Open Problems label projection mini benchmark",
        aliases=("lp",),
        inputs=(FileSpec("train.h5ad"), FileSpec("test.h5ad")),
        output=FileSpec("prediction.h5ad", {"obs": ["label_pred"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"obs": ["label"]}),
        scorer=_score_label_projection,
        instruction=_LABEL_PROJECTION_INSTRUCTION,
        output_schema=_LABEL_PROJECTION_SCHEMA,
        metric_config=_metric_config("accuracy", "outputs/prediction.h5ad"),
        discover=discover_label_projection,
        controls=_controls.make_label_projection_controls,
    ),
    TaskRecord(
        task_id="op_svg_mini",
        task_type="spatially_variable_genes",
        title="Open Problems spatially variable genes mini benchmark",
        aliases=("svg",),
        inputs=(FileSpec("dataset.h5ad"),),
        output=FileSpec("output.h5ad", {"var": ["pred_spatial_var_score"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"var": ["true_spatial_var_score"]}),
        scorer=_score_svg,
        instruction=_SVG_INSTRUCTION,
        output_schema=_SVG_SCHEMA,
        metric_config=_metric_config("kendall_tau", "outputs/output.h5ad", fallback="spearman_r"),
        discover=discover_spatially_variable_genes,
        controls=_controls.make_svg_controls,
    ),
    TaskRecord(
        task_id="op_denoising_mini",
        task_type="denoising",
        title="Open Problems denoising mini benchmark",
        aliases=("denoise",),
        inputs=(FileSpec("train.h5ad"),),
        output=FileSpec("denoised.h5ad", {"layers": ["denoised"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"layers": ["counts"]}),
        scorer=_score_denoising,
        instruction=_DENOISING_INSTRUCTION,
        output_schema=_DENOISING_SCHEMA,
        metric_config=_metric_config("mse", "outputs/denoised.h5ad"),
        discover=discover_denoising,
        controls=None,  # denoising ships no local controls -> raw scores only
        solution_role="test",  # the held-out molecular split is the hidden truth (train is the input)
    ),
    TaskRecord(
        task_id="op_dimensionality_reduction_mini",
        task_type="dimensionality_reduction",
        title="Open Problems dimensionality reduction mini benchmark",
        aliases=("dimred",),
        inputs=(FileSpec("dataset.h5ad"),),
        output=FileSpec("embedding.h5ad", {"obsm": ["X_emb"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"layers": ["normalized"]}),
        scorer=_score_dimred,
        instruction=_DIMRED_INSTRUCTION,
        output_schema=_DIMRED_SCHEMA,
        metric_config=_metric_config("trustworthiness", "outputs/embedding.h5ad"),
        discover=discover_dimensionality_reduction,
        controls=None,
    ),
    TaskRecord(
        task_id="op_rare_celltype_mini",
        task_type="rare_celltype",
        title="BioPulse rare cell-type annotation mini (derived from Open Problems label-projection data)",
        inputs=(FileSpec("train.h5ad"), FileSpec("test.h5ad")),
        output=FileSpec("prediction.h5ad", {"obs": ["label_pred"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"obs": ["label"]}),
        scorer=_score_rare_celltype,
        instruction=_RARE_CELLTYPE_INSTRUCTION,  # rare-specific macro-F1 framing
        output_schema=_LABEL_PROJECTION_SCHEMA,  # same prediction.h5ad contract as label_projection
        metric_config=_metric_config("macro_f1", "outputs/prediction.h5ad"),
        discover=None,  # not OP-discovered; derived from the label_projection pack
        controls=None,
        source="biopulse_derived",
        is_openproblems=False,
    ),
    TaskRecord(
        task_id="op_label_projection_dkd",
        task_type="label_projection_dkd",
        title="Open Problems label projection — dkd (full cellxgene_census benchmark)",
        aliases=("lp_dkd",),
        inputs=(FileSpec("train.h5ad"), FileSpec("test.h5ad")),
        output=FileSpec("prediction.h5ad", {"obs": ["label_pred"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"obs": ["label"]}),
        scorer=_score_label_projection,
        instruction=_LABEL_PROJECTION_INSTRUCTION,
        output_schema=_LABEL_PROJECTION_SCHEMA,
        metric_config=_metric_config("accuracy", "outputs/prediction.h5ad"),
        # Full-dataset variant of label_projection, aligned with OP's published dkd leaderboard. The pack
        # is built from a hand-built discovery dict, so source discovery is unused; controls are omitted
        # because OP's leaderboard already supplies the random/oracle anchors.
        discover=None,
        controls=None,
    ),
    TaskRecord(
        task_id="op_denoising_immune",
        task_type="denoising_immune",
        title="Open Problems denoising — immune_cells (openproblems_v1) full benchmark",
        aliases=("denoise_immune",),
        inputs=(FileSpec("train.h5ad"),),
        output=FileSpec("denoised.h5ad", {"layers": ["denoised"], "uns": ["method_id"]}),
        solution=FileSpec("solution.h5ad", {"layers": ["counts"]}),
        scorer=_score_denoising,
        instruction=_DENOISING_INSTRUCTION,
        output_schema=_DENOISING_SCHEMA,
        metric_config=_metric_config("mse", "outputs/denoised.h5ad"),
        # Full openproblems_v1/immune_cells denoising dataset, aligned with OP's published board.
        # Discovery is hand-built from op_cache; the held-out molecular split is the hidden truth.
        discover=None,
        controls=None,
        solution_role="test",
    ),
)


REGISTRY: dict[str, TaskRecord] = {record.task_type: record for record in _RECORDS}
_ALIASES: dict[str, str] = {alias: record.task_type for record in _RECORDS for alias in record.aliases}


def get(task_type: str) -> TaskRecord:
    """Look up a TaskRecord by task_type or alias (e.g. ``svg`` -> spatially_variable_genes). Raises a
    clear ValueError for unknown task types."""
    canonical = _ALIASES.get(task_type, task_type)
    try:
        return REGISTRY[canonical]
    except KeyError:
        known = sorted(REGISTRY) + sorted(_ALIASES)
        raise ValueError(f"unknown task_type: {task_type!r} (known: {known})") from None


def required_outputs(task_type: str) -> list[str]:
    """The output files a run must produce: the task's output file under ``outputs/`` plus the report."""
    return [f"outputs/{get(task_type).output.filename}", "outputs/report.md"]
