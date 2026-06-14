# biopulse-core

[![CI](https://github.com/yyw-informatics/BioPulse_core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yyw-informatics/BioPulse_core/actions/workflows/ci.yml)

`biopulse-core` is the evaluation core for BioPulse: a small, framework-agnostic library for scoring AI
agents on single-cell bioinformatics tasks derived from [Open Problems](https://openproblems.bio).

The project focuses on the parts of AI evaluation that need to be dependable: faithful metric
reproduction, explicit output contracts, artifact-boundary checks, benchmark-pack generation, and run
provenance. It intentionally contains no agent loop. A CodeAct loop, LangGraph graph, vendor agent, or
deterministic baseline can all use the same benchmark packs and scorers.

## What It Contains

- `biopulse.tasks.registry`: one `TaskRecord` per task, covering file contracts, scorer dispatch,
  source discovery, control generation, and task-facing prose.
- `biopulse.scorers`: deterministic scorers for label projection, spatially variable genes, denoising,
  dimensionality reduction, and a derived rare-cell-type task.
- `biopulse.builder`: benchmark-pack builders and validators, including random/oracle control outputs
  for normalized scoring.
- `biopulse.runner`: baseline execution, evidence bundles, token/cost artifacts, and process-plane
  summaries for auditable runs.
- `baselines/`: deterministic reference agents used to validate scorer behavior end to end.
- `scripts/`: reproducible pack and leaderboard ingestion utilities for Open Problems resources.

## Why This Matters

For AI evaluation work, the score is only useful if the benchmark boundary is explicit. BioPulse exposes
only staged public inputs to the agent-facing surface, keeps hidden solutions in scorer-only artifacts,
checks outputs for forbidden references, validates each task schema before scoring, and records enough
provenance to review what ran.

The core design choice is explicitness: each task owns its scorer and contracts. There is no generic
metric driver that tries to infer how a task should be graded from field names.

## Install

```bash
git clone https://github.com/yyw-informatics/BioPulse_core.git
cd BioPulse_core
python -m pip install -e ".[dev]"
```

The package name is `biopulse-core`; the import package is `biopulse`.

## Basic Use

```python
from biopulse.tasks.registry import get

record = get("label_projection")
result = record.scorer("benchmark_packs/op_label_projection_mini", "runs/example")
print(result["final_score"], result["passed"])
```

## Build Mini Benchmark Packs

```bash
python scripts/build_benchmark_packs.py \
  --openproblems-root external/openproblems \
  --out benchmark_packs
```

## Run a Baseline

The baseline scripts behave like small deterministic agents: they read only the staged public inputs and
write the same `outputs/` artifacts expected from an AI agent.

| Script | Task | Behavior |
| --- | --- | --- |
| `label_projection_majority.py` | label projection | predicts the most frequent training label |
| `label_projection_knn.py` | label projection | fits a kNN classifier in PCA or expression space |
| `svg_variance_baseline.py` | spatially variable genes | ranks genes by expression variance |
| `svg_random_baseline.py` | spatially variable genes | assigns seeded random scores |
| `bad_agent_stub.py` | any | produces no prediction artifact to exercise schema failures |

```bash
python -m biopulse.runner.run_baseline \
  --benchmark benchmark_packs/op_label_projection_mini \
  --baseline label_projection_majority \
  --run-id lp_majority_001 \
  --overwrite
```

The run directory includes `run_manifest.json`, `evaluator_results.json`, `evidence_bundle.json`,
`agent_profile.json`, `token_usage.json`, and `cost_summary.json`.

## Example Evaluation Report

A BioPulse run is reported on two planes: whether the scientific artifact is correct, and whether the
process record is auditable and avoids exposing hidden inputs.

Example: `label_projection_majority` on `op_label_projection_mini`.

| Plane | Metric | Value |
| --- | --- | --- |
| Scientific Artifact | `passed` | `true` |
| Scientific Artifact | `final_score` / accuracy | `0.1472` |
| Scientific Artifact | normalized accuracy | `0.1032` |
| Scientific Artifact | macro-F1 | `0.0103` |
| Scientific Artifact | schema valid | `1.0` |
| Scientific Artifact | safety gate passed | `true` |
| Agent Process / Provenance | exit code | `0` |
| Agent Process / Provenance | wall time | `0.55s` |
| Agent Process / Provenance | hidden ground truth excluded | `true` |
| Agent Process / Provenance | files available to agent | `input/train.h5ad`, `input/test.h5ad`, task docs |
| Agent Process / Provenance | files produced by agent | `outputs/prediction.h5ad`, `outputs/report.md` |
| Agent Process / Provenance | cost | `$0.00` for deterministic baseline |

The same run is stored as a single evidence object:

```json
{
  "schema_version": "biopulse.evidence.v1",
  "run_id": "readme_lp_majority",
  "task_id": "op_label_projection_mini",
  "agent_id": "label_projection_majority",
  "hidden_ground_truth_excluded": true,
  "scientific_artifact_plane": {
    "passed": true,
    "final_score": 0.1472,
    "safety_gate_passed": true,
    "metrics": {
      "accuracy": 0.1472,
      "accuracy_scaled": 0.1032,
      "macro_f1": 0.0103,
      "schema_valid": 1.0,
      "report_present": 1.0
    },
    "violations": []
  },
  "process_plane": {
    "agent_surface": "deterministic_baseline",
    "exit_code": 0,
    "wall_time_seconds": 0.55,
    "files_available_to_agent": ["input/test.h5ad", "input/train.h5ad", "instruction.md", "output_schema.md"],
    "files_produced_by_agent": ["outputs/prediction.h5ad", "outputs/report.md"]
  }
}
```

A commit-friendly redacted copy of this result is available under
`examples/label_projection_majority_demo/`. It excludes `.h5ad` files, hidden ground truth, empty logs,
and local absolute paths.

## Tests

```bash
python -m pytest -q
```

Most tests run without external data: characterization snapshots cover scorer branches, and golden-pack
tests pin generated task prose/configuration. Fidelity tests that require vendored Open Problems data
are expected to skip unless those resources are available locally.
