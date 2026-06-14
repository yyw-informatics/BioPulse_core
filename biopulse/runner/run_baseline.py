"""Reference-run harness: drive a deterministic baseline through the full scoring + provenance pipeline.

Copies a pack's public inputs into a fresh run workspace, runs a baseline script from ``baselines/`` as
the stand-in agent, then records the run manifest, runtime/token/cost artifacts, the evaluator result,
and the evidence bundle. The baselines give the scorer known-answer runs and exercise the same contract
a real agent front-end would drive.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from biopulse.scorers.common import read_task_yaml
from biopulse.tasks.registry import get as get_task

from .cost import write_runtime_artifacts
from .evidence import copy_public_to_workspace, evidence_bundle, list_files, utc_now, write_json

BASELINE_SCRIPTS = {
    "label_projection_majority": "label_projection_majority.py",
    "label_projection_knn": "label_projection_knn.py",
    "svg_variance": "svg_variance_baseline.py",
    "svg_random": "svg_random_baseline.py",
    "bad_agent_stub": "bad_agent_stub.py",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_baseline(
    benchmark: Path | str,
    baseline: str,
    run_id: str,
    runs_dir: Path | str = "runs",
    overwrite: bool = False,
) -> dict:
    benchmark_path = Path(benchmark)
    runs_root = Path(runs_dir)
    run_dir = runs_root / run_id
    workspace = run_dir / "workspace"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run already exists: {run_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    task = read_task_yaml(benchmark_path / "task.yaml")
    task_id = str(task.get("task_id", benchmark_path.name))
    task_type = str(task.get("task_type", ""))
    copy_public_to_workspace(benchmark_path, workspace)
    files_available = list_files(workspace)

    if baseline not in BASELINE_SCRIPTS:
        raise ValueError(f"Unknown baseline {baseline!r}. Valid baselines: {sorted(BASELINE_SCRIPTS)}")
    script = repo_root() / "baselines" / BASELINE_SCRIPTS[baseline]
    if not script.exists():
        raise FileNotFoundError(f"Baseline script not found: {script}")
    command = [sys.executable, str(script)]

    start = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
    ended = utc_now()
    wall_time = time.perf_counter() - started

    (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    files_after = list_files(workspace)
    files_produced = sorted(path for path in files_after if path not in files_available)

    manifest = {
        "schema_version": "biopulse.run_manifest.v1",
        "run_id": run_id,
        "task_id": task_id,
        "task_type": task_type,
        "agent_id": baseline,
        "agent_surface": "deterministic_baseline",
        "benchmark": str(benchmark_path),
        "workspace": str(workspace),
        "command": " ".join(command),
        "start_time_utc": start,
        "end_time_utc": ended,
        "wall_time_seconds": wall_time,
        "exit_code": completed.returncode,
        "files_available_to_agent": files_available,
        "files_produced_by_agent": files_produced,
        "stdout_path": "stdout.txt",
        "stderr_path": "stderr.txt",
        "hidden_ground_truth_excluded": "hidden" not in {part.lower() for path in files_after for part in Path(path).parts}
        and "solution.h5ad" not in {Path(path).name.lower() for path in files_after},
    }
    write_json(run_dir / "run_manifest.json", manifest)

    # Agent runtime + token/cost contract. Deterministic baselines use no LLM,
    # so token usage is recorded as "none" (zero tokens, zero cost), never estimated.
    write_runtime_artifacts(run_dir, run_id, baseline, "baseline")

    evaluator_results = score_run(benchmark_path, run_dir, task_type, run_id)
    write_json(run_dir / "evaluator_results.json", evaluator_results)
    write_json(run_dir / "evidence_bundle.json", evidence_bundle(manifest, evaluator_results))
    return evaluator_results


def score_run(benchmark: Path, run_dir: Path, task_type: str, run_id: Optional[str] = None) -> dict:
    # The registry holds each scorer as an opaque callable; dispatch is one lookup (raises on unknown).
    return get_task(task_type).scorer(benchmark, run_dir, run_id=run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a baseline against a BioPulse benchmark pack")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINE_SCRIPTS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = run_baseline(args.benchmark, args.baseline, args.run_id, args.runs_dir, overwrite=args.overwrite)
    print(f"{args.run_id}: passed={result['passed']} final_score={result['final_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
