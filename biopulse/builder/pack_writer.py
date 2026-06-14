from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from biopulse.tasks.registry import TaskRecord
from biopulse.tasks.registry import get as get_task

from .openproblems_adapter import discover_task
from .validate_pack import validate_pack


def build_packs(openproblems_root: Path | str, out: Path | str, tasks: Iterable[str], prefer: str = "resources_test", dataset_id: Optional[str] = None) -> list[Path]:
    output_root = Path(out)
    output_root.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []
    for task in tasks:
        normalized = get_task(task).task_type  # registry resolves aliases (lp/svg/denoise/dimred)
        discovery = discover_task(openproblems_root, normalized, prefer=prefer, dataset_id=dataset_id)
        built.append(write_pack(normalized, discovery, output_root))
    return built


def write_pack(task_type: str, discovery: dict, output_root: Path, overwrite: bool = True) -> Path:
    record = get_task(task_type)
    pack_dir = output_root / record.task_id
    if pack_dir.exists() and overwrite:
        shutil.rmtree(pack_dir)

    public_input = pack_dir / "public" / "input"
    hidden_truth = pack_dir / "hidden" / "ground_truth"
    metrics_dir = pack_dir / "metrics"
    baselines_dir = pack_dir / "baselines" / "baseline_outputs"
    environment_dir = pack_dir / "environment"
    for directory in [public_input, hidden_truth, metrics_dir, baselines_dir, environment_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    selected = discovery["selected"]
    repo_path = Path(discovery["repo_path"])
    selected_files = selected["files"]

    # Copy the public inputs (each FileSpec's name maps to the discovery role of the same stem, e.g.
    # train.h5ad <- role "train") and the hidden solution (from record.solution_role: usually
    # "solution", but denoising's hidden truth is the held-out "test" split).
    for spec in record.inputs:
        role = spec.filename[: -len(".h5ad")] if spec.filename.endswith(".h5ad") else spec.filename
        if role not in selected_files:
            raise KeyError(
                f"task {record.task_type!r}: input {spec.filename!r} maps to discovery role {role!r}, "
                f"which is not among the discovered roles {sorted(selected_files)}"
            )
        _copy(repo_path / selected_files[role], public_input / spec.filename)
    _copy(repo_path / selected_files[record.solution_role], hidden_truth / record.solution.filename)

    _write_text(pack_dir / "task.yaml", _task_yaml(record, selected["dataset_id"]))
    _write_text(pack_dir / "public" / "instruction.md", record.instruction)
    _write_text(pack_dir / "public" / "output_schema.md", record.output_schema)
    _write_text(metrics_dir / "metric_config.yaml", record.metric_config)
    _write_text(metrics_dir / "score.py", _metric_script(record))
    _write_text(pack_dir / "evidence_spec.yaml", _evidence_spec(record.task_type))
    _write_text(environment_dir / "environment.yml", _environment_yaml())
    _write_json(pack_dir / "source_manifest.json", discovery)

    # Deterministic OP control outputs (random/oracle anchors) for normalized scoring. No-op for tasks
    # without local controls (denoising / dimred). Kept outside public/, so never shown to the agent.
    from .controls import write_controls

    write_controls(pack_dir)

    result = validate_pack(pack_dir)
    if not result["valid"]:
        raise RuntimeError(f"Generated invalid benchmark pack {pack_dir}: {result['violations']}")
    return pack_dir


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(record: TaskRecord, dataset_id: str) -> str:
    return f"""task_id: {record.task_id}
task_type: {record.task_type}
title: {record.title}
source: {record.source}
is_openproblems: {str(record.is_openproblems).lower()}
source_dataset_id: {dataset_id}
public_dir: public
hidden_dir: hidden/ground_truth
metrics_dir: metrics
"""


def _metric_script(record: TaskRecord) -> str:
    # The generated shim imports the scorer by the module path recorded on the registry callable.
    module_path = record.scorer.__module__
    return f"""from pathlib import Path\nimport json\nimport sys\n\nfrom {module_path} import score\n\nif __name__ == \"__main__\":\n    benchmark = Path(sys.argv[1])\n    run_dir = Path(sys.argv[2])\n    result = score(benchmark, run_dir)\n    print(json.dumps(result, indent=2, sort_keys=True))\n"""


def _evidence_spec(task_type: str) -> str:
    return f"""task_type: {task_type}\nrequired_evidence:\n  - run_id\n  - task_id\n  - agent_id\n  - start_time_utc\n  - end_time_utc\n  - command\n  - files_available_to_agent\n  - files_produced_by_agent\n  - stdout\n  - stderr\n  - hidden_ground_truth_excluded\n  - evaluator_results\n"""


def _environment_yaml() -> str:
    return """name: biopulse-pack-runtime\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.12\n  - anndata\n  - h5py\n  - numpy\n  - pandas\n  - scipy\n  - scikit-learn\n  - pyyaml\n"""
