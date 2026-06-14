from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file())


def copy_public_to_workspace(benchmark: Path, workspace: Path) -> None:
    public = benchmark / "public"
    if not public.exists():
        raise FileNotFoundError(f"Benchmark public directory missing: {public}")
    workspace.mkdir(parents=True, exist_ok=True)
    for item in public.iterdir():
        destination = workspace / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL file: one JSON object per line (e.g. the process_events time-series)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def evidence_bundle(manifest: dict[str, Any], evaluator_results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "biopulse.evidence.v1",
        "run_id": manifest.get("run_id"),
        "task_id": manifest.get("task_id"),
        "agent_id": manifest.get("agent_id"),
        "process_plane": manifest,
        "scientific_artifact_plane": evaluator_results,
        "hidden_ground_truth_excluded": manifest.get("hidden_ground_truth_excluded", False),
        "created_at_utc": utc_now(),
    }
