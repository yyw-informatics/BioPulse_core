"""Characterization harness for scorer result snapshots.

The snapshots capture the normalized result dictionary for each scorer outcome: metric values,
violations in order, warnings, pass/fail flags, safety-gate status, and final score.

A "case" is one branch: a build() that constructs a (benchmark_dir, run_dir) triggering exactly that
return path, plus the scorer to run. Cases live in ``cases_<scorer>.py`` next to this file; the pytest
entry is ``tests/test_characterization.py``.

Regenerate snapshots after an intentional behavior change:
    BIOPULSE_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_characterization.py -q
Then review the diff before committing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

SNAP_DIR = Path(__file__).parent / "snapshots"


@dataclass
class Case:
    name: str  # unique, stable snapshot filename
    scorer: Callable[..., dict]  # the scorer's score() function
    build: Callable[[Path], Tuple[Path, Path]]  # (tmp_path) -> (benchmark_dir, run_dir)


def _round(value):
    # bool is an int subclass; keep it as a bool for snapshot readability.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def normalize_result(result: dict) -> dict:
    """Deterministic, platform-stable view of a result dict: drop the per-run id + timestamp, round
    every float to 6 decimal places, sort metric keys, and preserve violation/warning order."""
    snapshot = dict(result)
    snapshot.pop("run_id", None)
    snapshot.pop("scored_at_utc", None)
    snapshot["final_score"] = _round(snapshot.get("final_score", 0.0))
    snapshot["metrics"] = {key: _round(val) for key, val in sorted(snapshot.get("metrics", {}).items())}
    snapshot["violations"] = list(snapshot.get("violations", []))
    snapshot["warnings"] = list(snapshot.get("warnings", []))
    return snapshot


def capture(case: Case, tmp_path: Path) -> dict:
    benchmark_dir, run_dir = case.build(tmp_path)
    result = case.scorer(benchmark_dir, run_dir, run_id="characterization")
    return normalize_result(result)


def snapshot_path(name: str) -> Path:
    return SNAP_DIR / f"{name}.json"


def write_snapshot(name: str, data: dict) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path(name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(name: str) -> dict:
    return json.loads(snapshot_path(name).read_text(encoding="utf-8"))


def update_mode() -> bool:
    return os.environ.get("BIOPULSE_UPDATE_SNAPSHOTS", "") not in ("", "0", "false")
