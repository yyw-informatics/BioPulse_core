"""Characterization (golden-master) harness — snapshot each scorer's FULL result dict per branch.

A golden-master safety net for the scorers. The regular scorer tests assert a few substrings; this pins
the ENTIRE normalized result_dict (every metric key+value, violations in order, warnings, passed,
safety_gate_passed, final_score) for every return path in every scorer, so any unintended change in
observable behavior fails a snapshot diff.

A "case" is one branch: a build() that constructs a (benchmark_dir, run_dir) triggering exactly that
return path, plus the scorer to run. Cases live in ``cases_<scorer>.py`` next to this file; the pytest
entry is ``tests/test_characterization.py``.

Regenerate snapshots after an INTENTIONAL behavior change:
    BIOPULSE_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_characterization.py -q
Then eyeball the diff before committing — an unexpected snapshot change is a real regression.
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
    name: str  # unique, stable -> snapshot filename (convention: "<scorer>__<branch>")
    scorer: Callable[..., dict]  # the scorer's score() function
    build: Callable[[Path], Tuple[Path, Path]]  # (tmp_path) -> (benchmark_dir, run_dir)


def _round(value):
    # bool is an int subclass — keep it a bool, don't coerce to 0.0/1.0
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def normalize_result(result: dict) -> dict:
    """Deterministic, platform-stable view of a result dict: drop the per-run id + timestamp, round
    every float (final_score + each metric) to 6 dp, sort metric keys. violations/warnings KEEP their
    order — order is part of the scorer contract — and task_id stays (deterministic per fixture)."""
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
