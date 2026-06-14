"""Characterization snapshots — golden-master coverage of every scorer branch.

See ``tests/characterization/charsnap.py`` for the harness. Each scorer contributes a ``cases_*.py``
module defining ``CASES: list[Case]``; this entry runs them all against committed snapshots.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("numpy")

# The harness + case modules live in tests/characterization/ and import each other as top-level
# modules; put that dir on sys.path (at front, so they win) rather than making tests/ a package.
HERE = Path(__file__).parent / "characterization"
sys.path.insert(0, str(HERE))

import charsnap  # noqa: E402

CASE_MODULES = [
    "cases_label_projection",
    "cases_spatially_variable_genes",
    "cases_denoising",
    "cases_dimensionality_reduction",
    "cases_rare_celltype",
]


def _load_cases() -> list:
    cases: list = []
    for module_name in CASE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue  # a missing module is caught explicitly by test_all_case_modules_present
        cases.extend(getattr(module, "CASES", []))
    return cases


ALL_CASES = _load_cases()


def test_all_case_modules_present() -> None:
    """Every scorer must contribute a case module — guards against a silently-missing file that would
    quietly shrink coverage (the snapshot suite would still go green with a hole in it)."""
    missing = []
    for module_name in CASE_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    assert not missing, f"missing characterization case modules: {missing}"


def test_case_names_are_unique() -> None:
    names = [case.name for case in ALL_CASES]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate characterization case names (snapshots would collide): {duplicates}"


@pytest.mark.parametrize("case", ALL_CASES, ids=[case.name for case in ALL_CASES])
def test_characterization_matches_snapshot(case, tmp_path: Path) -> None:
    got = charsnap.capture(case, tmp_path)
    if charsnap.update_mode():
        charsnap.write_snapshot(case.name, got)
        pytest.skip(f"snapshot updated: {case.name}")
    snap = charsnap.snapshot_path(case.name)
    assert snap.exists(), f"no snapshot for {case.name}; regenerate with BIOPULSE_UPDATE_SNAPSHOTS=1"
    assert got == charsnap.load_snapshot(case.name)
