from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class DatasetCandidate:
    task_type: str
    dataset_id: str
    root: Path
    files: Mapping[str, Path]
    total_bytes: int

    def to_manifest(self, repo_root: Path) -> dict:
        return {
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "root": _rel(self.root, repo_root),
            "total_bytes": self.total_bytes,
            "files": {role: _rel(path, repo_root) for role, path in self.files.items()},
        }


class OpenProblemsDiscoveryError(RuntimeError):
    """Raised when Open Problems test resources cannot be discovered."""


def discover_task(openproblems_root: Path | str, task_type: str, prefer: str = "resources_test", dataset_id: Optional[str] = None) -> dict:
    # Lazy import: the registry imports the discover_* fns from this module, so importing it at module
    # level would be a cycle. The record's `discover` callable resolves aliases (svg, dimred) too.
    from biopulse.tasks.registry import get

    record = get(task_type)
    if record.discover is None:
        raise ValueError(f"task_type {task_type!r} has no Open Problems source discovery")
    return record.discover(Path(openproblems_root), prefer=prefer, dataset_id=dataset_id)


def discover_label_projection(openproblems_root: Path, prefer: str = "resources_test", dataset_id: Optional[str] = None) -> dict:
    repo = openproblems_root / "task_label_projection"
    resource_root = _resource_root(repo, prefer)
    candidates = _discover_by_exact_names(
        resource_root,
        task_type="label_projection",
        roles={"train.h5ad": "train", "test.h5ad": "test", "solution.h5ad": "solution"},
        required_roles={"train", "test", "solution"},
    )
    selected = _select_candidate(candidates, dataset_id=dataset_id)
    return _manifest(repo, resource_root, selected, candidates)


def discover_spatially_variable_genes(openproblems_root: Path, prefer: str = "resources_test", dataset_id: Optional[str] = None) -> dict:
    repo = openproblems_root / "task_spatially_variable_genes"
    resource_root = _resource_root(repo, prefer)
    candidates = _discover_by_exact_names(
        resource_root,
        task_type="spatially_variable_genes",
        roles={"dataset.h5ad": "dataset", "solution.h5ad": "solution"},
        required_roles={"dataset", "solution"},
    )
    selected = _select_candidate(candidates, dataset_id=dataset_id)
    return _manifest(repo, resource_root, selected, candidates)


def discover_denoising(openproblems_root: Path, prefer: str = "resources_test", dataset_id: Optional[str] = None) -> dict:
    repo = openproblems_root / "task_denoising"
    resource_root = _resource_root(repo, prefer)
    candidates = _discover_by_exact_names(
        resource_root,
        task_type="denoising",
        roles={"train.h5ad": "train", "test.h5ad": "test"},
        required_roles={"train", "test"},
    )
    selected = _select_candidate(candidates, dataset_id=dataset_id)
    return _manifest(repo, resource_root, selected, candidates)


def discover_dimensionality_reduction(openproblems_root: Path, prefer: str = "resources_test", dataset_id: Optional[str] = None) -> dict:
    repo = openproblems_root / "task_dimensionality_reduction"
    resource_root = _resource_root(repo, prefer)
    candidates = _discover_by_exact_names(
        resource_root,
        task_type="dimensionality_reduction",
        roles={"dataset.h5ad": "dataset", "solution.h5ad": "solution"},
        required_roles={"dataset", "solution"},
    )
    selected = _select_candidate(candidates, dataset_id=dataset_id)
    return _manifest(repo, resource_root, selected, candidates)


def _resource_root(repo: Path, prefer: str) -> Path:
    if not repo.exists():
        raise OpenProblemsDiscoveryError(f"Missing Open Problems repository: {repo}")
    preferred = repo / prefer
    if preferred.exists():
        return preferred
    fallback = repo / "resources_test"
    if fallback.exists():
        return fallback
    raise OpenProblemsDiscoveryError(f"Missing resources directory under {repo}; expected {prefer}/ or resources_test/")


def _discover_by_exact_names(resource_root: Path, task_type: str, roles: Mapping[str, str], required_roles: set[str]) -> List[DatasetCandidate]:
    grouped: Dict[Path, Dict[str, Path]] = {}
    for path in sorted(resource_root.rglob("*.h5ad")):
        role = roles.get(path.name.lower())
        if role is None:
            continue
        grouped.setdefault(path.parent, {})[role] = path

    candidates: List[DatasetCandidate] = []
    for group_root, files in grouped.items():
        if not required_roles.issubset(files.keys()):
            continue
        total_bytes = sum(path.stat().st_size for path in files.values())
        candidates.append(
            DatasetCandidate(
                task_type=task_type,
                dataset_id=group_root.name,
                root=group_root,
                files=dict(files),
                total_bytes=total_bytes,
            )
        )
    return sorted(candidates, key=lambda item: (item.total_bytes, item.dataset_id))


def _select_candidate(candidates: Iterable[DatasetCandidate], dataset_id: Optional[str]) -> DatasetCandidate:
    all_candidates = list(candidates)
    if not all_candidates:
        raise OpenProblemsDiscoveryError("No complete dataset groups found under the resources directory")
    if dataset_id:
        matches = [candidate for candidate in all_candidates if candidate.dataset_id == dataset_id or dataset_id in str(candidate.root)]
        if not matches:
            available = ", ".join(candidate.dataset_id for candidate in all_candidates)
            raise OpenProblemsDiscoveryError(f"Dataset id {dataset_id!r} not found. Available candidates: {available}")
        return matches[0]
    return all_candidates[0]


def _manifest(repo: Path, resource_root: Path, selected: DatasetCandidate, candidates: List[DatasetCandidate]) -> dict:
    h5ad_files = sorted(resource_root.rglob("*.h5ad"))
    return {
        "source": "openproblems",
        "repo_name": repo.name,
        "repo_path": str(repo),
        "resource_root": str(resource_root),
        "selected": selected.to_manifest(repo),
        "candidates": [candidate.to_manifest(repo) for candidate in candidates],
        "all_h5ad_files": [_rel(path, repo) for path in h5ad_files],
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
