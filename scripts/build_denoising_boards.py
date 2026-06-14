"""ETL: leaderboard.yaml (immune_cells, MSE) + the L3 method menu methods.yaml for the denoising task,
from Open Problems' published artifacts. No methods are rerun. MSE is MINIMIZED, so the board header
carries ``maximize: false`` and the placement consumer ranks ascending. Sibling of
``build_label_projection_boards.py`` (denoising has different controls + metric direction)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONTROLS = {"no_denoising", "perfect_denoising"}
DEFAULT_DATASET = "openproblems_v1/immune_cells"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_leaderboard(score_uns: Path, dataset_uns: Path, dataset: str = DEFAULT_DATASET) -> dict:
    """Merge OP's per-metric rows for one dataset into one ranked entry per method (ascending MSE)."""
    merged: dict[str, dict] = {}
    for row in _load(score_uns):
        if row.get("dataset_id") != dataset:
            continue
        entry = merged.setdefault(
            row["method_id"], {"method_id": row["method_id"], "normalization_id": row.get("normalization_id")}
        )
        mids, mvals = row["metric_ids"], row["metric_values"]
        if isinstance(mids, str):  # scalar metric row
            entry[mids] = mvals
        else:  # list metric row
            for mid, mval in zip(mids, mvals):
                entry[mid] = mval

    # MSE: lower is better -> sort ascending; entries lacking an mse sort last.
    entries = sorted(merged.values(), key=lambda e: e.get("mse", float("inf")))
    for rank, entry in enumerate(entries, start=1):
        entry["is_control"] = entry["method_id"] in CONTROLS
        entry["rank"] = rank

    meta = next((d for d in _load(dataset_uns) if d.get("dataset_id") == dataset), {})
    header = {
        "dataset_id": dataset,
        "dataset_name": meta.get("dataset_name"),
        "metric_primary": "mse",
        "metric_min": 0,
        "metric_max": None,
        "maximize": False,
        "n_methods": len(entries),
        "n_controls": sum(1 for e in entries if e["is_control"]),
        "source": "openproblems published results (no reruns)",
    }
    return {"schema_version": "biopulse.leaderboard.v1", "dataset": header, "entries": entries}


def build_methods(methods_dir: Path, scored_ids: set[str]) -> dict:
    """Harvest the method menu from the OP method configs. ``scored_ids`` flags which appear on the board."""
    methods = []
    for cfg_path in sorted(Path(methods_dir).glob("*/config.vsh.yaml")):
        cfg = _load(cfg_path)
        info = cfg.get("info", {}) or {}
        links = cfg.get("links", {}) or {}
        refs = (cfg.get("references", {}) or {}).get("doi", [])
        if isinstance(refs, str):
            refs = [refs]
        packages: list[str] = []
        for engine in cfg.get("engines", []) or []:
            for setup in engine.get("setup", []) or []:
                pkg = setup.get("packages")
                if isinstance(pkg, str):
                    packages.append(pkg)
                elif isinstance(pkg, list):
                    packages.extend(pkg)
        methods.append(
            {
                "id": cfg.get("name"),
                "label": cfg.get("label"),
                "summary": cfg.get("summary"),
                "description": (cfg.get("description") or "").strip(),
                "references_doi": list(refs) if refs else [],
                "repository": links.get("repository"),
                "documentation": links.get("documentation"),
                "preferred_normalization": info.get("preferred_normalization"),
                "packages": sorted(set(packages)),
                "has_score": cfg.get("name") in scored_ids,
            }
        )
    return {"schema_version": "biopulse.method_menu.v1", "task": "denoising", "n_methods": len(methods), "methods": methods}


def main() -> int:
    base = _repo_root() / "op_cache/task_denoising"
    parser = argparse.ArgumentParser(description="Build denoising leaderboard.yaml + methods.yaml from OP results")
    parser.add_argument("--results", type=Path, default=base / "results")
    parser.add_argument("--methods-dir", type=Path, default=base / "methods")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=base / "boards/denoising_immune")
    args = parser.parse_args()

    leaderboard = build_leaderboard(args.results / "score_uns.yaml", args.results / "dataset_uns.yaml", args.dataset)
    scored = {e["method_id"] for e in leaderboard["entries"]}
    methods = build_methods(args.methods_dir, scored)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.yaml").write_text(yaml.safe_dump(leaderboard, sort_keys=False), encoding="utf-8")
    (args.out / "methods.yaml").write_text(yaml.safe_dump(methods, sort_keys=False), encoding="utf-8")

    print(f"leaderboard: {leaderboard['dataset']['n_methods']} methods ({leaderboard['dataset']['n_controls']} controls) for {args.dataset}")
    print("best 3 (lowest MSE):", [(e["method_id"], round(e.get("mse", 0), 4)) for e in leaderboard["entries"][:3]])
    print(f"methods menu: {methods['n_methods']} denoisers; on board: {sum(m['has_score'] for m in methods['methods'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
