"""ETL: build the dkd ``leaderboard.yaml`` + the L3 method menu ``methods.yaml`` from Open Problems'
published artifacts. No methods are rerun — we read OP's ``score_uns`` and the method configs and emit
two pack-ready YAML files. Part of the OP-ingestion pipeline (lives in biopulse-core per the reuse
boundary; biopulse-langgraph only consumes the emitted files).

Defaults read from ``op_cache/task_label_projection`` and write to ``op_cache/.../boards/<dataset>``:
  results/score_uns.yaml                 OP per-(dataset, method, metric) scores
  results/dataset_uns.yaml               OP dataset metadata
  op_repo/src/methods/*/config.vsh.yaml  the method configs (the menu source)

In ``score_uns`` accuracy is a scalar-``metric_ids`` row and f1 is a list-``metric_ids`` row
(``[f1_macro, f1_micro, f1_weighted]``); the two are joined per ``(dataset_id, method_id)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONTROLS = {"true_labels", "majority_vote", "random_labels"}
DEFAULT_DATASET = "cellxgene_census/dkd"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_leaderboard(score_uns: Path, dataset_uns: Path, dataset: str = DEFAULT_DATASET) -> dict:
    """Merge OP's per-metric rows for one dataset into one ranked entry per method."""
    merged: dict[str, dict] = {}
    for row in _load(score_uns):
        if row.get("dataset_id") != dataset:
            continue
        entry = merged.setdefault(
            row["method_id"], {"method_id": row["method_id"], "normalization_id": row.get("normalization_id")}
        )
        mids, mvals = row["metric_ids"], row["metric_values"]
        if isinstance(mids, str):  # scalar metric row (accuracy)
            entry[mids] = mvals
        else:  # list metric row (f1_macro / f1_micro / f1_weighted)
            for mid, mval in zip(mids, mvals):
                entry[mid] = mval

    entries = sorted(merged.values(), key=lambda e: e.get("accuracy", float("-inf")), reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["is_control"] = entry["method_id"] in CONTROLS
        entry["rank"] = rank  # over all methods incl. controls; the placement consumer re-ranks vs real methods

    meta = next((d for d in _load(dataset_uns) if d.get("dataset_id") == dataset), {})
    header = {
        "dataset_id": dataset,
        "dataset_name": meta.get("dataset_name"),
        "metric_primary": "accuracy",
        "metric_min": 0,
        "metric_max": 1,
        "maximize": True,
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
                "has_dkd_score": cfg.get("name") in scored_ids,
            }
        )
    return {
        "schema_version": "biopulse.method_menu.v1",
        "task": "label_projection",
        "n_methods": len(methods),
        "methods": methods,
    }


def main() -> int:
    base = _repo_root() / "op_cache/task_label_projection"
    parser = argparse.ArgumentParser(description="Build dkd leaderboard.yaml + methods.yaml from OP results")
    parser.add_argument("--results", type=Path, default=base / "results")
    parser.add_argument("--methods-dir", type=Path, default=base / "op_repo/src/methods")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=base / "boards/label_projection_dkd")
    args = parser.parse_args()

    leaderboard = build_leaderboard(args.results / "score_uns.yaml", args.results / "dataset_uns.yaml", args.dataset)
    scored_ids = {e["method_id"] for e in leaderboard["entries"]}
    methods = build_methods(args.methods_dir, scored_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.yaml").write_text(yaml.safe_dump(leaderboard, sort_keys=False), encoding="utf-8")
    (args.out / "methods.yaml").write_text(yaml.safe_dump(methods, sort_keys=False), encoding="utf-8")

    n_ctrl = leaderboard["dataset"]["n_controls"]
    print(f"leaderboard: {leaderboard['dataset']['n_methods']} methods ({n_ctrl} controls) for {args.dataset}")
    print("top 3:", [(e["method_id"], round(e.get("accuracy", 0), 4)) for e in leaderboard["entries"][:3]])
    print(f"methods menu: {methods['n_methods']} methods; on dkd board: {sum(m['has_dkd_score'] for m in methods['methods'])}")
    print(f"wrote {args.out}/leaderboard.yaml + methods.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
