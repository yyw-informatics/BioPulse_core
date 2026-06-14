"""Harness-level policy for the value-of-harness experiment (the L1-L4 ladder).

The ladder isolates *how much scaffolding wraps the model*:

  - **L1** — bare loop, no network (enforced): the agent works from the public inputs alone.
  - **L2** — L1 + network minus a contamination blacklist (answer-revealing hosts); every fetch logged.
  - **L3** — L2 + the curated method menu injected into the agent (legitimate domain knowledge).
  - **L4** — L3 + a forced method-research / fitness-analysis step before the agent commits.

The contamination blacklist (L2+) blocks answer-revealing hosts at the network layer; the per-host
rationale is documented inline on ``CONTAMINATION_BLACKLIST`` below. L3+ injects method knowledge as a
pinned offline menu, never live.

This module is the single source of policy truth. Consumers import :func:`harness_policy`.
"""

from __future__ import annotations

from typing import Any

CONTAMINATION_BLACKLIST = [
    "github.com",  # OP task repos (solution + scoring code); host-level blocking can't single out a path
    "githubusercontent.com",  # raw.githubusercontent.com OP content
    "openproblems.bio",  # the Open Problems website (results / leaderboards)
    "openproblems-data",  # the openproblems-data S3 bucket (hosts solution.h5ad + published results)
    "cziscience.com",  # CELLxGENE (cellxgene.cziscience.com) — the original source of the labelled cells
    "cellxgene-census",  # the CELLxGENE Census data buckets
]

_LEVEL_ORDER = ("L1", "L2", "L3", "L4")


def harness_policy(level: str) -> dict[str, Any]:
    """Resolve a harness level (L1-L4) to its run policy.

    Returns the network ``mode`` (``block_all`` / ``blacklist``), the host ``blacklist``, whether the
    curated method menu is injected (L3+), and whether a forced method-research step is required (L4).
    """
    normalized = (level or "L1").upper()
    if normalized == "L1":
        return {
            "level": "L1",
            "network_allowed": False,
            "mode": "block_all",
            "blacklist": [],
            "injected_resources": False,
            "force_research": False,
        }
    if normalized in ("L2", "L3", "L4"):
        return {
            "level": normalized,
            "network_allowed": True,
            "mode": "blacklist",
            "blacklist": list(CONTAMINATION_BLACKLIST),
            "injected_resources": normalized in ("L3", "L4"),
            "force_research": normalized == "L4",
        }
    raise ValueError(f"unknown harness level: {level!r} (expected L1, L2, L3, or L4)")


def aggregate_harness_ladder(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a (model x level) grid into the value-of-harness view.

    Each cell is ``{model, level, summary}`` where summary carries per-run means. Returns per-model
    scores at each level plus the **harness lift** (top level - L1), ordered by L1 capability so
    diminishing returns can be inspected directly.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        summary = cell.get("summary", {})
        by_model.setdefault(cell["model"], {})[cell["level"]] = {
            "score_mean": round(float(summary.get("score_mean", 0.0)), 4),
            "score_std": round(float(summary.get("score_std", 0.0)), 4),
            "pass_rate": round(float(summary.get("pass_rate", 0.0)), 3),
            "mean_cost_usd": round(float(summary.get("mean_cost_per_run_usd", 0.0)), 6),
            "mean_iterations": round(float(summary.get("mean_iterations", 0.0)), 1),
            "mean_code_error_rate": round(float(summary.get("mean_code_error_rate", 0.0)), 4),
            "total_blocked_fetches": int(summary.get("total_blocked_fetches", 0)),
            "n_succeeded": int(summary.get("n_succeeded", summary.get("n_repeats", 0))),
            "n_repeats": int(summary.get("n_repeats", 0)),
        }

    models: list[dict[str, Any]] = []
    for model, levels in by_model.items():
        entry: dict[str, Any] = {"model": model, "levels": levels}
        if "L1" in levels:
            top_level = max(levels, key=lambda lvl: _LEVEL_ORDER.index(lvl) if lvl in _LEVEL_ORDER else -1)
            entry["l1_score"] = levels["L1"]["score_mean"]
            entry["top_level"] = top_level
            entry["harness_lift"] = round(levels[top_level]["score_mean"] - levels["L1"]["score_mean"], 4)
            entry["cost_delta_usd"] = round(levels[top_level]["mean_cost_usd"] - levels["L1"]["mean_cost_usd"], 6)
        models.append(entry)

    models.sort(key=lambda r: r.get("l1_score", 0.0))  # x-axis = base-model capability at L1
    return {"schema_version": "biopulse.harness_ladder.v1", "models": models}
