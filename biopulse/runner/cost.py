"""Agent runtime + token/cost contract.

Additive, deterministic, offline. Produces three per-run artifacts that turn the
bare ``agent_id`` string into real provenance:

- ``agent_profile.json``  -- who/what ran (agent_kind, provider, model, runtime)
- ``token_usage.json``    -- token counts per model, with observed|estimated|none provenance
- ``cost_summary.json``   -- cost computed offline from a checked-in price table

Cost is advisory: it is computed here, never read from a provider, and never gates
``final_score`` / ``passed`` / ``safety_gate_passed`` (those stay in evaluator_results.json).
Token field names mirror Inspect AI's ModelUsage so the optional Inspect bridge is 1:1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

#: Sentinel model id for runs with no LLM (deterministic baselines). Always priced at zero.
NO_LLM_MODEL = "none/deterministic"

#: Cached input is billed far below fresh input. These are standard approximations applied to the
#: per-model input price: a cache *read* ~10% of input, a cache *write* (creation) ~1.25x. Exact
#: rates are provider/model-specific; cost is advisory, so approximate cache pricing is preferable to
#: assuming no discount.
#: Convention: a run's ``input_tokens`` is the TOTAL input (cached reads/writes are a subset of it),
#: so provider adapters must normalize to that before we split off the cached portions here.
CACHE_READ_PRICE_MULTIPLIER = 0.1
CACHE_WRITE_PRICE_MULTIPLIER = 1.25


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_price_table_path() -> Path:
    return _repo_root() / "pricing" / "price_table.json"


def load_price_table(path: Path | str | None = None) -> dict[str, Any]:
    """Load the offline price table. Falls back to an empty table (cost 0) if missing."""
    table_path = Path(path) if path else default_price_table_path()
    if not table_path.exists():
        return {"schema_version": "biopulse.price_table.v1", "pricing_version": "unknown", "currency": "USD", "prices": {}}
    return json.loads(table_path.read_text(encoding="utf-8"))


def none_model_usage() -> dict[str, Any]:
    """A ModelUsage record for a run that used no LLM. None (not 0) marks 'not reported'."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_cache_read": None,
        "input_tokens_cache_write": None,
        "reasoning_tokens": None,
        "measurement_method": "none",  # observed | estimated | none
        "estimator": None,  # required (non-null) when measurement_method == "estimated"
        "source_ref": None,  # where observed counts were read from
    }


def agent_profile(
    run_id: str,
    agent_id: str,
    agent_kind: str,
    model: str = NO_LLM_MODEL,
    provider: str = "none",
    runtime: str = "python_subprocess",
    adapter_config: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": "biopulse.agent_profile.v1",
        "run_id": run_id,
        "agent_id": agent_id,
        "agent_kind": agent_kind,  # baseline | agent_command | llm_agent | inspect
        "provider": provider,
        "model": model,  # "<provider>/<model>" join key; NO_LLM_MODEL for non-LLM runs
        "runtime": runtime,
        "adapter_config": adapter_config or {},
        "tags": tags or [],
    }


def token_usage(
    run_id: str,
    model_usage: Optional[dict[str, dict[str, Any]]] = None,
    role_usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a token_usage document. Defaults to a single no-LLM record."""
    if model_usage is None:
        model_usage = {NO_LLM_MODEL: none_model_usage()}
    return {
        "schema_version": "biopulse.token_usage.v1",
        "run_id": run_id,
        "model_usage": model_usage,
        "role_usage": role_usage or {},  # phase -> ModelUsage; empty until agents emit phase events
    }


def cost_summary(run_id: str, token_usage_doc: dict[str, Any], price_table: dict[str, Any]) -> dict[str, Any]:
    """Compute cost offline = tokens x unit price, per model. Advisory only; never gates scoring."""
    prices = price_table.get("prices", {})
    pricing_version = price_table.get("pricing_version", "unknown")
    breakdown: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    total_cost = 0.0
    cost_is_estimated = False

    for model, usage in token_usage_doc.get("model_usage", {}).items():
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("input_tokens_cache_read", 0) or 0)
        cache_write = int(usage.get("input_tokens_cache_write", 0) or 0)
        method = str(usage.get("measurement_method", "none"))
        price = prices.get(model, {})
        ppi = float(price.get("price_per_1k_input", 0.0))
        ppo = float(price.get("price_per_1k_output", 0.0))
        # input_tokens is the total input; the cached read/write portions are billed at a discount.
        uncached_in = max(0, in_tok - cache_read - cache_write)
        cost = (
            (uncached_in / 1000.0) * ppi
            + (cache_read / 1000.0) * ppi * CACHE_READ_PRICE_MULTIPLIER
            + (cache_write / 1000.0) * ppi * CACHE_WRITE_PRICE_MULTIPLIER
            + (out_tok / 1000.0) * ppo
        )
        if method == "estimated" and cost > 0.0:
            cost_is_estimated = True
        total_in += in_tok
        total_out += out_tok
        total_cost += cost
        breakdown.append(
            {
                "model": model,
                "input_tokens": in_tok,
                "cached_input_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "output_tokens": out_tok,
                "price_per_1k_input": ppi,
                "price_per_1k_output": ppo,
                "computed_cost_usd": round(cost, 6),
                "measurement_method": method,
            }
        )

    return {
        "schema_version": "biopulse.cost_summary.v1",
        "run_id": run_id,
        "pricing_version": pricing_version,
        "cost_is_estimated": cost_is_estimated,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round(total_cost, 6),
        "model_breakdown": breakdown,
    }


def write_runtime_artifacts(
    run_dir: Path,
    run_id: str,
    agent_id: str,
    agent_kind: str,
    model: str = NO_LLM_MODEL,
    provider: str = "none",
    model_usage: Optional[dict[str, dict[str, Any]]] = None,
    price_table: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write agent_profile.json, token_usage.json, cost_summary.json into run_dir."""
    from .evidence import write_json  # local import to avoid a cycle

    table = price_table if price_table is not None else load_price_table()
    usage = token_usage(run_id, model_usage=model_usage)
    profile = agent_profile(run_id, agent_id, agent_kind, model=model, provider=provider)
    summary = cost_summary(run_id, usage, table)
    write_json(run_dir / "agent_profile.json", profile)
    write_json(run_dir / "token_usage.json", usage)
    write_json(run_dir / "cost_summary.json", summary)
    return summary
