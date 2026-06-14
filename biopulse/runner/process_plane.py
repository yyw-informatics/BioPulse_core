"""Agent Process Plane: behavioral metrics derived from a run's process-event time series.

The Scientific Artifact Plane asks whether the submitted artifact is correct. This plane summarizes how
the agent produced it: model calls, code execution, code failures, network access, and whether the run
finished cleanly or stopped at an iteration cap. These metrics are derived from ``process_events.jsonl``
events emitted by the agent loop, so behavior is measured from structured traces.
"""

from __future__ import annotations

from typing import Any


def summarize_process(
    events: list[dict[str, Any]],
    *,
    finished: bool,
    iterations: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Reduce the process-event stream to a behavioral summary for the Agent Process Plane."""
    model_calls = [e for e in events if e.get("event_type") == "model_call"]
    code_execs = [e for e in events if e.get("event_type") == "code_exec"]
    web_fetches = [e for e in events if e.get("event_type") == "web_fetch"]
    # Separate allowed fetches from blocked attempts. Allowed fetches are audit targets; blocked fetches
    # indicate the contamination policy was exercised.
    allowed_fetches = [e for e in web_fetches if not e.get("details", {}).get("blocked")]
    blocked_fetches = [e for e in web_fetches if e.get("details", {}).get("blocked")]

    n_code_failures = sum(1 for e in code_execs if not e.get("details", {}).get("ok"))
    n_timeouts = sum(1 for e in code_execs if e.get("details", {}).get("timed_out"))
    # Turns where the model responded without runnable code and did not finish. These indicate format
    # non-compliance or unsuccessful planning/tool-use attempts.
    n_no_code_turns = sum(
        1
        for e in model_calls
        if not e.get("details", {}).get("has_code") and not e.get("details", {}).get("is_finish")
    )

    return {
        "schema_version": "biopulse.process_summary.v1",
        "iterations": iterations,
        "max_iterations": max_iterations,
        "hit_iteration_cap": iterations >= max_iterations and not finished,
        "finished_cleanly": bool(finished),  # ended on an explicit FINISH, not by running out
        "n_model_calls": len(model_calls),
        "n_code_execs": len(code_execs),
        "n_code_failures": n_code_failures,
        "code_error_rate": round(n_code_failures / len(code_execs), 4) if code_execs else 0.0,
        "n_no_code_turns": n_no_code_turns,
        "n_timeouts": n_timeouts,
        "n_web_fetches": len(allowed_fetches),  # network reaches that went through (audit for leakage)
        "n_blocked_fetches": len(blocked_fetches),  # blocked attempts to answer sources (contamination caught)
        "n_events": len(events),
    }
