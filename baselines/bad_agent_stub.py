#!/usr/bin/env python
"""Negative-control baseline used to exercise scorer failure paths.

Writes a report but no prediction artifact, so the scorer fails the schema gate (missing required
output) rather than awarding a score. This confirms the harness rejects runs that produce no answer
artifact.
"""
from __future__ import annotations

from pathlib import Path


def main() -> int:
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/report.md").write_text(
        "# Incomplete run\n\nThis agent did not produce a prediction artifact.\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
