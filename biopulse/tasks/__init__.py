"""Task registry: one record per task_type, the single place a task is defined.

See ``biopulse/tasks/registry.py``. The registry collapses the ~15 ``task_type ==`` dispatch sites
(scorer selection, required outputs, pack generation, source discovery, control generation, pack
validation) into one ``TaskRecord`` lookup. Architectural invariant: the registry holds each scorer as
a task-specific callable and never reads its internals.
"""

from .registry import REGISTRY, FileSpec, TaskRecord, get, required_outputs

__all__ = ["REGISTRY", "FileSpec", "TaskRecord", "get", "required_outputs"]
