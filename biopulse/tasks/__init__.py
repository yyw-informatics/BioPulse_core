"""Task registry: one record per task_type, the central place a task is defined.

See ``biopulse/tasks/registry.py``. The registry keeps scorer selection, required outputs, pack
generation, source discovery, control generation, and pack validation aligned through one
``TaskRecord`` lookup.
"""

from .registry import REGISTRY, FileSpec, TaskRecord, get, required_outputs

__all__ = ["REGISTRY", "FileSpec", "TaskRecord", "get", "required_outputs"]
