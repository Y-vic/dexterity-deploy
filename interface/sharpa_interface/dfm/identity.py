"""Native ready identity retained across explicit transport recovery."""

from __future__ import annotations


def native_identity(metadata):
    fields = ("model_id", "task_id", "run_id", "prompt", "checkpoint_path")
    if any(not isinstance(metadata.get(field), str) or not metadata[field] for field in fields):
        raise ValueError("Native ready requires model/task/run/prompt/checkpoint identity")
    return tuple(metadata[field] for field in fields)
