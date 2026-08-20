"""executed_steps semantics — FROZEN.

Both UR and PND currently disagree on what `executed_steps` means when it
lands at the server:

  - PND legacy executor: cumulative,
        executed_steps = execute_start + executed_in_slice
  - UR YNS (sharpa_policy_v3_client/action_node.py): slice-local,
        executed_steps = executed_in_slice

Chosen: cumulative. Rationale — the server needs a monotonically increasing
step count to compute regret/lag; slice-local resets on every replan and
makes the server-side scheduler brittle when replans occur inside a plan.

Every embodiment builds its ExecutionFeedback through this function. If a
model retraining wants to change this, bump the ExecutionFeedback
msg version instead of silently changing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProgress:
    execute_start: int
    executed_in_slice: int
    revision: int

    @property
    def executed_steps(self) -> int:
        # FROZEN: cumulative.
        return self.execute_start + self.executed_in_slice


def make_progress(execute_start: int, executed_in_slice: int, revision: int) -> ExecutionProgress:
    if execute_start < 0 or executed_in_slice < 0:
        raise ValueError(
            f"non-negative required: execute_start={execute_start}, "
            f"executed_in_slice={executed_in_slice}"
        )
    return ExecutionProgress(
        execute_start=execute_start,
        executed_in_slice=executed_in_slice,
        revision=revision,
    )
