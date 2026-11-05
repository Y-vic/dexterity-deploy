"""Command source arbitration for the dual UR5 outlet.

`ur_node` is the ONLY process that writes to the UR arms, so it must decide
which of the three command producers -- bias, vive, actor -- is allowed to
drive at any moment. That decision follows the UR state machine owned by
`status_node`:

    stopped  -> no commands accepted
    homing   -> ONLY bias accepted (active reset transaction)
    homed    -> ONLY bias accepted (hold / correction)
    running  -> ONLY the active teleop-or-deploy source accepted

Keeping this as a pure function of (ur_state, active_source, command_source)
means it is fully testable without RTDE, ROS, or hardware. `ur_node` holds no
arbitration logic of its own; it just asks this module.

Rejections are explicit and carry a reason string. Silently dropping a command
would make a stalled arm indistinguishable from a disconnected producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class UrState(IntEnum):
    """Mirrors ur_msgs/ControlStatus UR_* constants."""

    STOPPED = 0
    HOMED = 1
    RUNNING = 2
    HOMING = 3
    STARTING = 4


class Source(IntEnum):
    """Mirrors ur_msgs/ArmCommand SOURCE_* constants."""

    UNKNOWN = 0
    BIAS = 1
    VIVE = 2
    ACTOR = 3


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.accepted


ACCEPT = Decision(True)


def arbitrate(
    ur_state: UrState,
    active_source: Source,
    command_source: Source,
) -> Decision:
    """Decide whether `command_source` may drive the arms right now.

    `active_source` is the source `status_node` has designated for the RUNNING
    state; it is ignored outside RUNNING.
    """
    if command_source == Source.UNKNOWN:
        return Decision(False, "command has no source; refusing to guess")

    if ur_state == UrState.STOPPED:
        return Decision(
            False,
            f"UR is stopped; {command_source.name.lower()} command rejected",
        )

    if ur_state == UrState.STARTING:
        return Decision(
            False,
            f"UR is starting; {command_source.name.lower()} command rejected",
        )

    if ur_state in (UrState.HOMING, UrState.HOMED):
        # Only bias may move the arms before they are declared running. This is
        # what makes "power on" safe: a stale actor plan or a live vive stream
        # cannot start moving the robot just because the node came up.
        if command_source == Source.BIAS:
            return ACCEPT
        return Decision(
            False,
            f"UR is {ur_state.name.lower()}; only bias may command "
            f"(got {command_source.name.lower()})",
        )

    if ur_state == UrState.RUNNING:
        if command_source == active_source:
            return ACCEPT
        return Decision(
            False,
            f"active source is {active_source.name.lower()}; "
            f"rejected {command_source.name.lower()}",
        )

    return Decision(False, f"unknown UR state {ur_state!r}")
