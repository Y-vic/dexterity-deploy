"""Command source arbitration for the SharpA hardware outlet."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SharpaState(IntEnum):
    STOPPED = 0
    RUNNING = 1
    HOMING = 2
    HOMED = 3
    STARTING = 4


class Source(IntEnum):
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
    sharpa_state: SharpaState,
    active_source: Source,
    command_source: Source,
) -> Decision:
    if command_source == Source.UNKNOWN:
        return Decision(False, "command has no source; refusing to guess")
    if sharpa_state == SharpaState.STOPPED:
        return Decision(
            False,
            f"SharpA is stopped; {command_source.name.lower()} command rejected",
        )
    if sharpa_state == SharpaState.STARTING:
        return Decision(
            False,
            f"SharpA is starting; {command_source.name.lower()} command rejected",
        )
    if sharpa_state in (SharpaState.HOMING, SharpaState.HOMED):
        if command_source == Source.BIAS:
            return ACCEPT
        return Decision(
            False,
            f"SharpA is {sharpa_state.name.lower()}; only bias may command "
            f"(got {command_source.name.lower()})",
        )
    if command_source == active_source:
        return ACCEPT
    return Decision(
        False,
        f"active source is {active_source.name.lower()}; "
        f"rejected {command_source.name.lower()}",
    )
