"""Shared UR5 + SharpA kinematics.

The single forward/inverse kinematics entry point for the whole system. Both
teleop and deploy import from here so the observation path and the execution
path can never drift apart.

Import the facade:

    from ur_common.kinematics import UrCommonKinematics, SafetyConfig

`kinematics` is not imported eagerly here: it pulls in the compiled UR5 ikfast
bindings, which are CPython 3.10 only. Pure-geometry consumers can use
`ur_common.geometry` on any interpreter.
"""

__version__ = "0.1.0"
