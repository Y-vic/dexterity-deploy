from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import ipaddress
import json
import math
from typing import Any, Sequence


ROBOT_IPS = {
    "left": "192.168.56.20",
    "right": "192.168.56.10",
}
NORMAL_SAFETY_MODE = 1
RUNNING_ROBOT_MODE = 7
STOPPED_RUNTIME_STATE = 1
MAX_DISTANCE_MM = 5.0
MAX_SPEED_M_S = 0.02
MAX_ACCELERATION_M_S2 = 0.1


@dataclass(frozen=True)
class JogRequest:
    side: str
    robot_ip: str
    axis: str
    distance_mm: float
    speed_m_s: float = 0.01
    acceleration_m_s2: float = 0.05

    def __post_init__(self) -> None:
        if self.side not in ROBOT_IPS:
            raise ValueError("side must be left or right")
        try:
            address = ipaddress.ip_address(self.robot_ip)
        except ValueError as exc:
            raise ValueError("robot_ip must be a valid IPv4 address") from exc
        if not isinstance(address, ipaddress.IPv4Address):
            raise ValueError("robot_ip must be a valid IPv4 address")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        distance = _finite_float(self.distance_mm, "distance_mm")
        if distance == 0.0 or abs(distance) > MAX_DISTANCE_MM:
            raise ValueError(
                f"distance_mm must be nonzero and within +/-{MAX_DISTANCE_MM:g}"
            )
        speed = _bounded_positive(
            self.speed_m_s,
            "speed_m_s",
            MAX_SPEED_M_S,
        )
        acceleration = _bounded_positive(
            self.acceleration_m_s2,
            "acceleration_m_s2",
            MAX_ACCELERATION_M_S2,
        )
        object.__setattr__(self, "robot_ip", str(address))
        object.__setattr__(self, "distance_mm", distance)
        object.__setattr__(self, "speed_m_s", speed)
        object.__setattr__(self, "acceleration_m_s2", acceleration)

    @property
    def confirmation_token(self) -> str:
        return f"{self.side}:{self.axis}:{self.distance_mm:+g}mm"


@dataclass(frozen=True)
class RobotSnapshot:
    actual_q: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    robot_mode: int
    safety_mode: int
    runtime_state: int


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(output):
        raise ValueError(f"{field} must be a finite number")
    return output


def _bounded_positive(value: object, field: str, maximum: float) -> float:
    output = _finite_float(value, field)
    if output <= 0.0 or output > maximum:
        raise ValueError(f"{field} must be in (0, {maximum:g}]")
    return output


def _finite_vector(value: Sequence[object], field: str) -> tuple[float, ...]:
    if len(value) != 6:
        raise ValueError(f"{field} must contain exactly 6 values")
    return tuple(_finite_float(component, field) for component in value)


def build_target_pose(
    current_tcp_pose: Sequence[object],
    request: JogRequest,
) -> tuple[float, ...]:
    target = list(_finite_vector(current_tcp_pose, "current_tcp_pose"))
    target[{"x": 0, "y": 1, "z": 2}[request.axis]] += (
        request.distance_mm / 1000.0
    )
    return tuple(target)


def validate_execution_snapshot(snapshot: RobotSnapshot) -> None:
    if snapshot.safety_mode != NORMAL_SAFETY_MODE:
        raise RuntimeError(
            f"safety mode must be NORMAL ({NORMAL_SAFETY_MODE}); "
            f"received {snapshot.safety_mode}"
        )
    if snapshot.robot_mode != RUNNING_ROBOT_MODE:
        raise RuntimeError(
            f"robot mode must be RUNNING ({RUNNING_ROBOT_MODE}); "
            f"received {snapshot.robot_mode}"
        )
    if snapshot.runtime_state != STOPPED_RUNTIME_STATE:
        raise RuntimeError(
            f"PolyScope program must be STOPPED ({STOPPED_RUNTIME_STATE}); "
            f"received {snapshot.runtime_state}"
        )


def _read_snapshot(receiver: Any) -> RobotSnapshot:
    if not receiver.isConnected():
        raise RuntimeError("RTDE receive interface is not connected")
    return RobotSnapshot(
        actual_q=_finite_vector(receiver.getActualQ(), "actual_q"),
        tcp_pose=_finite_vector(receiver.getActualTCPPose(), "tcp_pose"),
        robot_mode=int(receiver.getRobotMode()),
        safety_mode=int(receiver.getSafetyMode()),
        runtime_state=int(receiver.getRuntimeState()),
    )


def _load_rtde() -> tuple[Any, Any]:
    try:
        receive_module = importlib.import_module("rtde_receive")
        control_module = importlib.import_module("rtde_control")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ur-rtde is unavailable; run: "
            "python3 -m pip install --user --no-deps ur-rtde==1.6.0"
        ) from exc
    return receive_module, control_module


def _disconnect(interface: Any) -> None:
    disconnect = getattr(interface, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception:
            pass


def _stop_control(control: Any, *, emergency: bool) -> None:
    if emergency:
        try:
            control.stopL(0.1)
        except Exception:
            pass
    try:
        control.stopScript()
    except Exception:
        pass
    _disconnect(control)


def _result_payload(
    request: JogRequest,
    snapshot: RobotSnapshot,
    target_pose: Sequence[float],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "side": request.side,
        "robot_ip": request.robot_ip,
        "dry_run": dry_run,
        "robot_mode": snapshot.robot_mode,
        "safety_mode": snapshot.safety_mode,
        "runtime_state": snapshot.runtime_state,
        "actual_q": snapshot.actual_q,
        "current_tcp_pose": snapshot.tcp_pose,
        "target_tcp_pose": tuple(target_pose),
        "distance_mm": request.distance_mm,
        "speed_m_s": request.speed_m_s,
        "acceleration_m_s2": request.acceleration_m_s2,
        "confirmation_token": request.confirmation_token,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first Cartesian jog for the SharpA UR5 pair."
    )
    parser.add_argument("--side", choices=tuple(ROBOT_IPS), required=True)
    parser.add_argument("--robot-ip")
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    parser.add_argument("--speed-m-s", type=float, default=0.01)
    parser.add_argument("--acceleration-m-s2", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    try:
        request = JogRequest(
            side=options.side,
            robot_ip=options.robot_ip or ROBOT_IPS[options.side],
            axis=options.axis,
            distance_mm=options.distance_mm,
            speed_m_s=options.speed_m_s,
            acceleration_m_s2=options.acceleration_m_s2,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if options.execute and options.confirm != request.confirmation_token:
        parser.error(
            "--execute requires --confirm "
            f"{request.confirmation_token!r} from a successful dry-run"
        )

    receive_module, control_module = _load_rtde()
    receiver = None
    control = None
    movement_started = False
    try:
        receiver = receive_module.RTDEReceiveInterface(request.robot_ip, 10.0)
        snapshot = _read_snapshot(receiver)
        target_pose = build_target_pose(snapshot.tcp_pose, request)
        print(
            json.dumps(
                _result_payload(
                    request,
                    snapshot,
                    target_pose,
                    dry_run=not options.execute,
                ),
                separators=(",", ":"),
            ),
            flush=True,
        )
        if not options.execute:
            return 0

        validate_execution_snapshot(snapshot)
        flags = control_module.RTDEControlInterface.FLAG_UPLOAD_SCRIPT
        control = control_module.RTDEControlInterface(
            request.robot_ip,
            125.0,
            flags,
            50013,
            -1,
        )
        live_snapshot = _read_snapshot(receiver)
        if live_snapshot.safety_mode != NORMAL_SAFETY_MODE:
            raise RuntimeError("safety mode changed before movement")
        if live_snapshot.robot_mode != RUNNING_ROBOT_MODE:
            raise RuntimeError("robot mode changed before movement")
        movement_started = True
        succeeded = control.moveL(
            list(target_pose),
            request.speed_m_s,
            request.acceleration_m_s2,
            False,
        )
        if succeeded is not True:
            raise RuntimeError("RTDE moveL did not report success")
        final_pose = _finite_vector(receiver.getActualTCPPose(), "final_tcp_pose")
        position_error = math.dist(final_pose[:3], target_pose[:3])
        if position_error > 0.001:
            raise RuntimeError(
                f"final TCP position error is {position_error * 1000.0:.3f} mm"
            )
        print(
            json.dumps(
                {
                    "moved": True,
                    "side": request.side,
                    "robot_ip": request.robot_ip,
                    "final_tcp_pose": final_pose,
                    "position_error_mm": position_error * 1000.0,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        movement_started = False
        return 0
    finally:
        if control is not None:
            _stop_control(control, emergency=movement_started)
        if receiver is not None:
            _disconnect(receiver)


if __name__ == "__main__":
    raise SystemExit(main())
