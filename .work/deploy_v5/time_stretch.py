"""Path-preserving execution slowdown for MoT action chunks.

The model emits ten samples at 10 Hz.  Multiplying their displacement by a
speed factor makes the robot slower, but also prevents it from reaching the
predicted wrist and hand endpoints.  That changes the task, especially during
grasp closure.  This module instead resamples the complete path onto more 10 Hz
samples: velocity drops while every original endpoint remains on the path and
the final endpoint is unchanged.
"""

from __future__ import annotations

import os

import numpy as np

import repr_bridge as rb


TIME_SCALE_ENV = "SHARPA_DEPLOY_TIME_SCALE"


def read_time_scale(environ=None) -> float:
    """Read the execution-rate scale.  One is native speed; 0.3 is 30%."""
    raw = (environ if environ is not None else os.environ).get(TIME_SCALE_ENV, "")
    text = str(raw).strip()
    if not text:
        return 1.0
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{TIME_SCALE_ENV}={raw!r} is not a number") from exc
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{TIME_SCALE_ENV}={raw!r} must be in (0, 1]")
    return value


def stretched_length(horizon: int, time_scale: float) -> int:
    """Number of 10 Hz commands needed to traverse ``horizon`` at the scale."""
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(time_scale) or not 0.0 < float(time_scale) <= 1.0:
        raise ValueError(f"time_scale must be in (0, 1], got {time_scale!r}")
    return int(np.ceil(int(horizon) / float(time_scale)))


def stretch_chunk(
    chunk56: np.ndarray,
    current_hands: dict[str, np.ndarray],
    time_scale: float,
) -> np.ndarray:
    """Resample a raw relative-wrist/absolute-hand chunk without moving its end.

    Wrist channels have a zero anchor because the representation is relative to
    the current wrist pose.  Hand channels are absolute, so their anchor is the
    measured current joint vector.  Linear interpolation is performed in the
    exact representation consumed by ``decode_action_chunk``.
    """
    chunk = np.asarray(chunk56, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != rb.ACTION_DIM:
        raise ValueError(f"expected [horizon, {rb.ACTION_DIM}], got {chunk.shape}")
    output_length = stretched_length(chunk.shape[0], time_scale)
    if output_length == chunk.shape[0]:
        return np.array(chunk, copy=True)

    anchor = np.zeros((rb.ACTION_DIM,), dtype=np.float32)
    for side in rb.SIDES:
        hand = np.asarray(current_hands[side], dtype=np.float32).reshape(-1)
        if hand.shape != (rb.HAND_DIM,) or not np.isfinite(hand).all():
            raise ValueError(f"current {side} hand must be finite [{rb.HAND_DIM}]")
        anchor[rb.SLICES[f"{side}_hand_joints"]] = hand

    source = np.vstack([anchor, chunk]).astype(np.float64)
    source_t = np.arange(chunk.shape[0] + 1, dtype=np.float64)
    target_t = np.minimum(
        np.arange(1, output_length + 1, dtype=np.float64) * float(time_scale),
        float(chunk.shape[0]),
    )
    stretched = np.empty((output_length, rb.ACTION_DIM), dtype=np.float32)
    for column in range(rb.ACTION_DIM):
        stretched[:, column] = np.interp(
            target_t, source_t, source[:, column]
        ).astype(np.float32)

    # Make endpoint preservation bit-exact instead of relying on interpolation
    # rounding.  This is the property the diagnostic experiment depends on.
    stretched[-1] = chunk[-1]
    return stretched
