"""Two independent speed controls for serving: an operator SCALE and a CEILING.

Both exist because the measurements say the policy is not the thing running hot.

  * `dist_compare.py`: over 54 sampled chunks, the per-step motion the executed
    slice commands is AT OR BELOW the corpus at every percentile -- left wrist p99
    10.2 mm vs the corpus's 13.5 mm, max 11.3 vs 18.8; right wrist p99 11.5 vs
    13.1, max 12.4 vs 16.3. Hands likewise.
  * `gt_envelope.py`: the corpus's own per-step wrist speed varies 5.3x by phase
    (right wrist 1.2 mm/step in the quiet approach, 6.4 mm/step through the
    unscrew) and single steps reach 18.8 mm -- about 190 mm/s at the wrist.
  * `step_ratio.py`: the model's profile is as front-loaded as the corpus's
    (1.40x vs 1.39x), and the executed slice's speed is 1.09x GT on average,
    median 0.84x.

So "the hands sped up once the bottle was held, faster than during the descent"
is the TASK's own speed profile reproduced faithfully: the grasp-and-unscrew phase
is where the corpus is fastest. Nothing is malfunctioning, which is exactly why it
needs an operator control rather than a bug fix -- 190 mm/s near a held bottle is
legitimately unsafe to stand next to while the policy is still being trusted.

SCALE (`SHARPA_DEPLOY_SPEED_SCALE`, default 1.0) slows everything by a constant
factor. It costs progress per chunk proportionally and nothing else: the next
chunk re-anchors on wherever the arm actually reached, so a slowed chunk is not a
mis-aligned one. The anchor-lag relation (EXECUTE_START == RTC_STEPS) is untouched
because it is about WHICH steps are sent, not how far each one goes.

CEILING is derived from the corpus envelope and is inert inside it. It caps a
sampling excursion past anything the data ever asked for -- measured to engage on
0 of 54 in-distribution chunks, which is the point: it should be silent until
something is wrong.

Order matters: ceiling first, then scale. The ceiling is a statement about the
model's output; the scale is a statement about how fast the operator wants the
robot to move. Applying the scale first would let a slowed-down excursion slip
under the ceiling and never be reported.
"""

import os

import numpy as np

import repr_bridge as rb


# Corpus envelope, measured by gt_envelope.py over 6 episodes of unscrew_cap,
# on GT chunks built by the training code at the training stride. Worst single
# step: 18.53 mm (left wrist), 16.40 mm (right), 0.488 rad (left hand), 0.365 rad
# (right hand), all per 100 ms period.
CORPUS_MAX_WRIST_STEP_MM = 18.6
CORPUS_MAX_HAND_STEP_RAD = 0.49

# 1.25x the corpus worst case. Sampling noise around a legitimately fast window
# must not trip the ceiling, and a runaway must be caught well before the
# joint-space clamp in actor_ik_node (ARM_MAX_STEP_RAD 0.25 rad = 2.5 rad/s,
# most of a UR5's rated joint speed, and HAND_MAX_STEP_RAD 0.4 rad which is
# BELOW the corpus's own widest finger step -- neither is a task-aware bound).
HEADROOM = 1.25

MAX_WRIST_STEP_MM = CORPUS_MAX_WRIST_STEP_MM * HEADROOM   # 23.25 mm / 100 ms
MAX_HAND_STEP_RAD = CORPUS_MAX_HAND_STEP_RAD * HEADROOM   # 0.6125 rad / 100 ms

SCALE_ENV = "SHARPA_DEPLOY_SPEED_SCALE"


def read_scale(environ=None) -> float:
    """The operator's speed scale, from the environment. 1.0 = full speed.

    Rejected rather than clamped when out of range: a typo that silently becomes
    a valid speed is worse than a server that refuses to start. Zero is refused
    too -- "don't move" is what dry_run is for, and a zero-motion chunk that
    still reports success would look like a healthy policy holding position.
    """
    raw = (environ if environ is not None else os.environ).get(SCALE_ENV, "")
    text = str(raw).strip()
    if not text:
        return 1.0
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{SCALE_ENV}={raw!r} is not a number") from exc
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            f"{SCALE_ENV}={raw!r} must be in (0, 1]; 1.0 is full speed"
        )
    return value


def _scale_wrist(chunk: np.ndarray, side: str, factor: float) -> None:
    """Scale one wrist's relative displacement about the anchor, in place.

    Translation and rotation are scaled by the same factor. Slowing translation
    while leaving rotation alone would be a different path, not a slower one --
    and a different path through a grasp is worse than a slow one.
    """
    for key in (f"{side}_wrist_trans", f"{side}_wrist_rotvec"):
        sl = rb.SLICES[key]
        chunk[:, sl] = chunk[:, sl] * np.float32(factor)


def _scale_hand(chunk: np.ndarray, side: str, factor: float) -> None:
    """Scale one hand's motion about its FIRST row, in place.

    Hand joints are absolute, so the first row is the posture the chunk starts
    from and must not move; only the excursion away from it is scaled.
    """
    sl = rb.SLICES[f"{side}_hand_joints"]
    block = np.asarray(chunk[:, sl], dtype=np.float64)
    first = block[0:1]
    chunk[:, sl] = (first + (block - first) * factor).astype(np.float32)


def wrist_steps_mm(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step wrist translation in mm. Step 0 is measured from the anchor."""
    block = np.asarray(
        chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64
    ) * 1000.0
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    return np.linalg.norm(np.diff(with_anchor, axis=0), axis=1)


def hand_steps_rad(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step max joint change of one hand, rad. Absolute, so no anchor row."""
    block = np.asarray(chunk[:, rb.SLICES[f"{side}_hand_joints"]], dtype=np.float64)
    return np.abs(np.diff(block, axis=0)).max(axis=1)


def govern(chunk56: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """Apply the corpus ceiling, then the operator scale. Returns (chunk, report).

    ``chunk56`` is ``[horizon, 56]`` in RAW v5 units as the model emits it:
    anchor-relative wrists, absolute hands. The returned chunk is the same layout,
    safe to hand to ``decode_action_chunk``.

    ``report["capped"]`` lists the channels the ceiling engaged on -- normally
    empty. ``report["scale"]`` echoes the scale actually applied.
    """
    chunk = np.array(chunk56, dtype=np.float32, copy=True)
    if chunk.ndim != 2 or chunk.shape[1] != rb.ACTION_DIM:
        raise ValueError(f"expected [horizon, {rb.ACTION_DIM}], got {chunk.shape}")
    if not np.isfinite(scale) or not 0.0 < float(scale) <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale!r}")

    report: dict = {"capped": [], "scale": float(scale),
                    "max_wrist_step_mm": 0.0, "max_hand_step_rad": 0.0}

    for side in rb.SIDES:
        steps = wrist_steps_mm(chunk, side)
        worst = float(steps.max()) if steps.size else 0.0
        report["max_wrist_step_mm"] = max(report["max_wrist_step_mm"], worst)
        if worst > MAX_WRIST_STEP_MM:
            factor = MAX_WRIST_STEP_MM / worst
            _scale_wrist(chunk, side, factor)
            report["capped"].append(
                {"channel": f"{side}_wrist", "step_mm": round(worst, 2),
                 "cap_mm": round(MAX_WRIST_STEP_MM, 2), "factor": round(factor, 4)}
            )

    for side in rb.SIDES:
        steps = hand_steps_rad(chunk, side)
        worst = float(steps.max()) if steps.size else 0.0
        report["max_hand_step_rad"] = max(report["max_hand_step_rad"], worst)
        if worst > MAX_HAND_STEP_RAD:
            factor = MAX_HAND_STEP_RAD / worst
            _scale_hand(chunk, side, factor)
            report["capped"].append(
                {"channel": f"{side}_hand", "step_rad": round(worst, 4),
                 "cap_rad": round(MAX_HAND_STEP_RAD, 4), "factor": round(factor, 4)}
            )

    if float(scale) != 1.0:
        for side in rb.SIDES:
            _scale_wrist(chunk, side, float(scale))
            _scale_hand(chunk, side, float(scale))
    return chunk, report
