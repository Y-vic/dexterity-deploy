"""SharpA policy server v3.

Serves a MoT action checkpoint over the frozen SharpA v3 contract. The
representation it speaks is v5 (56D, relative wrist + absolute hand qpos), which
is what ties this file to a class of checkpoint rather than to one of them: any
run using the same representation is served by registering it, with no change
here. See ``repr_bridge`` for the layout and ``launch.sh`` for what a run
supplies.

Four endpoints, exactly as the workstation's frozen client expects them
(``interface/sharpa_interface/server/``):

    GET  /metadata   sharpa_policy_server.v3, including the metadata_format that
                     tells the workstation which buffers to fill
    POST /reset      new session; returns the AUTHORITATIVE metadata_format
    WS   /infer      binary msgpack, one observation in, one action out
    GET  /healthz    liveness only; the workstation polls this to decide the
                     model came up, and its ssh relay watches it to notice the
                     model going away

Wire rules that are not negotiable, each one having its own failure mode:

  * binary msgpack both ways, 64 MiB per message cap;
  * on /infer the server sends NOTHING until the client's first observation
    arrives -- a server that greets with metadata deadlocks the client, which is
    waiting to send;
  * every response echoes session_id/request_id/action_id/revision unchanged,
    because the client matches on identity and silently drops a mismatch;
  * ``execution.rtc`` is mandatory under action schema v5.

WHAT THIS SERVER OWNS, and the workstation deliberately does not:

The public interface carries ABSOLUTE wrist poses. This checkpoint was trained
on RELATIVE ones (v5), so the conversion lives here -- ``policy_client_node``
rejects a server that declares a non-absolute contract, by design: relative
representations are model-private.

TWO DIFFERENT ANCHORS, which is the thing most easily got wrong here. The ACTION
chunk is composed onto the CURRENT measured pose. The STATE rows are measured
against a pose ``STATE_ANCHOR_OFFSET`` frames further back, and are read from the
observation's own history so their spacing matches the 30 Hz the corpus was
recorded at. Their normalization statistics differ for the same reason.

Concurrency: connections are threaded, because the client holds its /infer
websocket open while calling /reset on a second connection -- a serial accept
loop deadlocks there. Inference itself is serialized by a lock.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# The numeric bridge lives beside this file so it can be unit-tested without a
# GPU, msgpack, or the training environment. It owns every representation
# conversion; nothing in this file does arithmetic on a pose.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repr_bridge import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_STRIDE,
    HAND_DIM,
    SLICES,
    STATE_ANCHOR_OFFSET,
    TRAIN_FPS,
    STATE_HISTORY_OFFSETS,
    build_state_rows,
    decode_action_chunk,
)
from speed_guard import (  # noqa: E402
    MAX_HAND_STEP_RAD,
    MAX_WRIST_STEP_MM,
    SCALE_ENV as SCALE_ENV_NAME,
    govern,
    read_scale,
)
from time_stretch import (  # noqa: E402
    TIME_SCALE_ENV as TIME_SCALE_ENV_NAME,
    read_time_scale,
    stretch_chunk,
    stretched_length,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("mot_server")

# --- schema constants, mirroring sharpa_interface.server.metadata ---------
SERVER_SCHEMA = "sharpa_policy_server.v3"
OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"
ACTION_SCHEMA = "sharpa_policy_action.v5"
METADATA_FORMAT_SCHEMA = "sharpa_policy_metadata_format.v1"
RESET_SCHEMA = "sharpa_policy_reset.v1"
TRANSPORT = "websocket+binary_msgpack"

IMAGE_NAMES = ("ego_cam", "left_wrist_cam", "right_wrist_cam")
SENSOR_NAMES = ("tau", "wrench", "deformation")

ERROR_SCHEMA = "sharpa_policy_error.v1"

MAX_MESSAGE_SIZE = 64 * 1024 * 1024
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# The canonical absolute-EEF ABI the workstation validates against. Any
# mismatch here and policy_client_node refuses to connect -- which is the
# intended behaviour, so these strings are copied exactly rather than derived.
EEF_CONTRACT = {
    "schema": "sharpa62_direct_absolute_eef.v1",
    "observation_eef_def": "absolute",
    "action_eef_def": "absolute",
    "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
    "server_xyz_direction_transform": "none",
}

# --- the trained schedule -------------------------------------------------
# ACTION_HORIZON/ACTION_STRIDE/TRAIN_FPS are imported from repr_bridge, not
# restated here: they describe what the 56D vectors MEAN, so two copies could
# drift and the wrong one would still produce well-shaped output.
#
# FREQUENCY_HZ is derived, and it is the one number in this file that a reader
# is most likely to "fix" wrongly. It must equal the rate the chunk's steps are
# spaced at -- fps/stride = 30/3 = 10 Hz -- because the client executes one step
# per period. Setting it to TRAIN_FPS would run every chunk 3x too fast, which
# reads as an aggressive policy rather than a units bug.
FREQUENCY_HZ = TRAIN_FPS / ACTION_STRIDE  # 10.0
TIME_SCALE = read_time_scale()
EXECUTION_HORIZON = stretched_length(ACTION_HORIZON, TIME_SCALE)

# --- deliberately sequential execution ------------------------------------
# RTC_STEPS = 0 is the diagnostic mode requested for model validation. The
# client waits for inference, executes the complete chunk, publishes done, and
# only then starts the next inference. The arm therefore holds its last command
# during inference instead of hiding latency by prefetching another chunk.
#
# This is intentionally slow and visibly stepped. It removes observation-to-
# takeover lag from the experiment, so a remaining boundary jump belongs to the
# model/observation path rather than RTC timing. Restore a measured non-zero RTC
# only after the policy can complete the task in this mode.
RTC_STEPS = 0
EXECUTE_START = RTC_STEPS
EXECUTE_LENGTH = EXECUTION_HORIZON - EXECUTE_START

# The pair above is only correct while these hold. Asserted at import: a future
# edit that raises RTC_STEPS without re-deriving EXECUTE_START would restore the
# silent retraction, and nothing downstream can see it.
if EXECUTE_START != RTC_STEPS:
    raise ValueError(
        f"EXECUTE_START {EXECUTE_START} must equal RTC_STEPS {RTC_STEPS}: the "
        "anchor lag at take-over is exactly RTC_STEPS steps"
    )
if not 0 < EXECUTE_LENGTH <= EXECUTION_HORIZON:
    raise ValueError(
        f"EXECUTE_LENGTH {EXECUTE_LENGTH} outside (0, {EXECUTION_HORIZON}]"
    )
if not 0 <= RTC_STEPS < EXECUTE_LENGTH:
    raise ValueError(
        f"RTC_STEPS {RTC_STEPS} must be < EXECUTE_LENGTH {EXECUTE_LENGTH}; the "
        "client rejects rtc >= execute_length"
    )


# --- msgpack, matching sharpa_interface.server.serialization exactly ------
# The ndarray descriptor is part of the wire contract, not an implementation
# detail: the client decodes `{__ndarray__, data, dtype, shape}` and rejects any
# other field. Encoding a plain list instead would be silently accepted as a
# list, and the client's validator would then reject the observation for a
# type reason that names the wrong cause.
import msgpack


def _encode_numpy(value):
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise ValueError(f"unsupported numpy dtype: {dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(order="C"),
            b"dtype": dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot msgpack-encode {type(value).__name__}")


def _descriptor_value(value, key: bytes):
    text_key = key.decode("ascii")
    if key in value:
        return value[key]
    if text_key in value:
        return value[text_key]
    raise ValueError(f"missing ndarray descriptor key: {text_key}")


def _decode_numpy(value):
    if b"__ndarray__" not in value and "__ndarray__" not in value:
        return value
    data = _descriptor_value(value, b"data")
    dtype_value = _descriptor_value(value, b"dtype")
    if isinstance(dtype_value, bytes):
        dtype_value = dtype_value.decode("ascii")
    shape = tuple(int(d) for d in _descriptor_value(value, b"shape"))
    return np.frombuffer(data, dtype=np.dtype(dtype_value)).reshape(shape).copy(order="C")


def packb(payload) -> bytes:
    encoded = msgpack.packb(
        payload, default=_encode_numpy, use_bin_type=True, strict_types=False
    )
    if len(encoded) > MAX_MESSAGE_SIZE:
        raise ValueError(f"encoded message is {len(encoded)} bytes")
    return encoded


def unpackb(payload):
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError(f"received message is {len(payload)} bytes")
    return msgpack.unpackb(
        payload, object_hook=_decode_numpy, raw=False, strict_map_key=False
    )


# --- what this model asks the workstation to send -------------------------
# The metadata_format is a REQUEST, not a description: the workstation fills
# exactly these buffers and no others. Declaring a stream the model does not
# read costs bandwidth on every frame (a JPEG per camera at 30 Hz) and, worse,
# invites a future reader to assume it is used.
#
# This checkpoint reads ONE image and ONE state row:
#
#   image   hand_action_dataset.__getitem__ takes `video_frames[start_frame]`,
#           a single frame from a single view. The T-Rex corpus's view is
#           `observation.images.head_left` (hand_action_corpus.TREX_CAMERA_KEY),
#           i.e. the head/ego camera. The two wrist cameras exist on this robot
#           but this checkpoint never saw them, so they are declared off.
#   state   STATE_HISTORY_OFFSETS is (9, 6, 3, 0), so FOUR rows, all measured
#           against the single anchor STATE_ANCHOR_OFFSET frames back. Every one
#           of those five frames is read out of the observation's OWN history,
#           which is why history_len is 12 rather than 0: buffering them here
#           instead would space the rows one inference apart (~500 ms) instead of
#           three frames apart (100 ms) and inflate every displacement ~5x.
#   sensor  tau/wrench/deformation are not in the 56D representation at all.
#           All off.
IMAGE_SIZE = (256, 256)

# Which body the prompt claims to be driving, as a key into the training code's
# EMBODIMENT_PHRASES ("robot" -> "a robot's two Sharpa hands"). Both the T-Rex
# pretrain windows and the UR-Sharpa post-train windows rendered this phrase,
# because dispatch is on whether a window carries a robot span and these are
# genuinely Sharpa hands on both. It is NOT cosmetic: the checkpoint's
# prompt_template has an {embodiment} slot and the backbone raises rather than
# defaulting when nothing fills it.
EMBODIMENT = "robot"

# Per-inference diagnostics, opt-in via SHARPA_DEPLOY_DIAGNOSE=1. Kept behind a
# flag rather than deleted after use: the numbers it prints (the chunk in
# NORMALIZED units, beside the metric path length) are what distinguish a model
# sampling out of distribution from a normalizer or layout error, and that
# question recurs on every new checkpoint.
DIAGNOSE = os.environ.get("SHARPA_DEPLOY_DIAGNOSE", "") not in ("", "0", "false")

# The operator's speed scale, read once at import so a bad value fails before the
# port is bound rather than on the first inference. 1.0 is full speed.
#
# This is a comfort/safety control, NOT a fix for anything: measured, the policy's
# commanded per-step motion sits at or below the corpus at every percentile
# (`dist_compare.py`), while the corpus itself reaches ~190 mm/s at the wrist and
# varies 5.3x by task phase (`gt_envelope.py`). The fast unscrew phase is the task,
# faithfully reproduced. Scaling it down trades progress per chunk for a speed a
# person can stand beside; it does not disturb the anchor-lag relation, which is
# about WHICH steps are sent, not how far each goes. See speed_guard.py.
SPEED_SCALE = read_scale()


def build_metadata_format(format_id: str) -> dict[str, Any]:
    """The buffer request. One ego frame, one state row, no tactile."""
    off = {"history_len": 0, "current": False}
    return {
        "schema": METADATA_FORMAT_SCHEMA,
        "format_id": format_id,
        "image": {
            "ego_cam": {"history_len": 0, "current": True},
            "left_wrist_cam": dict(off),
            "right_wrist_cam": dict(off),
        },
        "state": {
            # STATE_ANCHOR_OFFSET frames of history, so the four rows at
            # STATE_HISTORY_OFFSETS and their shared anchor can all be read at
            # the TRAINING rate. The workstation publishes state at 30 Hz --
            # exactly the corpus rate -- into a 19-deep buffer, so 12 fits.
            #
            # Asking for 0 here and buffering per-request on the server instead
            # would space the rows one inference apart rather than 3 frames,
            # inflating every displacement ~5x. See extract_state_rows.
            "history_len": STATE_ANCHOR_OFFSET,
            "current": True,
            # Wrist joint angles are not read: the representation is EEF-based,
            # and the arm is a UR5 here but was a Vega-1 in training, so the
            # joint vectors are not even the same length. eef only.
            "left_wrist": {"joint": False, "eef": True},
            "right_wrist": {"joint": False, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": dict(off),
            "wrench": dict(off),
            "deformation": dict(off),
        },
    }


def build_server_metadata(
    *,
    format_id: str,
    model_id: str,
    checkpoint_id: str,
    checkpoint_path: str,
    task_id: str,
    run_id: str,
    prompt: str,
    dataset_path: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    """The /metadata body.

    ``policy_family`` is matched case-insensitively against the workstation's
    configured provider when one is set, and ``prompt`` is compared against the
    dashboard's when BOTH are non-empty -- an empty prompt on either side skips
    that check (client.py:49), which is what lets this ship before the task
    prompt is settled.
    """
    return {
        "schema": SERVER_SCHEMA,
        "policy_family": "mot_action_v5",
        # The launcher directory name, which is what the dashboard passed to
        # launch.sh and what it verifies against. policy_client_node reads
        # `model_id` (or `model_name`) and refuses to connect on a mismatch --
        # a missing field counts as a mismatch, so this is mandatory even though
        # validate_server_metadata does not list it.
        "model_id": model_id,
        "model_name": model_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "task_id": task_id,
        "run_id": run_id,
        # Only checkpoint_path, dataset_path and prompt may be empty; every
        # other string field is validated nonempty (metadata.py:148).
        "dataset_path": dataset_path,
        "prompt": prompt,
        "transport": TRANSPORT,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        # The four paths are compared for exact equality by the client, so they
        # are not configurable: they name where this server actually serves.
        "host": host,
        "port": port,
        "infer_path": "/infer",
        "health_path": "/healthz",
        "metadata_path": "/metadata",
        "reset_path": "/reset",
        "max_message_size": MAX_MESSAGE_SIZE,
        "execution_mode": "synchronous",
        "metadata_format": build_metadata_format(format_id),
        # The public ABI declaration. Absolute on both sides: the relative v5
        # representation is resolved inside this server and never crosses the
        # boundary. policy_client_node.validate_server_eef_contract checks each
        # of these five values and refuses to connect on any mismatch.
        "interface": {"eef_contract": dict(EEF_CONTRACT)},
        "action_horizon": ACTION_HORIZON,
        "action_stride": ACTION_STRIDE,
        "frequency_hz": FREQUENCY_HZ,
    }


# --- observation -> model input -------------------------------------------
def _require(mapping: Any, key: str, path: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"observation is missing {path}.{key}")
    return mapping[key]


def extract_current_state(observation: dict[str, Any]) -> tuple[dict, dict]:
    """Pull the current wrist poses and hand joints out of one observation.

    Returns ``(pose9_by_side, hand_qpos_by_side)``. Raises rather than
    substituting a default: a zero pose would be a valid-looking pose at the
    robot's origin, and the resulting relative chunk would command a large
    motion toward wherever the arm actually is.
    """
    state = _require(observation, "state", "observation")
    current = _require(state, "current", "observation.state")
    if current is None:
        raise ValueError("observation.state.current is None; the workstation sent no state")

    poses: dict[str, np.ndarray] = {}
    hands: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        wrist = _require(current, f"{side}_wrist", "observation.state.current")
        eef = _require(wrist, "eef", f"observation.state.current.{side}_wrist")
        if eef is None:
            raise ValueError(f"{side}_wrist.eef is None but the metadata_format requested it")
        # eef_def None is equivalent to absolute; anything else must not reach
        # here, because the public interface does not carry relative poses.
        eef_def = wrist.get("eef_def")
        if eef_def not in (None, "absolute"):
            raise ValueError(f"{side}_wrist.eef_def must be absolute, got {eef_def!r}")
        poses[side] = np.asarray(eef, dtype=np.float32).reshape(-1)[:9]

        hand_joint = _require(current, "hand_joint", "observation.state.current")
        joints = _require(hand_joint, side, "observation.state.current.hand_joint")
        if joints is None:
            raise ValueError(f"hand_joint.{side} is None but the metadata_format requested it")
        hands[side] = np.asarray(joints, dtype=np.float32).reshape(-1)[:HAND_DIM]
    return poses, hands


def extract_state_rows(
    observation: dict[str, Any],
    offsets: tuple[int, ...] = STATE_HISTORY_OFFSETS,
    anchor_offset: int = STATE_ANCHOR_OFFSET,
) -> tuple[dict[str, np.ndarray], list[tuple[dict, dict]]]:
    """Read the state rows and their anchor from the observation's OWN history.

    Returns ``(anchor_pose9_by_side, rows)`` with one
    ``(pose9_by_side, hand_qpos_by_side)`` per offset, in ``offsets`` order.

    WHY THIS READS THE WORKSTATION'S HISTORY INSTEAD OF A SERVER-SIDE RING.
    The offsets are in TRAINING frames at 30 fps, so row spacing is 100 ms and
    the four rows span 300 ms. The workstation publishes UR state at 30 Hz
    (ur_node publish_hz) into a 19-deep buffer, so its history is at exactly the
    training rate. A server-side ring filled once per /infer would instead be
    spaced one inference apart (~500 ms at RTC=5), stretching the same four
    rows over ~1.5 s and inflating every displacement ~5x -- finite, correctly
    shaped, and wrong. So the rate has to come from the workstation.

    INDEXING. ``select(history_len, current=True)`` returns the newest frame as
    ``current`` and the preceding ``history_len`` frames as ``history``, OLDEST
    FIRST and excluding current. With ``history_len == anchor_offset`` a frame
    ``k`` back is therefore ``history[anchor_offset - k]`` for k >= 1, and
    ``current`` for k == 0. The anchor at k == anchor_offset is ``history[0]``.
    """
    state = _require(observation, "state", "observation")
    current = _require(state, "current", "observation.state")
    history = _require(state, "history", "observation.state")
    if current is None:
        raise ValueError("observation.state.current is None; the workstation sent no state")
    if history is None:
        raise ValueError(
            "observation.state.history is None, but this policy needs "
            f"{anchor_offset} history frames. The metadata_format must request "
            f"history_len={anchor_offset}."
        )

    def pose_at(offset: int, side: str) -> np.ndarray:
        if offset == 0:
            return np.asarray(
                current[f"{side}_wrist"]["eef"], dtype=np.float32
            ).reshape(-1)[:9]
        stack = np.asarray(history[f"{side}_wrist"]["eef"], dtype=np.float32)
        if stack.ndim != 2 or stack.shape[0] < anchor_offset:
            raise ValueError(
                f"state history for {side}_wrist is {stack.shape}; need at least "
                f"[{anchor_offset}, 9]"
            )
        return stack[anchor_offset - offset].reshape(-1)[:9]

    def hand_at(offset: int, side: str) -> np.ndarray:
        if offset == 0:
            return np.asarray(
                current["hand_joint"][side], dtype=np.float32
            ).reshape(-1)[:HAND_DIM]
        stack = np.asarray(history["hand_joint"][side], dtype=np.float32)
        if stack.ndim != 2 or stack.shape[0] < anchor_offset:
            raise ValueError(
                f"state history for hand_joint.{side} is {stack.shape}; need at "
                f"least [{anchor_offset}, {HAND_DIM}]"
            )
        return stack[anchor_offset - offset].reshape(-1)[:HAND_DIM]

    # The history stack carries one eef_def for the whole stack; the boundary is
    # absolute-only, so anything else here is a contract violation upstream.
    for side in ("left", "right"):
        eef_def = history[f"{side}_wrist"].get("eef_def")
        if eef_def not in (None, "absolute"):
            raise ValueError(
                f"state history {side}_wrist.eef_def must be absolute, got {eef_def!r}"
            )

    anchor = {side: pose_at(anchor_offset, side) for side in ("left", "right")}
    rows = [
        (
            {side: pose_at(offset, side) for side in ("left", "right")},
            {side: hand_at(offset, side) for side in ("left", "right")},
        )
        for offset in offsets
    ]
    return anchor, rows


def extract_ego_jpeg(observation: dict[str, Any]) -> bytes:
    """The one image this checkpoint reads, as JPEG bytes."""
    image = _require(observation, "image", "observation")
    ego = _require(image, "ego_cam", "observation.image")
    current = _require(ego, "current", "observation.image.ego_cam")
    if current is None:
        raise ValueError("observation.image.ego_cam.current is None; no ego frame arrived")
    data = current.get("data") if isinstance(current, dict) else current
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"ego_cam frame data must be JPEG bytes, got {type(data).__name__}")
    return bytes(data)


# --- the model ------------------------------------------------------------
class MotPolicy:
    """Load the checkpoint once and sample chunks from it.

    Imports of flagscale are deferred to ``load`` so that ``--dry-run`` can
    exercise the whole transport, including the workstation's validators,
    without a GPU or a 4B checkpoint. That mode is how the wire contract gets
    tested cheaply; it is not a fallback, and it never silently substitutes for
    a real model (see ``--dry-run``'s help).
    """

    def __init__(self, config_path: str, checkpoint_path: str, normalizer_path: str) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.normalizer_path = normalizer_path
        self.model = None
        self.normalizer = None
        self._torch = None

    def load(self) -> None:
        import torch
        from omegaconf import OmegaConf

        # train_qwen3_5_gr00t parses its own argv; leaving this process's flags in
        # place makes it fail on unknown arguments.
        saved_argv, sys.argv = sys.argv, [sys.argv[0]]
        try:
            from flagscale.models.vla.action_model.hand_action_norm import (
                HandActionNormalizer,
            )
            from flagscale.models.vla.pretrained_config import PreTrainedConfig
            from flagscale.train.train_qwen3_5_gr00t import (
                TrainConfig,
                make_dataset,
                make_policy,
            )
        finally:
            sys.argv = saved_argv

        self._torch = torch
        logger.info("loading train config from %s", self.config_path)
        config = TrainConfig.from_hydra_config(OmegaConf.load(self.config_path))
        policy_config = PreTrainedConfig.from_train_config(config)

        # make_policy needs dataset.meta for the action/state dimensions. This is
        # the one place serving still touches the dataset, and it is metadata
        # only -- no window is ever read.
        logger.info("building dataset metadata ...")
        dataset = make_dataset(config, policy_config, seed=0)

        logger.info("loading the 4B policy from %s ...", self.checkpoint_path)
        started = time.time()
        model = make_policy(policy_config, dataset.meta)

        # LOAD THE WEIGHTS. make_policy only BUILDS the model -- it ends at
        # TrainablePolicy.from_config, so the VLM comes from `base_vlm` but the
        # 1.26B action expert is left RANDOMLY INITIALIZED. The training entry
        # point loads them separately, further down its own function
        # (train_qwen3_5_gr00t.py, the `init_from` branch), which is why building
        # a policy the same way training does is not enough here.
        #
        # This was a real defect and it is worth naming precisely, because
        # everything that normally catches a bad deploy passed: the model loaded,
        # /metadata validated, identity matched, the chunk had the right shape and
        # dtype, every value was finite, latency was normal, and 43 contract tests
        # were green. The only symptom was that the actions were nonsense -- a
        # random expert maps noise to a well-shaped chunk, and multiplying it by
        # the normalizer's 19.1 mm/sigma even lands it in a plausible-looking
        # millimetre range. Nothing anywhere compared the output against the
        # distribution it was supposed to come from.
        #
        # Uses the training code's own loader so serving cannot drift from it, and
        # the config's `init_from` is deliberately NOT consulted: that names the
        # PRETRAIN checkpoint this run was post-trained from, so honouring it would
        # serve the wrong weights while every log line named the right ones.
        from flagscale.train.utils.train_utils import load_model_state_fsdp2

        weights_dir = Path(self.checkpoint_path) / "pretrained_model"
        if not weights_dir.is_dir():
            raise RuntimeError(
                f"{weights_dir} does not exist; --checkpoint wants the directory "
                "holding pretrained_model/, e.g. .../checkpoints/001500"
            )
        logger.info("loading expert + VLM weights from %s", weights_dir)
        load_model_state_fsdp2(model, weights_dir)

        # eval, not train: dropout off, and the backbone re-enables use_cache,
        # which is what makes the one-VLM-pass sampling path work.
        model.eval()
        self.model = model
        logger.info("policy ready in %.0fs", time.time() - started)

        if not getattr(model, "action_is_mot", False):
            raise RuntimeError("config does not build a MoT action expert")

        expert = model.action_model
        if int(expert.config.action_horizon) != ACTION_HORIZON:
            raise RuntimeError(
                f"checkpoint horizon {expert.config.action_horizon} != {ACTION_HORIZON}"
            )
        if int(expert.config.action_dim) != ACTION_DIM:
            raise RuntimeError(
                f"checkpoint action_dim {expert.config.action_dim} != {ACTION_DIM}"
            )
        # The state anchor is on the EXPERT config, not the data config, exactly
        # so serving cannot disagree with training about what a state row means.
        anchor = int(expert.config.state_anchor_offset)
        if anchor != STATE_ANCHOR_OFFSET:
            raise RuntimeError(
                f"checkpoint state_anchor_offset {anchor} != {STATE_ANCHOR_OFFSET}"
            )

        logger.info("loading normalizer from %s", self.normalizer_path)
        self.normalizer = HandActionNormalizer.load(self.normalizer_path)
        logger.info("normalizer repr_version=%s", getattr(self.normalizer, "repr_version", "?"))

    def infer(self, state56: np.ndarray, ego_jpeg: bytes, prompt: str) -> np.ndarray:
        """One chunk, ``[ACTION_HORIZON, ACTION_DIM]`` in RAW v5 units.

        Normalization happens here on both sides, with the two different
        statistics the checkpoint was fitted under: ``normalize_state`` for the
        input row and ``denormalize`` for the output chunk. Using one for both
        is the single most likely silent error in this file, so they are named
        explicitly rather than wrapped in a shared helper.
        """
        torch = self._torch
        import io

        from PIL import Image

        # A LIST of PIL Images, not a numpy array. The list path of
        # _predict_action_mot passes example["image"] straight to
        # build_qwenvl_inputs, whose signature is list[list[Image.Image]] and
        # which hands each element to qwen_vl_utils -- that calls
        # .startswith() on it, so an ndarray raises there rather than being
        # converted. (The dict path would have gone through _to_pil and
        # accepted an array; the two paths are not interchangeable.)
        #
        # Resized to IMAGE_SIZE here because training resized to the same
        # data.hand_action.image_size before the processor saw it.
        image = Image.open(io.BytesIO(ego_jpeg)).convert("RGB").resize(IMAGE_SIZE)

        # [rows, 56] -> normalized [rows, 56]. One set of statistics for every
        # row: they are the same quantity on the same layout at the same anchor,
        # so per-row scales would give one quantity several meanings and make
        # the row-to-row differences meaningless. normalize_state broadcasts
        # over the leading dimension, so no [None] is needed here -- adding one
        # would hand the expert a [1, rows, 56] it reads as a batch of rows.
        rows = np.asarray(state56, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != ACTION_DIM:
            raise ValueError(f"state must be [rows, {ACTION_DIM}], got {rows.shape}")
        state = self.normalizer.normalize_state(
            torch.from_numpy(rows), embodiment=EMBODIMENT
        )

        # `lang` is the BARE INSTRUCTION, not a rendered prompt. The checkpoint
        # carries prompt_template in its own config and build_qwenvl_inputs
        # substitutes {instruction} itself, so passing a pre-rendered sentence
        # here would nest it inside the template and shift every sample off the
        # length distribution the model trained on.
        #
        # `_embodiment` is MANDATORY for this checkpoint, not optional. Its
        # template contains an {embodiment} slot, and build_qwenvl_inputs RAISES
        # when the slot is present and no phrase was derived -- deliberately, so
        # one corpus cannot borrow another's prompt. The phrase is looked up in
        # EMBODIMENT_PHRASES; "robot" renders "a robot's two Sharpa hands",
        # which is exactly what these UR-Sharpa windows rendered in training
        # (dispatch is on carrying a robot span, and these genuinely are Sharpa
        # hands, so the UR corpus got the same phrase T-Rex did).
        batch = [
            {
                "image": [image],
                "lang": prompt,
                "state": state,
                "_embodiment": EMBODIMENT,
            }
        ]
        with torch.no_grad():
            output = self.model._predict_action_mot(batch)

        chunk = output[next(iter(output))]
        raw = chunk.float().cpu()
        chunk = self.normalizer.denormalize(raw, embodiment=EMBODIMENT)
        chunk = chunk[0].numpy() if chunk.ndim == 3 else chunk.numpy()
        chunk = np.asarray(chunk, dtype=np.float32)

        # Diagnostic, off by default. Logs the chunk in NORMALIZED units next to
        # the denormalized one, which is the only way to tell a model that is
        # sampling badly from a normalizer applied wrongly: the expert's output
        # should sit within a couple of sigma, so |raw| >> 2 says the weights are
        # producing out-of-distribution values and no scaling choice downstream
        # can rescue it, while a sane raw next to an insane metric output puts the
        # fault in the statistics or the layout.
        if DIAGNOSE:
            flat = raw[0] if raw.ndim == 3 else raw
            flat = np.asarray(flat, dtype=np.float32)
            per_step = np.linalg.norm(np.diff(chunk[:, 0:3], axis=0), axis=1)
            logger.info(
                "DIAG raw |mean|=%.3f max|.|=%.3f p99=%.3f | left_trans path=%.1fmm "
                "net=%.1fmm | hand max|d|/step=%.3f rad",
                float(np.abs(flat).mean()),
                float(np.abs(flat).max()),
                float(np.percentile(np.abs(flat), 99)),
                float(per_step.sum() * 1000),
                float(np.linalg.norm(chunk[-1, 0:3] - chunk[0, 0:3]) * 1000),
                float(np.abs(np.diff(chunk[:, 12:34], axis=0)).max()),
            )
        return chunk


# --- websocket framing ----------------------------------------------------
# Hand-rolled rather than pulled from a library: the training environment is a
# conda env owned by another project, and adding a dependency to it to serve one
# socket is a worse trade than 60 lines of RFC 6455. Only what the client uses
# is implemented -- binary frames, close, ping/pong -- and anything else is a
# protocol error rather than a silent no-op.
def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def ws_read_message(sock: socket.socket) -> tuple[int, bytes]:
    """One whole message, reassembling continuation frames.

    Returns ``(opcode, payload)`` where opcode is the FIRST frame's, per RFC
    6455: continuations carry opcode 0 and inherit the meaning of the frame that
    started the message.
    """
    opcode_first: int | None = None
    payload = bytearray()
    while True:
        header = _recv_exactly(sock, 2)
        fin = bool(header[0] & 0x80)
        opcode = header[0] & 0x0F
        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exactly(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exactly(sock, 8))[0]
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"frame announces {length} bytes; limit is {MAX_MESSAGE_SIZE}")
        # A client MUST mask; refusing an unmasked frame is required, and here it
        # also catches a non-websocket peer writing raw bytes at the socket.
        mask = _recv_exactly(sock, 4) if masked else b""
        data = _recv_exactly(sock, length) if length else b""
        if masked:
            data = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))

        if opcode in (0x8, 0x9, 0xA):
            # Control frames are never fragmented and never part of a message.
            return opcode, bytes(data)
        if opcode_first is None:
            opcode_first = opcode
        payload.extend(data)
        if fin:
            return opcode_first, bytes(payload)
        if len(payload) > MAX_MESSAGE_SIZE:
            raise ValueError("reassembled message exceeds the 64 MiB limit")


def ws_send(sock: socket.socket, payload: bytes, opcode: int = 0x2) -> None:
    """Send one unfragmented, unmasked frame. Servers must not mask."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    sock.sendall(bytes(header) + payload)


def ws_accept_key(client_key: str) -> str:
    import base64
    import hashlib

    digest = hashlib.sha1((client_key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


# --- session --------------------------------------------------------------
class Session:
    """One rollout's identity and the pose history that belongs to it.

    A session is created by /reset and is the ONLY thing that resets: the loaded
    model outlives every session, which is what makes repeated rollouts cheap
    (a 4B reload is minutes). Everything that would carry state across a
    rollout boundary lives here and dies with it.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # No pose ring here on purpose. State history comes from the
        # observation, at the workstation's 30 Hz -- see extract_state_rows.
        self.request_count = 0
        self.created_ns = time.time_ns()


class PolicyServer:
    def __init__(
        self,
        policy: MotPolicy,
        *,
        metadata: dict[str, Any],
        format_id: str,
        prompt: str,
        dry_run: bool,
    ) -> None:
        self.policy = policy
        self.metadata = metadata
        self.format_id = format_id
        self.prompt = prompt
        self.dry_run = dry_run
        self.ready = False
        # Guards the session pointer, which /reset replaces and /infer reads.
        self._lock = threading.Lock()
        # Serializes whole inferences. Connections are threaded (a websocket and
        # short HTTP requests must overlap, see main), so two /infer sockets
        # could otherwise interleave pushes into one session's pose history and
        # produce a state row anchored on the other socket's frame. The model
        # could not run two batches concurrently on one GPU anyway.
        self._inference_lock = threading.Lock()
        self._session: Session | None = None

    # --- endpoints --------------------------------------------------------
    def handle_metadata(self) -> dict[str, Any]:
        return self.metadata

    def handle_reset(self, body: dict[str, Any]) -> dict[str, Any]:
        """Start a new session and return the AUTHORITATIVE metadata_format.

        The client treats this response, not /metadata, as the format for
        request 0 -- a stateful policy may switch formats between rollouts. This
        one does not, so the same format is returned, but the client's rule is
        honoured rather than assumed.
        """
        requested = str(body.get("session_id") or "").strip()
        session_id = requested or uuid.uuid4().hex
        with self._lock:
            self._session = Session(session_id)
        logger.info("reset: session=%s", session_id)
        return {
            "schema": RESET_SCHEMA,
            "session_id": session_id,
            "request_id": 0,
            "reset": True,
            "metadata_format": build_metadata_format(self.format_id),
        }

    def handle_infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """One observation in, one action chunk out."""
        schema = observation.get("schema")
        if schema != OBSERVATION_SCHEMA:
            raise ValueError(f"observation.schema must be {OBSERVATION_SCHEMA}, got {schema!r}")

        session_id = str(observation.get("session_id") or "")
        request_id = int(observation.get("request_id", 0))
        with self._lock:
            session = self._session
        if session is None:
            raise ValueError("no active session; call /reset first")
        if session_id != session.session_id:
            # Identity mismatches are dropped silently by the client, so a wrong
            # session would look like a hang. Fail loudly on this side instead.
            raise ValueError(
                f"observation session_id {session_id!r} does not match the active "
                f"session {session.session_id!r}"
            )

        # The push, the anchor read and the sample are one atomic unit: a second
        # request pushing between the push and the anchor read would shift this
        # request's anchor by a frame, silently changing what the state row means.
        with self._inference_lock:
            # The CURRENT pose is the action anchor: the chunk is composed onto
            # where the arm actually is right now.
            poses, hands = extract_current_state(observation)

            # The STATE rows and their own anchor come from the workstation's
            # history, at the training frame rate. Two different anchors on
            # purpose -- the action anchors at the window start, the state 12
            # frames behind it, which is why their statistics differ too.
            state_anchor, rows = extract_state_rows(observation)

            # [len(STATE_HISTORY_OFFSETS), 56], oldest row first. The expert
            # rejects any other row count -- see STATE_HISTORY_OFFSETS.
            state56 = build_state_rows(state_anchor, rows)
            prompt = str(observation.get("prompt") or "") or self.prompt

            started = time.perf_counter()
            if self.dry_run:
                # A hold-still chunk: zero relative motion, hands where they are.
                # Decoded, this reproduces the current pose exactly, which is the
                # only safe thing a transport test can ask the arm to do.
                chunk = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
                for side in ("left", "right"):
                    chunk[:, SLICES[f"{side}_hand_joints"]] = hands[side]
            else:
                ego = extract_ego_jpeg(observation)
                chunk = self.policy.infer(state56, ego, prompt)
            latency = time.perf_counter() - started

            if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
                raise RuntimeError(
                    f"model returned {chunk.shape}, expected {(ACTION_HORIZON, ACTION_DIM)}"
                )
            if not np.all(np.isfinite(chunk)):
                raise RuntimeError("model returned a non-finite chunk")

            # Corpus ceiling, then the operator's scale. Inert at SPEED_SCALE=1
            # unless the sampled chunk exceeds anything the corpus contains, which
            # measured 0 of 54 in-distribution windows -- so a `capped` line in the
            # log is a real excursion and worth reading.
            chunk, speed = govern(chunk, SPEED_SCALE)
            if speed["capped"]:
                logger.warning(
                    "speed ceiling engaged: %s (max step %.1fmm / %.3frad)",
                    speed["capped"],
                    speed["max_wrist_step_mm"],
                    speed["max_hand_step_rad"],
                )

            # Slow the complete path in time rather than shrinking it in space.
            # At 0.3, ten model points become 34 commands at the same safe 10 Hz;
            # wrist and hand endpoints remain exactly what the policy predicted.
            chunk = stretch_chunk(chunk, hands, TIME_SCALE)
            decoded = decode_action_chunk(chunk, poses)
            session.request_count += 1
        # Log the newest row's displacement from the state anchor: it is the one
        # number that shows whether the history is arriving at the training rate.
        # Training measured p99 of a 12-frame span at ~5 cm; a value far above
        # that means the rows are spaced wider than 30 Hz.
        span_m = float(np.linalg.norm(state56[-1][SLICES["left_wrist_trans"]]))
        logger.info(
            "infer: request=%d latency=%.3fs rows=%d left_span=%.1fmm",
            request_id,
            latency,
            state56.shape[0],
            span_m * 1000.0,
        )
        return build_action_response(
            session_id=session.session_id,
            request_id=request_id,
            decoded=decoded,
            latency_s=latency,
        )


# --- model output -> interface action dict --------------------------------
def build_action_response(
    *,
    session_id: str,
    request_id: int,
    decoded: dict[str, np.ndarray],
    latency_s: float,
) -> dict[str, Any]:
    """Pack decoded absolute arrays into ``sharpa_policy_action.v5``.

    Every field the client's ``parse_policy_action`` requires is present, and
    every array is the full ``action_length`` -- the client does its own
    execution-slice truncation, so sending a pre-sliced array would be
    rejected for a length mismatch.

    ``execute_start`` is EXECUTE_START, not 0. Re-planning from the measured
    pose does NOT by itself make the chunk current: the pose is measured when
    the observation is taken and the chunk takes over RTC_STEPS later, so its
    first EXECUTE_START steps describe motion the arm has already performed.
    Sending them commands a retraction. See the RTC_STEPS derivation above.

    The arrays stay ``action_length`` long: the client slices them itself and
    rejects a pre-sliced array for a length mismatch.
    """
    return {
        "schema": ACTION_SCHEMA,
        "session_id": session_id,
        "request_id": int(request_id),
        # A fresh identity per chunk. The client matches execution_done against
        # it, so reusing an id across chunks would make a stale done look valid.
        "action_id": uuid.uuid4().hex,
        "revision": 0,
        "timestamp_ns": time.time_ns(),
        "execution": {
            "frequency_hz": float(FREQUENCY_HZ),
            "action_length": int(EXECUTION_HORIZON),
            "execute_start": int(EXECUTE_START),
            "execute_length": int(EXECUTE_LENGTH),
            "rtc": int(RTC_STEPS),
        },
        "action": {
            "left_wrist": {
                # joint is None: this model speaks EEF, and the arm's joint
                # vector is embodiment-specific (a UR5 has 6, the trained Vega
                # had 7). action_ik turns the EEF pose into UR5 joints.
                "joint": None,
                "eef": decoded["left_eef"],
                "eef_def": "absolute",
            },
            "right_wrist": {
                "joint": None,
                "eef": decoded["right_eef"],
                "eef_def": "absolute",
            },
            "hand_joint": {
                "left": decoded["left_hand"],
                "right": decoded["right_hand"],
            },
        },
        # Reserved for predicted modalities. The keys must exist even while every
        # value is None, so a future server can fill them without a schema bump.
        "auxiliary": {
            "video": {"ego": None, "left_wrist": None, "right_wrist": None},
            "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
        },
        "diagnostics": {"inference_latency_s": round(float(latency_s), 6)},
        # None means "keep the current format". This policy never switches.
        "next_metadata_format": None,
    }


# --- transport ------------------------------------------------------------
def _read_http_request(sock: socket.socket) -> tuple[str, str, dict[str, str], bytes]:
    """Read one HTTP request. Returns (method, path, headers, body)."""
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed before the request head was complete")
        buffer.extend(chunk)
        if len(buffer) > 64 * 1024:
            raise ValueError("HTTP request head too large")
    head, _, rest = bytes(buffer).partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, _, tail = lines[0].partition(" ")
    path = tail.rpartition(" ")[0].strip() or "/"
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()

    length = int(headers.get("content-length") or 0)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"content-length {length} exceeds the limit")
    body = bytearray(rest)
    while len(body) < length:
        chunk = sock.recv(min(65536, length - len(body)))
        if not chunk:
            raise ConnectionError("peer closed mid-body")
        body.extend(chunk)
    return method.upper(), path.split("?")[0], headers, bytes(body[:length])


def _send_http(sock: socket.socket, status: str, body: bytes, content_type: str) -> None:
    head = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("latin-1")
    sock.sendall(head + body)


def serve_connection(server: PolicyServer, sock: socket.socket, address) -> None:
    try:
        method, path, headers, body = _read_http_request(sock)
    except (ConnectionError, ValueError, OSError) as exc:
        logger.debug("bad request from %s: %s", address, exc)
        return

    try:
        if path == "/healthz":
            # Liveness only, and deliberately NOT readiness: the ssh relay exits
            # the moment this stops returning 2xx, so it must answer during the
            # minutes the checkpoint is loading or the workstation would tear the
            # tunnel down mid-startup. Readiness is reported in the body.
            state = b'{"status":"ok"}' if server.ready else b'{"status":"loading"}'
            _send_http(sock, "200 OK", state, "application/json")
            return

        if path == "/metadata":
            if not server.ready:
                _send_http(sock, "503 Service Unavailable", b"model is still loading", "text/plain")
                return
            _send_http(sock, "200 OK", packb(server.handle_metadata()), "application/msgpack")
            return

        if path == "/reset":
            if method != "POST":
                _send_http(sock, "405 Method Not Allowed", b"POST only", "text/plain")
                return
            if not server.ready:
                _send_http(sock, "503 Service Unavailable", b"model is still loading", "text/plain")
                return
            payload = unpackb(body) if body else {}
            if not isinstance(payload, dict):
                payload = {}
            _send_http(sock, "200 OK", packb(server.handle_reset(payload)), "application/msgpack")
            return

        if path == "/infer":
            key = headers.get("sec-websocket-key")
            if headers.get("upgrade", "").lower() != "websocket" or not key:
                _send_http(sock, "400 Bad Request", b"/infer requires a websocket upgrade", "text/plain")
                return
            if not server.ready:
                _send_http(sock, "503 Service Unavailable", b"model is still loading", "text/plain")
                return
            accept = ws_accept_key(key)
            sock.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "\r\n"
                ).encode("latin-1")
            )
            _serve_infer_socket(server, sock, address)
            return

        _send_http(sock, "404 Not Found", b"unknown path", "text/plain")
    except Exception as exc:  # noqa: BLE001 - one bad request must not end the server
        logger.exception("error handling %s %s", method, path)
        try:
            _send_http(sock, "500 Internal Server Error", str(exc).encode(), "text/plain")
        except OSError:
            pass


def _serve_infer_socket(server: PolicyServer, sock: socket.socket, address) -> None:
    """Serve one /infer session until the client closes.

    Nothing is sent on connect. The client sends the first observation and
    blocks until an answer arrives, so a server greeting would deadlock it.
    """
    logger.info("infer socket open from %s", address)
    while True:
        try:
            opcode, payload = ws_read_message(sock)
        except (ConnectionError, OSError, ValueError) as exc:
            logger.info("infer socket closed: %s", exc)
            return

        if opcode == 0x8:
            ws_send(sock, b"", opcode=0x8)
            return
        if opcode == 0x9:
            ws_send(sock, payload, opcode=0xA)
            continue
        if opcode == 0xA:
            continue
        if opcode != 0x2:
            # Text frames are not part of this contract; msgpack is binary.
            logger.warning("ignoring unexpected websocket opcode 0x%x", opcode)
            continue

        try:
            observation = unpackb(payload)
            response = server.handle_infer(observation)
        except Exception as exc:  # noqa: BLE001
            # An error response keeps the request_id so the client attributes the
            # failure to its own pending request rather than timing out. The
            # shape is fixed: `error` must be an OBJECT carrying code, message
            # and retryable, or the client's own validator raises on the error
            # path and reports a validation fault instead of this cause.
            logger.exception("inference failed")
            observation = locals().get("observation")
            request_id = 0
            if isinstance(observation, dict):
                try:
                    request_id = int(observation.get("request_id", 0) or 0)
                except (TypeError, ValueError):
                    request_id = 0
            response = {
                "schema": ERROR_SCHEMA,
                "request_id": request_id,
                "error": {
                    "code": type(exc).__name__,
                    "message": str(exc) or repr(exc),
                    # Nothing here is safely retryable: a bad observation will be
                    # bad again, and a model fault needs a human. Saying True
                    # would make the client spin on a permanent failure.
                    "retryable": False,
                },
            }
        try:
            ws_send(sock, packb(response))
        except OSError as exc:
            logger.info("failed to send response: %s", exc)
            return


# --- entry point ----------------------------------------------------------
def resolve_run(task_config_root: Path, task_name: str, run_id: str) -> tuple[dict, dict]:
    """Find one run in ``<task_config_root>/<task_name>/task.json``.

    The workstation passes the task name and run_id it showed the operator, so a
    miss here means the dashboard's catalog and this file disagree -- worth
    failing loudly rather than falling back to "the only run", which would start
    a different checkpoint than the one the operator picked.
    """
    path = task_config_root / task_name / "task.json"
    if not path.is_file():
        raise SystemExit(f"no task config at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    for entry in runs:
        if str(entry.get("run_id")) == run_id:
            return payload, entry
    available = ", ".join(sorted(str(e.get("run_id")) for e in runs)) or "<none>"
    raise SystemExit(f"run_id {run_id!r} not in {path}; available: {available}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--task-config-root", required=True)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; the workstation reaches this through an ssh relay, so "
             "the default deliberately does NOT expose the model on the network",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="serve the full wire contract WITHOUT loading the model, answering "
             "every request with a hold-still chunk. For testing the transport "
             "and the workstation's validators; it never stands in for a real "
             "model, and the log says so on every request",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asset_root = Path(args.asset_root)
    payload, run = resolve_run(Path(args.task_config_root), args.task_name, args.run_id)

    task = payload.get("task") or {}
    prompt = str(task.get("prompt") or "")
    checkpoint = str(run.get("checkpoint") or "")
    checkpoint_path = str((asset_root / checkpoint).resolve()) if checkpoint else ""
    config_path = str((asset_root / str(run.get("config") or "")).resolve())
    normalizer_path = str((asset_root / str(run.get("normalizer") or "")).resolve())

    logger.info("run_id=%s model=%s", args.run_id, run.get("model"))
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config=%s", config_path)
    logger.info("normalizer=%s", normalizer_path)
    if SPEED_SCALE != 1.0:
        # Loud, because a scaled server looks like an underperforming policy to
        # anyone reading only the robot: progress per chunk drops by this factor.
        logger.warning(
            "SPEED SCALE %.2f -- every commanded step is %.0f%% of the policy's "
            "own motion (%s)", SPEED_SCALE, SPEED_SCALE * 100.0, SCALE_ENV_NAME
        )
    else:
        logger.info("speed scale 1.00 (full); ceiling at %.1fmm/%.3frad per step",
                    MAX_WRIST_STEP_MM, MAX_HAND_STEP_RAD)
    if TIME_SCALE != 1.0:
        logger.warning(
            "TIME SCALE %.2f -- full %d-step policy path resampled to %d steps "
            "at %.1f Hz; endpoints are unchanged (%s)",
            TIME_SCALE,
            ACTION_HORIZON,
            EXECUTION_HORIZON,
            FREQUENCY_HZ,
            TIME_SCALE_ENV_NAME,
        )
    else:
        logger.info("time scale 1.00 (native %d steps at %.1f Hz)",
                    ACTION_HORIZON, FREQUENCY_HZ)
    if not prompt:
        # Allowed: the client skips the prompt comparison when either side is
        # empty. Logged because an empty prompt reaches the VLM as an empty
        # instruction, which is a real conditioning change, not a no-op.
        logger.warning("task prompt is EMPTY; the VLM will be conditioned on no instruction")

    # format_id must change whenever the requested buffers change, so the client
    # cannot pair a cached format with a different rollout. Tied to the run.
    format_id = f"mot_action_v5.{args.run_id}"
    metadata = build_server_metadata(
        format_id=format_id,
        # From the run entry, so it is the same string the dashboard passed
        # to launch.sh and the same one it verifies against.
        model_id=str(run.get("model") or ""),
        checkpoint_id=str(run.get("run_id") or ""),
        checkpoint_path=checkpoint_path,
        task_id=str(task.get("id") or args.task_name),
        run_id=str(run.get("run_id") or ""),
        prompt=prompt,
        dataset_path=str((asset_root / str(task.get("dataset") or "")).resolve())
        if task.get("dataset")
        else "",
        host=args.host,
        port=int(args.port),
    )

    policy = MotPolicy(config_path, checkpoint_path, normalizer_path)
    server = PolicyServer(
        policy,
        metadata=metadata,
        format_id=format_id,
        prompt=prompt,
        dry_run=bool(args.dry_run),
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, int(args.port)))
    listener.listen(8)
    logger.info("listening on %s:%d", args.host, args.port)

    # Bind BEFORE loading. The workstation starts polling /healthz immediately
    # and its relay exits if the port refuses a connection, so the socket has to
    # be answering during the minutes a 4B takes to load. /healthz reports
    # "loading" and the other endpoints answer 503 until ready flips.
    def _load() -> None:
        if server.dry_run:
            logger.warning("--dry-run: NO MODEL LOADED, every chunk holds position")
            server.ready = True
            return
        try:
            policy.load()
            server.ready = True
            logger.info("model ready; accepting inference")
        except Exception:
            logger.exception("model failed to load; the server will keep answering 503")

    threading.Thread(target=_load, name="load", daemon=True).start()

    def _handle(sock: socket.socket, address) -> None:
        try:
            sock.settimeout(None)
            serve_connection(server, sock, address)
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    try:
        while True:
            sock, address = listener.accept()
            # A THREAD PER CONNECTION IS REQUIRED, not a scaling choice.
            #
            # The client holds its /infer websocket open for the whole rollout
            # and calls /reset on a SEPARATE connection (client.connect() opens
            # the socket, then PolicyClientCore.start() calls reset). A serial
            # accept loop is blocked reading the websocket at that moment, so
            # the /reset connection is never accepted and both sides wait
            # forever -- a startup deadlock with no error on either end.
            #
            # Serializing INFERENCE is a different concern and is handled by the
            # lock in PolicyServer, which is what actually protects the session's
            # pose history from interleaved requests.
            threading.Thread(
                target=_handle, args=(sock, address), name="conn", daemon=True
            ).start()
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
        return 0
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
