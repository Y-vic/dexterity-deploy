"""Explicit checkpoint prompts for offline recorded-observation validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[2] / "scripts/validate_dfm_policy.py"
SPEC = importlib.util.spec_from_file_location("validate_dfm_policy", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Client:
    def __init__(self, prompt):
        self.current_metadata = {
            "prompt": prompt,
            "native_interface": {
                "observation": {"state": {}},
                "camera_bindings": {"left_wrist": "left_wrist_cam"},
                "action": {},
                "execution": {},
            },
            "metadata_format": {"format_id": "test-format"},
            "interface_revision": 1,
        }
        self.reset_calls = []
        self.observations = []

    def connect(self):
        return self.current_metadata

    def reset(self, session_id, request_id):
        self.reset_calls.append((session_id, request_id))

    def update_interface(self, proposal):
        return {"queued": True}

    def infer_action(self, value, *, expected_session_id, expected_request_id):
        self.observations.append(value)
        state = value["state"]["current"]
        revision = 1 if expected_request_id < 2 else 2
        full = np.concatenate(
            (state["left_wrist"]["eef"], state["right_wrist"]["eef"],
             state["hand_joint"]["left"], state["hand_joint"]["right"])
        )
        full = np.repeat(full[None], 30, axis=0)
        action = SimpleNamespace(
            left_wrist=SimpleNamespace(eef=full[:, :9]),
            right_wrist=SimpleNamespace(eef=full[:, 9:18]),
            hand_joint=SimpleNamespace(left=full[:, 18:40], right=full[:, 40:62]),
            execution=SimpleNamespace(execute_start=5 if revision == 2 else 0,
                                      execute_length=10 if revision == 2 else 30,
                                      action_length=30),
            diagnostics={"native_interface_revision": revision,
                         "camera_from_root": {"ego_cam": np.eye(4).tolist()}},
            action_id=f"action-{expected_request_id}",
            revision=revision,
        )
        return action, 0.001


@pytest.fixture
def episode(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    Image.new("RGB", (12, 8), (20, 40, 60)).save(root / "image.png")
    metadata = {
        "episode_id": "recorded-test",
        "instruction": "Real recorded instruction.",
        "observations": [{
            "images": {view: "image.png" for view in ("ego", "left_wrist", "right_wrist")},
            "sources": {view: {"image_timestamp_unix_ns": 100}
                        for view in ("ego", "left_wrist", "right_wrist")},
            "measured_arm_timestamp_unix_ns": 100,
            "measured_hand_timestamp_unix_ns": 100,
        }],
    }
    (root / "observation_metadata.json").write_text(json.dumps(metadata))
    np.savez(root / "current_canonical_observations.npz",
             states=np.zeros((1, 62), np.float32), time_unix_ns=np.array([100]))
    return root


def test_default_recorded_prompt_mismatch_fails_before_reset_or_inference(episode, tmp_path):
    client = Client("Sim checkpoint instruction.")
    with pytest.raises(ValueError, match="instruction differs"):
        MODULE.validate_recorded_policy(client, episode, tmp_path / "report")
    assert not client.reset_calls and not client.observations


def test_explicit_checkpoint_prompt_preserves_recorded_provenance(episode, tmp_path):
    selected = "Sim checkpoint instruction."
    client = Client(selected)
    output = tmp_path / "report"
    report = MODULE.validate_recorded_policy(client, episode, output, prompt=selected)
    assert report["recorded_instruction"] == "Real recorded instruction."
    assert report["effective_instruction"] == selected
    assert [value["prompt"] for value in client.observations] == [selected] * 3
    assert output.with_suffix(".json").is_file() and output.with_suffix(".npz").is_file()


@pytest.mark.parametrize("prompt", ["Wrong instruction.", "", " "])
def test_invalid_explicit_prompt_cannot_reset_or_infer_another_checkpoint(episode, tmp_path, prompt):
    client = Client("Sim checkpoint instruction.")
    with pytest.raises(ValueError):
        MODULE.validate_recorded_policy(client, episode, tmp_path / "report", prompt=prompt)
    assert not client.reset_calls and not client.observations
