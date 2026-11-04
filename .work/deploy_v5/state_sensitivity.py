"""Does the state channel actually reach the expert's decision?

The naive version of this test cannot work: sampling draws fresh noise per call,
so two calls differ even on identical input (measured: 4.6-6.1 mm spread on the
same window). So the noise is PINNED here and the only thing that varies is the
state. Then:

    same noise, same state   -> must be bit-identical (proves the pin works)
    same noise, other state  -> must differ, and by more than the noise spread

If the third case comes back ~identical to the first, the expert is ignoring the
state rows and the 4x56 tensor is decoration -- which would be the same class of
defect as the random expert: every shape correct, every value plausible, nothing
downstream able to report it.

Runs on the remote against a model resident in THIS process, so it needs the GPU
free -- it is not a client of the serving process.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import repr_bridge as rb


def load_model(config_path: str, checkpoint_path: str):
    """Build and load the policy the same way the server does."""
    from omegaconf import OmegaConf

    saved_argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from flagscale.models.vla.pretrained_config import PreTrainedConfig
        from flagscale.train.train_qwen3_5_gr00t import (
            TrainConfig,
            make_dataset,
            make_policy,
        )
        from flagscale.train.utils.train_utils import load_model_state_fsdp2
    finally:
        sys.argv = saved_argv

    config = TrainConfig.from_hydra_config(OmegaConf.load(config_path))
    policy_config = PreTrainedConfig.from_train_config(config)
    dataset = make_dataset(config, policy_config, seed=0)
    model = make_policy(policy_config, dataset.meta)
    # The load the server was missing. Without it this test would compare two
    # random networks and conclude the state does nothing.
    load_model_state_fsdp2(model, Path(checkpoint_path) / "pretrained_model")
    model.eval()
    return model


def perturbed_state(state56: np.ndarray) -> np.ndarray:
    """A materially different state: wrists displaced, hands closed.

    Chosen to be large but still physical -- 5 cm of wrist travel over the state
    span and hand joints driven toward a fist. If the expert reads state at all,
    a change this size cannot leave the chunk untouched.
    """
    other = state56.copy()
    for side in ("left", "right"):
        other[:, rb.SLICES[f"{side}_wrist_trans"]] += np.array(
            [0.05, -0.03, 0.04], dtype=np.float32
        )
        other[:, rb.SLICES[f"{side}_hand_joints"]] += 0.5
    return other


def main() -> None:
    import io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    run = (
        "/share/project/zjk/orca/outputs/"
        "mot_action_v5_ur_unscrew_16gpu_alr3p5e4_vlr3p5e5_aw0p15_bs8_wu150_st4000"
    )
    checkpoint = f"{run}/checkpoints/001500"
    model = load_model(f"{run}/hydra/.hydra/config.yaml", checkpoint)
    normalizer = HandActionNormalizer.load(
        "/share/project/zjk/orca/outputs/shared_mot_action_v5_ur_unscrew/hand_action_norm.json"
    )

    anchor, rows, jpeg, gt = ct.build_window(0, 620)
    state56 = rb.build_state_rows(anchor, rows)
    image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((256, 256))

    def sample(state_rows: np.ndarray, noise: torch.Tensor) -> np.ndarray:
        state = normalizer.normalize_state(torch.from_numpy(state_rows))
        batch = [
            {
                "image": [image],
                "lang": ct.UR_TASK,
                "state": state,
                "_embodiment": "robot",
            }
        ]
        with torch.no_grad():
            out = model._predict_action_mot(batch, noise=noise)
        chunk = out[next(iter(out))].float().cpu()
        chunk = normalizer.denormalize(chunk)
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.numpy().astype(np.float32)

    torch.manual_seed(0)
    noise = torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")

    base = sample(state56, noise)
    repeat = sample(state56, noise)
    other = sample(perturbed_state(state56), noise)

    def spread(a, b, label):
        wrist = np.abs(a[:, 0:12] - b[:, 0:12]).max()
        hand = np.abs(a[:, 12:56] - b[:, 12:56]).max()
        print(
            f"  {label:28} wrist max|d|={wrist*1000:8.3f}mm  hand max|d|={hand:7.4f}rad"
        )

    print("\n=== state sensitivity, noise pinned ===")
    spread(base, repeat, "same state, same noise")
    spread(base, other, "PERTURBED state, same noise")

    identical = np.array_equal(base, repeat)
    print(f"\n  pin works (bit-identical repeat): {identical}")
    wrist_delta = float(np.abs(base[:, 0:12] - other[:, 0:12]).max())
    hand_delta = float(np.abs(base[:, 12:56] - other[:, 12:56]).max())
    print(f"  state changes the chunk        : "
          f"{wrist_delta > 1e-5 or hand_delta > 1e-5}")


if __name__ == "__main__":
    main()
