"""The governor on REAL sampled chunks: inert at 1.0, exact at a scale.

Confirms on the served checkpoint's own output, not synthetic input, that
(a) at SPEED_SCALE=1.0 the chunk that reaches decode is byte-identical to what
the model emitted, and (b) at a scale the commanded per-step motion falls by
exactly that factor while the path direction is unchanged.

Runs on the remote with its own model copy.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import phase_sweep as ps
import repr_bridge as rb
import server as srv
import speed_guard as sg


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+", default=[300, 900, 1350])
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/004000")
    args = parser.parse_args()

    model = ps.ct_load(args.checkpoint)
    normalizer = HandActionNormalizer.load(ps.NORM)
    torch.manual_seed(0)
    noise = torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")

    def sample(state56, jpeg):
        image = Image.open(_io.BytesIO(jpeg)).convert("RGB").resize((256, 256))
        state = normalizer.normalize_state(torch.from_numpy(state56))
        batch = [{"image": [image], "lang": ct.UR_TASK, "state": state,
                  "_embodiment": "robot"}]
        with torch.no_grad():
            out = model._predict_action_mot(batch, noise=noise)
        chunk = out[next(iter(out))].float().cpu()
        chunk = normalizer.denormalize(chunk)
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.numpy().astype(np.float32)

    print(f"\n=== governor on real chunks, scale {args.scale} ===")
    print(f"    ceiling {sg.MAX_WRIST_STEP_MM:.2f}mm / {sg.MAX_HAND_STEP_RAD:.3f}rad "
          f"per step; slice [{srv.EXECUTE_START}, "
          f"{srv.EXECUTE_START + srv.EXECUTE_LENGTH})\n")
    for start in args.starts:
        anchor, hrows, jpeg, _gt = ct.build_window(args.episode, start)
        raw = sample(rb.build_state_rows(anchor, hrows), jpeg)
        full, rep_full = sg.govern(raw, 1.0)
        slow, rep_slow = sg.govern(raw, args.scale)
        identical = np.array_equal(full, raw)
        print(f"  --- start {start} ---")
        print(f"    scale 1.0: identical to raw = {identical}, "
              f"capped = {rep_full['capped']}")
        for side in ("left", "right"):
            f = sg.wrist_steps_mm(full, side)[srv.EXECUTE_START:]
            s = sg.wrist_steps_mm(slow, side)[srv.EXECUTE_START:]
            ratio = float(s.mean() / max(f.mean(), 1e-9))
            # Direction of the executed leg, before and after.
            def leg(chunk):
                t = np.asarray(
                    chunk[:, rb.SLICES[f"{side}_wrist_trans"]], np.float64)
                v = t[-1] - t[srv.EXECUTE_START]
                n = np.linalg.norm(v)
                return v / n if n > 1e-12 else v
            cos = float(np.dot(leg(full), leg(slow)))
            print(f"    {side}: mean step {f.mean():6.2f} -> {s.mean():6.2f} mm "
                  f"(ratio {ratio:.3f}, want {args.scale:.3f}); "
                  f"direction cos {cos:.6f}")
        print(f"    max step seen: {rep_slow['max_wrist_step_mm']:.2f}mm "
              f"/ {rep_slow['max_hand_step_rad']:.4f}rad")
    print("\n  ratio == scale and cos == 1 means the governor slows motion without")
    print("  bending the path. capped == [] at 1.0 means it is inert in normal use.")


if __name__ == "__main__":
    main()
