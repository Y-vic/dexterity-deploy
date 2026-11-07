"""Launch the native DFM InferenceServer without ROS or hardware connections."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from dfm.inference.serving.configuration import DeploymentConfig, TaskConfig
from dfm.inference.serving.server import InferenceServer, load_server
from sharpa_interface.dfm.environment import NativeEnvironmentAdapter
from sharpa_interface.dfm.model_runtime import SerializedPolicy


class DeploymentInferenceServer(InferenceServer):
    def __init__(self, loaded, environment):
        super().__init__(loaded.policy, loaded.config, demonstrations=loaded.demonstrations)
        self.environment = environment
        self.warmup_result = None

    def health(self):
        return {
            **super().health(),
            "run_id": self.config.task.run_id,
            "checkpoint_path": self.config.model.checkpoint_path,
            "policy_transport": "dfm_native",
            "interface_revision": self.interface_revision,
            "calibration_id": self.environment.calibration.calibration_id,
            "warmup": self.warmup_result,
        }


def resolve_run(config_root, asset_root, task_name, run_id):
    if not re.fullmatch(r"ur_[A-Za-z0-9._-]+", task_name):
        raise ValueError("Expected a configured UR task name")
    payload = json.loads((Path(config_root) / task_name / "task.json").read_text())
    if payload.get("schema") != "sharpa.deploy.task_config.v2":
        raise ValueError("Unsupported task catalog schema")
    runs = [run for run in payload["runs"] if run["run_id"] == run_id]
    if len(runs) != 1:
        raise ValueError("run_id must identify exactly one configured run")
    run = runs[0]
    if (
        run.get("model") != "dfm_flow"
        or run.get("status") != "complete"
        or run.get("deployment", {}).get("eligible") is not True
    ):
        raise ValueError("Run is not eligible for native DFM Flow deployment")
    root = Path(asset_root).resolve()
    checkpoint = (root / run["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or not checkpoint.is_dir():
        raise ValueError(f"Checkpoint does not exist inside the configured asset root: {checkpoint}")
    return payload["task"], run, checkpoint


def deployment_config(asset_root, checkpoint, task, run, *, device, precision, port, execute_steps):
    saved = json.loads((checkpoint / "config.json").read_text())
    processor = json.loads((checkpoint / "dfm_processor.json").read_text())
    if (
        saved.get("state_dim"),
        saved.get("action_dim"),
        saved.get("action_hz"),
        saved.get("action_horizon"),
        saved.get("action_representation"),
    ) != (62, 62, 30, 30, "flow"):
        raise ValueError("Deployment requires a physical 62D 30Hz/30-step Flow checkpoint")
    if (
        processor.get("camera_views") != ["ego", "left_wrist", "right_wrist"]
        or not (checkpoint / "action_expert").is_dir()
    ):
        raise ValueError("Deployment requires the saved three-camera Flow processor and expert")
    config = DeploymentConfig.load(Path(asset_root) / "configs/inference/dfm_sharpa62.yaml")
    return replace(
        config,
        model=replace(
            config.model, checkpoint_path=str(checkpoint), decoder="flow", device=device, precision=precision
        ),
        interface=replace(
            config.interface, execution=replace(config.interface.execution, execute_length=execute_steps)
        ),
        task=TaskConfig(task_id=task["id"], run_id=run["run_id"], prompt=run.get("prompt", task.get("prompt", ""))),
        server=replace(config.server, host="127.0.0.1", port=port),
    )


def warmup(service):
    """Exercise first/cached inference at saved pixel budgets, then reset history.

    Probe targets remain inside this model process. Warmup happens before the
    health endpoint is bound and does not enter the user/Actor request counters.
    """
    from PIL import Image

    processor = json.loads((Path(service.config.model.checkpoint_path) / "dfm_processor.json").read_text())
    images, sizes = {}, {}
    aspect = 16 / 9
    for view in service.adapter.views:
        budget = processor["image_sizes"][view]
        minimum, maximum = budget["min_pixels"], budget["max_pixels"]
        if type(minimum) is not int or type(maximum) is not int or not 0 < minimum <= maximum <= 16_777_216:
            raise ValueError("Invalid checkpoint camera pixel budget")
        height = int(math.sqrt(maximum / aspect) // 32) * 32
        width = int(math.sqrt(maximum * aspect) // 32) * 32
        if not minimum <= height * width <= maximum:
            raise ValueError("Camera pixel budget cannot represent a 16:9 resize grid")
        images[service.config.interface.camera_bindings[view]] = Image.new("RGB", (width, height), (127, 127, 127))
        sizes[view] = [width, height]
    model_ego = np.zeros(62, np.float32)
    model_ego[[3, 7, 12, 16]] = 1
    wire = service.environment.calibration.convert(model_ego, to_model=False)
    state, snapshot = service.environment.capture(wire, service.config.interface)
    seconds, diagnostics = [], []
    started = time.perf_counter()
    try:
        service.policy.set_seed(service.config.model.inference.seed)
        service.policy.reset(service.config.task.prompt, demonstrations=service.demonstrations)
        for index in range(2):
            observation = service.adapter.observation(
                {
                    "format_id": service.config.interface.observation.format_id,
                    "images": images,
                    "state": state,
                    "timestamp_s": float(index),
                    "observation_id": f"startup-warmup:{index}",
                },
                snapshot,
            )
            before = time.perf_counter()
            plan = service.policy.predict(observation)
            service.adapter.action(plan, snapshot)
            seconds.append(time.perf_counter() - before)
            diagnostics.append(plan.diagnostics_dict())
    finally:
        service.policy.set_seed(service.config.model.inference.seed)
        service.policy.reset(service.config.task.prompt, demonstrations=service.demonstrations)
    return {
        "requests": 2,
        "seconds": time.perf_counter() - started,
        "request_seconds": seconds,
        "request_diagnostics": diagnostics,
        "image_sizes": sizes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("run_id")
    parser.add_argument("--asset-root", default=os.environ.get("SHARPA_DEPLOY_ASSET_ROOT"))
    parser.add_argument("--task-config-root", default=os.environ.get("SHARPA_DEPLOY_TASK_CONFIG_ROOT"))
    parser.add_argument("--calibration", default=os.environ.get("SHARPA_DEPLOY_CALIBRATION"))
    parser.add_argument("--data-contract", default=os.environ.get("SHARPA_DEPLOY_DATA_CONTRACT"))
    parser.add_argument("--device", default=os.environ.get("SHARPA_DEPLOY_DEVICE", "cuda:0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHARPA_DEPLOY_PORT", "5500")))
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument(
        "--check", action="store_true", help="Validate physical/artifact contracts without loading model weights"
    )
    args = parser.parse_args(argv)
    if not args.asset_root or not args.task_config_root or not args.calibration or not args.data_contract:
        parser.error("asset root, task catalog, measured UR calibration and native data contract are required")
    task, run, checkpoint = resolve_run(args.task_config_root, args.asset_root, args.task, args.run_id)
    config = deployment_config(
        args.asset_root,
        checkpoint,
        task,
        run,
        device=args.device,
        precision=args.precision,
        port=args.port,
        execute_steps=args.execute_steps,
    )
    environment = NativeEnvironmentAdapter(args.calibration, args.data_contract)
    environment.validate_interface(config.interface)
    if args.check:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "task_id": config.task.task_id,
                    "run_id": config.task.run_id,
                    "prompt": config.task.prompt,
                    "calibration_id": environment.calibration.calibration_id,
                    "native_interface": config.interface.to_dict(),
                    "policy_transport": "dfm_native",
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(f"DFM native loading: task={config.task.task_id} run={config.task.run_id}", flush=True)
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dfm-model")
    loaded = None
    try:
        # CUDA allocation and compiler setup belong to the same worker that
        # handles both warmup paths and every subsequent episode operation.
        loaded = executor.submit(load_server, config).result()
        loaded.policy = SerializedPolicy(loaded.policy, executor)
        service = DeploymentInferenceServer(loaded, environment)
        print(f"DFM model loaded: seconds={time.perf_counter() - started:.3f}; warmup requests=2", flush=True)
        service.warmup_result = warmup(service)
        print(f"DFM warmup complete: seconds={service.warmup_result['seconds']:.3f}", flush=True)
        with service.create_server() as server:
            print(
                f"DFM native ready: task={config.task.task_id} run={config.task.run_id} port={config.server.port}",
                flush=True,
            )
            server.serve_forever()
    finally:
        if loaded is not None:
            if isinstance(loaded.policy, SerializedPolicy):
                loaded.policy.close()
            else:
                executor.submit(loaded.policy.close).result()
        executor.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
