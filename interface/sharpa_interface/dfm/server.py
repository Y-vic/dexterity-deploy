"""HTTP/MsgPack deployment service using the existing DFM policy loader.

This process owns only model inference; it has no ROS or hardware connection.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sharpa_interface.dfm.calibration import DFMCalibration
from sharpa_interface.dfm.policy import DFMDeploymentPolicy
from sharpa_interface.server.serialization import packb, unpackb


def create_app(adapter, *, warmup=True):
    from aiohttp import web

    app = web.Application(client_max_size=adapter.metadata["max_message_size"])
    owner = None
    owner_token = None
    last_id = -1
    last_digest = last_response = None
    reset_generation = 0
    operation_lock = asyncio.Lock()
    model_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dfm-inference")
    ready = False
    warmup_result = None

    async def model_operation(method, *args):
        # One worker retains the CUDA/compiler execution context for warmup,
        # resets and every real prediction. The network loop stays responsive.
        return await asyncio.get_running_loop().run_in_executor(model_worker, method, *args)

    async def model_lifecycle(app):
        nonlocal ready, warmup_result
        try:
            if warmup:
                print("DFM inference warmup started: requests=2", flush=True)
                warmup_result = await model_operation(adapter.warmup)
                print(
                    f"DFM inference warmup complete: seconds={warmup_result['seconds']:.3f} "
                    f"requests={warmup_result['requests']} image_sizes={warmup_result['image_sizes']}",
                    flush=True,
                )
            ready = True
            print(
                f"DFM deployment ready: task={adapter.metadata['task_id']} run={adapter.metadata['run_id']} "
                f"calibration={adapter.calibration.calibration_id}",
                flush=True,
            )
            yield
        finally:
            ready = False
            model_worker.shutdown(wait=True)

    app.cleanup_ctx.append(model_lifecycle)

    def error_payload(message, value=None):
        return {
            "schema": "sharpa_policy_error.v1",
            "reset_required": True,
            "session_id": value.get("session_id") if isinstance(value, dict) else None,
            "request_id": value.get("request_id") if isinstance(value, dict) else None,
            "error": {"code": "reset_required", "message": str(message), "retryable": False},
        }

    def response(value, status=200):
        return web.Response(body=packb(value), content_type="application/msgpack", status=status)

    async def health(request):
        return web.json_response(
            {
                "ready": ready,
                "session_active": owner is not None,
                "task_id": adapter.metadata["task_id"],
                "run_id": adapter.metadata["run_id"],
                "calibration_id": adapter.calibration.calibration_id,
                "warmup": warmup_result,
            }
        )

    async def metadata(request):
        return response(adapter.metadata)

    async def reset(request):
        nonlocal reset_generation
        try:
            value = unpackb(await request.read())
            if not isinstance(value, dict):
                raise ValueError("Reset body must be an object")
            async with operation_lock:
                if owner is None or value.get("owner_token") != owner_token:
                    return response(error_payload("Another session owns this policy", value), status=409)
                result = await model_operation(adapter.reset, value)
                reset_generation += 1
            return response(result)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return response(error_payload(exc), status=400)

    async def infer(request):
        nonlocal owner, owner_token, last_id, last_digest, last_response
        if owner is not None:
            return response({"error": "Another client owns this policy"}, status=409)
        import re

        token = request.query.get("owner_token", "")
        if not re.fullmatch(r"[a-f0-9]{32}", token):
            return response(error_payload("Connect with the required session owner token"), status=400)
        socket = web.WebSocketResponse(max_msg_size=adapter.metadata["max_message_size"], autoclose=False)
        owner = socket
        owner_token = token
        last_id, last_digest, last_response = -1, None, None
        generation = reset_generation
        try:
            await socket.prepare(request)
            async with operation_lock:
                await model_operation(adapter.invalidate)
            async for message in socket:
                if message.type != web.WSMsgType.BINARY:
                    await socket.close(code=1003, message=b"Expected binary MsgPack")
                    break
                value = None
                try:
                    value = unpackb(message.data)
                    if not isinstance(value, dict):
                        raise ValueError("Observation must be an object")
                    request_id = value.get("request_id")
                    if type(request_id) is not int or request_id < 0:
                        raise ValueError("request_id must be a nonnegative integer")
                    async with operation_lock:
                        if adapter.session is None or value.get("session_id") != adapter.session:
                            raise ValueError("Reset and use the active session before infer")
                        if generation != reset_generation:
                            generation = reset_generation
                            last_id, last_digest, last_response = -1, None, None
                        digest = hashlib.sha256(message.data).digest()
                        if request_id == last_id and digest == last_digest:
                            await socket.send_bytes(last_response)
                            continue
                        if request_id <= last_id:
                            raise ValueError("Request IDs must increase; duplicate observations must be identical")
                        result = await model_operation(adapter.infer, value)
                        try:
                            encoded = packb(result)
                        except Exception:
                            await model_operation(adapter.invalidate)
                            raise
                        last_id, last_digest, last_response = request_id, digest, encoded
                        await socket.send_bytes(encoded)
                except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                    async with operation_lock:
                        await model_operation(adapter.invalidate)
                        last_id, last_digest, last_response = -1, None, None
                    await socket.send_bytes(packb(error_payload(exc, value)))
        finally:
            # The same worker finishes predict before invalidate/reset, including
            # disconnected sockets; another client cannot reset an in-flight model.
            if owner is socket:
                async with operation_lock:
                    await model_operation(adapter.invalidate)
                owner = None
                owner_token = None
            await socket.close()
        return socket

    app.router.add_get("/healthz", health)
    app.router.add_get("/metadata", metadata)
    app.router.add_post("/reset", reset)
    app.router.add_get("/infer", infer)
    return app


def resolve_run(config_root, asset_root, task_name, run_id):
    # IDs are never interpreted as paths outside the configured task root.
    import re

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
        or run.get("status") not in {"complete", "in_progress"}
        or run.get("deployment", {}).get("eligible") is not True
    ):
        raise ValueError("Run is not eligible for the DFM Flow deployment adapter")
    checkpoint = (Path(asset_root) / run["checkpoint"]).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    return payload["task"], run, checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("run_id")
    parser.add_argument("--asset-root", default=os.environ.get("SHARPA_DEPLOY_ASSET_ROOT"))
    parser.add_argument("--task-config-root", default=os.environ.get("SHARPA_DEPLOY_TASK_CONFIG_ROOT"))
    parser.add_argument("--calibration", default=os.environ.get("SHARPA_DEPLOY_CALIBRATION"))
    parser.add_argument("--device", default=os.environ.get("SHARPA_DEPLOY_DEVICE", "cuda:0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHARPA_DEPLOY_PORT", "5500")))
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument(
        "--require-download-verification",
        action="store_true",
        default=os.environ.get("SHARPA_DEPLOY_REQUIRE_DOWNLOAD_VERIFICATION") == "1",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate artifacts/contracts without allocating model weights"
    )
    args = parser.parse_args(argv)
    if not args.asset_root or not args.task_config_root or not args.calibration:
        parser.error("asset root, task config root and measured calibration are required")
    task, run, checkpoint = resolve_run(args.task_config_root, args.asset_root, args.task, args.run_id)
    calibration = DFMCalibration.load(args.calibration)
    if args.require_download_verification:
        verification = json.loads((checkpoint / ".download_verified.json").read_text())
        if verification.get("verified") is not True:
            raise ValueError("Checkpoint download has not completed integrity verification")
    config = json.loads((checkpoint / "config.json").read_text())
    processor = json.loads((checkpoint / "dfm_processor.json").read_text())
    if (
        config.get("action_dim") != 62
        or config.get("state_dim") != 62
        or config.get("action_horizon") != 30
        or config.get("action_hz") != 30
        or config.get("action_representation") != "flow"
        or processor.get("camera_views") != ["ego", "left_wrist", "right_wrist"]
        or not (checkpoint / "action_expert").is_dir()
    ):
        raise ValueError("Checkpoint is not the supported trained sim-real Flow contract")
    if not 1 <= args.port <= 65535 or not 1 <= args.execute_steps <= 30:
        raise ValueError("Invalid service port or execution length")
    if args.check:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "task_id": task["id"],
                    "run_id": run["run_id"],
                    "calibration_id": calibration.calibration_id,
                    "frequency_hz": 30,
                    "action_length": 30,
                },
                ensure_ascii=False,
            )
        )
        return 0
    from aiohttp import web

    from dfm.inference.factory import load_policy

    print(f"DFM deployment loading: task={task['id']} run={run['run_id']}", flush=True)
    loaded_at = time.perf_counter()
    policy = load_policy(
        checkpoint,
        decoder="flow",
        device=args.device,
        bf16=args.precision == "bfloat16",
        execute_steps=args.execute_steps,
        max_context_length=8192,
        max_new_tokens=512,
        euler_steps=10,
    )
    adapter = DFMDeploymentPolicy(
        policy,
        calibration,
        checkpoint=checkpoint,
        task=task,
        run=run,
        port=args.port,
        execute_steps=args.execute_steps,
    )
    print(f"DFM model loaded: seconds={time.perf_counter() - loaded_at:.3f}; preparing inference", flush=True)
    try:
        web.run_app(create_app(adapter), host="127.0.0.1", port=args.port, print=None)
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
