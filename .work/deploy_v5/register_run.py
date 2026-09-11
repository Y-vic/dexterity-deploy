#!/usr/bin/env python3
"""Add or replace one deployable run in a task config.

    python scripts/deploy/task_config/register_run.py \
        --task ur_orca_action \
        --run-id mot_v5_head100h_030000 \
        --model mot_action_v5_head100h \
        --checkpoint outputs/<exp>/checkpoints/030000 \
        --config outputs/<exp>/hydra/.hydra/config.yaml \
        --normalizer outputs/shared_mot_action_v5_head100h/hand_action_norm.json

The workstation reads these files over ssh and shows one row per run. A run is
only clickable when EVERY one of the following holds -- the dashboard checks all
of them and prints the failures it finds (model_manager.catalog):

    * the task directory name starts with ``ur_``;
    * ``scripts/deploy/<model>/launch.sh`` exists;
    * ``<asset_root>/<checkpoint>`` is a directory;
    * ``deployment.eligible`` is true;
    * ``status`` is ``complete`` or ``in_progress``.

``run_id`` is the immutable identity the workstation passes back to launch.sh,
and it must be globally unique within the task: two runs sharing one would make
the started model depend on list order. Replacing an existing run_id therefore
requires ``--replace``, so a typo cannot silently repoint a run at a different
checkpoint.

Paths are stored RELATIVE to the asset root, because the workstation resolves
them against its own profile's asset_root, which need not equal this checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "sharpa.deploy.task_config.v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task directory name, e.g. ur_orca_action")
    parser.add_argument("--run-id", required=True, help="globally unique, immutable")
    parser.add_argument("--model", required=True, help="must equal the launcher directory name")
    parser.add_argument("--checkpoint", required=True, help="path relative to the asset root")
    parser.add_argument("--config", required=True, help="hydra config.yaml, relative to asset root")
    parser.add_argument("--normalizer", required=True, help="hand_action_norm.json, relative")
    parser.add_argument("--prompt", default="", help="task prompt; may be left empty for now")
    parser.add_argument("--dataset", default="", help="optional dataset path, relative")
    parser.add_argument("--task-id", default="", help="defaults to the task directory name")
    parser.add_argument("--variant", default="")
    parser.add_argument("--stage", default="deploy")
    parser.add_argument(
        "--status",
        default="complete",
        choices=("complete", "in_progress"),
        help="anything else makes the dashboard refuse to launch the run",
    )
    parser.add_argument(
        "--not-eligible",
        action="store_true",
        help="register the run but leave it un-launchable (deployment.eligible=false)",
    )
    parser.add_argument("--reason", default="", help="why it is not eligible, shown in the UI")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite an existing run with this run_id instead of refusing",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="task config root; defaults to this script's directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.task.startswith("ur_"):
        # Not a style rule: the dashboard's compatible flag is literally
        # task_name.startswith("ur_"), so any other name registers fine and then
        # shows up permanently greyed out with "不是 UR task".
        raise SystemExit(
            f"task name {args.task!r} must start with 'ur_' or the workstation will "
            "mark every run in it incompatible"
        )

    task_dir = Path(args.root) / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "task.json"

    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema": SCHEMA, "task": {}, "runs": []}

    payload.setdefault("schema", SCHEMA)
    task = payload.setdefault("task", {})
    task.setdefault("id", args.task_id or args.task)
    # The prompt is task-level, and the server reports it in /metadata. An empty
    # one is allowed: the client only compares prompts when both sides are
    # non-empty, so this can ship before the wording is settled.
    if args.prompt or "prompt" not in task:
        task["prompt"] = args.prompt
    if args.dataset or "dataset" not in task:
        task["dataset"] = args.dataset

    runs = payload.setdefault("runs", [])
    entry = {
        "id": args.run_id,
        "run_id": args.run_id,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "normalizer": args.normalizer,
        "variant": args.variant,
        "stage": args.stage,
        "status": args.status,
        "deployment": {
            "eligible": not args.not_eligible,
            "reason": args.reason or ("" if not args.not_eligible else "未标记为可部署"),
        },
    }

    existing = [i for i, run in enumerate(runs) if str(run.get("run_id")) == args.run_id]
    if existing and not args.replace:
        raise SystemExit(
            f"run_id {args.run_id!r} already exists in {path}. Pass --replace only if "
            "you intend to repoint it; a duplicate would make the launched model "
            "depend on list order."
        )
    if existing:
        for index in reversed(existing[1:]):
            del runs[index]
        runs[existing[0]] = entry
        action = "replaced"
    else:
        runs.append(entry)
        action = "added"

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{action} run {args.run_id!r} in {path}")
    print(f"  model      {args.model}")
    print(f"  checkpoint {args.checkpoint}")
    print(f"  eligible   {entry['deployment']['eligible']}")
    if not task["prompt"]:
        print("  NOTE: task prompt is empty; the VLM gets no instruction text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
