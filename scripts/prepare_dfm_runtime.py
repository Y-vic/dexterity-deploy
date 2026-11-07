"""Copy this deployment adapter and measured calibration to an inference host.

Only the model runtime bundle under DFM logs/ is created. No model or hardware
process is started. Run this again after changing the adapter or calibration.
"""

import argparse
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="SSH alias; omit for the local inference host")
    parser.add_argument("--dfm-root", required=True)
    parser.add_argument("--python", required=True, help="Absolute Python path in the DFM CUDA environment")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--data-contract", required=True, help="Saved native physical camera/training declaration")
    args = parser.parse_args(argv)
    if not Path(args.dfm_root).is_absolute() or not Path(args.python).is_absolute():
        parser.error("DFM root and Python executable must be absolute paths")
    if args.host and (args.host.startswith("-") or ":" in args.host):
        parser.error("Use a configured SSH alias")
    calibration = Path(args.calibration).resolve()
    if not calibration.is_file():
        parser.error("Measured calibration JSON does not exist")
    data_contract = Path(args.data_contract).resolve()
    if not data_contract.is_file():
        parser.error("Native data contract JSON does not exist")
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.dfm_root) / "logs/deploy-runtime"
    temporary_root = root / "runtime/tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dfm-runtime-", dir=temporary_root) as temporary:
        bundle = Path(temporary)
        shutil.copytree(
            root / "interface/sharpa_interface",
            bundle / "interface/sharpa_interface",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(root / "scripts/deploy", bundle / "scripts/deploy")
        shutil.copytree(root / "config/model_tasks", bundle / "config/model_tasks")
        shutil.copyfile(calibration, bundle / "config/dfm_calibration.json")
        shutil.copyfile(data_contract, bundle / "config/real101_data_contract.json")
        environment = (
            f"export SHARPA_DEPLOY_PYTHON={shlex.quote(args.python)}\n"
            f'export PATH={shlex.quote(str(Path(args.python).parent))}:"$PATH"\n'
            f'export PYTHONPATH={shlex.quote(args.dfm_root)}:"${{PYTHONPATH:-}}"\n'
            f"export SHARPA_DEPLOY_ASSET_ROOT={shlex.quote(args.dfm_root)}\n"
            f"export SHARPA_DEPLOY_TASK_CONFIG_ROOT={shlex.quote(str(destination / 'config/model_tasks'))}\n"
        )
        (bundle / "environment.sh").write_text(environment)
        if args.host:
            subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", args.host, "mkdir -p -- " + shlex.quote(str(destination))],
                check=True,
            )
            target = args.host + ":" + str(destination) + "/"
        else:
            destination.mkdir(parents=True, exist_ok=True)
            target = str(destination) + "/"
        subprocess.run(["rsync", "-a", "--protect-args", str(bundle) + "/", target], check=True)
        # These owned directories are complete snapshots. Remove superseded
        # catalogs/adapters only inside them; leave other runtime assets intact.
        for relative in ("interface/sharpa_interface", "scripts/deploy", "config/model_tasks"):
            subprocess.run(
                ["rsync", "-a", "--delete", "--protect-args", str(bundle / relative) + "/", target + relative + "/"],
                check=True,
            )
    print(f"Prepared {args.host + ':' if args.host else ''}{destination}; no service started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
