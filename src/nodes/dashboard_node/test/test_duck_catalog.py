"""Enumerate the shipped duck catalogue without ROS, models, or remote hosts."""

import json
from pathlib import Path

import pytest

from dashboard_node.model_manager import RemoteModelManager, ServerProfile


@pytest.mark.parametrize("machine,has_checkpoints", [("ps", True), ("ur", False)])
def test_duck_catalog_keeps_eight_prompts_and_disables_missing_machine_assets(
    tmp_path, machine, has_checkpoints
):
    source = Path(__file__).resolve().parents[4] / "config/model_tasks/ur_duck_placement/task.json"
    task = json.loads(source.read_text())
    bundle, assets, tasks = tmp_path / "bundle", tmp_path / machine, tmp_path / "tasks"
    launcher = bundle / "scripts/deploy/dfm_flow/launch.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("exit 0\n")
    task_file = tasks / "ur_duck_placement/task.json"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(json.dumps(task))
    for entry in task["runs"]:
        if has_checkpoints:
            (assets / entry["checkpoint"]).mkdir(parents=True)
        for relative in entry["deployment"]["required_files"]:
            dependency = bundle / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.touch()
    profile = ServerProfile.from_mapping(machine, {
        "label": machine,
        "local": True,
        "repo_root": str(bundle),
        "asset_root": str(assets),
        "task_config_root": str(tasks),
    })
    manager = RemoteModelManager(profiles={machine: profile}, default_server=machine)
    try:
        catalogue = manager.catalog(server_id=machine)
    finally:
        manager.close()
    assert len(catalogue["tasks"]) == 1
    runs = {entry["id"]: entry for entry in catalogue["tasks"][0]["runs"]}
    assert set(runs) == {
        "real101", "sim100_real101", "sim200_real101", "sim1k_real101",
        "sim2k_real101", "sim4k_real101", "sim8k_real101", "sim8k",
    }
    real_prompt = "Pick up the yellow ducks and place them on the pink plate."
    sim_prompt = "Pick up all the yellow ducks on the table and place them in the plate."
    for name, entry in runs.items():
        assert entry["prompt"] == (sim_prompt if name == "sim8k" else real_prompt)
        assert entry["run_id"].endswith("-2gpu-40k-20261106")
        assert entry["checkpoint_path"] == str(
            assets / "logs" / entry["run_id"] / "checkpoints/checkpoint-40000"
        )
        assert entry["checkpoint_exists"] is has_checkpoints
        assert entry["launchable"] is has_checkpoints
        if not has_checkpoints:
            assert "checkpoint 不存在" in entry["disabled_reason"]
