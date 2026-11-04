"""One hardware graph for collection, policy, HIL and replay."""
from ur_launch.launch_config import build_launch_description


def generate_launch_description():
    return build_launch_description("workbench")
