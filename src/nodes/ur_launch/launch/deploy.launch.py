"""Start the policy deployment graph."""

from ur_launch.launch_config import build_launch_description


def generate_launch_description():
    return build_launch_description("deploy")
