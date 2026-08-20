from setuptools import find_packages, setup

package_name = "ws_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Workstation core observation, policy, and actor sync nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "obs_sync = ws_core.obs_sync:main",
            "policy_client = ws_core.policy_client:main",
            "action_ik = ws_core.action_ik:main",
            "action_execute = ws_core.action_execute:main",
            "replay = ws_core.replay:main",
        ],
    },
)
