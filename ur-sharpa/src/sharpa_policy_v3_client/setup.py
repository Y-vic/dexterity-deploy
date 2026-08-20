from glob import glob
from setuptools import find_packages, setup


package_name = "sharpa_policy_v3_client"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SharpA Deployment Team",
    maintainer_email="sharpa-deploy@example.com",
    description="Isolated SharpA policy server v3 workstation client.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mock_policy_server = sharpa_policy_v3_client.mock_server:main",
            "policy_client = sharpa_policy_v3_client.policy_client_node:main",
            "policy_node = sharpa_policy_v3_client.policy_client_node:main",
            "state_node = sharpa_policy_v3_client.state_node:main",
            "ur_node = sharpa_policy_v3_client.ur_node:main",
            "sharpa_node = sharpa_policy_v3_client.sharpa_node:main",
            "zed_node = sharpa_policy_v3_client.zed_node:main",
            "action_node = sharpa_policy_v3_client.action_node:main",
            "ur_safe_jog = sharpa_policy_v3_client.ur_safe_jog:main",
        ],
    },
)
