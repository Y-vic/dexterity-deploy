import os
from glob import glob

from setuptools import find_packages, setup

package_name = "quest_node"


def tree_data_files(root_directory: str) -> list[tuple[str, list[str]]]:
    data_files: list[tuple[str, list[str]]] = []
    for root, _directories, filenames in os.walk(root_directory):
        if not filenames:
            continue
        destination = os.path.join("share", package_name, root)
        sources = [os.path.join(root, filename) for filename in filenames]
        data_files.append((destination, sources))
    return data_files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            ["package.xml", "README.md", "THIRD_PARTY_NOTICES.md"],
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/config",
            glob("config/*.conf") + glob("config/*.yaml"),
        ),
    ]
    + tree_data_files("web")
    + tree_data_files("windows"),
    install_requires=["casadi>=3.7.2", "pin-pink>=3.1.0", "setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Quest WebVR mocap, PND retargeting and Adam command gate.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "quest = quest_node.quest:main",
            "quest_webvr = quest_node.quest_webvr:main",
            "quest_retarget = quest_node.quest_retarget:main",
            "quest_command = quest_node.quest_command:main",
        ],
    },
)
