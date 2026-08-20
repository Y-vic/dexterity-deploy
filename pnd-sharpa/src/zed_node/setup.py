from glob import glob

from setuptools import find_packages, setup

package_name = "zed_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/zed.launch.py"]),
        ("share/" + package_name + "/web", glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="ZED observation and output status node.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "zed = zed_node.zed_node:main",
        ],
    },
)
