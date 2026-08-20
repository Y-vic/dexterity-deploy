from setuptools import find_packages, setup

package_name = "obs_node"

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
    description="TCP observation sender for robot state, tactile data, and ZED metadata.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "obs_node = obs_node.obs_node:main",
        ],
    },
)
