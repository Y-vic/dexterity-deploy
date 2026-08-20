from setuptools import find_packages, setup

package_name = "ws_io"

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
    description="Workstation TCP I/O bridge nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot_states = ws_io.robot_states:main",
            "robot_tactile = ws_io.robot_tactile:main",
            "robot_vision = ws_io.robot_vision:main",
        ],
    },
)
