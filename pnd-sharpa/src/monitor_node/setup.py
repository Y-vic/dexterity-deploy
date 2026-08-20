from setuptools import find_packages, setup

package_name = "monitor_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/monitor_node.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Collect local robot telemetry into columnar NPZ recording samples.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "monitor = monitor_node.recording_monitor:main",
        ],
    },
)
