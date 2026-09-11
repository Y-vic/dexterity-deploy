from setuptools import find_packages, setup

package_name = "recording_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "DEMO.md"]),
        ("share/" + package_name + "/launch", ["launch/recording_node.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "Pillow"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "recording_node = recording_node.recording_node:main",
        "demo_node = recording_node.demo_node:main",
    ]},
)
