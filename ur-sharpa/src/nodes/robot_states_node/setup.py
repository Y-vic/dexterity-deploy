from setuptools import find_packages, setup

package_name = "robot_states_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/robot_states_node.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={"console_scripts": ["robot_states_node = robot_states_node.robot_states_node:main"]},
)
