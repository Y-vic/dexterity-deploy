from setuptools import find_packages, setup

package_name = "actor_ik_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/actor_ik_node.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "actor_ik_node = actor_ik_node.actor_ik_node:main",
            "hand_tracking_node = actor_ik_node.hand_tracking:main",
        ]
    },
)
