from setuptools import find_packages, setup

package_name = "manus_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/manus_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Manus SDK acquisition and Sharpa retargeted-joint publisher.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "manus = manus_node.manus_node:main",
        ],
    },
)
