from setuptools import find_packages, setup

package_name = "bias_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/bias.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Bias UI and Adam-managed target publisher.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bias = bias_node.bias_node:main",
        ],
    },
)
