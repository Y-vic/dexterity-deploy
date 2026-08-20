from setuptools import find_packages, setup

package_name = "actor_node"

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
    description="TCP action receiver that publishes Adam and Sharpa command topics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "actor_node = actor_node.actor_node:main",
        ],
    },
)
