from setuptools import find_packages, setup

package_name = "deploy_common"

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
    description="Shared TCP frame protocol helpers for deploy nodes.",
    license="Apache-2.0",
)
