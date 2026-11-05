from setuptools import setup

package_name = "ur_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ur-sharpa",
    maintainer_email="ur.sharpa@example.com",
    description="Dual UR5 RTDE hardware node: the single command outlet for both arms.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ur_node = ur_node.ur_node:main",
        ],
    },
)
