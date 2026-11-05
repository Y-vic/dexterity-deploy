from setuptools import setup

package_name = "vive_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vive_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ur-sharpa",
    maintainer_email="ur.sharpa@example.com",
    description="Vive tracker teleop: wrist poses -> ur_common IK -> dual UR5 joint targets.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vive_node = vive_node.vive_node:main",
        ],
    },
)
