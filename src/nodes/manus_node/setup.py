from setuptools import setup

package_name = "manus_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/manus_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ur-sharpa",
    maintainer_email="ur.sharpa@example.com",
    description="Manus glove acquisition plus hand retargeting subprocess -> 44D SharpA HandCommand.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "manus_node = manus_node.manus_node:main",
        ],
    },
)
