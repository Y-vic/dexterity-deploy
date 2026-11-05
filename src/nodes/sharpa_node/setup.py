from setuptools import find_packages, setup

package_name = "sharpa_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sharpa_node.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={"console_scripts": ["sharpa_node = sharpa_node.sharpa_node:main"]},
)
