from setuptools import find_packages, setup

package_name = "monitor_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/monitor_node.launch.py"]),
    ],
    package_data={package_name: ["static/index.html"]},
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={"console_scripts": ["monitor_node = monitor_node.monitor_node:main"]},
)
