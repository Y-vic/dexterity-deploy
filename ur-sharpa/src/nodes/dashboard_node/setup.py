from setuptools import setup

package_name = "dashboard_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # The page is a single self-contained HTML file shipped next to the sources:
    # no build step, no npm, no CDN.
    package_data={package_name: ["dashboard.html"]},
    include_package_data=True,
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=False,
    maintainer="ur-sharpa",
    maintainer_email="ur.sharpa@example.com",
    description="Machine, remote model and policy dashboard for dual UR5 + SharpA.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dashboard_node = dashboard_node.dashboard_node:main",
        ],
    },
)
