from setuptools import find_packages, setup

package_name = "ws_dashboard"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["dashboard.html"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnd-humanoid",
    maintainer_email="pnd.humanoid@example.com",
    description="Six-panel live dashboard for the PND-SharpA pipeline.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dashboard = ws_dashboard.dashboard:main",
        ],
    },
)
