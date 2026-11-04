from setuptools import find_packages, setup

package_name = "ur_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["../../config/protection.yaml"]),
    ],
    # The compiled ikfast bindings and the hand URDFs must land next to the
    # Python sources, otherwise kinematics.py cannot find them at runtime.
    package_data={
        package_name: [
            "ikfastpy/*.so",
            "ikfastpy/*.xml",
            "assets/hand_urdf/*/*.urdf",
            "assets/command_calibration.json",
            "assets/sharpa_firmware_limits.json",
            "assets/ego_reference.js",
            "assets/workbench/*",
        ],
    },
    include_package_data=True,
    install_requires=["setuptools", "PyYAML"],
    tests_require=["pytest"],
    zip_safe=False,
    maintainer="ur-sharpa",
    maintainer_email="ur.sharpa@example.com",
    description="Shared UR5 + SharpA kinematics: the single FK/IK entry point for teleop and deploy.",
    license="Apache-2.0",
)
