from setuptools import find_packages, setup

setup(
    name="ur_launch",
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ur_launch"]),
        ("share/ur_launch", ["package.xml"]),
        (
            "share/ur_launch/launch",
            ["launch/ur.launch.py", "launch/teleop.launch.py", "launch/deploy.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    entry_points={},
)
