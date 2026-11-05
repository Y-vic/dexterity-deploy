from setuptools import setup

setup(
    name="ae01_node", version="0.1.0", packages=["ae01_node"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/ae01_node"]),
                ("share/ae01_node", ["package.xml"])],
    install_requires=["setuptools"], tests_require=["pytest"], zip_safe=True,
    maintainer="ur-sharpa", maintainer_email="ur.sharpa@example.com", license="Apache-2.0",
    description="AE01 Avatar SDK ROBOT input for gated SharpA teleoperation and HIL.",
    entry_points={"console_scripts": ["ae01_node = ae01_node.ae01_node:main"]},
)
