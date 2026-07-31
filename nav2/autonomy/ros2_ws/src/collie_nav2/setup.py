from glob import glob
import os

from setuptools import find_packages, setup


package_name = "collie_nav2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml") + glob("config/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Wendy Labs",
    maintainer_email="robotics@wendy.sh",
    description="Guarded Nav2 return-to-home gateway for collie-demo",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "gateway = collie_nav2.gateway:main",
        ],
    },
)
