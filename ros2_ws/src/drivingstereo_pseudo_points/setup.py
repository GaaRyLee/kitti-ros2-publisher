from glob import glob

from setuptools import setup

package_name = "drivingstereo_pseudo_points"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "pseudo_points_writer = drivingstereo_pseudo_points.pseudo_points_writer:main",
        ],
    },
)
