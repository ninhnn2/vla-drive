from setuptools import setup

package = "drive_ros"

setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package]),
                ("share/" + package, ["package.xml"]),
                ("share/" + package + "/launch", ["launch/drive.launch.py"])],
    install_requires=["setuptools"],
    zip_safe=True,
    description="ROS 2 nodes around drive/",
    license="MIT",
    entry_points={"console_scripts": [
        "sim = drive_ros.sim_node:main",
        "driver = drive_ros.drive_node:main",
        "recorder = drive_ros.recorder_node:main",
        "teleop = drive_ros.teleop_node:main",
    ]},
)
