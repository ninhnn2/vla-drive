"""The simulated car and the driver together, optionally with the recorder.

    ros2 launch drive_ros drive.launch.py policy:=cnn checkpoint:=v1 track:=trefoil record:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = {"track": "peanut", "reverse": "false", "obstacles": "", "show": "true",
             "policy": "pure-pursuit", "checkpoint": "", "latency_ms": "0.0",
             "record": "false", "name": "ros"}


def generate_launch_description():
    arg = LaunchConfiguration

    def typed(name, kind):
        return ParameterValue(arg(name), value_type=kind)

    track = {"track": arg("track"), "reverse": typed("reverse", bool)}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default) for name, default in ARGUMENTS.items()],
        Node(package="drive_ros", executable="sim", output="screen",
             parameters=[{**track, "obstacles": typed("obstacles", str), "show": typed("show", bool)}]),
        Node(package="drive_ros", executable="driver", output="screen",
             parameters=[{**track, "policy": arg("policy"), "checkpoint": typed("checkpoint", str),
                          "latency_ms": typed("latency_ms", float)}]),
        Node(package="drive_ros", executable="recorder", output="screen", condition=IfCondition(arg("record")),
             parameters=[{"name": arg("name")}]),
    ])
