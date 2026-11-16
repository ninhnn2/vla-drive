"""The topics the nodes share, and the conversions between their messages and arrays.

    /drive/image   sensor_msgs/Image          the camera, rgb8
    /drive/state   geometry_msgs/TwistStamped  linear.x: speed in m/s; angular.z: steering position, -1..1
    /drive/range   sensor_msgs/Range          distance to the nearest thing ahead, metres
    /drive/pose    geometry_msgs/Pose2D       where the car is; only a simulator publishes it
    /drive/cmd     geometry_msgs/Twist        linear.x: throttle, angular.z: steer, both -1..1
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image

IMAGE, STATE, RANGE, POSE, CMD = "/drive/image", "/drive/state", "/drive/range", "/drive/pose", "/drive/cmd"


def param(node, name: str, default):
    """Declare a parameter and read it as the type of `default`.

    Typed loosely on purpose: `-p latency_ms:=200` is an integer to ROS 2, and a
    strictly-typed float parameter would refuse it.
    """
    value = node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value
    return type(default)(value)


def image_to_msg(image: np.ndarray, stamp) -> Image:
    """A uint8 HxWx3 RGB array as a message."""
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, "camera"
    msg.height, msg.width = image.shape[:2]
    msg.encoding, msg.step = "rgb8", 3 * image.shape[1]
    msg.data = image.tobytes()
    return msg


def msg_to_image(msg: Image) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise ValueError(f"expected an rgb8 image, got {msg.encoding}")
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]


def action_to_msg(action) -> Twist:
    """`[steer, throttle]` as a command."""
    msg = Twist()
    msg.angular.z, msg.linear.x = float(action[0]), float(action[1])
    return msg


def msg_to_action(msg: Twist) -> np.ndarray:
    return np.array([msg.angular.z, msg.linear.x])


def seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def spin(make_node) -> None:
    """Run one node until Ctrl-C, then let it clean up (`node.close()`)."""
    rclpy.init()
    node = make_node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():                # not the Ctrl-C that landed in the middle of a message
            raise
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
