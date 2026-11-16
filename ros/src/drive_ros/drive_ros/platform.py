"""The car behind the ROS 2 topics, as the control loop sees it (drive.platform.Platform)."""
from __future__ import annotations

from geometry_msgs.msg import Pose2D, Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, Range

from drive.types import Observation, VehicleState

from . import topics


class RosPlatform:
    def __init__(self, node: Node):
        self.node = node
        self.start = node.get_clock().now()
        self.image, self.image_t = None, 0.0
        self.state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.pose = None                 # only a simulator sends one
        self.distance = 0.0              # nothing heard yet: assume something is right ahead
        self.has_state = False
        self.cmd_pub = node.create_publisher(Twist, topics.CMD, 1)
        node.create_subscription(Image, topics.IMAGE, self.on_image, 1)
        node.create_subscription(TwistStamped, topics.STATE, self.on_state, 1)
        node.create_subscription(Range, topics.RANGE, self.on_range, 1)
        node.create_subscription(Pose2D, topics.POSE, self.on_pose, 1)

    @property
    def t(self) -> float:
        """Seconds since this node started."""
        return (self.node.get_clock().now() - self.start).nanoseconds * 1e-9

    @property
    def ready(self) -> bool:
        """True once a camera frame and a state have arrived."""
        return self.image is not None and self.has_state

    def on_image(self, msg: Image) -> None:
        self.image = topics.msg_to_image(msg)
        # When the frame was taken, on this node's clock: its age is part of the latency.
        self.image_t = topics.seconds(msg.header.stamp) - self.start.nanoseconds * 1e-9

    def on_state(self, msg: TwistStamped) -> None:
        self.state = VehicleState(0.0, 0.0, 0.0, msg.twist.linear.x, msg.twist.angular.z)
        self.has_state = True

    def on_range(self, msg: Range) -> None:
        self.distance = msg.range

    def on_pose(self, msg: Pose2D) -> None:
        self.pose = (msg.x, msg.y, msg.theta)

    def observe(self, task: str) -> Observation:
        return Observation(self.image_t, [self.image], self.state.speed, self.state.steer, task, pose=self.pose)

    def range_m(self) -> float:
        return self.distance

    def apply(self, action, dt: float) -> None:
        self.cmd_pub.publish(topics.action_to_msg(action))
