"""The simulated car as a ROS 2 node: publishes its sensors, obeys /drive/cmd.

    ros2 run drive_ros sim --ros-args -p track:=trefoil -p show:=true
"""
from __future__ import annotations

import cv2
import numpy as np
from geometry_msgs.msg import Pose2D, Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, Range

from drive.config import LoopConfig
from drive.sim import Track, World
from drive.types import FULL_BRAKE
from drive.viewer import Hud

from . import topics

CMD_TIMEOUT_S = 0.5        # brake when the commands stop coming: a car must not coast on a dead link
IMAGE_EVERY = 2            # ticks between camera frames: 25 Hz at the 50 Hz step


class SimNode(Node):
    def __init__(self):
        super().__init__("drive_sim")
        track = Track.named(topics.param(self, "track", "peanut"),
                            topics.param(self, "reverse", False))
        # Cones on the track, as metres along it: "8" or "8,15".
        cones = [float(s) for s in topics.param(self, "obstacles", "").split(",") if s]
        self.world = World(track, tuple(track.obstacle(s) for s in cones))
        self.hud = Hud(self.world) if topics.param(self, "show", False) else None
        self.dt = LoopConfig().dt
        self.cmd, self.cmd_time, self.ticks = np.array([0.0, FULL_BRAKE]), -1.0, 0

        self.image_pub = self.create_publisher(Image, topics.IMAGE, 1)
        self.state_pub = self.create_publisher(TwistStamped, topics.STATE, 1)
        self.range_pub = self.create_publisher(Range, topics.RANGE, 1)
        self.pose_pub = self.create_publisher(Pose2D, topics.POSE, 1)
        self.create_subscription(Twist, topics.CMD, self.on_cmd, 1)
        self.create_timer(self.dt, self.step)
        self.get_logger().info(f"simulated car on a {track.length:.0f} m track; waiting for {topics.CMD}")

    def on_cmd(self, msg: Twist) -> None:
        self.cmd, self.cmd_time = np.clip(topics.msg_to_action(msg), -1.0, 1.0), self.world.t

    def step(self) -> None:
        world = self.world
        fresh = world.t - self.cmd_time <= CMD_TIMEOUT_S
        world.apply(self.cmd if fresh else [self.cmd[0], FULL_BRAKE], self.dt)
        stamp = self.get_clock().now().to_msg()

        state = TwistStamped()
        state.header.stamp = stamp
        state.twist.linear.x, state.twist.angular.z = world.state.speed, world.state.steer
        self.state_pub.publish(state)
        distance = Range()
        distance.header.stamp, distance.radiation_type = stamp, Range.INFRARED
        distance.max_range, distance.range = world.range_sensor.max_range_m, world.range_m()
        self.range_pub.publish(distance)
        self.pose_pub.publish(Pose2D(x=world.state.x, y=world.state.y, theta=world.state.yaw))

        self.ticks += 1
        if self.ticks % IMAGE_EVERY == 0:
            image = world.image()
            self.image_pub.publish(topics.image_to_msg(image, stamp))
            if self.hud is not None and self.ticks % (2 * IMAGE_EVERY) == 0:
                top = self.hud.map_view(world)
                view = np.zeros((max(image.shape[0], top.shape[0]), image.shape[1] + top.shape[1], 3), np.uint8)
                view[:image.shape[0], :image.shape[1]] = image
                view[:top.shape[0], image.shape[1]:] = top
                cv2.imshow("drive: simulated car", cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        if self.hud is not None:
            cv2.destroyAllWindows()


def main() -> None:
    topics.spin(SimNode)
