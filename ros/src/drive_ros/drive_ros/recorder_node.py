"""Record a drive from the topics into an episode; Ctrl-C saves it.

    ros2 run drive_ros recorder --ros-args -p name:=ros

Writes <project>/episodes/<name>-NNN/, the same files `./drive.sh collect` writes but
without labels: `./drive.sh label` makes those from the commands that followed each frame.
"""
from __future__ import annotations

from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from drive.config import LoopConfig
from drive.learn.project import Project
from drive.recorder import Recorder

from . import topics


class RecorderNode(Node):
    def __init__(self):
        super().__init__("drive_recorder")
        episodes = Project(topics.param(self, "project", str(Project().home))).episodes
        name = topics.param(self, "name", "ros")
        hz = topics.param(self, "hz", 10.0)
        task = topics.param(self, "task", LoopConfig().task)
        index = len(list(episodes.glob(f"{name}-*")))
        self.recorder = Recorder(episodes / f"{name}-{index:03d}", task, hz)
        self.start = None                # the first recorded frame is t = 0
        self.state = self.cmd = None
        self.create_subscription(Image, topics.IMAGE, self.on_image, 1)
        self.create_subscription(TwistStamped, topics.STATE, self.on_state, 1)
        self.create_subscription(Twist, topics.CMD, self.on_cmd, 1)
        self.get_logger().info(f"recording to {self.recorder.directory} from the first command on; Ctrl-C saves")

    def on_state(self, msg: TwistStamped) -> None:
        self.state = (msg.twist.linear.x, msg.twist.angular.z)

    def on_cmd(self, msg: Twist) -> None:
        self.cmd = topics.msg_to_action(msg)

    def on_image(self, msg: Image) -> None:
        if self.state is None or self.cmd is None:       # nobody is driving yet
            return
        stamp = topics.seconds(msg.header.stamp)
        self.start = stamp if self.start is None else self.start
        if self.recorder.due(stamp - self.start):
            self.recorder.add(stamp - self.start, topics.msg_to_image(msg).copy(), *self.state, self.cmd)

    def close(self) -> None:
        frames = len(self.recorder.t)
        if frames < 2:
            print("nothing recorded: no commands were heard")
            return
        print(f"{frames} frames ({self.recorder.t[-1]:.0f} s) saved to {self.recorder.save()}")


def main() -> None:
    topics.spin(RecorderNode)
