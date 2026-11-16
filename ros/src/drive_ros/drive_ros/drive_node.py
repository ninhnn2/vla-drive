"""The control loop as a ROS 2 node: a policy and the safety supervisor, publishing /drive/cmd.

    ros2 run drive_ros driver --ros-args -p policy:=cnn -p checkpoint:=v1

It is drive.loop.DriveLoop, unchanged, on a platform made of topics: the policy infers
on its own thread, the loop runs at 50 Hz, the supervisor has the last word.
"""
from __future__ import annotations

from collections import Counter

from rclpy.node import Node

from drive.config import CameraParams, LoopConfig, SafetyLimits, VehicleParams
from drive.control import ThreadedWorker
from drive.learn.project import Project
from drive.loop import DriveLoop
from drive.policy import make_policy
from drive.safety import SafetySupervisor
from drive.sim import Track
from drive.sim.render import Camera

from . import topics
from .platform import RosPlatform


class DriveNode(Node):
    def __init__(self):
        super().__init__("drive_driver")
        name = topics.param(self, "policy", "pure-pursuit")
        checkpoint = topics.param(self, "checkpoint", "")      # a version (v1) or a file, for cnn
        project = Project(topics.param(self, "project", str(Project().home)))
        # Only the pure-pursuit teacher uses these: it follows the map, so they must match the simulator's.
        track = Track.named(topics.param(self, "track", "peanut"),
                            topics.param(self, "reverse", False))
        speed = topics.param(self, "speed", 0.0)               # m/s; 0: the policy's own
        extra_latency = topics.param(self, "latency_ms", 0.0)  # pretend the model is this much slower
        max_plan_age = topics.param(self, "max_plan_age_ms", 600.0)

        policy = make_policy(name, track=track, vehicle=VehicleParams(), camera=Camera(CameraParams()),
                             checkpoint=str(project.checkpoint(checkpoint)) if checkpoint else None,
                             target_speed_mps=speed or None)
        self.platform = RosPlatform(self)
        self.loop = DriveLoop(self.platform, ThreadedWorker(policy, extra_latency / 1000),
                              SafetySupervisor(SafetyLimits(max_plan_age_s=max_plan_age / 1000)), LoopConfig())
        self.reasons: Counter = Counter()
        self.create_timer(self.loop.config.dt, self.step)
        self.create_timer(2.0, self.report)
        self.get_logger().info(f"policy {name}; waiting for {topics.IMAGE} and {topics.STATE}")

    def step(self) -> None:
        if self.platform.ready:
            self.reasons.update(self.loop.step().decision.reasons)

    def report(self) -> None:
        if not self.platform.ready:
            return
        overrides = ", ".join(f"{reason} {count * self.loop.config.dt:.1f} s" for reason, count in self.reasons.items())
        self.get_logger().info(f"speed {self.platform.state.speed:.2f} m/s, "
                               f"{len(self.loop.worker.latencies_s)} inferences"
                               + (f"; safety overrode: {overrides}" if overrides else ""))
        self.reasons.clear()

    def close(self) -> None:
        # No goodbye command: the car brakes by itself when the commands stop (its watchdog).
        self.loop.worker.close()


def main() -> None:
    topics.spin(DriveNode)
