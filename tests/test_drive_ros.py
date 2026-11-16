"""ros/src/drive_ros: the nodes, run together in one process. Needs ROS 2: `make test`."""
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

try:
    import rclpy
except ImportError:
    rclpy = None

if rclpy is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros/src/drive_ros"))
    from builtin_interfaces.msg import Time
    from rclpy.executors import SingleThreadedExecutor

    from drive.learn.label import label, read_labels
    from drive.learn.validate import validate
    from drive_ros import topics
    from drive_ros.drive_node import DriveNode
    from drive_ros.recorder_node import RecorderNode
    from drive_ros.sim_node import SimNode
    from drive_ros.teleop_node import MAX_THROTTLE, stick


@unittest.skipIf(rclpy is None, "ROS 2 is not installed (use the container: make test)")
class MessageTests(unittest.TestCase):
    def test_image_round_trip(self):
        image = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        msg = topics.image_to_msg(image, Time(sec=3, nanosec=500_000_000))
        self.assertEqual((msg.height, msg.width, msg.encoding, msg.step), (24, 32, "rgb8", 96))
        np.testing.assert_array_equal(topics.msg_to_image(msg), image)
        self.assertEqual(topics.seconds(msg.header.stamp), 3.5)
        msg.encoding = "bgr8"
        with self.assertRaisesRegex(ValueError, "rgb8"):
            topics.msg_to_image(msg)

    def test_action_round_trip(self):
        msg = topics.action_to_msg([0.25, -0.5])
        self.assertEqual((msg.angular.z, msg.linear.x), (0.25, -0.5))
        np.testing.assert_array_equal(topics.msg_to_action(msg), [0.25, -0.5])

    def test_teleop_stick(self):
        np.testing.assert_allclose(stick(480, 360, 960, 720), [0.0, 0.0])
        np.testing.assert_allclose(stick(0, 0, 960, 720), [1.0, MAX_THROTTLE])       # top left: full left, go
        np.testing.assert_allclose(stick(960, 720, 960, 720), [-1.0, -1.0])          # bottom right: right, brake
        np.testing.assert_allclose(stick(720, 180, 960, 720), [-0.5, 0.5 * MAX_THROTTLE])


@unittest.skipIf(rclpy is None, "ROS 2 is not installed (use the container: make test)")
class NodeTests(unittest.TestCase):
    def run_nodes(self, project: Path, seconds: float, *parameters: str):
        """Spin the simulator, the driver and the recorder together; returns them."""
        os.environ["ROS_DOMAIN_ID"] = "87"               # away from any nodes already running
        arguments = [f"project:={project}", *parameters]
        rclpy.init(args=["--ros-args", *[word for p in arguments for word in ("-p", p)]])
        nodes = []
        try:
            nodes += [SimNode(), DriveNode(), RecorderNode()]
            executor = SingleThreadedExecutor()
            for node in nodes:
                executor.add_node(node)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            with redirect_stdout(io.StringIO()):
                for node in nodes:
                    node.close()
        finally:
            for node in nodes:
                node.destroy_node()
            rclpy.shutdown()
        return nodes

    def test_teacher_drives_the_simulated_car_and_the_drive_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            sim, driver, recorder = self.run_nodes(Path(d), 7.0, "track:=oval")
            world = sim.world
            s, lateral, _ = world.track.project(world.state.x, world.state.y)
            self.assertGreater(s, 4.0)                                   # it went somewhere
            self.assertGreater(world.state.speed, 0.8)
            self.assertLess(abs(lateral), 0.10)                          # and stayed on the centerline
            self.assertGreater(len(driver.loop.worker.latencies_s), 50)

            episode = Path(d) / "episodes/ros-000"
            data = np.load(episode / "episode.npz")
            self.assertGreater(len(data["t"]), 40)
            self.assertEqual(data["images"].shape[1:], (240, 320, 3))
            self.assertEqual(data["t"][0], 0.0)
            self.assertGreater(data["states"][-1, 0], 0.8)               # the recorded speed is the car's
            self.assertTrue((np.abs(data["actions"]) <= 1.0).all() and data["actions"][:, 1].max() > 0.1)
            self.assertIsNone(read_labels(episode))                      # a ROS recording comes unlabelled
            label(Path(d) / "episodes")
            report = validate(Path(d) / "episodes", log=lambda line: None)
            self.assertEqual([f for f in report["failures"] if "one episode only" not in f], [])

    def test_the_car_brakes_when_commands_stop(self):
        with tempfile.TemporaryDirectory() as d:
            # A plan can never be fresh enough: the supervisor brakes, and so nothing moves.
            sim, driver, recorder = self.run_nodes(Path(d), 2.0, "max_plan_age_ms:=1", "obstacles:=8")
            self.assertEqual(sim.world.state.speed, 0.0)
            self.assertEqual(len(sim.world.obstacles), 1)


if __name__ == "__main__":
    unittest.main()
