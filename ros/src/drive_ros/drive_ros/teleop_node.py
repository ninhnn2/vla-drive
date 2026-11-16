"""Drive by hand: a window with the camera; hold the left mouse button to drive.

    ros2 run drive_ros teleop

The pointer is the stick: left and right of the middle steer, above the middle is
throttle, below is brake. Releasing the button coasts with the wheels straight.
Space brakes, q quits. With the recorder running, this is how a person's drives
become training data.
"""
from __future__ import annotations

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

from drive.types import FULL_BRAKE

from . import topics

WINDOW, SCALE = "drive: teleop", 3
MAX_THROTTLE = 0.6          # the top of the window; full power is too lively by hand


def stick(x: int, y: int, width: int, height: int) -> np.ndarray:
    """The pointer's place in the window as `[steer, throttle]`."""
    steer = -(x - width / 2) / (width / 2)
    throttle = (height / 2 - y) / (height / 2)
    return np.array([min(max(steer, -1.0), 1.0),
                     min(max(throttle, -1.0), 1.0) * (MAX_THROTTLE if throttle > 0 else 1.0)])


class TeleopNode(Node):
    def __init__(self):
        super().__init__("drive_teleop")
        self.image = np.zeros((240, 320, 3), np.uint8)
        self.action, self.held = np.zeros(2), False
        self.cmd_pub = self.create_publisher(Twist, topics.CMD, 1)
        self.create_subscription(Image, topics.IMAGE, self.on_image, 1)
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.create_timer(0.05, self.step)

    def on_image(self, msg: Image) -> None:
        self.image = topics.msg_to_image(msg)

    def on_mouse(self, event, x, y, flags, _) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.held = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.held = False
        height, width = (SCALE * n for n in self.image.shape[:2])
        self.action = stick(x, y, width, height) if self.held else np.zeros(2)

    def step(self) -> None:
        frame = cv2.resize(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR), None, fx=SCALE, fy=SCALE,
                           interpolation=cv2.INTER_NEAREST)
        height, width = frame.shape[:2]
        cv2.line(frame, (width // 2, 0), (width // 2, height), (90, 90, 90), 1)
        cv2.line(frame, (0, height // 2), (width, height // 2), (90, 90, 90), 1)
        cv2.putText(frame, f"steer {self.action[0]:+.2f}  throttle {self.action[1]:+.2f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, "hold the left button to drive   space: brake   q: quit", (12, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if key == ord(" "):
            self.action = np.array([self.action[0], FULL_BRAKE])
        self.cmd_pub.publish(topics.action_to_msg(self.action))

    def close(self) -> None:
        cv2.destroyAllWindows()


def main() -> None:
    topics.spin(TeleopNode)
