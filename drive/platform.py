"""What the control loop needs from a car, simulated or real.

`drive.sim.World` is the built-in one. A CARLA or ROS 2 adapter is another: read the
camera and speed in `observe`, the forward range in `range_m`, and send
`[steer, throttle]` to the vehicle in `apply`. Nothing else in `drive/` changes.
"""
from __future__ import annotations

from typing import Protocol

from .types import Observation, VehicleState


class Platform(Protocol):
    t: float                    # seconds, on the platform's own clock
    state: VehicleState         # the loop reads only `.speed`

    def observe(self, task: str) -> Observation:
        """Read the sensors now."""

    def range_m(self) -> float:
        """Distance to the nearest thing ahead, for the safety supervisor."""

    def apply(self, action, dt: float) -> None:
        """Send `[steer, throttle]` to the motors and let `dt` seconds pass."""
