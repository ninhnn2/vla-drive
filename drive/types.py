"""The data that moves between the simulator, the policies and the control loop.

Units are SI. An action is `[steer, throttle]`, both in [-1, 1]: steer +1 is full left,
throttle +1 is full power and -1 is full brake (there is no reverse).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STEER, THROTTLE = 0, 1
ACTION_DIM = 2
FULL_BRAKE = -1.0


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float      # rad, counter-clockwise from +x
    speed: float    # m/s, never negative
    steer: float    # the wheels' actual position in [-1, 1]; lags the command


@dataclass
class Observation:
    """What a policy sees at one instant."""

    t: float                       # when the sensors were read
    images: list[np.ndarray]       # uint8 HxWx3 RGB, in camera order
    speed: float
    steer: float
    task: str
    #: (x, y, yaw) in the world. Only a simulator knows it; policies that read it are
    #: teachers for collecting data, not something a real car can run.
    pose: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ActionChunk:
    """A policy's plan: `actions[i]` is meant for time `t_obs + i * dt`."""

    actions: np.ndarray            # float (N, 2)
    dt: float
    t_obs: float

    def __post_init__(self):
        if self.actions.ndim != 2 or self.actions.shape[1] != ACTION_DIM or not len(self.actions):
            raise ValueError(f"a chunk is (N, {ACTION_DIM}) with N >= 1, got {self.actions.shape}")
        if self.dt <= 0:
            raise ValueError("chunk dt must be positive")

    @property
    def horizon_s(self) -> float:
        return (len(self.actions) - 1) * self.dt

    def sample(self, elapsed_s: float) -> np.ndarray | None:
        """The action `elapsed_s` after the chunk's first one, or None past its end."""
        position = max(elapsed_s, 0.0) / self.dt
        last = len(self.actions) - 1
        if position > last + 1e-9:
            return None
        i = min(int(position), max(last - 1, 0))
        w = min(position - i, 1.0)
        return (1.0 - w) * self.actions[i] + w * self.actions[min(i + 1, last)]
