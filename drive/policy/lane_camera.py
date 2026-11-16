"""A camera-only lane follower: find the road in the image, steer at its middle.

Classical vision, no map and no pose, so this one could run on a real car with a
calibrated camera. It cannot predict, so its chunk repeats one action: watch how much
sooner it fails than a real plan as the inference latency grows.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams
from ..sim.render import Camera
from ..types import ActionChunk, Observation
from .base import Policy
from .pure_pursuit import pursuit_steer


def road_mask(image: np.ndarray) -> np.ndarray:
    """Asphalt and paint, as opposed to grass: wherever green does not dominate."""
    rgb = image.astype(np.int16)
    return rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]) < 20


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each stretch of True in a 1-D boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], row, [False]]).astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


class LaneCameraPolicy(Policy):
    name = "lane-camera"

    def __init__(self, camera: Camera, vehicle: VehicleParams = VehicleParams(), *, target_speed_mps: float = 1.0,
                 chunk_size: int = 15, chunk_dt_s: float = 1 / 30, lookaheads_m=(1.1, 0.9, 0.7, 0.5)):
        self.camera, self.vehicle = camera, vehicle
        self.target_speed = target_speed_mps
        self.chunk_size, self.chunk_dt = chunk_size, chunk_dt_s
        centre = camera.params.width // 2
        ahead = np.where(camera.ground[:, centre], camera.forward_m[:, centre], np.inf)
        self.rows = [int(np.argmin(np.abs(ahead - d))) for d in lookaheads_m]
        self.column = centre
        self.steer = 0.0

    def target(self, image: np.ndarray) -> tuple[float, float] | None:
        """The road's middle at the farthest lookahead that shows road, in the car's frame."""
        mask = road_mask(image) & self.camera.ground
        for row in self.rows:
            found = [r for r in runs(mask[row]) if r[1] - r[0] >= 12]
            if found:
                # Of several stretches (another part of the track in view), stay on ours.
                start, end = min(found, key=lambda r: abs((r[0] + r[1]) // 2 - self.column))
                self.column = (start + end) // 2
                return float(self.camera.forward_m[row, self.column]), float(self.camera.left_m[row, self.column])
        return None

    def infer(self, obs: Observation) -> ActionChunk:
        point = self.target(obs.images[0])
        if point is None:
            speed = 0.3                                   # lost the road: keep turning, crawl
        else:
            self.steer = pursuit_steer(*point, self.vehicle)
            speed = self.target_speed * (1.0 - 0.5 * abs(self.steer))
        throttle = min(max(2.0 * (speed - obs.speed) + 0.1 * speed, -1.0), 1.0)
        return ActionChunk(np.tile([self.steer, throttle], (self.chunk_size, 1)), self.chunk_dt, obs.t)
