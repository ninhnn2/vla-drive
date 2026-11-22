"""The front camera on the CARLA car: where it is mounted, and how a ground point lands in its picture.

The mount is given in drive/'s vehicle frame: metres forward of the REAR AXLE (the point
the car model and the teacher work with), metres above the ground, and a pitch down.
The same numbers spawn the sensor in CARLA (platform.py) and draw a label's path on its
frames (learn/browse.py), so the drawing matches the camera that took the picture.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True)
class CarlaCameraParams:
    """320x240 like the simulator's camera, so the same SmolVLA settings fit both.

    Mounted over the front of the bonnet. At the windscreen (2.4 m, 1.4 m, 8 deg down) the
    bonnet filled the bottom quarter of every frame, a quarter of the pixels showing
    nothing about the road; here none of it is in view.

    Exposed one stop down. CARLA's auto exposure washes the road out at noon: measured on
    the road below the horizon, brightness 211 and Laplacian variance (sharpness, the
    lane marks) 30 at the default, 171 and 52 one stop down; at sunset 160/53 against
    118/50. Without the bloom nothing changed; manual exposure was worse (242/6)."""
    width: int = 320
    height: int = 240
    hfov_deg: float = 90.0
    forward_m: float = 2.9        # ahead of the rear axle (1.5 m ahead of a Model 3's centre)
    height_m: float = 1.5         # above the ground
    pitch_down_deg: float = 6.0
    exposure_compensation: float = -1.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "CarlaCameraParams":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    @property
    def focal_px(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    def project(self, forward, left) -> np.ndarray:
        """(N, 2) pixel (u, v) of ground points `forward`, `left` metres from the rear axle;
        NaN for points behind the camera."""
        forward = np.asarray(forward, dtype=np.float64) - self.forward_m
        left = np.asarray(left, dtype=np.float64)
        down = np.full_like(forward, self.height_m)
        pitch = math.radians(self.pitch_down_deg)
        # Camera axes: depth along the optical axis (tilted down by `pitch`), right, down.
        depth = forward * math.cos(pitch) + down * math.sin(pitch)
        below = -forward * math.sin(pitch) + down * math.cos(pitch)
        right = -left
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.width / 2.0 + self.focal_px * right / depth
            v = self.height / 2.0 + self.focal_px * below / depth
        pixels = np.stack([u, v], axis=-1)
        pixels[depth <= 0.1] = np.nan
        return pixels
