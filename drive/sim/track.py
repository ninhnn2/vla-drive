"""A closed track as an evenly sampled centerline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius_m: float = 0.12


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


#: Closed curves as (x, y) of an angle; all are driven counter-clockwise unless reversed.
SHAPES = {
    # About 22 m around, left and right bends, tightest radius about 1 m.
    "peanut": lambda t: (4.6 * np.cos(t), 2.2 * np.sin(t) + 1.1 * np.sin(t) ** 3 * np.cos(2 * t)),
    "oval": lambda t: (4.0 * np.cos(t), 2.4 * np.sin(t)),
    "trefoil": lambda t: ((3.6 + 0.6 * np.cos(3 * t)) * np.cos(t), (3.6 + 0.6 * np.cos(3 * t)) * np.sin(t)),
}


class Track:
    def __init__(self, centerline: np.ndarray, width_m: float = 0.9, spacing_m: float = 0.02):
        points = np.asarray(centerline, dtype=np.float64)
        closed = np.vstack([points, points[:1]])
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
        self.length = float(along[-1])
        n = int(round(self.length / spacing_m))
        self.ds = self.length / n
        s = np.arange(n) * self.ds
        self.xy = np.stack([np.interp(s, along, closed[:, 0]), np.interp(s, along, closed[:, 1])], axis=1)
        step = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)
        self.heading = np.arctan2(step[:, 1], step[:, 0])
        turn = wrap_angle(np.roll(self.heading, -1) - np.roll(self.heading, 1))
        self.curvature = turn / (2 * self.ds)      # 1/m, positive turns left
        self.width_m = width_m

    @classmethod
    def named(cls, name: str = "peanut", reverse: bool = False) -> "Track":
        """One of `SHAPES`; `reverse` drives it clockwise, which swaps left and right bends."""
        if name not in SHAPES:
            raise ValueError(f"unknown track {name!r}; choose from {', '.join(SHAPES)}")
        t = np.linspace(0.0, 2 * np.pi, 4000, endpoint=False)
        points = np.stack(SHAPES[name](t), axis=1)
        return cls(points[::-1] if reverse else points)

    @classmethod
    def default(cls) -> "Track":
        return cls.named("peanut")

    def _index(self, s) -> np.ndarray:
        return np.floor(np.asarray(s) / self.ds).astype(int) % len(self.xy)

    def at(self, s) -> tuple:
        """(x, y, heading) at arc length `s`; `s` wraps and may be an array."""
        i = self._index(s)
        w = (np.asarray(s) / self.ds) % 1.0
        j = (i + 1) % len(self.xy)
        xy = self.xy[i] * (1 - w)[..., None] + self.xy[j] * w[..., None]
        return xy[..., 0], xy[..., 1], self.heading[i]

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """(s, lateral offset, track heading) of the nearest centerline point.

        The offset is positive to the left of the direction of travel.
        """
        delta = np.array([x, y]) - self.xy
        i = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        heading = float(self.heading[i])
        tx, ty = np.cos(heading), np.sin(heading)
        dx, dy = delta[i]
        return (i * self.ds + tx * dx + ty * dy) % self.length, float(tx * dy - ty * dx), heading

    def sharpest(self, s: float, ahead_m: float) -> float:
        """Largest |curvature| between `s` and `s + ahead_m`."""
        i = self._index(s + np.arange(0.0, ahead_m, self.ds))
        return float(np.abs(self.curvature[i]).max())

    def obstacle(self, s: float, lateral_m: float = 0.0, radius_m: float = 0.12) -> Obstacle:
        x, y, heading = self.at(s)
        return Obstacle(float(x - np.sin(heading) * lateral_m), float(y + np.cos(heading) * lateral_m), radius_m)
