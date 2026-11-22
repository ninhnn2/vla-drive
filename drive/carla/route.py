"""A route through a town: an open centerline, the junctions on it, and what to say there.

A CARLA route runs from A to B, so unlike `drive.sim.Track` it does not wrap. It may
cross itself (the same junction twice), so projecting the car onto it searches near
where the car last was, not the whole route.

What the driver is told depends on where it is. From `announce_m` before a junction until
it has left the junction, the command is that junction's maneuver; everywhere else it
is `follow`. So `follow` never covers a junction, and the same picture of an
intersection comes with a different sentence, and a different label, depending on
which way the route goes: the case a VLA's language input exists for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.track import wrap_angle

MANEUVERS = ("left", "right", "straight")
COMMANDS = ("follow",) + MANEUVERS

#: Several ways to say each command, so the model learns the meaning, not one string.
PHRASES = {
    "follow": ("follow the lane", "keep driving in your lane", "follow the road"),
    "left": ("turn left at the next intersection", "take the next left", "at the next intersection, turn left"),
    "right": ("turn right at the next intersection", "take the next right", "at the next intersection, turn right"),
    "straight": ("go straight at the next intersection", "continue straight through the next intersection",
                 "drive straight across the next intersection"),
}

#: A heading change through a junction beyond this is a turn; below it, going straight.
TURN_RAD = np.radians(35.0)


def maneuver(heading_in: float, heading_out: float) -> str:
    """left, right or straight, from the headings (drive/'s frame, counter-clockwise) into and out of a junction."""
    turn = float(wrap_angle(heading_out - heading_in))
    return "left" if turn > TURN_RAD else "right" if turn < -TURN_RAD else "straight"


@dataclass(frozen=True)
class Junction:
    s_entry: float          # metres along the route where it enters the junction
    s_exit: float           # ... and leaves it
    maneuver: str           # left, right or straight


class Route:
    def __init__(self, points, headings=None, width_m: float = 3.5, junctions=(), spacing_m: float = 0.25,
                 smooth_m: float = 2.0):
        """`points` (N, 2) along the lane centre, in drive/'s frame. `headings` (N,) are the
        lane's own directions there (CARLA's waypoints know them exactly); without them they
        are computed from the points."""
        points = np.asarray(points, dtype=np.float64)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6])
        points = points[keep]
        if len(points) < 2:
            raise ValueError("a route needs at least two distinct points")
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        self.length = float(along[-1])
        n = max(int(round(self.length / spacing_m)), 1) + 1
        self.ds = self.length / (n - 1)
        s = np.arange(n) * self.ds
        self.xy = np.stack([np.interp(s, along, points[:, 0]), np.interp(s, along, points[:, 1])], axis=1)
        if headings is not None:
            unwrapped = np.unwrap(np.asarray(headings, dtype=np.float64)[keep])
            self.heading = wrap_angle(np.interp(s, along, unwrapped))
        else:
            step = np.gradient(self.xy, axis=0)
            self.heading = np.arctan2(step[:, 1], step[:, 0])
        # Curvature from the heading's change, smoothed over `smooth_m`: piecewise-linear
        # points would otherwise put all of a bend's curvature on their corners.
        raw = np.gradient(np.unwrap(self.heading), self.ds)
        width = max(int(round(smooth_m / self.ds)), 1)
        self.curvature = np.convolve(np.pad(raw, width // 2, mode="edge"), np.ones(width) / width, mode="valid")[:n]
        self.width_m = float(width_m)
        self.junctions = tuple(sorted(junctions, key=lambda j: j.s_entry))

    def _index(self, s) -> np.ndarray:
        return np.clip(np.floor(np.asarray(s) / self.ds).astype(int), 0, len(self.xy) - 1)

    def at(self, s) -> tuple:
        """(x, y, heading) at arc length `s`, held at the ends; `s` may be an array."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        i = np.minimum(np.floor(s / self.ds).astype(int), len(self.xy) - 2)
        w = (s / self.ds - i)[..., None]
        xy = self.xy[i] * (1 - w) + self.xy[i + 1] * w
        return xy[..., 0], xy[..., 1], self.heading[i]

    def project(self, x: float, y: float, near_s: float | None = None, window_m: float = 30.0) -> tuple:
        """(s, lateral offset, heading) of the nearest centerline point, positive to the left.

        With `near_s`, only `window_m` either side of it is searched: a route that crosses
        itself has two candidates at the crossing, and the right one is the one nearby."""
        lo, hi = 0, len(self.xy)
        if near_s is not None:
            lo = int(max((near_s - window_m) / self.ds, 0))
            hi = int(min((near_s + window_m) / self.ds + 1, len(self.xy)))
        delta = np.array([x, y]) - self.xy[lo:hi]
        i = lo + int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        heading = float(self.heading[i])
        tx, ty = np.cos(heading), np.sin(heading)
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        s = min(max(i * self.ds + tx * dx + ty * dy, 0.0), self.length)
        return s, float(tx * dy - ty * dx), heading

    def sharpest(self, s: float, ahead_m: float) -> float:
        """Largest |curvature| between `s` and `s + ahead_m` (1/m)."""
        i = self._index(np.arange(s, s + max(ahead_m, self.ds), self.ds))
        return float(np.abs(self.curvature[i]).max())

    def junction_at(self, s: float, announce_m: float) -> int | None:
        """Index of the junction whose command applies at `s`, or None."""
        for index, junction in enumerate(self.junctions):
            if junction.s_entry - announce_m <= s <= junction.s_exit:
                return index
            if junction.s_entry - announce_m > s:
                break
        return None

    def command(self, s: float, announce_m: float = 30.0) -> str:
        """follow, or the maneuver of the junction ahead (or under the car)."""
        index = self.junction_at(s, announce_m)
        return "follow" if index is None else self.junctions[index].maneuver
