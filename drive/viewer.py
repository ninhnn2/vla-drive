"""A heads-up display of a simulated drive: the camera, the map, and what safety is doing.

`Output` is the window and video writer the other viewers use too.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .loop import DriveLoop, Tick
from .sim import World, step_vehicle
from .types import STEER, THROTTLE

WIDTH, HEIGHT, PANEL = 960, 440, 480
WHITE, RED, GREEN, CYAN, DARK = (240, 240, 240), (240, 70, 60), (90, 220, 110), (60, 220, 240), (24, 24, 28)
GREY, MAGENTA = (150, 150, 150), (240, 80, 210)


class Hud:
    def __init__(self, world: World):
        ground = world.ground
        self.scale = min(PANEL / ground.image.shape[1], 300 / ground.image.shape[0])
        self.map = cv2.resize(ground.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def _px(self, world: World, xy) -> np.ndarray:
        return (world.ground.to_px(np.asarray(xy)) * self.scale).round().astype(np.int32)

    def map_view(self, world: World, colour=WHITE, path=(), trail=()) -> np.ndarray:
        """The map from above with the car on it, `path` (world x, y points) in cyan and
        `trail`, where the car has been, in magenta."""
        top = self.map.copy()
        if len(trail) > 1:
            cv2.polylines(top, [self._px(world, trail)], False, MAGENTA, 2, cv2.LINE_AA)
        if len(path) > 1:
            cv2.polylines(top, [self._px(world, path)], False, CYAN, 2, cv2.LINE_AA)
        s, v = world.state, world.vehicle
        cos, sin = np.cos(s.yaw), np.sin(s.yaw)
        body = [(s.x + cos * f - sin * l, s.y + sin * f + cos * l)
                for f, l in ((v.wheelbase_m + 0.12, 0), (-0.07, v.width_m), (-0.07, -v.width_m))]
        cv2.fillPoly(top, [self._px(world, body)], colour, cv2.LINE_AA)
        return top

    def frame(self, world: World, loop: DriveLoop, tick: Tick) -> np.ndarray:
        canvas = np.full((HEIGHT, WIDTH, 3), DARK, np.uint8)
        canvas[:360, :PANEL] = cv2.resize(world.image(), (PANEL, 360), interpolation=cv2.INTER_LINEAR)

        # The plan: where the remaining actions of the current chunk would take the car.
        state, path = world.state, []
        for k in range(30):
            action = loop.executor.action(world.t + k * 0.05)
            if action is None:
                break
            state = step_vehicle(state, action, world.vehicle, 0.05)
            path.append((state.x, state.y))
        top = self.map_view(world, RED if tick.decision.reasons else WHITE, path)
        canvas[:top.shape[0], PANEL:PANEL + top.shape[1]] = top

        def text(line: str, x: int, y: int, colour=WHITE):
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        s, applied = world.state, tick.decision.action
        age = loop.executor.age_s(tick.t)
        text(f"t {tick.t:5.1f} s   speed {s.speed:4.2f} m/s   range {tick.range_m:4.2f} m", PANEL + 10, 330)
        text(f"steer {applied[STEER]:+.2f}   throttle {applied[THROTTLE]:+.2f}   plan age "
             + (f"{1000 * age:4.0f} ms" if np.isfinite(age) else "   -"), PANEL + 10, 356)
        if tick.decision.reasons:
            text("SAFETY: " + ", ".join(tick.decision.reasons), 10, 390, RED)
        else:
            text("policy in control", 10, 390, GREEN)
        text("cyan: the plan in the current chunk    q: quit    e: emergency stop", 10, 420, GREY)
        return canvas


class Output:
    """Where HUD frames go: a window, a video file, or both."""

    def __init__(self, show: bool, video: Path | None, fps: float, size=(WIDTH, HEIGHT)):
        self.show = show
        self.writer = None
        if video is not None:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
            if not self.writer.isOpened():
                raise OSError(f"cannot write video to {video}")

    def put(self, frame: np.ndarray) -> str:
        """Show one RGB frame; returns the key pressed in the window, or ''."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.writer is not None:
            self.writer.write(bgr)
        if not self.show:
            return ""
        cv2.imshow("drive", bgr)
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else ""

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
