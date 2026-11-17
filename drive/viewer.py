"""A heads-up display of a simulated drive: the camera, the map, and what safety is doing.

Below them, a debug panel shows what the policy is outputting: the chunk in use, with
the part skipped while it was computed and the action being played now, and the last
seconds of what the model asked for against what went to the motors.

`Output` is the window and video writer the other viewers use too.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .loop import DriveLoop, Tick
from .sim import World, step_vehicle
from .types import STEER, THROTTLE

WIDTH, HEIGHT, PANEL = 960, 440, 480
DEBUG_HEIGHT = 220                     # the panel under the HUD
HUD_HEIGHT = HEIGHT + DEBUG_HEIGHT
WHITE, RED, GREEN, CYAN, DARK = (240, 240, 240), (240, 70, 60), (90, 220, 110), (60, 220, 240), (24, 24, 28)
GREY, MAGENTA, ORANGE, SHADE = (150, 150, 150), (240, 80, 210), (245, 160, 50), (52, 52, 60)
YELLOW = (250, 235, 120)
HISTORY_S = 5.0                        # seconds shown in the strip chart


def text(canvas: np.ndarray, line: str, x: int, y: int, colour=WHITE, scale: float = 0.55) -> None:
    cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


class Plot:
    """A box on the canvas mapping (x in [x0, x1], value in [-1, 1]) to pixels."""

    def __init__(self, canvas, left: int, top: int, width: int, height: int, x0: float, x1: float):
        self.canvas, self.left, self.top, self.width, self.height = canvas, left, top, width, height
        self.x0, self.x1 = x0, max(x1, x0 + 1e-6)
        cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 70, 78), 1)
        for value, label in ((1.0, "+1"), (0.0, " 0"), (-1.0, "-1")):
            y = self.y(value)
            if value == 0.0:
                cv2.line(canvas, (left, y), (left + width, y), (70, 70, 78), 1)
            text(canvas, label, left - 26, y + 4, GREY, 0.4)

    def x(self, value: float) -> int:
        return int(round(self.left + (value - self.x0) / (self.x1 - self.x0) * self.width))

    def y(self, value: float) -> int:
        return int(round(self.top + (1.0 - min(max(value, -1.0), 1.0)) / 2.0 * self.height))

    def line(self, xs, values, colour, thickness: int = 2) -> None:
        points = np.array([(self.x(a), self.y(b)) for a, b in zip(xs, values)], np.int32)
        if len(points) > 1:
            cv2.polylines(self.canvas, [points], False, colour, thickness, cv2.LINE_AA)


class Hud:
    def __init__(self, world: World):
        ground = world.ground
        self.scale = min(PANEL / ground.image.shape[1], 300 / ground.image.shape[0])
        self.map = cv2.resize(ground.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        #: (t, command or None, applied, overridden) for the strip chart
        self.history: deque = deque()

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
        """The HUD and, under it, the debug panel: WIDTH x HUD_HEIGHT."""
        canvas = np.full((HUD_HEIGHT, WIDTH, 3), DARK, np.uint8)
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

        s, applied = world.state, tick.decision.action
        age = loop.executor.age_s(tick.t)
        text(canvas, f"t {tick.t:5.1f} s   speed {s.speed:4.2f} m/s   range {tick.range_m:4.2f} m", PANEL + 10, 330)
        text(canvas, f"steer {applied[STEER]:+.2f}   throttle {applied[THROTTLE]:+.2f}   plan age "
             + (f"{1000 * age:4.0f} ms" if np.isfinite(age) else "   -"), PANEL + 10, 356)
        if tick.decision.reasons:
            text(canvas, "SAFETY: " + ", ".join(tick.decision.reasons), 10, 390, RED)
        else:
            text(canvas, "policy in control", 10, 390, GREEN)
        text(canvas, "cyan: the plan in the current chunk    q: quit    e: emergency stop", 10, 420, GREY)
        self.debug_panel(canvas, loop, tick)
        return canvas

    def debug_panel(self, canvas: np.ndarray, loop: DriveLoop, tick: Tick) -> None:
        """What the policy outputs, under the HUD: the chunk in use, and the last seconds of commands."""
        top = HEIGHT
        cv2.line(canvas, (0, top), (WIDTH, top), (70, 70, 78), 1)
        command, applied = tick.command, tick.decision.action
        overridden = bool(tick.decision.reasons)
        self.history.append((tick.t, None if command is None else np.array(command, float), np.array(applied), overridden))
        while self.history and self.history[0][0] < tick.t - HISTORY_S:
            self.history.popleft()

        # The line a person reads first: what the model asked for, and what the motors got.
        if command is None:
            asked = "model: no action (no plan, or the plan ran out)"
        else:
            asked = f"model: steer {command[STEER]:+.2f}  throttle {command[THROTTLE]:+.2f}"
        text(canvas, asked, 10, top + 20, CYAN if command is not None else RED)
        got = f"-> motors: steer {applied[STEER]:+.2f}  throttle {applied[THROTTLE]:+.2f}"
        text(canvas, got + ("   (" + ", ".join(tick.decision.reasons) + ")" if overridden else ""),
             330, top + 20, RED if overridden else WHITE)

        # Left: the chunk in use, against time since its observation.
        executor, worker = loop.executor, loop.worker
        chunk, latencies = executor.chunk, worker.latencies_s
        if chunk is None:
            text(canvas, "waiting for the first chunk", 40, top + 110, GREY)
        else:
            horizon = max(chunk.horizon_s, chunk.dt)
            plot = Plot(canvas, 40, top + 40, 420, 130, 0.0, horizon)
            arrived = executor.arrived - executor.origin            # 0 without latency compensation
            if arrived > 0:                                         # actions meant for the time spent computing
                cv2.rectangle(canvas, (plot.x(0), plot.top + 1), (plot.x(min(arrived, horizon)), plot.top + plot.height - 1),
                              SHADE, -1)
            times = np.arange(len(chunk.actions)) * chunk.dt
            plot.line(times, chunk.actions[:, STEER], CYAN)
            plot.line(times, chunk.actions[:, THROTTLE], ORANGE)
            now = tick.t - executor.origin
            if 0 <= now <= horizon:
                cv2.line(canvas, (plot.x(now), plot.top), (plot.x(now), plot.top + plot.height), WHITE, 1)
                if command is not None:
                    for value, colour in ((command[STEER], CYAN), (command[THROTTLE], ORANGE)):
                        cv2.circle(canvas, (plot.x(now), plot.y(value)), 4, colour, -1, cv2.LINE_AA)
            last = f"{1000 * latencies[-1]:.0f} ms" if latencies else "-"
            text(canvas, f"chunk {len(chunk.actions)} x {chunk.dt:.3g} s from a {1000 * (tick.t - chunk.t_obs):.0f} ms old "
                         f"frame; #{len(latencies)} took {last}", 40, top + 190, GREY, 0.42)
            text(canvas, "cyan steer, orange throttle; grey: skipped; white: now", 40, top + 210, GREY, 0.42)

        # Right: the last seconds of what the model asked for and what the motors got.
        plot = Plot(canvas, 530, top + 40, 410, 130, tick.t - HISTORY_S, tick.t)
        ts = [h[0] for h in self.history]
        asked = [(t, c) for t, c, _, _ in self.history if c is not None]
        if len(asked) > 1:                          # the model, thick, under the motors, thin
            plot.line([t for t, _ in asked], [c[STEER] for _, c in asked], CYAN, 3)
            plot.line([t for t, _ in asked], [c[THROTTLE] for _, c in asked], ORANGE, 3)
        plot.line(ts, [a[STEER] for _, _, a, _ in self.history], WHITE, 1)
        plot.line(ts, [a[THROTTLE] for _, _, a, _ in self.history], YELLOW, 1)
        for t, _, _, over in self.history:
            if over:
                cv2.line(canvas, (plot.x(t), plot.top + plot.height - 6), (plot.x(t), plot.top + plot.height), RED, 1)
        text(canvas, f"last {HISTORY_S:.0f} s. thick: model; thin: motors", 530, top + 190, GREY, 0.42)
        text(canvas, "(white steer, yellow throttle); red: overridden", 530, top + 210, GREY, 0.42)


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
