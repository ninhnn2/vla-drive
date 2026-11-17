"""Record a drive: what the car saw, how it was moving, and what was sent to the motors."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .learn.label import write_labels


class Recorder:
    """Samples a drive at `hz` and writes one episode directory.

        episode.npz   t (N,)  images (N, H, W, 3) uint8 RGB  states (N, 2)  actions (N, 2)
        meta.json     the task, the rate, what the columns mean
        labels.npz    only when a teacher gave a plan with every frame (learn/label.py)

    The arrays line up by index: `images[i]` and `states[i]` were seen at `t[i]`, and
    `actions[i]` was on the motors then. That is the content of one LeRobot episode.
    """

    def __init__(self, directory: Path, task: str, hz: float = 10.0):
        self.directory, self.task, self.hz = Path(directory), task, hz
        self._next = 0.0
        self.t, self.images, self.states, self.actions, self.plans = [], [], [], [], []
        self.plan_dt: float | None = None

    def due(self, t: float) -> bool:
        """True when it is time for the next sample."""
        return t >= self._next - 1e-9

    def add(self, t: float, image: np.ndarray, speed: float, steer: float, action, plan=None) -> None:
        self._next += 1.0 / self.hz
        if self._next <= t:                      # after a pause, carry on from now
            self._next = t + 1.0 / self.hz
        self.t.append(t)
        self.images.append(image)
        self.states.append([speed, steer])
        self.actions.append(np.asarray(action, dtype=np.float32))
        if plan is not None:
            self.plans.append(plan.actions)
            self.plan_dt = plan.dt

    def save(self) -> Path:
        if self.plans and len(self.plans) != len(self.t):
            raise ValueError("a plan must come with every frame or with none")
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.directory / "episode.npz", t=np.array(self.t, np.float32),
                            images=np.array(self.images, np.uint8), states=np.array(self.states, np.float32),
                            actions=np.array(self.actions, np.float32))
        (self.directory / "meta.json").write_text(json.dumps({
            "task": self.task, "fps": self.hz, "frames": len(self.t),
            "state": ["speed_mps", "steer"], "action": ["steer", "throttle"],
            "image": "uint8 HxWx3 RGB"}, indent=2) + "\n")
        if self.plans:
            write_labels(self.directory, np.array(self.plans), np.ones(len(self.t), bool), self.plan_dt, "teacher")
        return self.directory


class ActionLog:
    """Every control tick of a drive as one CSV row: what the policy asked for, what went
    to the motors, and why they differ. For debugging a policy after the fact.

        t               seconds, on the platform's clock
        new_chunk       1 when an inference finished this tick
        chunk_t_obs     when the observation behind the chunk in use was taken (empty: none yet)
        plan_age_ms     t - chunk_t_obs
        cmd_steer, cmd_throttle    the chunk's action for t (empty: no plan, or it ran out)
        steer, throttle            what the supervisor sent to the motors
        overrides       the supervisor's reasons, separated by '+' (empty: the command went through)
        speed_mps, wheel_steer     the car's measured state
        lateral_m       distance from the centerline (a simulator only)
    """

    COLUMNS = ("t", "new_chunk", "chunk_t_obs", "plan_age_ms", "cmd_steer", "cmd_throttle", "steer", "throttle",
               "overrides", "speed_mps", "wheel_steer", "lateral_m")

    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)
        self.rows = 0

    def add(self, tick, chunk, speed: float, wheel_steer: float, lateral_m: float | None = None) -> None:
        def number(value, digits=4):
            return "" if value is None else round(float(value), digits)
        command, applied = tick.command, tick.decision.action
        self.writer.writerow([
            number(tick.t, 3), int(tick.new_chunk), number(chunk and chunk.t_obs, 3),
            "" if chunk is None else round(1000 * (tick.t - chunk.t_obs), 1),
            number(None if command is None else command[0]), number(None if command is None else command[1]),
            number(applied[0]), number(applied[1]), "+".join(tick.decision.reasons),
            number(speed), number(wheel_steer), number(lateral_m)])
        self.rows += 1

    def close(self) -> None:
        self.file.close()
