"""Record a drive: what the car saw, how it was moving, and what was sent to the motors."""
from __future__ import annotations

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
