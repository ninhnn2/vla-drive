"""One control tick: observe, hand the policy work, execute its plan, supervise, actuate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LoopConfig
from .control.executor import ChunkExecutor
from .platform import Platform
from .safety import Decision, SafetySupervisor


@dataclass
class Tick:
    t: float
    command: np.ndarray | None       # what the plan asked for
    decision: Decision               # what the supervisor let through
    range_m: float
    new_chunk: bool                  # an inference finished this tick


class DriveLoop:
    """Drives a platform (the simulator, CARLA, a real car) at `config.control_hz`."""

    def __init__(self, platform: Platform, worker, safety: SafetySupervisor | None = None,
                 config: LoopConfig = LoopConfig()):
        self.platform, self.worker, self.config = platform, worker, config
        self.safety = safety
        self.executor = ChunkExecutor(config.blend_s, config.latency_compensation)

    def step(self) -> Tick:
        platform, now = self.platform, self.platform.t
        chunk = self.worker.poll(now)
        if chunk is not None:
            self.executor.update(chunk, now)
        if not self.worker.busy:
            self.worker.submit(platform.observe(self.config.task), now)

        command = self.executor.action(now)
        distance = platform.range_m()
        if self.safety is not None:
            decision = self.safety.filter(command, speed=platform.state.speed, range_m=distance,
                                          plan_age_s=self.executor.age_s(now), dt=self.config.dt)
        else:                         # for experiments only: nothing guards the motors
            decision = Decision(np.zeros(2) if command is None else np.clip(command, -1.0, 1.0), ())
        platform.apply(decision.action, self.config.dt)
        return Tick(now, command, decision, distance, chunk is not None)
