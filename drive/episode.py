"""Build a simulated drive from a few settings, run it, and measure how it went."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .config import LoopConfig, SafetyLimits
from .control.inference import SimulatedWorker, ThreadedWorker
from .loop import DriveLoop, Tick
from .policy import POLICIES, make_policy
from .safety import SafetySupervisor
from .sim import Track, World


@dataclass(frozen=True)
class EpisodeConfig:
    policy: str = "pure-pursuit"
    checkpoint: str | None = None      # for the cnn policy: a file from ./drive.sh train
    track: str = "peanut"
    reverse: bool = False              # drive the track clockwise
    latency_ms: float = 100.0          # per inference; a VLA on the Orin Nano is 100-650
    jitter_ms: float = 0.0
    duration_s: float = 30.0
    target_speed_mps: float | None = None
    obstacles_s: tuple[float, ...] = ()   # metres along the track
    latency_compensation: bool = True
    safety: bool = True
    max_plan_age_ms: float = 600.0     # the supervisor's watchdog; see SafetyLimits
    #: Pace to the wall clock and infer on a real thread, as on the car.
    realtime: bool = False
    seed: int = 0


class Metrics:
    """Per-tick measurements only a simulator can make, and their summary."""

    def __init__(self, world: World):
        self.world = world
        self._s = world.track.project(world.state.x, world.state.y)[0]
        self.progress_m = 0.0
        self.lateral, self.speed = [], []
        self.reasons: Counter = Counter()
        self.collided = False
        self.min_range_m = float("inf")

    def add(self, tick: Tick) -> None:
        track, state = self.world.track, self.world.state
        s, lateral, _ = track.project(state.x, state.y)
        self.progress_m += (s - self._s + track.length / 2) % track.length - track.length / 2
        self._s = s
        self.lateral.append(lateral)
        self.speed.append(state.speed)
        self.reasons.update(tick.decision.reasons)
        self.collided |= self.world.collided()
        self.min_range_m = min(self.min_range_m, tick.range_m)

    def summary(self, dt: float) -> dict:
        lateral = np.abs(self.lateral)
        return {
            "duration_s": round(len(lateral) * dt, 2),
            "laps": round(self.progress_m / self.world.track.length, 2),
            "mean_speed_mps": round(float(np.mean(self.speed)), 2),
            "lateral_rms_m": round(float(np.sqrt(np.mean(lateral ** 2))), 3),
            "lateral_max_m": round(float(lateral.max()), 3),
            "off_track_pct": round(100 * float(np.mean(lateral > self.world.track.width_m / 2)), 1),
            "collided": self.collided,
            "min_range_m": round(self.min_range_m, 2),
            # Seconds each safety rule spent overriding the policy.
            "safety_s": {reason: round(count * dt, 2) for reason, count in sorted(self.reasons.items())},
        }


def build(config: EpisodeConfig, loop_config: LoopConfig = LoopConfig()) -> tuple[World, DriveLoop]:
    track = Track.named(config.track, config.reverse)
    world = World(track, tuple(track.obstacle(s) for s in config.obstacles_s), seed=config.seed)
    policy = make_policy(config.policy, track=track, vehicle=world.vehicle, camera=world.camera,
                         checkpoint=config.checkpoint, target_speed_mps=config.target_speed_mps)
    if config.realtime:
        worker = ThreadedWorker(policy, config.latency_ms / 1000)
    else:
        worker = SimulatedWorker(policy, config.latency_ms / 1000, config.jitter_ms / 1000, config.seed)
    loop_config = LoopConfig(**{**asdict(loop_config), "latency_compensation": config.latency_compensation})
    return world, DriveLoop(world, worker, SafetySupervisor(SafetyLimits(max_plan_age_s=config.max_plan_age_ms / 1000)) if config.safety else None, loop_config)


def run(config: EpisodeConfig, on_tick=None) -> dict:
    """Run one episode. `on_tick(world, loop, tick)` may return False to end it early."""
    world, loop = build(config)
    metrics = Metrics(world)
    deadline = time.perf_counter()
    try:
        while world.t < config.duration_s - 1e-9:
            tick = loop.step()
            metrics.add(tick)
            if on_tick is not None and on_tick(world, loop, tick) is False:
                break
            if config.realtime:
                # One tick per dt of wall time. After a stall (a window opening), carry on
                # from now: rushing to catch up would make the model look slower than it is.
                deadline = max(deadline + loop.config.dt, time.perf_counter())
                time.sleep(max(deadline - time.perf_counter(), 0.0))
    finally:
        loop.worker.close()
    latencies = loop.worker.latencies_s
    return {"config": asdict(config), **metrics.summary(loop.config.dt),
            "inferences": len(latencies),
            "latency_ms_mean": round(1000 * float(np.mean(latencies)), 1) if latencies else None}
