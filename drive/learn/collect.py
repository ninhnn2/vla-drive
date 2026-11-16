"""Record demonstrations from the pure-pursuit teacher.

A network trained only on perfect driving has never seen the car off the centerline,
so its first small error takes it somewhere it has no answer for, and the error grows.
The fix used here (as in DART, Laskey et al. 2017): disturb the steering that is
*applied*, so the car wanders, and record what the teacher *would* do from each of
those places. The label is always the teacher's clean plan.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..config import LoopConfig
from .label import PLAN_DT_S, PLAN_STEPS
from ..policy.pure_pursuit import PurePursuitPolicy
from ..recorder import Recorder
from ..sim import SHAPES, Track, World
from .project import Project, add_project_argument


class Disturbance:
    """A smooth random push on the steering: an Ornstein-Uhlenbeck process."""

    def __init__(self, strength: float, rng: np.random.Generator, correlation_s: float = 0.5):
        self.strength, self.rng, self.correlation_s = strength, rng, correlation_s
        self.value = 0.0

    def step(self, dt: float) -> float:
        decay = dt / self.correlation_s
        self.value += -self.value * decay + self.strength * math.sqrt(2 * decay) * self.rng.standard_normal()
        return self.value


def collect(out: Path, episodes: int = 40, duration_s: float = 30.0, tracks=("peanut",),
            steer_noise: float = 0.25, name: str = "sim", hz: float = 10.0, seed: int = 0, log=print) -> dict:
    """Write `episodes` drives to `out/<name>-NNN/`. Returns what the car went through."""
    out = Path(out)
    if any(out.glob(f"{name}-*")):
        raise ValueError(f"{out} already has episodes named {name}-*; choose another --name")
    config = LoopConfig()
    worst, frames = 0.0, 0
    for index in range(episodes):
        rng = np.random.default_rng([seed, index])
        # Alternate direction so left and right bends are equally common.
        track_name, reverse = tracks[index % len(tracks)], bool((index // len(tracks)) % 2)
        track = Track.named(track_name, reverse)
        world = World(track, seed=seed + index)
        world.place(rng.uniform(0, track.length), rng.uniform(-0.15, 0.15), rng.uniform(-0.2, 0.2))
        speed = rng.uniform(0.9, 1.8)
        teacher = PurePursuitPolicy(track, world.vehicle, target_speed_mps=speed,
                                    chunk_size=PLAN_STEPS, chunk_dt_s=PLAN_DT_S)
        recorder = Recorder(out / f"{name}-{index:03d}", config.task, hz)
        disturbance, applied = Disturbance(steer_noise, rng), np.zeros(2)
        while world.t < duration_s - 1e-9:
            if recorder.due(world.t):
                obs = world.observe(config.task)
                recorder.add(world.t, obs.images[0], obs.speed, obs.steer, applied, teacher.infer(obs))
            steer, throttle = teacher.act(world.state)
            applied = np.array([min(max(steer + disturbance.step(config.dt), -1.0), 1.0), throttle])
            world.apply(applied, config.dt)
            worst = max(worst, abs(track.project(world.state.x, world.state.y)[1]))
        recorder.save()
        frames += len(recorder.t)
        log(f"  episode {index + 1}/{episodes}: {track_name}{' reversed' if reverse else ''}, {speed:.1f} m/s")
    return {"episodes": episodes, "frames": frames, "max_lateral_m": round(worst, 3)}


def add_parser(sub):
    p = sub.add_parser("collect", help="record the teacher's drives in the simulator, with disturbed steering")
    add_project_argument(p)
    p.add_argument("--name", default="sim", help="episodes are written as <name>-000, <name>-001, ...")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--tracks", type=lambda s: s.split(","), default=["peanut"],
                   help=f"comma-separated, from {', '.join(SHAPES)} (default peanut)")
    p.add_argument("--steer-noise", type=float, default=0.25,
                   help="how hard the applied steering is disturbed; 0 records perfect driving only")
    p.add_argument("--seed", type=int, default=0)


def run(project: Project, args) -> int:
    print(f"recording {args.episodes} drives of {args.duration_s:.0f} s to {project.episodes}")
    done = collect(project.episodes, args.episodes, args.duration_s, tuple(args.tracks), args.steer_noise,
                   args.name, seed=args.seed)
    print(f"{done['frames']} frames in {done['episodes']} episodes; the disturbed car strayed up to "
          f"{100 * done['max_lateral_m']:.0f} cm from the centerline")
    return 0
