"""Record demonstrations from the pure-pursuit teacher.

A network trained only on perfect driving has never seen the car off the centerline,
so its first small error takes it somewhere it has no answer for, and the error grows.
The fix used here (as in DART, Laskey et al. 2017): disturb the steering that is
*applied*, so the car wanders, and record what the teacher *would* do from each of
those places. The label is always the teacher's clean plan.

`--show` watches it happen, in real time; the recording is the same with or without.

DAgger (Ross et al. 2011) goes one step further: `--driver cnn --checkpoint v1` lets a
trained network drive, through the same control loop and supervisor as `sim`, and the
teacher labels every frame of *its* drive. The data then covers the places the learner's
own mistakes take it, which no demonstration does. `--beta` hands the wheel to the teacher
on that share of the ticks, to keep a weak learner near the road.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import LoopConfig, SafetyLimits
from .label import PLAN_DT_S, PLAN_STEPS
from ..policy.pure_pursuit import PurePursuitPolicy
from ..recorder import Recorder
from ..sim import SHAPES, Track, World
from ..types import ActionChunk
from ..viewer import CYAN, DARK, GREY, HEIGHT, PANEL, RED, WHITE, WIDTH, Hud, Output
from .browse import plan_path
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


class Watch:
    """The drive being recorded, in a window or a video.

    Left, the camera with the teacher's plan (the label) drawn on the road; right, the
    map with the plan and, in magenta, where the disturbed car has been; below, how far the
    disturbance pushes the steering away from what the teacher wants.
    """

    def __init__(self, show: bool, video: Path | None, dt: float):
        self.show, self.dt = show, dt
        self.output = Output(show, video, fps=25)
        self.deadline = time.perf_counter()
        self.ticks = 0

    def start(self, world: World, title: str) -> None:
        self.world, self.title, self.hud = world, title, Hud(world)
        self.trail: list[tuple[float, float]] = []
        self.plan = np.empty((0, 2))

    def planned(self, plan: ActionChunk) -> None:
        """The path the plan takes the car on from here, in world coordinates."""
        s = self.world.state
        forward, left = plan_path(plan.actions, plan.dt, s.speed, s.steer, self.world.vehicle).T
        cos, sin = math.cos(s.yaw), math.sin(s.yaw)
        self.plan = np.stack([s.x + cos * forward - sin * left, s.y + sin * forward + cos * left], axis=1)

    def tick(self, wanted: float, applied: float, frames: int) -> bool:
        """After each control tick; False when the viewer asked to stop."""
        s = self.world.state
        self.ticks += 1
        if self.ticks % 5 == 0:
            self.trail.append((s.x, s.y))
        key = ""
        if self.ticks % 2 == 0:                       # 25 frames a second at the 50 Hz tick
            key = self.output.put(self.frame(wanted, applied, frames))
        if self.show:                                  # at the car's pace, not the PC's
            self.deadline = max(self.deadline + self.dt, time.perf_counter())
            time.sleep(max(self.deadline - time.perf_counter(), 0.0))
        return key not in ("q", "\x1b")

    def frame(self, wanted: float, applied: float, frames: int) -> np.ndarray:
        world, s = self.world, self.world.state
        canvas = np.full((HEIGHT, WIDTH, 3), DARK, np.uint8)

        image = world.image()
        scale = PANEL / image.shape[1]
        view = cv2.resize(image, (PANEL, round(image.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        if len(self.plan) > 1:                         # the plan, seen from where the car is now
            cos, sin = math.cos(s.yaw), math.sin(s.yaw)
            dx, dy = self.plan[:, 0] - s.x, self.plan[:, 1] - s.y
            pixels = world.camera.project(cos * dx + sin * dy, -sin * dx + cos * dy) * scale
            pixels = pixels[np.isfinite(pixels).all(axis=1)].round().astype(np.int32)
            if len(pixels) > 1:
                cv2.polylines(view, [pixels], False, CYAN, 2, cv2.LINE_AA)
        canvas[:view.shape[0], :PANEL] = view
        top = self.hud.map_view(world, WHITE, self.plan, self.trail + [(s.x, s.y)])
        canvas[:top.shape[0], PANEL:PANEL + top.shape[1]] = top

        def text(line: str, x: int, y: int, colour=WHITE):
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        lateral = world.track.project(s.x, s.y)[1]
        push = applied - wanted
        text(self.title, PANEL + 10, 322)
        text(f"t {world.t:4.1f} s   speed {s.speed:4.2f} m/s   off centre {100 * lateral:+4.0f} cm", PANEL + 10, 346)
        text(f"recorded {frames} frames", PANEL + 10, 370)
        text(f"steering: teacher wants {wanted:+.2f}   applied {applied:+.2f}   disturbance {push:+.2f}",
             10, 390, RED if abs(push) > 0.2 else WHITE)
        text("cyan: the teacher's plan from here, saved as the label   magenta: the car's path   q: stop",
             10, 420, GREY)
        return canvas

    def close(self) -> None:
        self.output.close()


class Mixer:
    """The world as the learner's control loop sees it, except at the wheels: on a `beta` share
    of the ticks the teacher's action goes there instead, and the disturbance is added on top."""

    def __init__(self, world: World, teacher: PurePursuitPolicy, beta: float, disturbance, rng):
        self.world, self.teacher, self.beta, self.disturbance, self.rng = world, teacher, beta, disturbance, rng
        self.applied = np.zeros(2)

    @property
    def t(self) -> float:
        return self.world.t

    @property
    def state(self):
        return self.world.state

    def observe(self, task: str):
        return self.world.observe(task)

    def range_m(self) -> float:
        return self.world.range_m()

    def apply(self, action, dt: float) -> None:
        action = np.asarray(action, float)
        if self.beta and self.rng.random() < self.beta:
            action = np.array(self.teacher.act(self.world.state))
        self.applied = np.array([min(max(action[0] + self.disturbance.step(dt), -1.0), 1.0), action[1]])
        self.world.apply(self.applied, dt)


def collect(out: Path, episodes: int = 40, duration_s: float = 30.0, tracks=("peanut",),
            steer_noise: float = 0.25, name: str = "sim", hz: float = 10.0, seed: int = 0, log=print,
            show: bool = False, video: Path | None = None, speeds: tuple[float, ...] = (0.9, 1.8),
            driver: Path | None = None, beta: float = 0.0, latency_ms: float = 100.0) -> dict:
    """Write `episodes` drives to `out/<name>-NNN/`. Returns what the car went through.

    `show` opens a window and runs in real time; `video` writes the same view to a file.
    Neither changes what is recorded. Quitting the window keeps the drives so far.

    `speeds` is the range each drive's target speed is drawn from, or one speed for all of
    them. The network never sees the target, only the speed the car is going at.

    `driver`, a CNN checkpoint, drives instead of the teacher (DAgger); the teacher still
    labels every frame. `beta` is the share of ticks the teacher drives instead.
    """
    learner = None
    if driver is not None:
        from ..control.inference import SimulatedWorker
        from ..loop import DriveLoop
        from ..policy.cnn import CnnPolicy
        from ..safety import SafetySupervisor
        learner = CnnPolicy(Path(driver))
    out = Path(out)
    if any(out.glob(f"{name}-*")):
        raise ValueError(f"{out} already has episodes named {name}-*; choose another --name, "
                         f"and another --seed unless you want the same drives again")
    config = LoopConfig()
    worst, frames, saved, stopped, off_road = 0.0, 0, 0, False, 0
    watch = Watch(show, video, config.dt) if show or video else None
    try:
        for index in range(episodes):
            rng = np.random.default_rng([seed, index])
            # Alternate direction so left and right bends are equally common.
            track_name, reverse = tracks[index % len(tracks)], bool((index // len(tracks)) % 2)
            track = Track.named(track_name, reverse)
            world = World(track, seed=seed + index)
            world.place(rng.uniform(0, track.length), rng.uniform(-0.15, 0.15), rng.uniform(-0.2, 0.2))
            speed = rng.uniform(speeds[0], speeds[-1])     # one draw either way: the same noise follows
            teacher = PurePursuitPolicy(track, world.vehicle, target_speed_mps=speed,
                                        chunk_size=PLAN_STEPS, chunk_dt_s=PLAN_DT_S)
            recorder = Recorder(out / f"{name}-{index:03d}", config.task, hz)
            disturbance, applied = Disturbance(steer_noise, rng), np.zeros(2)
            outside = 0                                # frames recorded with the car off the road
            loop = None
            if learner is not None:
                wheel = Mixer(world, teacher, beta, disturbance, np.random.default_rng([seed, index, 1]))
                loop = DriveLoop(wheel, SimulatedWorker(learner, latency_ms / 1000, 0.0, seed + index),
                                 SafetySupervisor(SafetyLimits()), config)
            where = f"{track_name}{' reversed' if reverse else ''}, {speed:.1f} m/s"
            if watch is not None:
                watch.start(world, f"{recorder.directory.name} ({index + 1}/{episodes})  {where}")
            while world.t < duration_s - 1e-9:
                if recorder.due(world.t):
                    obs = world.observe(config.task)
                    plan = teacher.infer(obs)
                    recorder.add(world.t, obs.images[0], obs.speed, obs.steer, applied, plan)
                    outside += abs(track.project(world.state.x, world.state.y)[1]) > track.width_m / 2
                    if watch is not None:
                        watch.planned(plan)
                if loop is None:                           # the teacher drives, disturbed
                    steer, throttle = teacher.act(world.state)
                    applied = np.array([min(max(steer + disturbance.step(config.dt), -1.0), 1.0), throttle])
                    world.apply(applied, config.dt)
                else:                                      # the learner drives; the teacher only labels
                    steer = teacher.act(world.state)[0]
                    loop.step()
                    applied = wheel.applied
                worst = max(worst, abs(track.project(world.state.x, world.state.y)[1]))
                if watch is not None and not watch.tick(steer, applied[0], len(recorder.t)):
                    stopped = True
                    break
            if loop is not None:
                loop.worker.close()
            if len(recorder.t) >= 2:                   # one frame is not an episode
                recorder.save()
                saved += 1
                frames += len(recorder.t)
                off_road += outside
                log(f"  episode {index + 1}/{episodes}: {where}"
                    + (f", stopped after {recorder.t[-1]:.0f} s" if stopped else ""))
            if stopped:
                break
    finally:
        if watch is not None:
            watch.close()
    return {"episodes": saved, "frames": frames, "max_lateral_m": round(worst, 3), "stopped": stopped,
            "off_road_frames": off_road}


def add_parser(sub):
    p = sub.add_parser("collect", help="record the teacher's drives in the simulator, with disturbed steering")
    add_project_argument(p)
    p.add_argument("--name", default="sim", help="episodes are written as <name>-000, <name>-001, ...")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--tracks", type=lambda s: s.split(","), default=["peanut"],
                   help=f"comma-separated, from {', '.join(SHAPES)} (default peanut)")
    p.add_argument("--speeds", type=lambda s: tuple(float(x) for x in s.split(",")), default=(0.9, 1.8),
                   metavar="LOW,HIGH", help="the range each drive's target speed is drawn from, in m/s, or one "
                                            "speed for every drive (default 0.9,1.8)")
    p.add_argument("--steer-noise", type=float,
                   help="how hard the applied steering is disturbed; 0 records perfect driving only "
                        "(default 0.25 when the teacher drives, 0 when a network does)")
    p.add_argument("--driver", choices=("teacher", "cnn"), default="teacher",
                   help="who drives: the teacher, or a trained network (--checkpoint) that the teacher labels: DAgger")
    p.add_argument("--checkpoint", help="with --driver cnn: a trained version (v1) or a checkpoint file")
    p.add_argument("--beta", type=float, default=0.0,
                   help="with --driver cnn: the share of ticks the teacher drives instead (default 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="the same seed records the same drives; change it with --name to get new ones")
    p.add_argument("--show", action="store_true",
                   help="watch the drives being recorded, in real time (30 s each); q stops and keeps them")
    p.add_argument("--video", type=Path, help="write the same view to an .mp4")


def run(project: Project, args) -> int:
    driver = None
    if args.driver == "cnn":
        if not args.checkpoint:
            raise ValueError("--driver cnn needs --checkpoint, a trained version (v1) or a checkpoint file")
        driver = project.checkpoint(args.checkpoint)
    elif args.checkpoint or args.beta:
        raise ValueError("--checkpoint and --beta are for --driver cnn")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError(f"--beta is a share of the ticks, from 0 to 1, not {args.beta}")
    noise = args.steer_noise if args.steer_noise is not None else (0.25 if driver is None else 0.0)
    print(f"recording {args.episodes} drives of {args.duration_s:.0f} s to {project.episodes}"
          + (f", driven by {driver}" + (f" ({100 * args.beta:.0f} % of the ticks by the teacher)" if args.beta else "")
             + "; the teacher labels every frame (DAgger)" if driver else ""))
    done = collect(project.episodes, args.episodes, args.duration_s, tuple(args.tracks), noise,
                   args.name, seed=args.seed, show=args.show, video=args.video, speeds=args.speeds,
                   driver=driver, beta=args.beta)
    print(f"{done['frames']} frames in {done['episodes']} episodes; the {'disturbed ' if driver is None else ''}car "
          f"strayed up to {100 * done['max_lateral_m']:.0f} cm from the centerline")
    share = 100 * done["off_road_frames"] / max(done["frames"], 1)
    print(f"  {done['off_road_frames']} frames ({share:.1f} %) were recorded with the car off the road")
    if share > 5:
        # A few frames off the road teach coming back; many teach driving on grass, where the teacher,
        # who knows the map, still has a plan that the picture cannot explain.
        print(f"  WARN  that is a lot: the disturbance is stronger than the teacher can correct. "
              f"Lower --steer-noise (now {noise:g})" if driver is None else
              "  WARN  that is a lot: the network leaves the road and the frames show grass the teacher's labels "
              "cannot explain. Give the teacher a share of the ticks: --beta 0.3")
    if args.video:
        print(f"  video: {args.video}")
    return 0
