"""Let a fine-tuned VLA drive in CARLA and score it: does it keep its lane, and does it turn
where it is told?

    ./vla.sh -m drive carla drive --town Town01 --checkpoint smolvla-carla-v1 --episodes 10 --seed 100 --show

Each drive is drawn as a `collect` episode is (generator `[seed, index]`): weather, start,
a random route, and the sentence the route gives at each moment. Use seeds the dataset was
not recorded with (collect used 0, 1 and 2): the routes are then new to the model, the town
is not; a town it never saw (Town02) asks more.

The policy sees what it was trained on: the camera's image, `[speed, steer]` and the
sentence. It drives through `DriveLoop` at the server's 20 Hz, each chunk arriving
`latency_s` of simulated time after its observation (`SimulatedWorker`): the server waits
for the model, so a slow PC costs wall time and changes nothing else. A `SafetySupervisor`
sized for a car stands between the chunks and the pedals.

A teacher rides along. It drives nothing: it says where the car is on the route and what
it would do there (its plan is drawn in green under the model's cyan one). With
`--checkpoint teacher` it drives instead, through the same loop, latency and supervisor:
the score a perfect student would get, and the check that the harness is not the problem.

Scored per drive: how far along the route the car got, how far from the lane centre,
whether it hit anything, and every junction it reached, passed (out of the junction and
still on the route) or failed (off the route or hit something on the way through), by
maneuver. A drive ends early on a collision, more than `off_route_m` from the route (a
wrong turn ends there), or stuck.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import LoopConfig, SafetyLimits
from ..control.inference import SimulatedWorker, ThreadedWorker
from ..learn.label import PLAN_DT_S, PLAN_STEPS
from ..loop import DriveLoop
from ..safety import SafetySupervisor
from .camera import CarlaCameraParams
from .collect import WEATHERS, Phrasing, green_lights
from .identify import spawn_on_route
from .model import CarModel
from .route import MANEUVERS
from .teacher import RouteTeacher

#: A car's limits, not the 1:10 car's (SafetyLimits' defaults): the dataset tops out at 8.9 m/s.
CAR_LIMITS = SafetyLimits(max_speed_mps=10.0, stop_margin_m=2.0)
#: Past a junction's exit by this much, the car is out of it.
EXIT_MARGIN_M = 5.0
#: A drive that fails this close before a junction failed that junction.
APPROACH_M = 10.0


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 10
    duration_s: float = 60.0
    seed: int = 100
    speeds: tuple = (4.0, 8.0)              # the riding teacher's target, for its plan; the model picks its own
    latency_s: float = 0.1                  # simulated time from an observation to its chunk
    off_route_m: float = 2.5                # this far from the route ends the drive (a wrong turn does)
    stuck_s: float = 5.0
    announce_m: float = 30.0
    weathers: tuple = WEATHERS
    hz: float = 10.0                        # frames per second in the window and the video
    replan_s: float = 0.0                   # LoopConfig.replan_s: 0 asks for a chunk as soon as one is back
    ensemble: bool = False                  # LoopConfig.ensemble: average the chunks covering a moment
    ensemble_k: float = 0.0
    realtime: bool = False                  # paced to the wall clock, the policy on its own thread: a chunk
                                            # arrives when it is done (latency_s unused), as on the car

    def loop_config(self, dt: float) -> LoopConfig:
        return LoopConfig(control_hz=1.0 / dt, replan_s=self.replan_s, ensemble=self.ensemble,
                          ensemble_k=self.ensemble_k)

    def limits(self) -> SafetyLimits:
        """CAR_LIMITS, a plan allowed to age by the time between asks on top of the latency: asked every
        500 ms, a chunk computed in 190 ms is 0.69 s old before the next, and 0.6 s braked the car hard
        for two or three ticks of every ten in --realtime. (On the simulated clock, at 100 ms, it is 0.55.)"""
        return replace(CAR_LIMITS, max_plan_age_s=CAR_LIMITS.max_plan_age_s + self.replan_s)


@dataclass
class Drive:
    """One drive's setting, drawn as a `collect` episode's is (generator `[seed, index]`)."""
    rng: np.random.Generator
    weather: str
    speed: float
    route: object
    platform: object
    teacher: RouteTeacher
    phrasing: Phrasing


def start_drive(session, seed: int, index: int, speeds: tuple, duration_s: float, weathers: tuple,
                model: CarModel, camera: CarlaCameraParams) -> Drive:
    """Weather, start, route and car for drive `index`, and the teacher riding along."""
    import carla
    from .routing import random_route, start_waypoint
    rng = np.random.default_rng([seed, index])
    weather = weathers[int(rng.integers(len(weathers)))]
    session.world.set_weather(getattr(carla.WeatherParameters, weather))
    speed = float(rng.uniform(speeds[0], speeds[-1]))
    start = start_waypoint(session.map, rng)
    route = random_route(start, speed * duration_s * 1.2 + 60.0, rng)
    platform = spawn_on_route(session, route, model, start, camera)
    teacher = RouteTeacher(route, model, target_speed_mps=speed, chunk_size=PLAN_STEPS, chunk_dt_s=PLAN_DT_S)
    return Drive(rng, weather, speed, route, platform, teacher, Phrasing(route, rng))


def score_junctions(junctions, s_end: float, failed: bool) -> list[tuple[str, bool]]:
    """(maneuver, passed) for the junctions a drive reached.

    Passed: the car got `EXIT_MARGIN_M` past the junction's exit. A drive that failed (off
    the route, a collision, stuck) inside a junction or `APPROACH_M` before it failed that
    junction. A junction the drive ran out of time in or before is not scored."""
    scored = []
    for junction in junctions:
        if s_end >= junction.s_exit + EXIT_MARGIN_M:
            scored.append((junction.maneuver, True))
        elif failed and s_end >= junction.s_entry - APPROACH_M:
            scored.append((junction.maneuver, False))
            break
        else:
            break
    return scored


class TimedPolicy:
    """The policy, with the wall time of each inference kept."""

    def __init__(self, policy):
        self.policy, self.ms = policy, []

    def infer(self, obs):
        started = time.perf_counter()
        chunk = self.policy.infer(obs)
        self.ms.append(1000 * (time.perf_counter() - started))
        return chunk


def evaluate(session, policy, config: EvalConfig, model: CarModel, camera: CarlaCameraParams = CarlaCameraParams(),
             log=print, show: bool = False, video: Path | None = None, name: str = "drive") -> dict:
    """Drive `config.episodes` routes with `policy` (None: the riding teacher); returns the drives and their summary."""
    from ..learn.browse import Geometry, draw
    from ..viewer import Output
    dt = session.tick_s
    every = int(round(1.0 / (config.hz * dt)))
    timed = TimedPolicy(policy)
    geometry = Geometry({"carla": {"camera": camera.to_json(), "vehicle": model.to_json()}})
    output = Output(show, video, fps=config.hz, size=(camera.width * 3, camera.height * 3)) if show or video else None
    drives, started = [], time.perf_counter()
    with green_lights(session.world), _closing(output):
        for index in range(config.episodes):
            d = start_drive(session, config.seed, index, config.speeds, config.duration_s, config.weathers,
                            model, camera)
            weather, route, platform, teacher, phrasing = d.weather, d.route, d.platform, d.teacher, d.phrasing
            timed.policy = teacher if policy is None else policy
            worker = ThreadedWorker(timed) if config.realtime else SimulatedWorker(timed, config.latency_s)
            loop = DriveLoop(platform, worker, SafetySupervisor(config.limits()), config.loop_config(dt))
            title = f"{name}-{index:03d}"
            if config.realtime:     # a thread's first inference sets CUDA up for it (to 0.7 s): not on the clock
                worker.submit(platform.observe(phrasing(0.0, config.announce_m)), platform.t)
                while worker.poll(platform.t) is None:
                    time.sleep(0.005)
            lateral, speeds, ended, still_s, s, frames, behind_s = [], [], "duration", 0.0, 0.0, 0, 0.0
            wall0 = time.perf_counter()
            try:
                while platform.t < config.duration_s - 1e-9:
                    state = platform.state
                    s, offset = teacher.locate(state)
                    teacher.s = s
                    task = phrasing(s, config.announce_m)
                    if platform.collided():
                        ended = "collision"
                    elif abs(offset) > config.off_route_m:
                        ended = "off_route"
                    elif teacher.done(s, state.speed) or s > route.length - teacher.stop_margin - 2.0:
                        ended = "route_end"
                    else:
                        wanted = teacher.speed_target(s, state.speed)
                        still_s = still_s + dt if state.speed < 0.2 and wanted > 1.0 else 0.0
                        ended = "stuck" if still_s > config.stuck_s else ended
                    if ended != "duration":
                        break
                    frames += platform.ticks % every == 0          # the video's frames of this drive
                    if output is not None and platform.ticks % every == 0:
                        key = output.put(_picture(platform, loop, teacher, state, s, offset, task, route, title,
                                                  weather, geometry, draw))
                        if key == "q":
                            raise KeyboardInterrupt("stopped from the window")
                    lateral.append(offset)
                    speeds.append(state.speed)
                    loop.step(task)
                    if config.realtime:                 # wait for the wall clock to catch the simulation up
                        ahead = wall0 + platform.t - time.perf_counter()
                        time.sleep(max(ahead, 0.0))
                        behind_s = max(behind_s, -ahead)
            finally:
                platform.close()
                worker.close()
            failed = ended in ("collision", "off_route", "stuck")
            junctions = score_junctions(route.junctions, s, failed)
            lateral_a = np.abs(np.array(lateral or [0.0]))
            drive = {"name": title, "weather": weather, "route_m": round(route.length, 1), "driven_m": round(s, 1),
                     "seconds": round(platform.t, 1), "frames": frames, "ended": ended, "mean_speed_mps": round(float(np.mean(speeds or [0])), 2),
                     "lateral_rms_m": round(float(np.sqrt(np.mean(lateral_a ** 2))), 3),
                     "lateral_max_m": round(float(lateral_a.max()), 3), "junctions": junctions}
            if config.realtime:
                drive["behind_wall_clock_max_s"] = round(behind_s, 3)
            drives.append(drive)
            marks = " ".join(f"{m[0].upper()}{'+' if ok else 'x'}" for m, ok in junctions) or "-"
            log(f"  {title}: {drive['seconds']:5.1f} s, {drive['driven_m']:5.0f} m, {weather}, "
                f"lateral rms {drive['lateral_rms_m']:.2f} m, junctions {marks}"
                + ("" if ended == "duration" else f", ended: {ended}")
                + (f", at most {1000 * behind_s:.0f} ms behind the clock" if config.realtime else ""))
    return {"drives": drives, "summary": summarise(drives, timed.ms, time.perf_counter() - started),
            "config": {"seed": config.seed, "episodes": config.episodes, "duration_s": config.duration_s,
                       "latency_s": config.latency_s, "off_route_m": config.off_route_m, "town": session.town,
                       "speeds": list(config.speeds), "replan_s": config.replan_s, "ensemble": config.ensemble,
                       "ensemble_k": config.ensemble_k, "realtime": config.realtime}}


def summarise(drives: list[dict], inference_ms: list[float], wall_s: float) -> dict:
    ended: dict[str, int] = {}
    for d in drives:
        ended[d["ended"]] = ended.get(d["ended"], 0) + 1
    by = {m: [ok for d in drives for mm, ok in d["junctions"] if mm == m] for m in MANEUVERS}
    seconds = sum(d["seconds"] for d in drives)
    km = sum(d["driven_m"] for d in drives) / 1000
    failures = sum(ended.get(k, 0) for k in ("collision", "off_route", "stuck"))
    return {
        "drives": len(drives), "ended": ended, "driven_km": round(km, 2), "failures": failures,
        "km_per_failure": round(km / failures, 2) if failures else None,
        "lateral_rms_m": round(float(np.sqrt(sum(d["lateral_rms_m"] ** 2 * d["seconds"] for d in drives)
                                             / max(seconds, 1e-9))), 3),
        "junctions": {m: {"passed": int(sum(v)), "reached": len(v)} for m, v in by.items()},
        "inference_ms_mean": round(float(np.mean(inference_ms)), 1) if inference_ms else None,
        "inference_ms_p90": round(float(np.percentile(inference_ms, 90)), 1) if inference_ms else None,
        "inference_ms_max": round(float(np.max(inference_ms)), 1) if inference_ms else None,
        "wall_s": round(wall_s, 1),
    }


def _picture(platform, loop, teacher, state, s, offset, task, route, title, weather, geometry, draw):
    """The frame with the model's plan (from now on) in cyan and the teacher's in green."""
    plan, action = None, np.zeros(2)
    chunk = loop.executor.chunk
    if chunk is not None:
        start = max(int((platform.t - loop.executor.origin) / chunk.dt), 0)
        plan = chunk.actions[start:] if start < len(chunk.actions) else None
        action = loop.executor.action(platform.t)
        action = np.zeros(2) if action is None else action
    reference = teacher.plan(state, s, platform.t)
    return draw(platform.image(), (state.speed, state.steer), action, plan, reference.dt,
                f"{title}  {weather}  t {platform.t:.1f} s", geometry, task,
                hint=f"{s:.0f} of {route.length:.0f} m   off the route {offset:+.2f} m   q: stop",
                reference=reference.actions, legend="cyan: the model's plan   green: the teacher's")


@contextmanager
def _closing(output):
    try:
        yield
    finally:
        if output is not None:
            output.close()
