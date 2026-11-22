"""Record demonstrations in CARLA: the teacher drives random routes through a town and is
told, sentence by sentence, where to turn.

    ./drive.sh carla collect --episodes 40 [--duration-s 60] [--project runs/carla]

Each episode is the same as one from `drive collect` (recorder.py), so validate, browse
and export take it unchanged, plus a sentence per frame:

    episode.npz   t, images (N, 240, 320, 3), states [speed, steer], actions [steer, throttle],
                  tasks (N,): what the driver was told at that frame,
                  poses (N, 3) rear axle x, y, yaw; route_s, lateral_m: where on the route (not trained on)
    labels.npz    the teacher's plan at every frame: 16 actions, 0.1 s apart
    meta.json     the town, the weather, the route and its junctions, the car model and the camera

The recipe is the simulator's (learn/collect.py): disturb the steering that is APPLIED
(DART), so the car wanders and the data shows how to come back, and label every frame
with the teacher's CLEAN plan from where the car is. The push is held within two
standard deviations: a longer streak beyond that (one took a car off a straight in
Town01) teaches nothing the shorter ones do not, and ends the episode. The plan's first action is exactly
what the teacher wanted then; its later ones come from the fitted car model.

`--show` puts every recorded frame in a window as it is recorded, the label's path drawn on
the road and the sentence above it (what `browse` shows afterwards); `--video` writes the
same to an .mp4. Neither changes what is recorded. q in the window stops, as Ctrl-C does;
`--resume` carries on.

An episode ends early on a collision, off its lane or stuck; its last 2 s are dropped
then, since they show a car heading into trouble with labels that pretend otherwise.

Every episode draws from its own generator, seeded `[seed, index]`, so `--resume` after a
server crash records exactly the episodes an unbroken run would have: it skips the ones
already written (their labels.npz is the last file saved) and does the rest.
Every traffic light is held green while recording (`green_lights`): the teacher does
not look at them, and a dataset of a car driving through red lights would teach that.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..learn.label import PLAN_DT_S, PLAN_STEPS
from ..recorder import Recorder
from .camera import CarlaCameraParams
from .identify import OrnsteinUhlenbeck, spawn_on_route
from .model import CarModel
from .route import PHRASES
from .teacher import RouteTeacher

#: Weathers drawn from by default: day and dusk, dry and wet. CARLA's night presets are
#: left out: at -quality-level=Low they are close to black.
WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon", "MidRainyNoon",
            "ClearSunset", "CloudySunset", "WetSunset", "SoftRainSunset")


@dataclass(frozen=True)
class CollectConfig:
    episodes: int = 40
    duration_s: float = 60.0
    speeds: tuple = (4.0, 8.0)              # m/s, each episode's target drawn from this range
    steer_noise: float = 0.06               # strength (standard deviation) of the push on the applied steering
    noise_cap: float = 2.0                  # ... held within this many standard deviations
    speed_noise: float = 0.0                # the speed the car is DRIVEN at, scaled by 1 + this (sd), bends too;
    speed_noise_s: float = 2.0              # ... wandering this slowly. The label keeps the true target
    hz: float = 10.0                        # recorded frames per second
    announce_m: float = 30.0                # a junction's command starts this far before it
    weathers: tuple = WEATHERS
    name: str = "carla"
    seed: int = 0
    off_lane_m: float = 1.2                 # the rear axle this far from the lane centre ends the episode
    stuck_s: float = 4.0
    resume: bool = False                    # keep the episodes already written and record the rest


class Phrasing:
    """One sentence per event: each junction keeps the wording it was given, and so does
    every stretch of `follow` in an episode."""

    def __init__(self, route, rng):
        self.route = route
        self.follow = PHRASES["follow"][int(rng.integers(len(PHRASES["follow"])))]
        self.junction = [PHRASES[j.maneuver][int(rng.integers(len(PHRASES[j.maneuver])))] for j in route.junctions]

    def __call__(self, s: float, announce_m: float) -> str:
        index = self.route.junction_at(s, announce_m)
        return self.follow if index is None else self.junction[index]


@contextmanager
def green_lights(world):
    """Every traffic light green and frozen while recording, and running again after.

    The teacher does not look at traffic lights. Left cycling, the dataset would show the
    car driving through red lights, labelled as the thing to do; a VLA sees the lights."""
    import carla
    for light in world.get_actors().filter("traffic.traffic_light"):
        light.set_state(carla.TrafficLightState.Green)
    world.freeze_all_traffic_lights(True)
    try:
        yield
    finally:
        world.freeze_all_traffic_lights(False)


def collect(session, out: Path, config: CollectConfig, model: CarModel, camera: CarlaCameraParams = CarlaCameraParams(),
            log=print, show: bool = False, video: Path | None = None) -> dict:
    """Write `config.episodes` drives to `out/<name>-NNN/`. Returns what they went through.

    `show` / `video`: every recorded frame, annotated, to a window / an .mp4 as it is taken."""
    import carla
    from ..learn.browse import Geometry, draw
    from ..viewer import Output
    from .routing import random_route, start_waypoint
    out = Path(out)
    written = _written(out, config) if config.resume else set()
    if not config.resume and any(out.glob(f"{config.name}-*")):
        raise ValueError(f"{out} already has episodes named {config.name}-*; choose another --name "
                         "(and another --seed unless you want the same drives again), or --resume "
                         "to finish a run the server died in")
    dt = session.tick_s
    every = int(round(1.0 / (config.hz * dt)))
    if abs(every * dt * config.hz - 1.0) > 1e-9:
        raise ValueError(f"{config.hz} Hz is not a whole number of {dt} s ticks")
    totals = {"episodes": 0, "frames": 0, "ended": {}, "commands": {}, "off_lane_frames": 0}
    started = time.perf_counter()
    geometry = Geometry({"carla": {"camera": camera.to_json(), "vehicle": model.to_json()}})
    output = Output(show, video, fps=config.hz, size=(camera.width * 3, camera.height * 3)) if show or video else None
    with green_lights(session.world), _closing(output):
        for index in range(config.episodes):
            if index in written:
                continue
            rng = np.random.default_rng([config.seed, index])
            weather = config.weathers[int(rng.integers(len(config.weathers)))]
            session.world.set_weather(getattr(carla.WeatherParameters, weather))
            speed = float(rng.uniform(config.speeds[0], config.speeds[-1]))
            start = start_waypoint(session.map, rng)
            route = random_route(start, speed * config.duration_s * 1.2 + 60.0, rng)
            platform = spawn_on_route(session, route, model, start, camera)
            teacher = RouteTeacher(route, model, target_speed_mps=speed, chunk_size=PLAN_STEPS, chunk_dt_s=PLAN_DT_S)
            phrasing = Phrasing(route, rng)
            name = f"{config.name}-{index:03d}"
            recorder = Recorder(out / name, phrasing.follow, config.hz, meta={
                "robot_type": f"carla:{model.blueprint}",
                "carla": {"town": session.town, "weather": weather, "target_speed_mps": round(speed, 2),
                          "steer_noise": config.steer_noise, "speed_noise": config.speed_noise,
                          "announce_m": config.announce_m, "tick_s": dt,
                          "seed": [config.seed, index], "route_m": round(route.length, 1), "traffic_lights": "green, frozen",
                          "junctions": [[round(j.s_entry, 1), round(j.s_exit, 1), j.maneuver] for j in route.junctions],
                          "vehicle": model.to_json(), "camera": camera.to_json()}})
            push = OrnsteinUhlenbeck(config.steer_noise, 0.5, rng)
            pace = OrnsteinUhlenbeck(config.speed_noise, config.speed_noise_s, rng) if config.speed_noise > 0 else None
            applied, ended, still_s, off_lane = np.zeros(2), "duration", 0.0, 0
            while platform.t < config.duration_s - 1e-9:
                state = platform.state
                s, lateral = teacher.locate(state)
                teacher.s = s
                scale = 1.0
                if pace is not None:                   # too fast or too slow, and the label brings it back
                    cap = config.noise_cap * config.speed_noise
                    scale = 1.0 + min(max(pace.step(dt), -cap), cap)
                steer, throttle = teacher.control(state, s, speed_scale=scale)
                if platform.ticks % every == 0:
                    task = phrasing(s, config.announce_m)
                    image, plan = platform.image(), teacher.plan(state, s, platform.t)
                    recorder.add(platform.t, image, state.speed, state.steer, applied, plan, task=task,
                                 extra={"poses": (state.x, state.y, state.yaw), "route_s": s, "lateral_m": lateral})
                    off_lane += abs(lateral) > route.width_m / 2 - model.width_m / 2
                    if output is not None:
                        title = f"{name}  {session.town}  {weather}  t {platform.t:.1f} s"
                        picture = draw(image, (state.speed, state.steer), applied, plan.actions, plan.dt, title,
                                       geometry, task, hint=f"recording   off the lane centre {lateral:+.2f} m   q: stop")
                        if output.put(picture) == "q":
                            raise KeyboardInterrupt("stopped from the window")
                limit = config.noise_cap * config.steer_noise
                applied = np.array([min(max(steer + min(max(push.step(dt), -limit), limit), -1.0), 1.0), throttle])
                if platform.collided():
                    ended = "collision"
                elif abs(lateral) > config.off_lane_m:
                    ended = "off_lane"
                elif teacher.done(s, state.speed):
                    ended = "route_end"
                else:
                    wanted = teacher.speed_target(s, state.speed)
                    still_s = still_s + dt if state.speed < 0.2 and wanted > 1.0 else 0.0
                    ended = "stuck" if still_s > config.stuck_s else ended
                if ended != "duration":
                    break
                platform.apply(applied, dt)
            platform.close()
            if ended in ("collision", "off_lane", "stuck"):
                recorder.truncate(int(round(2.0 * config.hz)))
            kept = len(recorder.t)
            totals["ended"][ended] = totals["ended"].get(ended, 0) + 1
            if kept >= 2 * config.hz:                  # under 2 s is not worth an episode
                recorder.save()
                totals["episodes"] += 1
                totals["frames"] += kept
                totals["off_lane_frames"] += off_lane
                for task in recorder.tasks:
                    command = next(c for c, words in PHRASES.items() if task in words)
                    totals["commands"][command] = totals["commands"].get(command, 0) + 1
            maneuvers = "".join(j.maneuver[0].upper() for j in route.junctions if j.s_entry < s)
            log(f"  {name}: {kept / config.hz:5.1f} s, {weather}, {speed:.1f} m/s, junctions {maneuvers or '-'}"
                + ("" if ended == "duration" else f", ended: {ended}") + ("" if kept >= 2 * config.hz else " (dropped)"))
    totals["seconds"] = round(time.perf_counter() - started, 1)
    totals["skipped"] = len(written)
    return totals


@contextmanager
def _closing(output):
    try:
        yield
    finally:
        if output is not None:
            output.close()


def _written(out: Path, config: CollectConfig) -> set:
    """Indices of the episodes a run with this name and seed has finished; refuses others'."""
    import json
    written = set()
    for directory in sorted(out.glob(f"{config.name}-*")):
        if not (directory / "labels.npz").exists():
            continue                                # cut off while saving: record it again
        seed = json.loads((directory / "meta.json").read_text()).get("carla", {}).get("seed")
        index = int(directory.name.rsplit("-", 1)[-1])
        if seed != [config.seed, index]:
            raise ValueError(f"{directory} was recorded with seed {seed}, not [{config.seed}, {index}]: "
                             "resume with the --seed of the run it belongs to")
        written.add(index)
    return written
