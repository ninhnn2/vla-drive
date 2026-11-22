"""DAgger with the riding teacher: the model drives, the teacher labels every frame it gets to.

    ./vla.sh -m drive carla dagger --town Town01 --checkpoint smolvla-carla-v3 --seed 300 \\
        --episodes 20 --speeds 6,6 --show

A model trained on the teacher's drives meets, driving itself, states the teacher never
visited: into a bend late and too fast, on the inside of a turn. This records those states
with what the teacher would do from each (its clean plan, the label `collect` gives every
frame), as episodes in `collect`'s format, to aggregate with the data the model learnt from
and train again (Ross, Gordon and Bagnell 2011, "A reduction of imitation learning ...").

The teacher takes over before the model leaves the lane, as a safety driver would: when the
rear axle, carried on at its heading and speed for `horizon_s`, would be `takeover_m` from
the route's centre. It drives until the car is back within `return_m`, its heading within
`return_rad` of the route's, for `settle_s`, then hands back to a fresh loop. Those frames
are recorded too: they are the recovery the dataset lacked. The takeovers are counted; km
per takeover is the measure of the model, the number a safety driver would give. A
collision, or the car `off_route_m` away despite the teacher, ends the episode, and its last
`drop_s` are dropped as `collect` drops a crash's.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..control.inference import SimulatedWorker
from ..loop import DriveLoop
from ..recorder import Recorder
from ..safety import SafetySupervisor
from ..sim.track import wrap_angle
from .camera import CarlaCameraParams
from .collect import WEATHERS, _closing, green_lights
from .evaluate import EvalConfig, TimedPolicy, start_drive
from .model import CarModel


@dataclass(frozen=True)
class DaggerConfig:
    episodes: int = 20
    duration_s: float = 60.0
    seed: int = 300                         # not the evaluation's 100, nor collect's 0-2
    speeds: tuple = (6.0, 6.0)
    latency_s: float = 0.1
    replan_s: float = 0.0
    ensemble: bool = False
    ensemble_k: float = 0.0
    horizon_s: float = 0.5                  # the takeover looks this far ahead ...
    takeover_m: float = 1.0                 # ... for the rear axle this far from the route's centre
    return_m: float = 0.3                   # handed back once this close ...
    return_rad: float = 0.1                 # ... this straight with the route ...
    settle_s: float = 1.0                   # ... for this long
    off_route_m: float = 2.5
    stuck_s: float = 5.0
    drop_s: float = 2.0
    announce_m: float = 30.0
    weathers: tuple = WEATHERS
    hz: float = 10.0
    name: str = "dagger"

    def eval_config(self) -> EvalConfig:
        return EvalConfig(replan_s=self.replan_s, ensemble=self.ensemble, ensemble_k=self.ensemble_k)


def drifting(lateral: float, heading_error: float, speed: float, horizon_s: float) -> float:
    """Where the rear axle would be, across the route, after `horizon_s` at its heading and speed."""
    return lateral + speed * math.sin(heading_error) * horizon_s


class Handover:
    """Who drives: the model until it is about to leave the lane, then the teacher until it is back."""

    def __init__(self, config: DaggerConfig):
        self.config, self.teacher_driving, self.takeovers, self._settled = config, False, 0, 0.0

    def update(self, lateral: float, heading_error: float, speed: float, dt: float) -> bool:
        """True when the model is to be handed the car back on this tick (a fresh loop)."""
        c = self.config
        if not self.teacher_driving:
            if abs(drifting(lateral, heading_error, speed, c.horizon_s)) > c.takeover_m:
                self.teacher_driving, self._settled = True, 0.0
                self.takeovers += 1
            return False
        back = abs(lateral) < c.return_m and abs(heading_error) < c.return_rad
        self._settled = self._settled + dt if back else 0.0
        if self._settled >= c.settle_s:
            self.teacher_driving = False
            return True
        return False


def dagger(session, policy, config: DaggerConfig, out: Path, model: CarModel,
           camera: CarlaCameraParams = CarlaCameraParams(), log=print, show: bool = False,
           video: Path | None = None, checkpoint: str = "") -> dict:
    """Record `config.episodes` drives of `policy` with the teacher's labels; returns the totals."""
    from ..learn.browse import Geometry, draw
    from ..viewer import Output
    dt = session.tick_s
    every = int(round(1.0 / (config.hz * dt)))
    timed = TimedPolicy(policy)
    geometry = Geometry({"carla": {"camera": camera.to_json(), "vehicle": model.to_json()}})
    output = Output(show, video, fps=config.hz, size=(camera.width * 3, camera.height * 3)) if show or video else None
    totals = {"episodes": 0, "frames": 0, "teacher_frames": 0, "takeovers": 0, "driven_m": 0.0, "ended": {}}
    started = time.perf_counter()
    loop_config = config.eval_config().loop_config(dt)
    with green_lights(session.world), _closing(output):
        for index in range(config.episodes):
            d = start_drive(session, config.seed, index, config.speeds, config.duration_s, config.weathers,
                            model, camera)
            platform, teacher, route = d.platform, d.teacher, d.route
            name = f"{config.name}-{index:03d}"
            recorder = Recorder(out / name, d.phrasing.follow, config.hz, meta={
                "robot_type": f"carla:{model.blueprint}",
                "carla": {"town": session.town, "weather": d.weather, "target_speed_mps": round(d.speed, 2),
                          "steer_noise": 0.0, "speed_noise": 0.0, "announce_m": config.announce_m, "tick_s": dt,
                          "seed": [config.seed, index], "route_m": round(route.length, 1),
                          "traffic_lights": "green, frozen",
                          "junctions": [[round(j.s_entry, 1), round(j.s_exit, 1), j.maneuver] for j in route.junctions],
                          "vehicle": model.to_json(), "camera": camera.to_json()},
                "dagger": {"checkpoint": checkpoint, "latency_s": config.latency_s, "replan_s": config.replan_s,
                           "ensemble": config.ensemble, "horizon_s": config.horizon_s,
                           "takeover_m": config.takeover_m, "return_m": config.return_m}})
            worker = SimulatedWorker(timed, config.latency_s)
            loop = DriveLoop(platform, worker, SafetySupervisor(config.eval_config().limits()), loop_config)
            hand = Handover(config)
            applied, ended, still_s, s = np.zeros(2), "duration", 0.0, 0.0
            try:
                while platform.t < config.duration_s - 1e-9:
                    state = platform.state
                    s, lateral, heading = route.project(state.x, state.y, teacher.s)
                    teacher.s = s
                    error = float(wrap_angle(state.yaw - heading))
                    task = d.phrasing(s, config.announce_m)
                    if platform.collided():
                        ended = "collision"
                    elif abs(lateral) > config.off_route_m:
                        ended = "off_route"
                    elif teacher.done(s, state.speed) or s > route.length - teacher.stop_margin - 2.0:
                        ended = "route_end"
                    else:
                        wanted = teacher.speed_target(s, state.speed)
                        still_s = still_s + dt if state.speed < 0.2 and wanted > 1.0 else 0.0
                        ended = "stuck" if still_s > config.stuck_s else ended
                    if ended != "duration":
                        break
                    if hand.update(lateral, error, state.speed, dt):          # the model's turn again
                        worker.close()
                        worker = SimulatedWorker(timed, config.latency_s)
                        loop = DriveLoop(platform, worker, SafetySupervisor(config.eval_config().limits()), loop_config)
                    if platform.ticks % every == 0:
                        image, plan = platform.image(), teacher.plan(state, s, platform.t)
                        recorder.add(platform.t, image, state.speed, state.steer, applied, plan, task=task,
                                     extra={"poses": (state.x, state.y, state.yaw), "route_s": s,
                                            "lateral_m": lateral, "teacher_driving": float(hand.teacher_driving)})
                        if output is not None:
                            who = f"TEACHER (takeover {hand.takeovers})" if hand.teacher_driving else "model"
                            picture = draw(image, (state.speed, state.steer), applied, None, plan.dt,
                                           f"{name}  {d.weather}  t {platform.t:.1f} s   driving: {who}",
                                           geometry, task, hint=f"{s:.0f} of {route.length:.0f} m   "
                                           f"off the route {lateral:+.2f} m   q: stop",
                                           reference=plan.actions, legend="green: the teacher's plan (the label)")
                            if output.put(picture) == "q":
                                raise KeyboardInterrupt("stopped from the window")
                    if hand.teacher_driving:
                        applied = np.asarray(teacher.control(state, s), dtype=np.float64)
                        platform.apply(applied, dt)
                    else:
                        tick = loop.step(task)
                        applied = tick.decision.action
            finally:
                platform.close()
                worker.close()
            if ended in ("collision", "off_route", "stuck"):
                recorder.truncate(int(round(config.drop_s * config.hz)))
            kept = len(recorder.t)
            teacher_frames = int(sum(recorder.extras.get("teacher_driving", [])))
            totals["ended"][ended] = totals["ended"].get(ended, 0) + 1
            totals["takeovers"] += hand.takeovers
            totals["driven_m"] += s
            if kept >= 2 * config.hz:
                recorder.save()
                totals["episodes"] += 1
                totals["frames"] += kept
                totals["teacher_frames"] += teacher_frames
            log(f"  {name}: {platform.t:5.1f} s, {s:4.0f} m, {d.weather}, {hand.takeovers} takeovers, "
                f"teacher drove {teacher_frames / max(kept, 1):.0%} of {kept} frames"
                + ("" if ended == "duration" else f", ended: {ended}") + ("" if kept >= 2 * config.hz else " (dropped)"))
    km = totals["driven_m"] / 1000
    totals["km_per_takeover"] = round(km / totals["takeovers"], 3) if totals["takeovers"] else None
    totals["driven_km"] = round(km, 2)
    totals["inference_ms_mean"] = round(float(np.mean(timed.ms)), 1) if timed.ms else None
    totals["seconds"] = round(time.perf_counter() - started, 1)
    return totals
