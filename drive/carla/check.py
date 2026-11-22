"""How good are the labels? Drive as `collect` does, and measure the car model's predictions.

    ./drive.sh carla check [--minutes 3]

A label is the teacher's plan rolled out with the car model, so the later actions of a
plan are only as right as the model's idea of where they take the car. Every 0.1 s this
takes the car's real state, rolls the model forward with the actions that were REALLY
applied (every tick), and compares with where the car really was 0.5, 1.0 and 1.5 s
later: position (m), heading (deg) and speed (m/s). It also reports how the teacher
drove, under the same steering disturbance as `collect`.
"""
from __future__ import annotations

import math

import numpy as np

from ..types import VehicleState
from .camera import CarlaCameraParams
from .collect import CollectConfig
from .identify import OrnsteinUhlenbeck, spawn_on_route
from .model import CarModel
from .teacher import RouteTeacher

HORIZONS_S = (0.5, 1.0, 1.5)


def drive(session, model: CarModel, config: CollectConfig, seconds: float, rng, log=print) -> list[np.ndarray]:
    """Collect-style drives (no pictures kept) for `seconds`; per tick: t, x, y, yaw, speed, steer, steer cmd, throttle cmd, lateral."""
    from .routing import random_route, start_waypoint
    dt, logs, driven = session.tick_s, [], 0.0
    while driven < seconds:
        speed = float(rng.uniform(config.speeds[0], config.speeds[-1]))
        start = start_waypoint(session.map, rng)
        route = random_route(start, speed * config.duration_s * 1.2 + 60.0, rng)
        platform = spawn_on_route(session, route, model, start, CarlaCameraParams(width=64, height=48))
        teacher = RouteTeacher(route, model, target_speed_mps=speed)
        push, rows, ended = OrnsteinUhlenbeck(config.steer_noise, 0.5, rng), [], ""
        while platform.t < config.duration_s and driven + platform.t < seconds:
            state = platform.state
            s, lateral = teacher.locate(state)
            teacher.s = s
            steer, throttle = teacher.control(state, s)
            limit = config.noise_cap * config.steer_noise
            action = (min(max(steer + min(max(push.step(dt), -limit), limit), -1.0), 1.0), throttle)
            rows.append((platform.t, state.x, state.y, state.yaw, state.speed, state.steer, *action, lateral))
            if platform.collided() or teacher.done(s, state.speed):
                ended = "collision" if platform.collided() else "route end"
                break
            platform.apply(action, dt)
        driven += platform.t
        platform.close()
        logs.append(np.array(rows))
        log(f"  drive {len(logs)}: {platform.t:5.1f} s at {speed:.1f} m/s" + (f", {ended}" if ended else ""))
    return logs


def prediction_errors(logs: list[np.ndarray], model: CarModel, dt: float, every: int = 2) -> dict:
    """Errors of the model's open-loop prediction at each horizon, over every `every`-th tick."""
    out = {h: {"position_m": [], "heading_deg": [], "speed_mps": []} for h in HORIZONS_S}
    longest = int(round(max(HORIZONS_S) / dt))
    for log in logs:
        for k in range(0, len(log) - longest, every):
            row = log[k]
            if row[4] < 0.5:                      # standing still: nothing to predict
                continue
            state = VehicleState(*row[1:6])
            for j in range(1, longest + 1):
                state = model.step(state, log[k + j - 1, 6:8], dt)
                horizon = round(j * dt, 6)
                if horizon in out:
                    real = log[k + j]
                    out[horizon]["position_m"].append(math.hypot(state.x - real[1], state.y - real[2]))
                    out[horizon]["heading_deg"].append(abs(math.degrees(math.remainder(state.yaw - real[3], 2 * math.pi))))
                    out[horizon]["speed_mps"].append(abs(state.speed - real[4]))
    return {f"{h:.1f}s": {name: {"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
                         for name, v in errors.items()} for h, errors in out.items()}


def summary(logs: list[np.ndarray], dt: float) -> dict:
    """How the teacher drove: mean speed and distance from the lane centre while moving."""
    rows = np.concatenate(logs)
    moving = rows[:, 4] > 1.0
    lateral = np.abs(rows[moving, 8])
    return {"seconds": round(len(rows) * dt, 1), "mean_speed_mps": round(float(rows[moving, 4].mean()), 2),
            "lateral_rms_m": round(float(np.sqrt(np.mean(lateral ** 2))), 3),
            "lateral_p99_m": round(float(np.percentile(lateral, 99)), 3)}
