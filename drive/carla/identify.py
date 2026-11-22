"""Fit the car model (model.py) to a car in CARLA, from drives that excite it.

    ./drive.sh carla identify [--vehicle vehicle.tesla.model3] [--minutes 6]

The teacher drives random routes while its speed target jumps every 6-14 s between 2
and 10 m/s, and smooth random pushes (Ornstein-Uhlenbeck) are added to the steering and
the throttle. Everything is logged at the server's tick (20 Hz), and fitted by least
squares:

    steering     w[k+1] - w[k] = alpha * (g * target[k] - w[k])      -> tau, gain g
                 (target = command x the speed's share of the lock; w = the wheels' mean angle)
    curvature    yaw_rate = v * tan(w) / L                           -> effective wheelbase L
    speed        a = gas (t0 + t1 v) - (c0 + c1 v + c2 v^2)            (samples without brake)
    brake        deceleration beyond that drag, per pedal bin          -> a monotonic table

The speed fit is judged by what labels need, not by its R^2: the speed it predicts
1.5 s ahead from the commands that really followed. On the first Model 3 log the
one-step R^2 was 0.44 (the measured acceleration is noisy) while that prediction was off
by 0.19 m/s at the median. A first-order lag on the pedals improved only the 90th
percentile (0.72 -> 0.50 m/s) and would need a state the observation does not carry.
The gain g is folded into the lock (`max_steer_rad`): the mean of two Ackermann wheels
turns less than the inner one, which is what CARLA's lock describes.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from .model import CarModel


class OrnsteinUhlenbeck:
    def __init__(self, strength: float, correlation_s: float, rng):
        self.strength, self.correlation_s, self.rng, self.value = strength, correlation_s, rng, 0.0

    def step(self, dt: float) -> float:
        decay = dt / self.correlation_s
        self.value += -self.value * decay + self.strength * math.sqrt(2 * decay) * self.rng.standard_normal()
        return self.value


def save_log(path, logged: dict, nominal: CarModel):
    """Keep `excite`'s drives (and the nominal model they were driven with) in one .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, columns=np.array(logged["columns"]), dt=logged["dt"],
                        nominal=json.dumps(nominal.to_json()),
                        phases=np.array(logged.get("phases", ["throttle"] * len(logged["drives"]))),
                        **{f"drive_{i:03d}": d for i, d in enumerate(logged["drives"])})
    return path


def load_log(path) -> tuple[dict, CarModel]:
    with np.load(path) as data:
        drives = [data[k] for k in sorted(data.files) if k.startswith("drive_")]
        phases = [str(p) for p in data["phases"]] if "phases" in data.files else ["throttle"] * len(drives)
        logged = {"drives": drives, "phases": phases, "columns": tuple(str(c) for c in data["columns"]),
                  "dt": float(data["dt"])}
        return logged, CarModel.from_json(json.loads(str(data["nominal"])))


def spawn_on_route(session, route, model, start_waypoint, camera):
    """A CarlaPlatform with its rear axle on the start of `route`, settled on the road."""
    import carla
    from .frames import xy_to_carla, yaw_to_carla
    from .platform import CarlaPlatform
    x, y, heading = (float(v) for v in route.at(0.0))
    cx, cy = xy_to_carla(x + model.rear_axle_m * math.cos(heading), y + model.rear_axle_m * math.sin(heading))
    transform = carla.Transform(carla.Location(x=cx, y=cy, z=start_waypoint.transform.location.z + 0.3),
                                carla.Rotation(yaw=yaw_to_carla(heading)))
    platform = CarlaPlatform(session, transform, model, camera)
    for _ in range(int(round(1.0 / session.tick_s))):     # drop onto the road and stand still
        platform.apply([0.0, -1.0], session.tick_s)
    platform.collisions.clear()
    platform.ticks, platform.t = 0, 0.0                    # the drive starts now
    return platform


def excite(session, model: CarModel, seconds: float, rng, log=print, steer_push: float = 0.07,
           throttle_push: float = 0.25, brake_pulses: bool = False, speeds: tuple = (2.0, 10.0),
           change_speed: bool = True, drive_s: float = 90.0) -> dict:
    """Drive random routes for `seconds` of simulated time; return the log as arrays.

    With `brake_pulses` a 1 s brake of 0.05-0.7 is held every 8-15 s instead of the
    teacher's throttle; those drives are for the brake table only (`fit`)."""
    from .camera import CarlaCameraParams
    from .routing import random_route, start_waypoint
    from .teacher import RouteTeacher
    rows, dt, driven = [], session.tick_s, 0.0
    while driven < seconds:
        start = start_waypoint(session.map, rng)
        route = random_route(start, 600.0, rng)
        platform = spawn_on_route(session, route, model, start, CarlaCameraParams(width=64, height=48))
        teacher = RouteTeacher(route, model, target_speed_mps=float(rng.uniform(*speeds)))
        steer_noise, gas_noise = OrnsteinUhlenbeck(steer_push, 0.5, rng), OrnsteinUhlenbeck(throttle_push, 1.0, rng)
        next_change, drive_rows = (float(rng.uniform(6, 14)) if change_speed else math.inf), []
        # Brake pulses at whatever speed the car is doing: the teacher alone brakes almost
        # only when slow, and a brake fitted on that says nothing about 9 m/s.
        next_brake, brake_until, brake = (float(rng.uniform(5, 15)) if brake_pulses else math.inf), -1.0, 0.0
        while platform.t < drive_s and driven + platform.t < seconds:
            if platform.t >= next_change:
                teacher.target_speed, next_change = float(rng.uniform(*speeds)), platform.t + float(rng.uniform(6, 14))
            if platform.t >= next_brake:
                brake, brake_until = float(rng.uniform(0.05, 0.7)), platform.t + 1.0
                next_brake = platform.t + float(rng.uniform(8, 15))
            state = platform.state
            steer, throttle = teacher.act(state)
            throttle = -brake if platform.t < brake_until else min(max(throttle + gas_noise.step(dt), -1.0), 1.0)
            action = (min(max(steer + steer_noise.step(dt), -1.0), 1.0), throttle)
            drive_rows.append((platform.t, state.x, state.y, state.yaw, state.speed, state.steer,
                               platform.yaw_rate(), action[0], action[1]))
            platform.apply(action, dt)
            if platform.collided() or teacher.done(teacher.s, state.speed):
                break
        bumped = platform.collided()
        if bumped:                                  # the last two seconds led into the bump: not the car's own motion
            drive_rows = drive_rows[:max(len(drive_rows) - int(2.0 / dt), 0)]
        driven += platform.t
        rows.append(np.array(drive_rows, dtype=np.float64))
        log(f"  drive {len(rows)}: {platform.t:5.1f} s on a {route.length:.0f} m route"
            + (" (ended in a collision)" if bumped else ""))
        platform.close()
    names = ("t", "x", "y", "yaw", "speed", "steer", "yaw_rate", "cmd_steer", "cmd_throttle")
    phase = "brake" if brake_pulses else "throttle"
    return {"drives": rows, "phases": [phase] * len(rows), "columns": names, "dt": dt}


def identify_drives(session, model: CarModel, seconds: float, rng, log=print, brake_share: float = 0.35) -> dict:
    """`excite` twice, as one log: drives like `collect`'s for the throttle, the drag and the
    steering, then drives with brake pulses for the brake table.

    The throttle phase drives as collect does (its steering disturbance, its speeds, no
    push on the throttle) because CARLA's throttle is not linear, and the line that
    predicts the car best is the one fitted where the teacher drives. Measured, fitted on
    one collect-style log and tested on another: 0.04-0.09 m/s of speed error at 1.5 s,
    against 0.08-0.19 m/s for a fit on drives with the throttle pushed at random."""
    log("  throttle and steering (driven as collect drives):")
    first = excite(session, model, seconds * (1 - brake_share), rng, log, steer_push=0.06, throttle_push=0.0,
                   speeds=(4.0, 8.0), change_speed=False, drive_s=60.0)
    log("  brake pulses:")
    second = excite(session, model, seconds * brake_share, rng, log, brake_pulses=True)
    return {**first, "drives": first["drives"] + second["drives"], "phases": first["phases"] + second["phases"]}


#: Coefficients of the speed fit that physics signs: the throttle's push (t0) and the
#: resistances (c0, c1, c2) are never negative. Unconstrained, the first Model 3 fit gave
#: c0 = -0.20 and c2 = -0.019: a car that creeps forward from standstill with no throttle,
#: and whose drag falls past 12 m/s. Its predictions in the fitted range were fine; the
#: model was wrong where it had no data. (A linear brake in the same fit came out as
#: 8.3 - 0.92 v: no brake at all at 9 m/s, for the same reason.)
NON_NEGATIVE = (0, 2, 3, 4)           # t0, c0, c1, c2 of (t0, t1, c0, c1, c2)
#: Brake pedal bins for the brake table.
BRAKE_BINS = (0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.01)


def _physical_fit(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares with NON_NEGATIVE coefficients >= 0: a coefficient that comes out
    negative is pinned at zero and the others are fitted again, until none does."""
    free = np.ones(features.shape[1], bool)
    while True:
        coef = np.zeros(features.shape[1])
        coef[free], *_ = np.linalg.lstsq(features[:, free], target, rcond=None)
        wrong = [i for i in NON_NEGATIVE if free[i] and coef[i] < 0]
        if not wrong:
            return coef
        free[min(wrong, key=lambda i: coef[i])] = False


def fit(logged: dict, nominal: CarModel, max_delay_ticks: int = 4) -> tuple[CarModel, dict]:
    """The model fitted to `logged` (from `excite`), and how well each part fits."""
    dt = logged["dt"]
    col = {name: i for i, name in enumerate(logged["columns"])}

    phases = logged.get("phases", ["throttle"] * len(logged["drives"]))

    def stacked(build, phase=None):
        """`build` over the drives (of one phase), concatenated."""
        parts = [build(d) for d, p in zip(logged["drives"], phases)
                 if len(d) > max_delay_ticks + 2 and phase in (None, p)]
        return [np.concatenate(p) for p in zip(*parts)]

    # Steering: first-order lag towards g x target, both as shares of the standstill lock.
    def steering(d):
        target = d[:-1, col["cmd_steer"]] * np.array([nominal.lock_share(v) for v in d[:-1, col["speed"]]])
        return target, d[:-1, col["steer"]], np.diff(d[:, col["steer"]])
    target, wheel, change = stacked(steering)
    (a1, a2), *_ = np.linalg.lstsq(np.stack([target, -wheel], axis=1), change, rcond=None)
    alpha, gain = float(a2), float(a1 / a2)
    tau = -dt / math.log(max(1.0 - alpha, 1e-6))
    steer_rms = float(np.sqrt(np.mean((change - (a1 * target - a2 * wheel)) ** 2)))

    # Curvature: yaw rate = v tan(w) / L, with w the measured mean wheel angle.
    def turning(d):
        moving = d[:, col["speed"]] > 1.0
        return (d[moving, col["speed"]] * np.tan(d[moving, col["steer"]] * nominal.max_steer_rad),
                d[moving, col["yaw_rate"]])
    sweep, yaw_rate = stacked(turning)
    inverse_l = float(sweep @ yaw_rate / (sweep @ sweep))
    curvature_rms = float(np.sqrt(np.mean((yaw_rate - inverse_l * sweep) ** 2)))

    # Speed, in two parts. Throttle and resistance from the samples WITHOUT brake: least
    # squares, then again without the samples more than 3 sigma (from the median absolute
    # deviation) off, which are take-offs through CARLA's dead zone and the jolts of a bump.
    # Fitting the brake in the same least squares let the brake pulses bend the throttle and
    # the drag, and made the model worse exactly where the teacher drives most.
    def longitudinal(d):
        v = d[:, col["speed"]]
        command = d[:-1, col["cmd_throttle"]]
        keep = (v[:-1] > 0.5) & (command >= 0.0)
        gas, speed = command[keep], v[:-1][keep]
        features = np.stack([gas, gas * speed, -np.ones_like(speed), -speed, -speed ** 2], 1)
        return features, (np.diff(v) / dt)[keep]
    features, accel = stacked(longitudinal, "throttle")
    coef, *_ = np.linalg.lstsq(features, accel, rcond=None)
    residual = accel - features @ coef
    inliers = np.abs(residual) < 3 * 1.4826 * np.median(np.abs(residual))
    coef = _physical_fit(features[inliers], accel[inliers])
    t0, t1, c0, c1, c2 = (float(c) for c in coef)

    # The brake: its deceleration beyond the fitted drag, MEAN per pedal bin, made monotonic.
    # The first tick of a pulse is left out: CARLA's brake answers a tick late. Not the
    # median: CARLA brakes in jolts (measured at 0.3-0.45 of pedal: median 1.3 m/s^2, 90th
    # percentile 22 m/s^2 in a single tick), and the speed lost over a second is their sum.
    def braking(d):
        v, command = d[:, col["speed"]], d[:-1, col["cmd_throttle"]]
        held = np.r_[False, (command[1:] < 0) & (np.abs(command[1:] - command[:-1]) < 1e-9)]
        keep = held & (v[:-1] > 1.0)
        speed = v[:-1][keep]
        drag = np.maximum(c0 + c1 * speed + c2 * speed ** 2, 0.0)
        return -command[keep], -(np.diff(v) / dt)[keep] - drag
    pedal, decel = stacked(braking, "brake") if "brake" in phases else (np.empty(0), np.empty(0))
    table = [(0.0, 0.0)]
    for low, high in zip(BRAKE_BINS[:-1], BRAKE_BINS[1:]):
        inside = (pedal >= low) & (pedal < high)
        if inside.sum() >= 10:
            table.append((float(np.median(pedal[inside])), max(float(np.mean(decel[inside])), table[-1][1])))
    if len(table) == 1:
        table = list(nominal.brake_table)
    brake_samples = int(len(pedal))

    fitted = replace(nominal, wheelbase_m=1.0 / inverse_l, max_steer_rad=nominal.max_steer_rad * gain,
                     steer_tau_s=tau, throttle_accel=(t0, t1), resistance=(c0, c1, c2),
                     brake_table=tuple((round(p, 3), round(d, 3)) for p, d in table))
    # The number that matters for labels: the speed the model predicts 1.5 s ahead, from the
    # real speed and the commands that really followed, against the speed the car reached.
    horizon = int(round(1.5 / dt))
    errors = []
    for d, phase in zip(logged["drives"], phases):
        if phase != "throttle":
            continue
        v, command = d[:, col["speed"]], d[:, col["cmd_throttle"]]
        for k in range(0, len(d) - horizon, 2):
            if v[k] < 0.5:
                continue
            speed = v[k]
            for j in range(horizon):
                speed = max(speed + fitted.accel(command[k + j], speed) * dt, 0.0)
            errors.append(abs(speed - v[k + horizon]))
    errors = np.array(errors)
    report = {"samples": int(len(change)), "seconds": round(len(change) * dt, 1),
              "steer_tau_s": round(tau, 3), "steer_gain": round(gain, 3), "steer_step_rms": round(steer_rms, 4),
              "wheelbase_m": round(1.0 / inverse_l, 3), "yaw_rate_rms": round(curvature_rms, 4),
              "accel_outliers_pct": round(100 * float(1 - inliers.mean()), 1),
              "pinned_at_zero": [name for i, name in enumerate(("t0", "t1", "c0", "c1", "c2"))
                                 if i in NON_NEGATIVE and coef[i] == 0.0],
              "brake_samples": brake_samples, "brake_table": [list(row) for row in fitted.brake_table],
              "speed_error_1_5s_mps": {"median": round(float(np.median(errors)), 3),
                                       "p90": round(float(np.percentile(errors, 90)), 3)}}
    return replace(fitted, fit=report), report
