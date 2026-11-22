"""How a CARLA car answers `[steer, throttle]`: the model the teacher plans with.

The teacher's labels are plans rolled out with this model (teacher.py), so a label is as
good as the model's prediction of the real car. `./drive.sh carla identify` fits it to the
car in CARLA and writes `vehicles/<blueprint>.json`; `./drive.sh carla check` measures
how far its predictions land from where the car really goes.

Lateral: the front wheels follow the command with a first-order lag. The command is a
share of the lock CARLA allows at the current speed (`steering_curve`, which shrinks the
lock as the car speeds up), and the path's curvature is tan(wheel angle) / wheelbase.

Longitudinal: the throttle's push grows linearly with the pedal, resistances grow with
speed, and the brake is a table, because CARLA's brake is anything but linear (measured:
0.25 of brake already decelerates a Model 3 at 4-5 m/s^2):

    a = throttle * (t0 + t1 v) - (c0 + c1 v + c2 v^2) - brake_decel(brake)

State `steer` is the front wheels' angle as a share of the standstill lock, so it stays
in [-1, 1] at every speed; a command of 1 at speed asks for less than that.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from ..types import STEER, THROTTLE, VehicleState

VEHICLES = Path(__file__).parent / "vehicles"


@dataclass(frozen=True)
class CarModel:
    blueprint: str = "vehicle.tesla.model3"
    wheelbase_m: float = 3.005
    max_steer_rad: float = math.radians(70.0)
    #: (speed km/h, share of the lock); CARLA's PhysicsControl.steering_curve.
    steering_curve: tuple = ((0.0, 1.0), (20.0, 0.9), (60.0, 0.8), (120.0, 0.7))
    steer_tau_s: float = 0.10
    throttle_accel: tuple = (4.0, 0.0)            # t0, t1
    resistance: tuple = (0.3, 0.0, 0.0)           # c0, c1, c2
    #: (brake pedal, deceleration m/s^2) pairs, increasing, from (0, 0); linear between them.
    brake_table: tuple = ((0.0, 0.0), (1.0, 8.0))
    #: How far the rear axle is behind the actor's origin (CARLA puts the origin mid-car).
    #: The model, the teacher and the camera mount all work from the rear axle.
    rear_axle_m: float = 1.5
    length_m: float = 4.79
    width_m: float = 2.16
    #: How it was fitted: samples, error of the fit; empty for the nominal model.
    fit: dict = field(default_factory=dict)

    # --- the car's answer -------------------------------------------------------------
    def lock_share(self, speed: float) -> float:
        """The share of the standstill lock CARLA allows at `speed` (m/s)."""
        kmh, share = np.array(self.steering_curve, dtype=float).T
        return float(np.interp(speed * 3.6, kmh, share))

    def wheel_angle(self, command: float, speed: float) -> float:
        """The front-wheel angle (rad) a steer command settles at, at this speed."""
        return min(max(command, -1.0), 1.0) * self.lock_share(speed) * self.max_steer_rad

    def accel(self, throttle: float, speed: float) -> float:
        """m/s^2 under drive/'s throttle in [-1, 1] (negative brakes) at `speed`."""
        t0, t1 = self.throttle_accel
        c0, c1, c2 = self.resistance
        gas, brake = max(throttle, 0.0), max(-throttle, 0.0)
        # Resistance and the brake only ever slow the car: never a push forwards (a fit with a
        # negative c0 would have it creep away from standstill), and, standing still, never
        # one backwards (step() keeps a moving car's speed from going below zero).
        drag = max(c0 + c1 * speed + c2 * speed * speed, 0.0)
        a = gas * (t0 + t1 * speed) - self.brake_decel(brake) - drag
        return a if speed > 0.0 else max(a, 0.0)

    def brake_decel(self, brake: float) -> float:
        pedal, decel = np.array(self.brake_table, dtype=float).T
        return float(np.interp(brake, pedal, decel))

    def throttle_for(self, accel: float, speed: float) -> float:
        """The throttle in [-1, 1] that gives `accel` at `speed`: the model, inverted."""
        coast = self.accel(0.0, speed) if speed > 0.0 else 0.0
        t0, t1 = self.throttle_accel
        if accel >= coast:
            return min((accel - coast) / max(t0 + t1 * speed, 1e-3), 1.0)
        pedal, decel = np.array(self.brake_table, dtype=float).T
        return -float(np.interp(coast - accel, decel, pedal))       # past the table's end: full brake

    def step(self, state: VehicleState, action, dt: float) -> VehicleState:
        """Hold `action` for `dt` seconds."""
        command = min(max(float(action[STEER]), -1.0), 1.0)
        throttle = min(max(float(action[THROTTLE]), -1.0), 1.0)
        x, y, yaw, speed = state.x, state.y, state.yaw, state.speed
        wheel = state.steer * self.max_steer_rad
        steps = max(1, math.ceil(dt / 0.025 - 1e-9))
        h = dt / steps
        settle = 1.0 - math.exp(-h / self.steer_tau_s)
        for _ in range(steps):
            wheel += (self.wheel_angle(command, speed) - wheel) * settle
            speed = max(speed + self.accel(throttle, speed) * h, 0.0)
            yaw += speed * math.tan(wheel) / self.wheelbase_m * h
            x += speed * math.cos(yaw) * h
            y += speed * math.sin(yaw) * h
        return VehicleState(x, y, yaw, speed, wheel / self.max_steer_rad)

    # --- files ----------------------------------------------------------------------
    def to_json(self) -> dict:
        data = asdict(self)
        data["max_steer_deg"] = round(math.degrees(data.pop("max_steer_rad")), 3)
        return data

    @classmethod
    def from_json(cls, data: dict) -> "CarModel":
        """A model from its JSON; fields an older version wrote and this one dropped are
        ignored, and fields it did not have yet keep their defaults."""
        data = dict(data)
        data["max_steer_rad"] = math.radians(data.pop("max_steer_deg"))
        known = {f.name for f in fields(cls)}
        data = {key: value for key, value in data.items() if key in known}
        for key in ("steering_curve", "throttle_accel", "resistance", "brake_table"):
            if key in data:
                data[key] = tuple(tuple(p) if isinstance(p, list) else p for p in data[key])
        return cls(**data)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or VEHICLES / f"{self.blueprint}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, blueprint_or_path: str) -> "CarModel":
        """A fitted model by blueprint (vehicles/<blueprint>.json) or file."""
        path = Path(blueprint_or_path)
        if not path.is_file():
            path = VEHICLES / f"{blueprint_or_path}.json"
        if not path.is_file():
            raise ValueError(f"no fitted model for {blueprint_or_path!r} in {VEHICLES}; "
                             "fit one with ./drive.sh carla identify --vehicle <blueprint>")
        return cls.from_json(json.loads(path.read_text()))
