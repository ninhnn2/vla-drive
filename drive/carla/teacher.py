"""The teacher in CARLA: pure pursuit along the route, sized for a real car, and its plans.

The simulator's teacher (`policy/pure_pursuit.py`) is tuned for a 1:10 car at 1-2 m/s:
a 0.5-1.4 m lookahead and a throttle that adds 0.1 x speed. Neither survives a 1.8 t car
at 8 m/s. This one aims at a point 3 m + 0.5 s x speed ahead (3-12 m) from the rear
axle, and no further than half the radius of the sharpest bend within that reach; plans
its speed from the bends within braking reach and the end of the route; and gets its
pedals by inverting the car model, so its plans and its commands come from one model.
It has no integral term: a plan must depend on the state it starts from only.

The lookahead was 4 m + 0.6 s x speed, without the bend limit, in the first Town01
recording: 4 of 40 episodes ended early. Pure pursuit pulls back to the lane in
proportion to 1 / lookahead^2, and at 7.5 m a 2 s streak of the DART push outpulled it
on a straight; and it cut the corners of junctions (0.13-0.18 m to the inside on
average) into a guardrail. A teacher that comes back weakly also teaches that.

`infer` is a label: the next `chunk_size` actions, `chunk_dt_s` apart, rolled out with the
model from the observed pose. How close that rollout lands to the real car is what
`./drive.sh carla check` measures.
"""
from __future__ import annotations

import math

import numpy as np

from ..policy.base import Policy
from ..sim.track import wrap_angle
from ..types import ActionChunk, Observation, VehicleState
from .model import CarModel
from .route import Route


class RouteTeacher(Policy):
    name = "carla-teacher"

    def __init__(self, route: Route, model: CarModel, *, target_speed_mps: float = 7.0, chunk_size: int = 16,
                 chunk_dt_s: float = 0.1, lateral_accel_mps2: float = 2.0, decel_mps2: float = 2.5,
                 accel_mps2: float = 2.0, speed_gain_per_s: float = 0.8, lookahead_m: float = 3.0,
                 lookahead_s: float = 0.5, max_lookahead_m: float = 12.0, bend_lookahead: float = 0.5,
                 stop_margin_m: float = 4.0):
        self.route, self.model = route, model
        self.target_speed = target_speed_mps
        self.chunk_size, self.chunk_dt = chunk_size, chunk_dt_s
        self.lateral_accel, self.decel, self.accel = lateral_accel_mps2, decel_mps2, accel_mps2
        self.speed_gain = speed_gain_per_s
        self.lookahead = (lookahead_m, lookahead_s, max_lookahead_m)
        self.bend_lookahead = bend_lookahead
        self.stop_margin = stop_margin_m
        self.s = 0.0                                # progress: where the car was last projected

    def locate(self, state: VehicleState, near_s: float | None = None) -> tuple[float, float]:
        """(s, lateral offset) of the car on the route, searching near its last progress."""
        s, lateral, _ = self.route.project(state.x, state.y, self.s if near_s is None else near_s)
        return s, lateral

    def speed_target(self, s: float, speed: float, scale: float = 1.0) -> float:
        """What to drive at here: the target, slower for the bends within braking reach, and a stop at the end.
        `scale` drives faster or slower than that, bends included (collect's speed noise); not the stop."""
        reach = speed * speed / (2 * self.decel) + 8.0
        bend = self.route.sharpest(s, reach)
        speed_ref = scale * min(self.target_speed, math.sqrt(self.lateral_accel / max(bend, 1e-4)))
        to_end = self.route.length - s - self.stop_margin
        return min(speed_ref, math.sqrt(2 * self.decel * max(to_end, 0.0)))

    def control(self, state: VehicleState, s: float, speed_scale: float = 1.0) -> tuple[float, float]:
        """[steer, throttle] from the car's state and its place `s` on the route; `speed_scale`: see speed_target."""
        base, per_s, longest = self.lookahead
        ahead = min(max(base + per_s * state.speed, base), longest)
        # In a bend, aim closer: no further than `bend_lookahead` x the bend's radius.
        ahead = max(min(ahead, self.bend_lookahead / max(self.route.sharpest(s, ahead), 1e-4)), base)
        tx, ty, _ = self.route.at(s + ahead)
        dx, dy = float(tx) - state.x, float(ty) - state.y
        bearing = float(wrap_angle(math.atan2(dy, dx) - state.yaw))
        distance = max(math.hypot(dx, dy), 1e-3)
        wheel = math.atan2(2.0 * self.model.wheelbase_m * math.sin(bearing), distance)
        lock = self.model.lock_share(state.speed) * self.model.max_steer_rad
        steer = min(max(wheel / lock, -1.0), 1.0)
        target = self.speed_target(s, state.speed, speed_scale)
        accel = min(max(self.speed_gain * (target - state.speed), -self.decel), self.accel)
        return steer, self.model.throttle_for(accel, state.speed)

    def act(self, state: VehicleState) -> tuple[float, float]:
        """One control tick; advances the teacher's progress along the route."""
        self.s, _ = self.locate(state)
        return self.control(state, self.s)

    def plan(self, state: VehicleState, s: float, t_obs: float) -> ActionChunk:
        """The next `chunk_size` actions from `state`, rolled out with the model."""
        actions = np.empty((self.chunk_size, 2))
        for i in range(self.chunk_size):
            actions[i] = self.control(state, s)
            state = self.model.step(state, actions[i], self.chunk_dt)
            s, _ = self.locate(state, near_s=s)
        return ActionChunk(actions, self.chunk_dt, t_obs)

    def infer(self, obs: Observation) -> ActionChunk:
        if obs.pose is None:
            raise ValueError("the teacher needs the simulator's pose; a real car has none")
        state = VehicleState(*obs.pose, obs.speed, obs.steer)
        s, _ = self.locate(state)
        return self.plan(state, s, obs.t)

    def done(self, s: float, speed: float) -> bool:
        """At the end of the route, stopped."""
        return self.route.length - s < self.stop_margin + 2.0 and speed < 0.3
