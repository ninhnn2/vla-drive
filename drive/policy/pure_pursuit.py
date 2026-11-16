"""A teacher that knows the map: pure pursuit on the centerline, rolled out into a chunk.

It reads the car's pose, which only the simulator has. Use it to see how the control
loop behaves with a good plan, and to record demonstrations for fine-tuning a VLA.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import VehicleParams
from ..sim.track import Track, wrap_angle
from ..sim.vehicle import step_vehicle
from ..types import ActionChunk, Observation, VehicleState
from .base import Policy


def pursuit_steer(forward_m: float, left_m: float, params: VehicleParams) -> float:
    """Normalized steering that arcs the rear axle through a point in the car's frame."""
    distance = math.hypot(forward_m, left_m)
    if distance < 1e-6:
        return 0.0
    wheel = math.atan2(2.0 * params.wheelbase_m * left_m / distance, distance)
    return min(max(wheel / params.max_steer_rad, -1.0), 1.0)


class PurePursuitPolicy(Policy):
    name = "pure-pursuit"

    def __init__(self, track: Track, vehicle: VehicleParams = VehicleParams(), *, target_speed_mps: float = 1.5,
                 chunk_size: int = 50, chunk_dt_s: float = 1 / 30, lateral_accel_mps2: float = 2.0):
        self.track, self.vehicle = track, vehicle
        self.target_speed = target_speed_mps
        self.chunk_size, self.chunk_dt = chunk_size, chunk_dt_s
        self.lateral_accel = lateral_accel_mps2

    def act(self, state: VehicleState) -> tuple[float, float]:
        s, _, _ = self.track.project(state.x, state.y)
        lookahead = min(max(0.4 + 0.35 * state.speed, 0.5), 1.4)
        tx, ty, _ = self.track.at(s + lookahead)
        dx, dy = float(tx) - state.x, float(ty) - state.y
        bearing = wrap_angle(math.atan2(dy, dx) - state.yaw)
        distance = math.hypot(dx, dy)
        steer = pursuit_steer(distance * math.cos(bearing), distance * math.sin(bearing), self.vehicle)
        # Slow down for the sharpest bend within braking reach.
        bend = self.track.sharpest(s, 1.5)
        speed = min(self.target_speed, math.sqrt(self.lateral_accel / max(bend, 1e-3)))
        return steer, min(max(2.0 * (speed - state.speed) + 0.1 * speed, -1.0), 1.0)

    def infer(self, obs: Observation) -> ActionChunk:
        if obs.pose is None:
            raise ValueError("pure pursuit needs the simulator's pose; a real car has none")
        state = VehicleState(*obs.pose, obs.speed, obs.steer)
        actions = np.empty((self.chunk_size, 2))
        for i in range(self.chunk_size):
            actions[i] = self.act(state)
            state = step_vehicle(state, actions[i], self.vehicle, self.chunk_dt)
        return ActionChunk(actions, self.chunk_dt, obs.t)
