"""Kinematic bicycle model with a lagging steering servo."""
from __future__ import annotations

import math

from ..config import VehicleParams
from ..types import STEER, THROTTLE, VehicleState


def step_vehicle(state: VehicleState, action, params: VehicleParams, dt: float) -> VehicleState:
    command = min(max(float(action[STEER]), -1.0), 1.0)
    throttle = min(max(float(action[THROTTLE]), -1.0), 1.0)
    steer = state.steer + (command - state.steer) * (1.0 - math.exp(-dt / params.steer_tau_s))
    accel = throttle * (params.max_accel_mps2 if throttle >= 0 else params.max_brake_mps2)
    speed = min(max(state.speed + (accel - params.drag_per_s * state.speed) * dt, 0.0), params.max_speed_mps)
    yaw = state.yaw + speed / params.wheelbase_m * math.tan(steer * params.max_steer_rad) * dt
    return VehicleState(state.x + speed * math.cos(yaw) * dt, state.y + speed * math.sin(yaw) * dt,
                        yaw, speed, steer)
