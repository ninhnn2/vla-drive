"""The supervisor between the policy and the motors.

It trusts nothing upstream: it sees only the commanded action, the measured speed and
its own range sensor, and it has the last word. Keep it this small on a real car, and
run it where a hung policy process cannot stop it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SafetyLimits
from .types import FULL_BRAKE, STEER, THROTTLE


@dataclass(frozen=True)
class Decision:
    action: np.ndarray               # what goes to the motors
    reasons: tuple[str, ...]         # why it differs from the command; empty if it does not


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits = SafetyLimits()):
        self.limits = limits
        self.estopped = False
        self._steer = 0.0

    def estop(self, engaged: bool = True) -> None:
        self.estopped = engaged

    def filter(self, command, *, speed: float, range_m: float, plan_age_s: float, dt: float) -> Decision:
        """Check one command. `command` is None when the executor has nothing to give."""
        limits = self.limits
        if self.estopped:
            return self._brake("estop")
        if command is None or plan_age_s > limits.max_plan_age_s:
            return self._brake("stale")
        steer, throttle = float(command[STEER]), float(command[THROTTLE])
        if not (math.isfinite(steer) and math.isfinite(throttle)):
            return self._brake("invalid")
        reasons = []
        steer, throttle = min(max(steer, -1.0), 1.0), min(max(throttle, -1.0), 1.0)

        swing = limits.max_steer_rate_per_s * dt
        if abs(steer - self._steer) > swing + 1e-9:
            steer = self._steer + math.copysign(swing, steer - self._steer)
            reasons.append("steer_rate")

        if speed > limits.max_speed_mps:
            throttle = min(throttle, max(-2.0 * (speed - limits.max_speed_mps), FULL_BRAKE))
            reasons.append("speed")

        # The fastest we may go and still stop `stop_margin_m` short of what the sensor sees.
        room = range_m - limits.stop_margin_m
        if room <= 0:
            throttle = FULL_BRAKE
            reasons.append("obstacle_stop")
        else:
            allowed = math.sqrt(2.0 * limits.planned_decel_mps2 * room)
            if speed > allowed:
                throttle = FULL_BRAKE
                reasons.append("obstacle_brake")
            elif speed > 0.8 * allowed and throttle > 0:
                throttle = 0.0
                reasons.append("obstacle_slow")

        self._steer = steer
        return Decision(np.array([steer, throttle]), tuple(reasons))

    def _brake(self, reason: str) -> Decision:
        """Full brake, wheels where they are."""
        return Decision(np.array([self._steer, FULL_BRAKE]), (reason,))
